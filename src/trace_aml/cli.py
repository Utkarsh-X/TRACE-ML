"""TRACE-AML CLI entrypoint."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trace_aml.core.config import Settings, load_settings
from trace_aml.core.health import run_health_checks
from trace_aml.core.ids import next_person_id
from trace_aml.core.logger import configure_logger
from trace_aml.core.models import AlertSeverity, HistoryQuery, PersonCategory, PersonLifecycleStatus, PersonRecord
from trace_aml.core.streaming import EventStreamPublisher, InMemoryEventStreamPublisher
from trace_aml.liveness.base import MiniFASNetStub
from trace_aml.pipeline.collect import capture_from_webcam, import_from_directory, person_image_dir
from trace_aml.pipeline.session import RecognitionSession
from trace_aml.pipeline.train import rebuild_embeddings
from trace_aml.quality.gating import decide_person_lifecycle
from trace_aml.recognizers.arcface import ArcFaceRecognizer
from trace_aml.store.analytics import AnalyticsStore
from trace_aml.store.vector_store import VectorStore

app = typer.Typer(help="TRACE-AML v3 tactical CLI", no_args_is_help=True, rich_markup_mode="rich")
person_app = typer.Typer(help="Person registry commands", no_args_is_help=True)
train_app = typer.Typer(help="Embedding build commands", no_args_is_help=True)
recognize_app = typer.Typer(help="Live recognition commands", no_args_is_help=True)
history_app = typer.Typer(help="Detection history commands", no_args_is_help=True)
report_app = typer.Typer(help="Reporting commands", no_args_is_help=True)
export_app = typer.Typer(help="Export commands", no_args_is_help=True)
events_app = typer.Typer(help="Entity-event stream commands", no_args_is_help=True)
alerts_app = typer.Typer(help="Alert stream commands", no_args_is_help=True)
incident_app = typer.Typer(help="Incident commands", no_args_is_help=True)
action_app = typer.Typer(help="Action audit commands", no_args_is_help=True)
service_app = typer.Typer(help="Service layer commands", no_args_is_help=True)

app.add_typer(person_app, name="person")
app.add_typer(train_app, name="train")
app.add_typer(recognize_app, name="recognize")
app.add_typer(history_app, name="history")
app.add_typer(report_app, name="report")
app.add_typer(export_app, name="export")
app.add_typer(events_app, name="events")
app.add_typer(alerts_app, name="alerts")
app.add_typer(incident_app, name="incident")
app.add_typer(action_app, name="action")
app.add_typer(service_app, name="service")

console = Console()


@dataclass
class Runtime:
    settings: Settings
    store: VectorStore
    analytics: AnalyticsStore
    stream_publisher: EventStreamPublisher


def _init_runtime(config: str | None) -> Runtime:
    settings = load_settings(config)
    configure_logger(settings)
    store = VectorStore(settings)
    analytics = AnalyticsStore(store)
    stream_publisher = InMemoryEventStreamPublisher()
    return Runtime(
        settings=settings,
        store=store,
        analytics=analytics,
        stream_publisher=stream_publisher,
    )


def _runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.obj.get("runtime")
    if runtime is None:
        raise typer.Exit(code=2)
    return runtime


def _recognizer(runtime: Runtime) -> ArcFaceRecognizer:
    recog = ArcFaceRecognizer(runtime.settings)
    if runtime.settings.liveness.enabled:
        recog.set_liveness_checker(
            MiniFASNetStub(
                model_path=runtime.settings.liveness.model_path,
                threshold=runtime.settings.liveness.threshold,
            )
        )
    return recog


def _banner() -> Panel:
    return Panel.fit(
        "[bold cyan]TRACE-AML v3[/bold cyan]\n"
        "[dim]Tactical CLI | Quality Core Hardening[/dim]",
        border_style="bright_black",
    )


@app.callback()
def main(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
) -> None:
    """TRACE-AML command root."""
    ctx.ensure_object(dict)
    ctx.obj["runtime"] = _init_runtime(config)


@app.command("doctor")
def doctor(ctx: typer.Context) -> None:
    """Run startup and dependency checks."""
    runtime = _runtime(ctx)
    checks = run_health_checks(runtime.settings)

    console.print(_banner())
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    failures = 0
    for check in checks:
        status_style = "green" if check.status == "OK" else "red"
        if check.status != "OK":
            failures += 1
        table.add_row(check.name, f"[{status_style}]{check.status}[/{status_style}]", check.detail)
    console.print(table)
    if failures:
        raise typer.Exit(code=1)


@person_app.command("add")
def person_add(
    ctx: typer.Context,
    name: str = typer.Option(..., prompt=True),
    category: PersonCategory = typer.Option(PersonCategory.criminal),
    severity: str = typer.Option("", help="Severity, e.g. 1-low / 2-medium / 3-high"),
    dob: str = typer.Option(""),
    gender: str = typer.Option(""),
    city: str = typer.Option(""),
    country: str = typer.Option(""),
    notes: str = typer.Option(""),
    capture_count: int = typer.Option(0, help="Capture N images from webcam"),
    capture_mode: str = typer.Option("auto", help="auto | manual"),
    capture_interval: float = typer.Option(0.35, help="Auto-capture interval seconds"),
    images_dir: str = typer.Option("", help="Import images from directory"),
) -> None:
    """Add a person and ingest images."""
    runtime = _runtime(ctx)
    existing_ids = [p["person_id"] for p in runtime.store.list_persons()]
    person_id = next_person_id(category, existing_ids)
    now = datetime.now(timezone.utc).isoformat()
    record = PersonRecord(
        person_id=person_id,
        name=name,
        category=category,
        severity=severity,
        dob=dob,
        gender=gender,
        last_seen_city=city,
        last_seen_country=country,
        notes=notes,
        lifecycle_state=PersonLifecycleStatus.draft,
        lifecycle_reason="awaiting_training",
        created_at=now,
        updated_at=now,
    )
    runtime.store.add_or_update_person(record)

    ingested = 0
    if capture_count > 0:
        captured = capture_from_webcam(
            runtime.settings,
            person_id,
            capture_count,
            auto=(capture_mode.lower() != "manual"),
            interval_seconds=capture_interval,
        )
        ingested += len(captured)
    if images_dir:
        imported = import_from_directory(runtime.settings, person_id, images_dir)
        ingested += len(imported)

    if ingested == 0:
        runtime.store.set_person_state(
            person_id=person_id,
            lifecycle_state=PersonLifecycleStatus.draft,
            lifecycle_reason="no_images_ingested",
            enrollment_score=0.0,
            valid_embeddings=0,
            valid_images=0,
            total_images=0,
        )

    console.print(
        Panel.fit(
            f"[green]Added[/green] {name} ({person_id})\n"
            f"Category: {category.value} | Images ingested: {ingested}",
            title="person add",
            border_style="cyan",
        )
    )


@person_app.command("list")
def person_list(ctx: typer.Context) -> None:
    """List registered persons with lifecycle state."""
    runtime = _runtime(ctx)
    persons = runtime.store.list_persons()
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("State")
    table.add_column("Emb", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Updated", style="dim")

    for p in persons:
        table.add_row(
            p.get("person_id", ""),
            p.get("name", ""),
            p.get("category", ""),
            p.get("lifecycle_state", "draft"),
            str(p.get("valid_embeddings", 0)),
            f"{float(p.get('enrollment_score', 0.0)):.2f}",
            p.get("updated_at", ""),
        )
    console.print(table)


@person_app.command("update")
def person_update(
    ctx: typer.Context,
    person_id: str = typer.Option(...),
    name: str | None = typer.Option(None),
    severity: str | None = typer.Option(None),
    dob: str | None = typer.Option(None),
    gender: str | None = typer.Option(None),
    city: str | None = typer.Option(None),
    country: str | None = typer.Option(None),
    notes: str | None = typer.Option(None),
) -> None:
    """Update a person record."""
    runtime = _runtime(ctx)
    current = runtime.store.get_person(person_id)
    if not current:
        raise typer.BadParameter(f"Person not found: {person_id}")

    updated = PersonRecord(
        person_id=current["person_id"],
        name=name if name is not None else current.get("name", ""),
        category=PersonCategory(current.get("category", "criminal")),
        severity=severity if severity is not None else current.get("severity", ""),
        dob=dob if dob is not None else current.get("dob", ""),
        gender=gender if gender is not None else current.get("gender", ""),
        last_seen_city=city if city is not None else current.get("last_seen_city", ""),
        last_seen_country=country if country is not None else current.get("last_seen_country", ""),
        notes=notes if notes is not None else current.get("notes", ""),
        lifecycle_state=PersonLifecycleStatus(current.get("lifecycle_state", "draft")),
        lifecycle_reason=current.get("lifecycle_reason", ""),
        enrollment_score=float(current.get("enrollment_score", 0.0)),
        valid_embeddings=int(current.get("valid_embeddings", 0)),
        created_at=current.get("created_at", datetime.now(timezone.utc).isoformat()),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    runtime.store.add_or_update_person(updated)
    console.print(f"[green]Updated[/green] {person_id}")


@person_app.command("capture")
def person_capture(
    ctx: typer.Context,
    person_id: str = typer.Option(..., help="Existing person id to append captures"),
    capture_count: int = typer.Option(20, help="Number of images to capture"),
    capture_mode: str = typer.Option("manual", help="auto | manual"),
    capture_interval: float = typer.Option(0.35, help="Auto-capture interval seconds"),
    rebuild: bool = typer.Option(True, help="Run train rebuild after capture"),
) -> None:
    """Capture additional images for an existing person."""
    runtime = _runtime(ctx)
    person = runtime.store.get_person(person_id)
    if not person:
        raise typer.BadParameter(f"Person not found: {person_id}")

    captured = capture_from_webcam(
        runtime.settings,
        person_id,
        capture_count,
        auto=(capture_mode.lower() != "manual"),
        interval_seconds=capture_interval,
    )
    capture_msg = (
        f"[green]Captured[/green] {len(captured)} images for {person.get('name', person_id)} ({person_id})"
    )
    if rebuild:
        recognizer = _recognizer(runtime)
        stats = rebuild_embeddings(runtime.settings, runtime.store, recognizer)
        capture_msg += (
            f"\nRebuild: embeddings={stats.embeddings_created}, "
            f"active={stats.active_persons}, ready={stats.ready_persons}, blocked={stats.blocked_persons}"
        )
    console.print(Panel.fit(capture_msg, title="person capture", border_style="cyan"))


@person_app.command("audit")
def person_audit(
    ctx: typer.Context,
    apply: bool = typer.Option(False, "--apply", help="Apply recommended lifecycle states"),
) -> None:
    """Audit person quality states and suggest lifecycle corrections."""
    runtime = _runtime(ctx)
    image_root = Path(runtime.settings.store.root) / "person_images"
    persons = runtime.store.list_persons()

    table = Table(title="Person Audit", box=box.SIMPLE_HEAVY)
    table.add_column("Person", style="cyan")
    table.add_column("Current")
    table.add_column("Recommended")
    table.add_column("Emb", justify="right")
    table.add_column("Valid/Total", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Issues")

    changes = 0
    for person in persons:
        person_id = str(person["person_id"])
        person_dir = image_root / person_id
        total_images = len(
            [
                file
                for file in person_dir.iterdir()
                if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        ) if person_dir.exists() else 0
        emb_count = runtime.store.count_embeddings(person_id)
        quality_rows = runtime.store.list_image_quality(person_id)
        valid_images = sum(1 for row in quality_rows if bool(row.get("passed", False)))
        avg_quality = (
            sum(float(row.get("quality_score", 0.0)) for row in quality_rows if bool(row.get("passed", False)))
            / max(1, valid_images)
        ) if valid_images else 0.0

        decision = decide_person_lifecycle(
            settings=runtime.settings,
            total_images=total_images,
            valid_images=valid_images,
            embeddings_count=emb_count,
            avg_quality=avg_quality,
        )
        current_state = str(person.get("lifecycle_state", "draft"))
        issues: list[str] = []
        if total_images == 0:
            issues.append("no_images")
        if emb_count == 0:
            issues.append("zero_embeddings")
        if valid_images < runtime.settings.quality.min_valid_images:
            issues.append("few_valid_images")
        if avg_quality < runtime.settings.quality.min_quality_score:
            issues.append("low_quality")
        issue_text = ",".join(issues) if issues else "none"

        if apply and current_state != decision.state.value:
            runtime.store.set_person_state(
                person_id=person_id,
                lifecycle_state=decision.state,
                lifecycle_reason=decision.reason,
                enrollment_score=decision.enrollment_score,
                valid_embeddings=emb_count,
                valid_images=valid_images,
                total_images=total_images,
            )
            changes += 1

        table.add_row(
            person_id,
            current_state,
            decision.state.value,
            str(emb_count),
            f"{valid_images}/{total_images}",
            f"{avg_quality:.2f}",
            issue_text,
        )
    console.print(table)
    if apply:
        console.print(f"[cyan]Applied state updates:[/cyan] {changes}")


@person_app.command("delete")
def person_delete(
    ctx: typer.Context,
    person_id: str = typer.Option(...),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Delete person and linked embeddings/detections."""
    runtime = _runtime(ctx)
    if not yes:
        confirmed = typer.confirm(f"Delete {person_id}? This removes embeddings and detections.")
        if not confirmed:
            raise typer.Exit()
    runtime.store.delete_person(person_id, delete_detections=True)

    img_dir = person_image_dir(runtime.settings, person_id)
    if img_dir.exists():
        shutil.rmtree(img_dir)
    console.print(f"[red]Deleted[/red] {person_id}")


@train_app.command("rebuild")
def train_rebuild(ctx: typer.Context) -> None:
    """Recompute embeddings and enrollment lifecycle states."""
    runtime = _runtime(ctx)
    recognizer = _recognizer(runtime)
    stats = rebuild_embeddings(runtime.settings, runtime.store, recognizer)
    table = Table(title="Embedding Rebuild", box=box.SIMPLE_HEAVY)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Persons total", str(stats.persons_total))
    table.add_row("Persons processed", str(stats.persons_processed))
    table.add_row("Embeddings created", str(stats.embeddings_created))
    table.add_row("Skipped images", str(stats.skipped_images))
    table.add_row("Active persons", str(stats.active_persons))
    table.add_row("Ready persons", str(stats.ready_persons))
    table.add_row("Blocked persons", str(stats.blocked_persons))
    console.print(table)


@recognize_app.command("live")
def recognize_live(ctx: typer.Context) -> None:
    """Run live recognition from laptop webcam."""
    runtime = _runtime(ctx)
    recognizer = _recognizer(runtime)
    session = RecognitionSession(
        runtime.settings,
        runtime.store,
        recognizer,
        stream_publisher=runtime.stream_publisher,
    )
    console.print(_banner())
    console.print("[dim]Press q on the OpenCV window to stop.[/dim]")
    session.run()


@history_app.command("query")
def history_query(
    ctx: typer.Context,
    start: str = typer.Option("", help="ISO UTC lower bound"),
    end: str = typer.Option("", help="ISO UTC upper bound"),
    person_id: str = typer.Option("", help="Filter by person id"),
    category: str = typer.Option("", help="Filter by category"),
    decision_state: str = typer.Option("", help="accept|review|reject"),
    min_confidence: float = typer.Option(0.0),
    limit: int = typer.Option(50),
) -> None:
    """Query detection history."""
    runtime = _runtime(ctx)
    query = HistoryQuery(
        start_ts=start or None,
        end_ts=end or None,
        person_id=person_id or None,
        category=category or None,
        decision_state=decision_state or None,
        min_confidence=min_confidence,
        limit=limit,
    )
    rows = runtime.analytics.history(query)
    table = Table(title="Detection History", box=box.SIMPLE_HEAVY)
    table.add_column("Timestamp", style="dim")
    table.add_column("Name")
    table.add_column("ID", style="cyan")
    table.add_column("Decision")
    table.add_column("Conf", justify="right")
    table.add_column("Smooth", justify="right")
    table.add_column("Track")
    for row in rows:
        table.add_row(
            str(row["timestamp_utc"]),
            str(row["name"]),
            str(row["person_id"]),
            str(row["decision_state"]),
            f"{float(row['confidence']):.2f}",
            f"{float(row['smoothed_confidence']):.2f}",
            str(row["track_id"]),
        )
    console.print(table)


@report_app.command("summary")
def report_summary(ctx: typer.Context) -> None:
    """Show aggregated session summary."""
    runtime = _runtime(ctx)
    summary = runtime.analytics.summary()
    table = Table(title="Top Detections", box=box.SIMPLE_HEAVY)
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Hits", justify="right")
    table.add_column("Avg confidence", justify="right")
    for row in summary.top_persons:
        table.add_row(
            str(row["name"]),
            str(row["category"]),
            str(row["hits"]),
            f"{row['avg_confidence']:.2f}",
        )

    decision_dist = ", ".join(
        [f"{k}:{v}" for k, v in sorted(summary.decision_distribution.items())]
    ) or "none"
    console.print(
        Panel.fit(
            f"Total detections: [bold]{summary.total_detections}[/bold]\n"
            f"Unique persons: [bold]{summary.unique_persons}[/bold]\n"
            f"Average confidence: [bold]{summary.avg_confidence:.2f}[/bold]\n"
            f"Decision distribution: [bold]{decision_dist}[/bold]\n"
            f"Blocked persons: [bold]{summary.blocked_persons}[/bold] | "
            f"Low-quality persons: [bold]{summary.low_quality_persons}[/bold]",
            title="Session Summary",
            border_style="cyan",
        )
    )
    console.print(table)


@report_app.command("quality")
def report_quality(ctx: typer.Context) -> None:
    """Show quality-focused enrollment and threshold diagnostics."""
    runtime = _runtime(ctx)
    low_quality = runtime.analytics.low_quality_enrollments(limit=30)
    impact = runtime.analytics.threshold_impact()

    table = Table(title="Low-quality / Non-active Enrollments", box=box.SIMPLE_HEAVY)
    table.add_column("Person", style="cyan")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Score", justify="right")
    table.add_column("Emb", justify="right")
    table.add_column("Valid/Total", justify="right")
    table.add_column("Reason")
    for row in low_quality:
        table.add_row(
            str(row["person_id"]),
            str(row["name"]),
            str(row["state"]),
            f"{float(row['enrollment_score']):.2f}",
            str(row["valid_embeddings"]),
            f"{row['valid_images']}/{row['total_images']}",
            str(row["reason"]),
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"Threshold impact bands\n"
            f"accept_band: {impact['accept_band']}\n"
            f"review_band: {impact['review_band']}\n"
            f"reject_band: {impact['reject_band']}",
            border_style="bright_black",
        )
    )


@export_app.command("csv")
def export_csv(
    ctx: typer.Context,
    output: str = typer.Option("", help="Output csv path"),
    start: str = typer.Option(""),
    end: str = typer.Option(""),
    person_id: str = typer.Option(""),
    category: str = typer.Option(""),
    decision_state: str = typer.Option(""),
    min_confidence: float = typer.Option(0.0),
    limit: int = typer.Option(10_000),
) -> None:
    """Export filtered detections to CSV."""
    runtime = _runtime(ctx)
    query = HistoryQuery(
        start_ts=start or None,
        end_ts=end or None,
        person_id=person_id or None,
        category=category or None,
        decision_state=decision_state or None,
        min_confidence=min_confidence,
        limit=limit,
    )
    output_path = (
        Path(output)
        if output
        else Path(runtime.settings.store.exports_dir)
        / f"detections_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    saved = runtime.analytics.export_csv(query, output_path)
    console.print(f"[green]Exported[/green] {saved}")


@events_app.command("tail")
def events_tail(
    ctx: typer.Context,
    limit: int = typer.Option(30, help="Number of latest events"),
    entity_id: str = typer.Option("", help="Filter by entity id"),
) -> None:
    """Show latest entity-linked event stream."""
    runtime = _runtime(ctx)
    rows = runtime.store.list_events(limit=limit, entity_id=entity_id or None)
    table = Table(title="Entity Event Stream", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim")
    table.add_column("Entity", style="cyan")
    table.add_column("Decision")
    table.add_column("Conf", justify="right")
    table.add_column("Track")
    table.add_column("Source", style="dim")
    for row in rows:
        table.add_row(
            str(row.get("timestamp_utc", "")),
            str(row.get("entity_id", "")),
            str(row.get("decision", "")),
            f"{float(row.get('confidence', 0.0)):.2f}",
            str(row.get("track_id", "")),
            str(row.get("source", "")),
        )
    console.print(table)


@alerts_app.command("tail")
def alerts_tail(
    ctx: typer.Context,
    limit: int = typer.Option(30, help="Number of latest alerts"),
    entity_id: str = typer.Option("", help="Filter by entity id"),
    severity: str = typer.Option("", help="Filter by severity low|medium|high"),
) -> None:
    """Show latest rule-generated alerts."""
    runtime = _runtime(ctx)
    rows = runtime.store.list_alerts(
        limit=limit,
        entity_id=entity_id or None,
        severity=severity or None,
    )
    table = Table(title="Alert Stream", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim")
    table.add_column("Severity")
    table.add_column("Entity", style="cyan")
    table.add_column("Type")
    table.add_column("Events", justify="right")
    table.add_column("Reason")
    for row in rows:
        level = str(row.get("severity", "")).upper()
        if level == "HIGH":
            sev = f"[red]{level}[/red]"
        elif level == "MEDIUM":
            sev = f"[yellow]{level}[/yellow]"
        else:
            sev = f"[green]{level}[/green]"
        table.add_row(
            str(row.get("timestamp_utc", "")),
            sev,
            str(row.get("entity_id", "")),
            str(row.get("type", "")),
            str(int(row.get("event_count", 1))),
            str(row.get("reason", "")),
        )
    console.print(table)


@incident_app.command("list")
def incident_list(
    ctx: typer.Context,
    status: str = typer.Option("", help="Filter by open|closed"),
    limit: int = typer.Option(50, help="Maximum incidents"),
) -> None:
    """List incidents for operator review."""
    runtime = _runtime(ctx)
    rows = runtime.store.list_incidents(limit=limit, status=status or None)
    table = Table(title="Incidents", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="cyan")
    table.add_column("Entity")
    table.add_column("Status")
    table.add_column("Severity")
    table.add_column("Alerts", justify="right")
    table.add_column("Start", style="dim")
    table.add_column("Last Seen", style="dim")
    table.add_column("Last Action", style="dim")
    for row in rows:
        table.add_row(
            str(row.get("incident_id", "")),
            str(row.get("entity_id", "")),
            str(row.get("status", "")),
            str(row.get("severity", "low")),
            str(int(row.get("alert_count", 0))),
            str(row.get("start_time", "")),
            str(row.get("last_seen_time", "")),
            str(row.get("last_action_at", "")),
        )
    console.print(table)


@incident_app.command("show")
def incident_show(
    ctx: typer.Context,
    id: str = typer.Option(..., "--id", help="Incident id"),
) -> None:
    """Show details and timeline for one incident."""
    runtime = _runtime(ctx)
    incident = runtime.store.get_incident(id)
    if not incident:
        raise typer.BadParameter(f"Incident not found: {id}")

    incident_alert_ids = set(str(v) for v in incident.get("alert_ids", []))
    alerts = runtime.store.list_alerts(limit=10_000, entity_id=str(incident.get("entity_id", "")))
    timeline = [row for row in alerts if str(row.get("alert_id", "")) in incident_alert_ids]
    timeline.sort(key=lambda r: str(r.get("timestamp_utc", "")))

    console.print(
        Panel.fit(
            f"Incident: [bold]{incident.get('incident_id','')}[/bold]\n"
            f"Entity: [bold]{incident.get('entity_id','')}[/bold]\n"
            f"Status: [bold]{incident.get('status','')}[/bold]\n"
            f"Severity: [bold]{incident.get('severity','low')}[/bold]\n"
            f"Start: {incident.get('start_time','')}\n"
            f"Last seen: {incident.get('last_seen_time','')}\n"
            f"Last action: {incident.get('last_action_at','')}\n"
            f"Alerts linked: {incident.get('alert_count', 0)}",
            title="Incident Detail",
            border_style="cyan",
        )
    )

    table = Table(title="Incident Timeline", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim")
    table.add_column("Alert ID", style="cyan")
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Reason")
    for row in timeline:
        table.add_row(
            str(row.get("timestamp_utc", "")),
            str(row.get("alert_id", "")),
            str(row.get("type", "")),
            str(row.get("severity", "")),
            str(row.get("reason", "")),
        )
    console.print(table)


@incident_app.command("close")
def incident_close(
    ctx: typer.Context,
    id: str = typer.Option(..., "--id", help="Incident id"),
) -> None:
    """Close an incident manually."""
    runtime = _runtime(ctx)
    ok = runtime.store.close_incident(id)
    if not ok:
        raise typer.BadParameter(f"Incident not found: {id}")
    console.print(f"[green]Closed incident[/green] {id}")


@incident_app.command("set-severity")
def incident_set_severity(
    ctx: typer.Context,
    id: str = typer.Option(..., "--id", help="Incident id"),
    severity: AlertSeverity = typer.Option(..., "--severity", help="low|medium|high"),
) -> None:
    """Set incident severity manually (operator control)."""
    runtime = _runtime(ctx)
    ok = runtime.store.set_incident_severity(id, severity.value)
    if not ok:
        raise typer.BadParameter(f"Incident not found: {id}")
    console.print(f"[green]Updated severity[/green] {id} -> {severity.value}")


@action_app.command("list")
def action_list(
    ctx: typer.Context,
    incident_id: str = typer.Option(..., "--incident-id", help="Incident id"),
    limit: int = typer.Option(50, help="Maximum actions"),
) -> None:
    """List action audit records for an incident."""
    runtime = _runtime(ctx)
    rows = runtime.store.get_actions(incident_id, limit=limit)
    table = Table(title=f"Actions for {incident_id}", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim")
    table.add_column("Action ID", style="cyan")
    table.add_column("Type")
    table.add_column("Trigger")
    table.add_column("Status")
    table.add_column("Reason")
    for row in rows:
        table.add_row(
            str(row.get("timestamp_utc", "")),
            str(row.get("action_id", "")),
            str(row.get("action_type", "")),
            str(row.get("trigger", "")),
            str(row.get("status", "")),
            str(row.get("reason", "")),
        )
    console.print(table)


@service_app.command("run")
def service_run(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
    live: bool = typer.Option(
        False,
        "--live",
        help="Run webcam recognition in-process (same SSE stream as the API; do not run 'recognize live' separately).",
    ),
) -> None:
    """Run TRACE-AML service API for UI integration."""
    runtime = _runtime(ctx)
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "uvicorn is not installed. Install service deps: pip install fastapi uvicorn"
        ) from exc

    from trace_aml.service.app import create_service_app

    api = create_service_app(
        settings=runtime.settings,
        store=runtime.store,
        stream_publisher=runtime.stream_publisher,
    )
    if live:
        import threading

        from trace_aml.pipeline.session import RecognitionSession

        def _live_worker() -> None:
            try:
                recognizer = _recognizer(runtime)
                session = RecognitionSession(
                    runtime.settings,
                    runtime.store,
                    recognizer,
                    stream_publisher=runtime.stream_publisher,
                )
                session.run_headless()
            except Exception as exc:  # pragma: no cover - hardware-dependent
                console.print(Panel.fit(f"[red]Live recognition stopped: {exc}[/red]", title="service run --live"))

        threading.Thread(target=_live_worker, name="trace-aml-live-recognition", daemon=True).start()
        console.print(
            Panel.fit(
                "[yellow]Live recognition thread started (webcam index 0).[/yellow]\n"
                "Do not run [bold]recognize live[/bold] in another terminal (camera conflict).",
                title="service run --live",
                border_style="yellow",
            )
        )
    console.print(
        Panel.fit(
            f"[cyan]Service starting[/cyan]\n"
            f"http://{host}:{port}",
            title="service run",
            border_style="cyan",
        )
    )
    uvicorn.run(api, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
