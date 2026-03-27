from pathlib import Path

from typer.testing import CliRunner

from trace_ml.cli import app
from trace_ml.core.config import load_settings
from trace_ml.core.models import (
    ActionRecord,
    ActionStatus,
    ActionTrigger,
    ActionType,
    AlertRecord,
    AlertSeverity,
    AlertType,
    DecisionState,
    EventRecord,
    IncidentRecord,
    IncidentStatus,
    PersonCategory,
    PersonLifecycleStatus,
    PersonRecord,
)
from trace_ml.store.vector_store import VectorStore


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
camera:
  device_index: 0
store:
  root: {tmp_path.as_posix()}/data
  vectors_dir: {tmp_path.as_posix()}/data/vectors
  screenshots_dir: {tmp_path.as_posix()}/data/screens
  exports_dir: {tmp_path.as_posix()}/data/exports
logging:
  file_path: {tmp_path.as_posix()}/data/logs/trace_ml.log
""".strip(),
        encoding="utf-8",
    )
    return cfg


def test_cli_person_add_list_and_report(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runner = CliRunner()

    add = runner.invoke(
        app,
        ["--config", str(cfg), "person", "add", "--name", "Alice", "--category", "criminal"],
    )
    assert add.exit_code == 0

    listing = runner.invoke(app, ["--config", str(cfg), "person", "list"])
    assert listing.exit_code == 0
    assert "PRC001" in listing.stdout

    summary = runner.invoke(app, ["--config", str(cfg), "report", "summary"])
    assert summary.exit_code == 0

    export = runner.invoke(app, ["--config", str(cfg), "export", "csv"])
    assert export.exit_code == 0


def test_cli_person_help_includes_capture(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg), "person", "--help"])
    assert result.exit_code == 0
    assert "capture" in result.stdout


def test_cli_person_audit_apply_and_quality_report(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.add_or_update_person(
        PersonRecord(
            person_id="PRC001",
            name="Needs Audit",
            category=PersonCategory.criminal,
        )
    )
    # Intentionally set an invalid active state for a person with no images/embeddings.
    store.set_person_state(
        person_id="PRC001",
        lifecycle_state=PersonLifecycleStatus.active,
        lifecycle_reason="forced_bad_state",
        enrollment_score=0.95,
        valid_embeddings=20,
        valid_images=20,
        total_images=20,
    )

    runner = CliRunner()
    audit = runner.invoke(app, ["--config", str(cfg), "person", "audit", "--apply"])
    assert audit.exit_code == 0
    assert "Applied state updates:" in audit.stdout

    listing = runner.invoke(app, ["--config", str(cfg), "person", "list"])
    assert listing.exit_code == 0
    assert "draft" in listing.stdout

    quality = runner.invoke(app, ["--config", str(cfg), "report", "quality"])
    assert quality.exit_code == 0
    assert "Threshold impact bands" in quality.stdout


def test_cli_events_tail(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.add_event(
        EventRecord(
            event_id="EVT-1",
            entity_id="UNK001",
            confidence=42.0,
            decision=DecisionState.review,
            track_id="T0001",
            is_unknown=True,
            source="webcam:0",
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg), "events", "tail", "--limit", "10"])
    assert result.exit_code == 0
    assert "UNK001" in result.stdout


def test_cli_alerts_tail(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.add_alert(
        AlertRecord(
            alert_id="ALT-1",
            entity_id="UNK001",
            type=AlertType.unknown_recurrence,
            severity=AlertSeverity.high,
            reason="UNKNOWN_RECURRENCE detected with 3 events",
            event_count=3,
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg), "alerts", "tail", "--limit", "10"])
    assert result.exit_code == 0
    assert "UNK001" in result.stdout
    assert "HIGH" in result.stdout
    assert "3 events" in result.stdout


def test_cli_incident_list_show_close(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.add_alert(
        AlertRecord(
            alert_id="ALT-9",
            entity_id="UNK009",
            type=AlertType.unknown_recurrence,
            severity=AlertSeverity.high,
            reason="UNKNOWN_RECURRENCE detected with 5 events",
            event_count=5,
        )
    )
    store.create_incident(
        IncidentRecord(
            incident_id="INC-9",
            entity_id="UNK009",
            status=IncidentStatus.open,
            start_time="2026-03-27T00:00:00+00:00",
            last_seen_time="2026-03-27T00:00:10+00:00",
            alert_ids=["ALT-9"],
            alert_count=1,
        )
    )

    runner = CliRunner()
    listing = runner.invoke(app, ["--config", str(cfg), "incident", "list"])
    assert listing.exit_code == 0
    assert "INC-9" in listing.stdout

    detail = runner.invoke(app, ["--config", str(cfg), "incident", "show", "--id", "INC-9"])
    assert detail.exit_code == 0
    assert "ALT-9" in detail.stdout

    close = runner.invoke(app, ["--config", str(cfg), "incident", "close", "--id", "INC-9"])
    assert close.exit_code == 0

    closed = runner.invoke(app, ["--config", str(cfg), "incident", "list", "--status", "closed"])
    assert closed.exit_code == 0
    assert "INC-9" in closed.stdout


def test_cli_incident_set_severity_and_action_list(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.create_incident(
        IncidentRecord(
            incident_id="INC-7",
            entity_id="PRC007",
            status=IncidentStatus.open,
            alert_ids=[],
            alert_count=0,
        )
    )
    store.insert_action(
        ActionRecord(
            action_id="ACT-1",
            incident_id="INC-7",
            action_type=ActionType.log,
            trigger=ActionTrigger.on_create,
            status=ActionStatus.success,
            reason="logged",
        )
    )

    runner = CliRunner()
    sev = runner.invoke(
        app,
        ["--config", str(cfg), "incident", "set-severity", "--id", "INC-7", "--severity", "high"],
    )
    assert sev.exit_code == 0
    assert "high" in sev.stdout

    listed = runner.invoke(app, ["--config", str(cfg), "incident", "list", "--status", "open"])
    assert listed.exit_code == 0
    assert "high" in listed.stdout

    actions = runner.invoke(app, ["--config", str(cfg), "action", "list", "--incident-id", "INC-7"])
    assert actions.exit_code == 0
    assert "ACT-1" in actions.stdout


def test_cli_incident_invalid_id_errors(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runner = CliRunner()

    show = runner.invoke(app, ["--config", str(cfg), "incident", "show", "--id", "INC-DOES-NOT-EXIST"])
    assert show.exit_code != 0

    close = runner.invoke(app, ["--config", str(cfg), "incident", "close", "--id", "INC-DOES-NOT-EXIST"])
    assert close.exit_code != 0
