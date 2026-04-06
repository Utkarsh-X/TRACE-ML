"""FastAPI service bridge for TRACE-AML read models."""

from __future__ import annotations

import json
from queue import Empty, Full, Queue
from typing import Any

from pydantic import BaseModel

from trace_aml.core.config import Settings
from trace_aml.core.models import AlertSeverity
from trace_aml.core.streaming import EventStreamPublisher, NullEventStreamPublisher, StreamEvent
from trace_aml.pipeline.live_overlay import get_live_overlay
from trace_aml.query.read_models import IntelligenceReadModelService
from trace_aml.store.vector_store import VectorStore


class IncidentSeverityPayload(BaseModel):
    severity: AlertSeverity


def _event_to_sse(event: StreamEvent) -> str:
    payload = {
        "topic": event.topic,
        "timestamp_utc": event.timestamp_utc,
        "payload": event.payload,
    }
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event.topic}\nid: {event.timestamp_utc}\ndata: {data}\n\n"


def create_service_app(
    settings: Settings,
    store: VectorStore,
    stream_publisher: EventStreamPublisher | None = None,
) -> Any:
    """Create FastAPI app lazily so CLI stays usable without web deps."""
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "Service layer requires FastAPI. Install extras: pip install fastapi uvicorn"
        ) from exc

    publisher = stream_publisher or NullEventStreamPublisher()
    read_models = IntelligenceReadModelService(store, publisher)
    app = FastAPI(
        title=f"{settings.app.name} Service",
        version="4.0.0",
        description="TRACE-AML UI/API bridge over intelligence read models.",
    )
    # Dev-friendly: allow the static mockup to call the API with ?api=http://127.0.0.1:8080
    # without a reverse proxy (EventSource + fetch need CORS when origins differ).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": settings.app.name,
            "environment": settings.app.environment,
            "version": "4.0.0",
            "status": "ok",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        snapshot = read_models.get_live_ops_snapshot(entity_limit=3, incident_limit=3, alert_limit=3)
        return snapshot.system_health.model_dump(mode="json")

    @app.get("/api/v1/live/overlay")
    def live_overlay() -> dict[str, Any]:
        """Latest normalized detection boxes from in-process live recognition (if running)."""
        return get_live_overlay()

    @app.get("/api/v1/live/snapshot")
    def live_snapshot(
        entity_limit: int = Query(default=12, ge=1, le=200),
        incident_limit: int = Query(default=12, ge=1, le=200),
        alert_limit: int = Query(default=12, ge=1, le=200),
    ) -> dict[str, Any]:
        snapshot = read_models.get_live_ops_snapshot(
            entity_limit=entity_limit,
            incident_limit=incident_limit,
            alert_limit=alert_limit,
        )
        return snapshot.model_dump(mode="json")

    @app.get("/api/v1/entities")
    def entities(
        limit: int = Query(default=100, ge=1, le=1000),
        type_filter: str = Query(default=""),
        status: str = Query(default=""),
    ) -> list[dict[str, Any]]:
        rows = read_models.list_entities(
            limit=limit,
            type_filter=type_filter or None,
            status=status or None,
        )
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/entities/{entity_id}")
    def entity(entity_id: str) -> dict[str, Any]:
        try:
            result = read_models.get_entity(entity_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.get("/api/v1/entities/{entity_id}/profile")
    def entity_profile(entity_id: str) -> dict[str, Any]:
        try:
            result = read_models.get_entity_profile(entity_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.get("/api/v1/entities/{entity_id}/timeline")
    def entity_timeline(
        entity_id: str,
        start: str = Query(default=""),
        end: str = Query(default=""),
        limit: int = Query(default=500, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        rows = read_models.get_entity_timeline(
            entity_id=entity_id,
            start=start or None,
            end=end or None,
            limit=limit,
        )
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/entities/{entity_id}/incidents")
    def entity_incidents(entity_id: str) -> list[dict[str, Any]]:
        rows = read_models.get_entity_incidents(entity_id)
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/incidents")
    def incidents(
        limit: int = Query(default=200, ge=1, le=5000),
        status: str = Query(default=""),
        entity_id: str = Query(default=""),
    ) -> list[dict[str, Any]]:
        rows = read_models.list_incidents(
            limit=limit,
            status=status or None,
            entity_id=entity_id or None,
        )
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/incidents/{incident_id}")
    def incident_detail(incident_id: str) -> dict[str, Any]:
        try:
            result = read_models.get_incident_detail(incident_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.patch("/api/v1/incidents/{incident_id}/severity")
    def incident_set_severity(incident_id: str, payload: IncidentSeverityPayload) -> dict[str, Any]:
        updated = store.set_incident_severity(incident_id, payload.severity.value)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")
        detail = read_models.get_incident_detail(incident_id)
        return detail.model_dump(mode="json")

    @app.post("/api/v1/incidents/{incident_id}/close")
    def incident_close(incident_id: str) -> dict[str, Any]:
        updated = store.close_incident(incident_id)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")
        detail = read_models.get_incident_detail(incident_id)
        return detail.model_dump(mode="json")

    @app.get("/api/v1/timeline")
    def global_timeline(
        start: str = Query(default=""),
        end: str = Query(default=""),
        limit: int = Query(default=1000, ge=1, le=20_000),
        entity_id: str = Query(default=""),
        incident_id: str = Query(default=""),
        kinds: list[str] = Query(default=[]),
    ) -> list[dict[str, Any]]:
        rows = read_models.get_global_timeline(
            start=start or None,
            end=end or None,
            limit=limit,
            entity_id=entity_id or None,
            incident_id=incident_id or None,
            kinds=kinds or None,
        )
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/alerts/recent")
    def recent_alerts(limit: int = Query(default=50, ge=1, le=5000)) -> list[dict[str, Any]]:
        rows = read_models.get_recent_alerts(limit=limit)
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/actions")
    def actions(
        incident_id: str = Query(default=""),
        limit: int = Query(default=200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        rows = read_models.list_actions(limit=limit, incident_id=incident_id or None)
        return [item.model_dump(mode="json") for item in rows]

    @app.get("/api/v1/events/stream")
    async def stream_events(
        request: Request,
        backfill: int = Query(default=20, ge=0, le=500),
        heartbeat_sec: float = Query(default=15.0, ge=2.0, le=60.0),
    ) -> StreamingResponse:
        queue: Queue[StreamEvent] = Queue(maxsize=512)

        def _listener(event: StreamEvent) -> None:
            try:
                queue.put_nowait(event)
            except Full:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except Exception:
                    return None

        unsubscribe = publisher.subscribe(_listener)

        async def _iterator():
            try:
                if backfill > 0:
                    for event in publisher.recent(limit=backfill):
                        yield _event_to_sse(event)
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = queue.get(timeout=float(heartbeat_sec))
                        yield _event_to_sse(event)
                    except Empty:
                        yield ": keepalive\n\n"
            finally:
                unsubscribe()

        return StreamingResponse(_iterator(), media_type="text/event-stream")

    return app
