"""FastAPI service bridge for TRACE-AML read models."""

import json
from contextlib import asynccontextmanager
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

    # ── Person management & training routes ──
    from trace_aml.service.person_api import create_person_router

    # ── Startup: ghost entity purge (lifespan) ───────────────────────────────
    @asynccontextmanager
    async def _lifespan(application: Any):
        """Run startup purge then yield control to the app."""
        if settings.pipeline.purge_ghost_entities_on_start:
            from loguru import logger
            min_ev = settings.pipeline.ghost_entity_min_events
            logger.info("Running startup ghost entity purge (min_events={})...", min_ev)
            purged = store.purge_ghost_entities(min_events=min_ev)
            logger.info("Startup purge done: {} ghost entities removed.", purged)
        yield  # app runs here
    # ─────────────────────────────────────────────────────────────────────────

    app = FastAPI(
        title=f"{settings.app.name} Service",
        version="4.0.0",
        description="TRACE-AML UI/API bridge over intelligence read models.",
        lifespan=_lifespan,
    )
    # Dev-friendly: allow the static mockup to call the API with ?api=http://127.0.0.1:8080
    # without a reverse proxy (EventSource + fetch need CORS when origins differ).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    person_router = create_person_router(settings, store)
    app.include_router(person_router)

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

    @app.get("/api/v1/live/mjpeg")
    async def live_mjpeg(
        request: Request,
        fps: int = Query(default=12, ge=1, le=30),
        quality: int = Query(default=80, ge=40, le=95),
    ) -> StreamingResponse:
        """Stream webcam frames as MJPEG for the Live Ops page camera panel.

        When the recognition pipeline is active (``service run --live``), frames
        are read from the shared ``CameraCapture`` buffer — no second camera is
        opened.  When recognition is *not* running, falls back to opening its
        own ``cv2.VideoCapture`` for a raw (no-detection) preview stream.
        """
        try:
            import cv2
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="OpenCV is not available in this runtime") from exc

        from trace_aml.pipeline.capture import CameraCapture

        use_shared = CameraCapture.is_active()
        own_cap: cv2.VideoCapture | None = None

        if not use_shared:
            own_cap = cv2.VideoCapture(int(settings.camera.device_index))
            if not own_cap.isOpened():
                raise HTTPException(
                    status_code=503,
                    detail=f"Camera device {settings.camera.device_index} is unavailable",
                )
            own_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(settings.camera.width))
            own_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(settings.camera.height))
            own_cap.set(cv2.CAP_PROP_FPS, float(settings.camera.fps))

        async def _iterator():
            import asyncio

            try:
                frame_interval = 1.0 / max(1, int(fps))
                encode_opts = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                _last_idx = -1
                while True:
                    if await request.is_disconnected():
                        break

                    frame = None
                    if use_shared:
                        packet = CameraCapture.get_latest_frame()
                        if packet is not None and packet.frame_index != _last_idx:
                            frame = packet.frame
                            _last_idx = packet.frame_index
                    else:
                        assert own_cap is not None
                        ok, f = own_cap.read()
                        if ok:
                            frame = f

                    if frame is None:
                        await asyncio.sleep(0.05)
                        continue

                    encoded_ok, buffer = cv2.imencode(".jpg", frame, encode_opts)
                    if not encoded_ok:
                        await asyncio.sleep(0.01)
                        continue

                    payload = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                        + payload
                        + b"\r\n"
                    )
                    await asyncio.sleep(frame_interval)
            finally:
                if own_cap is not None:
                    own_cap.release()

        return StreamingResponse(
            _iterator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
            },
        )

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
        limit: int = Query(default=20, ge=1, le=5000),
        skip: int = Query(default=0, ge=0),
        status: str = Query(default=""),
        entity_id: str = Query(default=""),
    ) -> list[dict[str, Any]]:
        rows = read_models.list_incidents(
            limit=limit,
            skip=skip,
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
            import asyncio
            try:
                if backfill > 0:
                    for event in publisher.recent(limit=backfill):
                        yield _event_to_sse(event)
                heartbeat_counter = 0.0
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = queue.get_nowait()
                        yield _event_to_sse(event)
                        heartbeat_counter = 0.0
                    except Empty:
                        # Non-blocking sleep — yields control to event loop
                        await asyncio.sleep(0.25)
                        heartbeat_counter += 0.25
                        if heartbeat_counter >= heartbeat_sec:
                            yield ": keepalive\n\n"
                            heartbeat_counter = 0.0
            finally:
                unsubscribe()

        return StreamingResponse(_iterator(), media_type="text/event-stream")

    # ── Static frontend mount ──
    # Serve the frontend UI from /ui/ so both API and UI run on the same port.
    # API routes are registered first and take priority over the static mount.
    import importlib.resources
    from pathlib import Path

    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Locate frontend directory relative to the package
    _frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if _frontend_dir.is_dir():
        @app.get("/ui")
        @app.get("/ui/")
        def _ui_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/live_ops/index.html")

        app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

    return app
