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
    session: Any | None = None,  # RecognitionSession instance (for camera control)
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

    # Add geo and quality routers
    from trace_aml.service.geo_api import create_geo_router
    from trace_aml.service.quality_api import create_quality_router
    
    if session is not None and hasattr(session, 'recognizer'):
        geo_router = create_geo_router(settings, store)
        app.include_router(geo_router)
        
        quality_router = create_quality_router(settings, store, session.recognizer)
        app.include_router(quality_router)

    # ── Portrait endpoint ────────────────────────────────────────────────────
    from trace_aml.store.portrait_store import PortraitStore
    from fastapi.responses import FileResponse as _FileResponse

    _portrait_store = (
        session.portrait_store
        if session is not None and hasattr(session, "portrait_store")
        else PortraitStore(settings)
    )

    @app.get(
        "/api/v1/entities/{entity_id}/portrait",
        response_class=_FileResponse,
        tags=["entities"],
        summary="Best-match face portrait for an entity",
    )
    def entity_portrait(entity_id: str) -> Any:
        """Return the highest-quality face crop captured for this entity.

        The portrait is extracted from the live camera frame at the moment of
        a confirmed *accept* match, cropped to the face bounding-box, padded,
        and stored as a 256×256 JPEG.  This endpoint is a pure filesystem
        read — no database queries.

        Returns 404 if no portrait has been captured for this entity yet.
        """
        path = _portrait_store.get_portrait_path(entity_id)
        if path is None:
            from fastapi import HTTPException as _HTTPException
            raise _HTTPException(
                status_code=404,
                detail=f"No portrait available for entity '{entity_id}' yet.",
            )
        return _FileResponse(
            str(path),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Entity-Id": entity_id,
            },
        )
    # ────────────────────────────────────────────────────────────────────────


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

    @app.get("/api/v1/config")
    def get_config() -> dict[str, Any]:
        """Get current live configuration."""
        return settings.model_dump(mode="json")

    @app.patch("/api/v1/config")
    def patch_config(payload: dict[str, Any]) -> dict[str, Any]:
        """Update live configuration. Only hot-tunable fields are supported."""
        # Deep update for nested Pydantic models
        # For MVP, we update recognition thresholds and rules cooldowns
        if "recognition" in payload:
            rec = payload["recognition"]
            for k, v in rec.items():
                if hasattr(settings.recognition, k):
                    setattr(settings.recognition, k, v)
        
        if "rules" in payload:
            rul = payload["rules"]
            for k, v in rul.items():
                if hasattr(settings.rules, k):
                    if isinstance(v, dict) and hasattr(settings.rules, k):
                        sub = getattr(settings.rules, k)
                        for sk, sv in v.items():
                            if hasattr(sub, sk):
                                setattr(sub, sk, sv)
                    else:
                        setattr(settings.rules, k, v)
        
        if "actions" in payload:
            act = payload["actions"]
            if "enabled" in act:
                settings.actions.enabled = bool(act["enabled"])
            if "cooldown_sec" in act:
                settings.actions.cooldown_sec = int(act["cooldown_sec"])

        return settings.model_dump(mode="json")

    # ── Camera Control Endpoints (Frontend-driven) ────────────────────────────
    @app.get("/api/v1/camera/status")
    def camera_status() -> dict[str, Any]:
        """Get current camera status (enabled/disabled)."""
        if session is None:
            return {
                "enabled": False,
                "camera_index": settings.camera.device_index,
                "resolution": f"{settings.camera.width}x{settings.camera.height}",
                "fps": settings.camera.fps,
                "message": "Live recognition session not available",
            }
        return session.get_camera_status()

    @app.post("/api/v1/camera/enable")
    def camera_enable() -> dict[str, Any]:
        """Enable camera capture. Called when user clicks 'Enable Camera' button."""
        if session is None:
            return {"status": "error", "message": "Live recognition session not available"}
        return session.enable_camera()

    @app.post("/api/v1/camera/disable")
    def camera_disable() -> dict[str, Any]:
        """Disable camera capture. Called when user clicks 'Disable Camera' button."""
        if session is None:
            return {"status": "error", "message": "Live recognition session not available"}
        return session.disable_camera()

    # ── Recognition/Inference Control Endpoints ────────────────────────────────
    @app.get("/api/v1/recognition/status")
    def recognition_status() -> dict[str, Any]:
        """Get current recognition/inference status."""
        if session is None:
            return {
                "enabled": False,
                "camera_enabled": False,
                "message": "Live recognition session not available",
            }
        return session.get_recognition_status()

    @app.post("/api/v1/recognition/enable")
    def recognition_enable() -> dict[str, Any]:
        """Enable face recognition inference. Requires camera to be enabled first."""
        if session is None:
            return {"status": "error", "message": "Live recognition session not available"}
        return session.enable_recognition()

    @app.post("/api/v1/recognition/disable")
    def recognition_disable() -> dict[str, Any]:
        """Disable face recognition inference (camera keeps running)."""
        if session is None:
            return {"status": "error", "message": "Live recognition session not available"}
        return session.disable_recognition()
    # ─────────────────────────────────────────────────────────────────────────

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

    class EntityMergePayload(BaseModel):
        target_entity_id: str

    @app.get("/api/v1/entities/{entity_id}/suggestions")
    def entity_suggestions(entity_id: str, threshold: float = Query(default=0.50)) -> list[dict[str, Any]]:
        return store.get_entity_suggestions(entity_id, threshold=threshold)

    @app.post("/api/v1/entities/{entity_id}/merge")
    def entity_merge(entity_id: str, payload: EntityMergePayload) -> dict[str, Any]:
        success = store.merge_entities(entity_id, payload.target_entity_id)
        if not success:
            raise HTTPException(status_code=400, detail="Merge operation failed")
        return {"status": "success", "source": entity_id, "target": payload.target_entity_id}

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

    @app.post("/api/v1/incidents/deduplicate")
    def incident_deduplicate() -> dict[str, Any]:
        """Remove duplicate incident records from the database.
        
        This endpoint triggers the deduplication process to remove any
        duplicate incident_id records that may have been created due to
        non-atomic operations.
        
        Returns a summary of the deduplication operation.
        """
        removed_count = store.deduplicate_incidents()
        total_incidents = len(store.list_incidents(limit=100_000))
        unique_incident_ids = len(set(
            str(row.get("incident_id", ""))
            for row in store.list_incidents(limit=100_000)
        ))
        return {
            "status": "success",
            "removed_duplicates": removed_count,
            "total_records": total_incidents,
            "unique_incidents": unique_incident_ids,
            "message": f"Removed {removed_count} duplicate record(s). Database now has {unique_incident_ids} unique incidents."
        }

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
