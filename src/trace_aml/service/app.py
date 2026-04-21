"""FastAPI service bridge for TRACE-AML read models."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

from fastapi import Query, Request
from fastapi.responses import FileResponse
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

    @app.post(
        "/api/v1/entities/{entity_id}/portrait",
        tags=["entities"],
        summary="Upload / replace the portrait for an entity",
    )
    async def entity_portrait_upload(entity_id: str, request: Request) -> dict[str, Any]:
        """Accept a multipart image upload and store it as the entity's portrait.

        This replaces whatever auto-captured portrait exists (or creates one
        from scratch), giving operators manual control over the face image
        shown in the UI.  The uploaded image is resized to 256×256 and saved
        as a JPEG alongside a metadata sidecar with score=1.0 so the
        auto-capture pipeline will never overwrite a manually-uploaded portrait.

        Returns 404 if the entity does not exist in the database.
        Returns 422 if no valid image file is found in the multipart body.
        """
        import cv2 as _cv2
        import json
        import numpy as np
        from datetime import datetime, timezone

        entity_row = store.get_entity(entity_id)
        if not entity_row:
            raise HTTPException(status_code=404, detail="Entity not found")

        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise HTTPException(
                status_code=422,
                detail="Request must be multipart/form-data with a 'file' field",
            )

        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="No 'file' field in multipart form")

        content = await upload.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=422,
                detail="Could not decode image — ensure you upload a valid JPEG or PNG file",
            )

        # Resize to standard portrait dimensions (256×256)
        portrait_size = 256
        img_resized = _cv2.resize(img, (portrait_size, portrait_size), interpolation=_cv2.INTER_LANCZOS4)

        # Write to portrait store directory
        portrait_dir = _portrait_store._dir
        safe_id = entity_id.replace("/", "_").replace("\\", "_")
        portrait_path = portrait_dir / f"{safe_id}.jpg"
        meta_path = portrait_dir / f"{safe_id}.meta.json"

        encode_params = [_cv2.IMWRITE_JPEG_QUALITY, 92]
        ok, buf = _cv2.imencode(".jpg", img_resized, encode_params)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode portrait image")

        portrait_path.write_bytes(buf.tobytes())

        # score=1.0 → auto-capture pipeline will never overwrite this portrait
        meta_payload = {
            "score": 1.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "manual_upload",
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        return {
            "status": "updated",
            "entity_id": entity_id,
        }
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
            # Interactive policy matrix — update on_create / on_update per severity
            if "on_create" in act:
                for sev, actions in act["on_create"].items():
                    if hasattr(settings.actions.on_create, sev):
                        setattr(settings.actions.on_create, sev, list(actions))
            if "on_update" in act:
                for sev, actions in act["on_update"].items():
                    if hasattr(settings.actions.on_update, sev):
                        setattr(settings.actions.on_update, sev, list(actions))

        if "notifications" in payload:
            notif = payload["notifications"]
            # Email channel
            if "email" in notif:
                em = notif["email"]
                email_cfg = settings.notifications.email
                for k in ("enabled", "smtp_host", "smtp_port", "smtp_user",
                          "sender_address", "sender_name", "use_tls", "attach_pdf",
                          "recipient_addresses"):
                    if k in em:
                        setattr(email_cfg, k, em[k])
                # Password from config file (for testing; env var takes precedence at runtime)
                if "smtp_password" in em:
                    email_cfg.smtp_password = em["smtp_password"]
            # WhatsApp channel
            if "whatsapp" in notif:
                wa = notif["whatsapp"]
                wa_cfg = settings.notifications.whatsapp
                for k in ("enabled", "bridge_url", "recipient_numbers",
                          "send_pdf", "send_text"):
                    if k in wa:
                        setattr(wa_cfg, k, wa[k])
            # PDF reports
            if "pdf_report" in notif:
                pdf = notif["pdf_report"]
                pdf_cfg = settings.notifications.pdf_report
                for k in ("enabled", "include_screenshots", "include_entity_portrait",
                          "max_detection_rows", "max_alert_rows"):
                    if k in pdf:
                        setattr(pdf_cfg, k, pdf[k])

        # Return config (password included for testing; env var takes precedence at runtime)
        cfg_dump = settings.model_dump(mode="json")
        return cfg_dump

    @app.patch("/api/v1/config/notifications")
    def patch_notifications(payload: dict[str, Any]) -> dict[str, Any]:
        """Shorthand for patching only the notifications block."""
        return patch_config({"notifications": payload})

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

    class EntityUpdatePayload(BaseModel):
        name: str | None = None
        category: str | None = None
        severity: str | None = None
        notes: str | None = None

    @app.patch("/api/v1/entities/{entity_id}")
    def entity_update(entity_id: str, payload: EntityUpdatePayload) -> dict[str, Any]:
        entity_row = store.get_entity(entity_id)
        if not entity_row:
            raise HTTPException(status_code=404, detail="Entity not found")
        is_known = str(entity_row.get("type", "")) == "known"
        person_id = str(entity_row.get("source_person_id", "")) or entity_id

        if is_known:
            current = store.get_person(person_id)
            if not current:
                raise HTTPException(status_code=404, detail="Underlying person not found")
            from trace_aml.core.models import PersonCategory, PersonLifecycleStatus, PersonRecord
            from datetime import datetime, timezone
            
            cat = payload.category if payload.category is not None else current.get("category", "criminal")
            updated = PersonRecord(
                person_id=current["person_id"],
                name=payload.name if payload.name is not None else current.get("name", ""),
                category=PersonCategory(cat),
                severity=payload.severity if payload.severity is not None else current.get("severity", ""),
                dob=current.get("dob", ""),
                gender=current.get("gender", ""),
                last_seen_city=current.get("last_seen_city", ""),
                last_seen_country=current.get("last_seen_country", ""),
                notes=payload.notes if payload.notes is not None else current.get("notes", ""),
                lifecycle_state=PersonLifecycleStatus(current.get("lifecycle_state", "draft")),
                lifecycle_reason=current.get("lifecycle_reason", ""),
                enrollment_score=float(current.get("enrollment_score", 0.0)),
                valid_embeddings=int(current.get("valid_embeddings", 0)),
                created_at=current.get("created_at", datetime.now(timezone.utc).isoformat()),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            escaped_id = updated.person_id.replace("'", "''")
            store.persons.delete(f"person_id = '{escaped_id}'")
            store.persons.add([store._filtered_row(store.persons, updated.model_dump())])
            store.set_person_state(
                person_id=updated.person_id,
                lifecycle_state=updated.lifecycle_state,
                lifecycle_reason=updated.lifecycle_reason,
                enrollment_score=updated.enrollment_score,
                valid_embeddings=updated.valid_embeddings,
                valid_images=int(current.get("valid_images", 0)),
                total_images=int(current.get("total_images", 0)),
            )
            return {"status": "updated", "entity_id": entity_id}
        else:
            if payload.name:
                from trace_aml.core.ids import next_person_id
                from trace_aml.core.models import PersonCategory, PersonLifecycleStatus, PersonRecord, EmbeddingRecord
                from trace_aml.pipeline.collect import person_image_dir
                import uuid
                from datetime import datetime, timezone
                
                existing_ids = [p["person_id"] for p in store.list_persons()]
                cat = payload.category or "criminal"
                new_person_id = next_person_id(PersonCategory(cat), existing_ids)
                now = datetime.now(timezone.utc).isoformat()
                
                record = PersonRecord(
                    person_id=new_person_id,
                    name=payload.name,
                    category=PersonCategory(cat),
                    severity=payload.severity or "",
                    dob=payload.dob or "",
                    gender=payload.gender or "",
                    last_seen_city=payload.city or "",
                    last_seen_country=payload.country or "",
                    notes=payload.notes or "",
                    lifecycle_state=PersonLifecycleStatus.active,
                    lifecycle_reason="promoted_from_unknown",
                    created_at=now,
                    updated_at=now,
                )
                store.add_or_update_person(record)
                person_image_dir(settings, new_person_id)
                
                profiles = store._query_rows(store.unknown_profiles, where=f"entity_id = '{store._escape(entity_id)}'")
                records = []
                for p in profiles:
                    records.append(EmbeddingRecord(
                        embedding_id=str(uuid.uuid4()),
                        person_id=new_person_id,
                        source_path="",
                        quality_score=float(p.get("quality_score", 0.0)),
                        quality_flags=[],
                        embedding=[float(v) for v in p.get("embedding", [])]
                    ))
                store.replace_person_embeddings(new_person_id, records)
                
                store.merge_entities(entity_id, new_person_id)
                return {"status": "promoted", "old_entity_id": entity_id, "new_entity_id": new_person_id}
            else:
                raise HTTPException(status_code=400, detail="Cannot edit unknown entity without providing a name to promote it.")

    @app.delete("/api/v1/entities/{entity_id}")
    def entity_delete(entity_id: str) -> dict[str, Any]:
        entity_row = store.get_entity(entity_id)
        if not entity_row:
            raise HTTPException(status_code=404, detail="Entity not found")
        is_known = str(entity_row.get("type", "")) == "known"
        if is_known:
            person_id = str(entity_row.get("source_person_id", "")) or entity_id
            store.delete_person(person_id, delete_detections=True)
            import shutil
            from trace_aml.pipeline.collect import person_image_dir
            img_dir = person_image_dir(settings, person_id)
            if img_dir.exists():
                shutil.rmtree(img_dir, ignore_errors=True)
        else:
            store.delete_unknown_entity(entity_id)
        return {"status": "deleted", "entity_id": entity_id}

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

    @app.post("/api/v1/system/factory-reset")
    def system_factory_reset() -> dict[str, Any]:
        """Wipe ALL data and reset to a pristine first-run state.

        Clears every LanceDB table, portrait files, screenshots, enrollment
        images, and exports. Rebuilds the empty gallery cache in-process so
        no restart is needed.

        SAFETY: The camera must be disabled before calling this endpoint.
        Running this while recognition is active would cause concurrent LanceDB
        writes from two threads, which corrupts the database.
        """
        if session.is_camera_enabled():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot reset while camera is active. "
                    "Disable camera from the Live Ops page first."
                ),
            )
        from loguru import logger
        from trace_aml.service.person_api import clear_enrollment_state
        result = store.factory_reset()
        # Reset all per-session runtime state so the fresh DB is reflected
        # immediately without needing a service restart.
        session.last_logged_at.clear()
        session.last_event_at.clear()
        session._committed_tracks.clear()
        session._seen_entity_ids.clear()
        session._entities_before_session.clear()
        session.event_feed.clear()
        session.recent_confidences.clear()
        session.decision_counters = {"accept": 0, "review": 0, "reject": 0}
        session.current_focus = "none"
        # Clear in-memory enrollment status dict so old person IDs don't linger
        clear_enrollment_state()
        logger.info("Factory reset complete via API: {}", result)
        return {"status": "success", "detail": result}

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

    @app.get("/api/v1/alerts")
    def list_alerts(
        limit: int = Query(default=100, ge=1, le=5000),
        entity_id: str = Query(default=""),
        severity: str = Query(default=""),
        acknowledged: str = Query(default=""),  # "true" | "false" | "" = all
    ) -> list[dict[str, Any]]:
        """List alerts with optional filtering by entity, severity, or acknowledged state."""
        rows = store.list_alerts(
            limit=limit,
            entity_id=entity_id or None,
            severity=severity or None,
        )
        if acknowledged == "true":
            rows = [r for r in rows if r.get("acknowledged")]
        elif acknowledged == "false":
            rows = [r for r in rows if not r.get("acknowledged")]
        return rows

    @app.patch("/api/v1/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(alert_id: str) -> dict[str, Any]:
        """Mark an alert as acknowledged."""
        found = store.acknowledge_alert(alert_id)
        if not found:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {"status": "acknowledged", "alert_id": alert_id}

    @app.get("/api/v1/actions")
    def actions(
        incident_id: str = Query(default=""),
        limit: int = Query(default=200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        rows = read_models.list_actions(limit=limit, incident_id=incident_id or None)
        return [item.model_dump(mode="json") for item in rows]

    # ── Report Endpoints ──────────────────────────────────────────────────────

    @app.post("/api/v1/incidents/{incident_id}/report")
    def generate_incident_report(incident_id: str) -> dict[str, Any]:
        """Generate a PDF+HTML incident report on demand."""
        from trace_aml.actions.pdf_handler import PdfReportHandler
        from trace_aml.core.models import ActionTrigger

        # Load incident
        inc_row = store.get_incident(incident_id)
        if inc_row is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        from trace_aml.core.models import IncidentRecord, IncidentStatus, AlertSeverity
        try:
            incident = IncidentRecord(**inc_row)
        except Exception:
            incident = IncidentRecord(
                incident_id=inc_row.get("incident_id", incident_id),
                entity_id=inc_row.get("entity_id", ""),
            )

        handler = PdfReportHandler(settings, store)
        context: dict[str, Any] = {}
        ok, reason = handler.execute(incident, ActionTrigger.on_create, context)

        if not ok:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=reason)

        pdf_path  = context.get("pdf_report_path", "")
        html_path = context.get("html_report_path", "")

        # Build URL-accessible path for the HTML companion
        html_url = ""
        if html_path:
            try:
                exports_base = Path(settings.notifications.pdf_report.output_dir).resolve()
                rel = Path(html_path).relative_to(exports_base)
                html_url = f"/ui/exports/{rel.as_posix()}"
            except Exception:
                html_url = ""

        return {
            "status": "generated",
            "incident_id": incident_id,
            "pdf_path": pdf_path,
            "html_url": html_url,
            "reason": reason,
        }

    @app.get("/api/v1/incidents/{incident_id}/reports")
    def list_incident_reports(incident_id: str) -> list[dict[str, Any]]:
        """List all previously generated reports for this incident."""
        exports_dir = Path(settings.notifications.pdf_report.output_dir).resolve()
        results = []
        if exports_dir.is_dir():
            for f in sorted(exports_dir.rglob(f"*{incident_id[-8:]}*.pdf"), reverse=True):
                try:
                    rel = f.relative_to(exports_dir)
                    results.append({
                        "filename": f.name,
                        "pdf_path": str(f),
                        "html_url": f"/ui/exports/{rel.with_suffix('.html').as_posix()}",
                        "size_bytes": f.stat().st_size,
                        "generated_at": f.stat().st_mtime,
                    })
                except Exception:
                    continue
        return results

    @app.get("/api/v1/entities/{entity_id}/reports")
    def list_entity_reports(entity_id: str) -> list[dict[str, Any]]:
        """List all previously generated reports for an entity across incidents."""
        exports_dir = Path(settings.notifications.pdf_report.output_dir).resolve()
        results = []
        if exports_dir.is_dir():
            for f in sorted(exports_dir.rglob(f"{entity_id}*.pdf"), reverse=True):
                try:
                    rel = f.relative_to(exports_dir)
                    results.append({
                        "filename": f.name,
                        "pdf_path": str(f),
                        "html_url": f"/ui/exports/{rel.with_suffix('.html').as_posix()}",
                        "size_bytes": f.stat().st_size,
                        "generated_at": f.stat().st_mtime,
                    })
                except Exception:
                    continue
        return results

    # ── Notification Test Endpoints ────────────────────────────────────────────

    @app.post("/api/v1/notifications/test/email")
    def test_email() -> dict[str, Any]:
        """Send a test email to verify SMTP configuration."""
        if not settings.notifications.email.enabled:
            return {"status": "skipped", "reason": "email_disabled"}
        if not settings.notifications.email.smtp_host:
            return {"status": "failed", "reason": "email_no_smtp_host"}
        if not settings.notifications.email.recipient_addresses:
            return {"status": "failed", "reason": "email_no_recipients"}

        from trace_aml.actions.email_handler import EmailHandler
        from trace_aml.core.models import (
            IncidentRecord, IncidentStatus, AlertSeverity, ActionTrigger
        )
        test_incident = IncidentRecord(
            incident_id="TEST-00000000",
            entity_id="TEST-ENTITY",
            status=IncidentStatus.open,
            severity=AlertSeverity.high,
            alert_count=1,
            summary="This is a test notification from TRACE-AML. If you received this, email alerts are working correctly.",
        )
        handler = EmailHandler(settings, store)
        # Use execute_sync to wait for actual delivery (blocking)
        ok, reason = handler.execute_sync(test_incident, ActionTrigger.on_create, {}, timeout=30.0)
        return {"status": "sent" if ok else "failed", "reason": reason}

    @app.post("/api/v1/notifications/test/whatsapp")
    def test_whatsapp() -> dict[str, Any]:
        """Send a test WhatsApp message to verify local bridge."""
        import httpx
        wa = settings.notifications.whatsapp
        if not wa.enabled:
            return {"status": "skipped", "reason": "whatsapp_disabled"}
        if not wa.recipient_numbers:
            return {"status": "failed", "reason": "whatsapp_no_recipients"}

        # Pre-flight: verify bridge is running and WhatsApp is actually connected
        try:
            resp = httpx.get(f"{wa.bridge_url.rstrip('/')}/status", timeout=5.0)
            bridge_status = resp.json()
        except Exception as exc:
            logger.error("[WA-TEST] Bridge unreachable at {}: {}", wa.bridge_url, exc)
            return {"status": "failed", "reason": f"bridge_unreachable: {exc}"}

        if not bridge_status.get("ready"):
            qr_hint = " Scan QR at http://localhost:3001/qr" if bridge_status.get("hasQr") else ""
            return {"status": "failed", "reason": f"whatsapp_not_ready:{qr_hint}"}

        from trace_aml.actions.whatsapp_handler import WhatsAppHandler
        from trace_aml.core.models import (
            IncidentRecord, IncidentStatus, AlertSeverity, ActionTrigger
        )
        test_incident = IncidentRecord(
            incident_id="TEST-00000000",
            entity_id="TEST-ENTITY",
            status=IncidentStatus.open,
            severity=AlertSeverity.high,
            alert_count=1,
            summary="This is a test notification from TRACE-AML. If you received this, WhatsApp alerts are working correctly.",
        )
        handler = WhatsAppHandler(settings, store)
        ok, reason = handler.execute_sync(test_incident, ActionTrigger.on_create, {}, timeout=60.0)
        return {"status": "sent" if ok else "failed", "reason": reason}

    @app.get("/api/v1/whatsapp/status")
    def whatsapp_status() -> dict[str, Any]:
        """Get local WhatsApp bridge connection status and QR code."""
        import httpx
        wa = settings.notifications.whatsapp
        if not wa.enabled:
            return {"enabled": False, "ready": False, "qr": None}
        try:
            resp = httpx.get(f"{wa.bridge_url.rstrip('/')}/status", timeout=5.0)
            data = resp.json()
            qr_resp = None
            if data.get("hasQr"):
                try:
                    qr_resp = httpx.get(f"{wa.bridge_url.rstrip('/')}/qr", timeout=5.0)
                    qr_data = qr_resp.json()
                    data["qr"] = qr_data.get("qrDataUrl")
                except Exception:
                    data["qr"] = None
            return data
        except Exception as exc:
            return {"enabled": True, "ready": False, "error": str(exc), "qr": None}

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

    # ── Exports static mount ──
    # Serve generated PDF/HTML reports from data/exports/ at /ui/exports/
    _exports_dir = Path(settings.notifications.pdf_report.output_dir).resolve()
    _exports_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/ui/exports",
        StaticFiles(directory=str(_exports_dir), html=False),
        name="exports",
    )

    return app
