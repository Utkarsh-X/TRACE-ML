"""Person management REST API router.

Endpoints for person CRUD, image upload, and embedding training —
the service-layer equivalent of CLI ``person add``, ``person capture``,
``train rebuild``, etc.

Auto-enrollment
---------------
When images are uploaded via ``POST /api/v1/persons/{id}/images``, the
endpoint automatically queues the person for incremental embedding
computation.  A single background ``EnrollmentWorker`` thread processes the
queue and calls ``enroll_person()`` for just that person — all other
persons' embeddings are left completely untouched.

This eliminates the need for the operator to manually press "Rebuild
Embeddings" after every registration.

Note: ``from __future__ import annotations`` is intentionally omitted —
FastAPI needs real type objects at runtime for parameter resolution.
"""

import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from trace_aml.core.config import Settings
from trace_aml.core.ids import next_person_id
from trace_aml.core.models import PersonCategory, PersonLifecycleStatus, PersonRecord
from trace_aml.pipeline.collect import person_image_dir
from trace_aml.store.vector_store import VectorStore
from trace_aml.store.data_vault import DataVault


# ── Request / response models ──────────────────────────────────────────

class PersonCreatePayload(BaseModel):
    name: str
    category: PersonCategory = PersonCategory.criminal
    severity: str = ""
    dob: str = ""
    gender: str = ""
    city: str = ""
    country: str = ""
    notes: str = ""


class PersonUpdatePayload(BaseModel):
    name: Optional[str] = None
    severity: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    notes: Optional[str] = None


class TrainRebuildPayload(BaseModel):
    scope: str = Field(default="all", pattern=r"^(all|new_only|person_id=.+)$")


# ── Shared full-rebuild training state ────────────────────────────────

_train_lock = threading.Lock()
_train_status: dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_started_at": None,
    "last_completed_at": None,
}


# ── Per-person incremental enrollment worker ──────────────────────────
#
# A single background thread drains a queue of person_ids that need
# (re-)enrollment.  Loading ArcFace once and reusing it across all
# jobs avoids the expensive ONNX model initialisation overhead.

_enrollment_queue: queue.Queue[str] = queue.Queue(maxsize=2_000)
_enrollment_lock = threading.Lock()
_enrollment_worker_started = False
_enrollment_per_person: dict[str, str] = {}  # pid → "queued"|"processing"|"done"|"error:…"
_enrollment_recognizer: Any = None  # lazily-created ArcFaceRecognizer, reused


def clear_enrollment_state() -> None:
    """Wipe in-memory enrollment status — called during factory reset."""
    with _enrollment_lock:
        _enrollment_per_person.clear()



def _start_enrollment_worker(settings: "Settings", store: "VectorStore") -> None:
    """Start the background enrollment thread (idempotent — safe to call multiple times)."""
    global _enrollment_worker_started

    with _enrollment_lock:
        if _enrollment_worker_started:
            return
        _enrollment_worker_started = True

    def _worker() -> None:
        global _enrollment_recognizer

        while True:
            try:
                person_id = _enrollment_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with _enrollment_lock:
                _enrollment_per_person[person_id] = "processing"

            try:
                # Lazily initialise the recognizer once and reuse across jobs.
                if _enrollment_recognizer is None:
                    from trace_aml.liveness.base import MiniFASNetStub
                    from trace_aml.recognizers.arcface import ArcFaceRecognizer

                    _enrollment_recognizer = ArcFaceRecognizer(settings)
                    if settings.liveness.enabled:
                        _enrollment_recognizer.set_liveness_checker(
                            MiniFASNetStub(
                                model_path=settings.liveness.model_path,
                                threshold=settings.liveness.threshold,
                            )
                        )

                from trace_aml.pipeline.train import enroll_person

                created, skipped = enroll_person(
                    person_id=person_id,
                    settings=settings,
                    store=store,
                    recognizer=_enrollment_recognizer,
                )
                with _enrollment_lock:
                    _enrollment_per_person[person_id] = "done"
                logger.info(
                    "Auto-enrollment complete: {} ({} embeddings, {} skipped)",
                    person_id,
                    created,
                    skipped,
                )

            except Exception as exc:
                logger.error("Auto-enrollment failed for {}: {}", person_id, exc)
                with _enrollment_lock:
                    _enrollment_per_person[person_id] = f"error: {exc}"
            finally:
                _enrollment_queue.task_done()

    t = threading.Thread(
        target=_worker,
        name="trace-aml-enrollment-worker",
        daemon=True,
    )
    t.start()


def _queue_enrollment(person_id: str) -> bool:
    """Queue a person for incremental enrollment.  Returns False if already queued."""
    with _enrollment_lock:
        current = _enrollment_per_person.get(person_id, "")
        if current in ("queued", "processing"):
            return False  # already in flight
        _enrollment_per_person[person_id] = "queued"
    try:
        _enrollment_queue.put_nowait(person_id)
        return True
    except queue.Full:
        with _enrollment_lock:
            _enrollment_per_person[person_id] = "error: queue full"
        logger.warning("Enrollment queue full — could not enqueue {}", person_id)
        return False


# ── Router factory ─────────────────────────────────────────────────────

def create_person_router(settings: "Settings", store: "VectorStore") -> Any:
    """Build and return a FastAPI APIRouter with all person management routes."""

    try:
        from fastapi import APIRouter, File, HTTPException, UploadFile
    except ImportError as exc:
        raise RuntimeError("Person API needs FastAPI: pip install fastapi") from exc

    # Start the enrollment worker the first time the router is created
    # (i.e., at service startup).
    _start_enrollment_worker(settings, store)

    router = APIRouter(prefix="/api/v1", tags=["persons"])

    # ── GET /api/v1/persons ────────────────────────────────────────────

    @router.get("/persons")
    def list_persons():
        """List all registered persons with enrollment data."""
        rows = store.list_persons()
        result: list[dict[str, Any]] = []
        for p in rows:
            img_dir = person_image_dir(settings, str(p["person_id"]))
            image_count = len([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]) if img_dir.exists() else 0
            result.append({
                "person_id": p.get("person_id", ""),
                "name": p.get("name", ""),
                "category": p.get("category", "criminal"),
                "severity": p.get("severity", ""),
                "lifecycle_state": p.get("lifecycle_state", "draft"),
                "lifecycle_reason": p.get("lifecycle_reason", ""),
                "enrollment_score": float(p.get("enrollment_score", 0.0)),
                "valid_embeddings": int(p.get("valid_embeddings", 0)),
                "valid_images": int(p.get("valid_images", 0)),
                "total_images": int(p.get("total_images", 0)),
                "image_count_on_disk": image_count,
                "created_at": p.get("created_at", ""),
                "updated_at": p.get("updated_at", ""),
            })
        return result

    # ── POST /api/v1/persons ───────────────────────────────────────────

    @router.post("/persons", status_code=201)
    def create_person(payload: PersonCreatePayload) -> dict[str, Any]:
        """Create a new person record. Returns the generated person_id."""
        existing_ids = [p["person_id"] for p in store.list_persons()]
        person_id = next_person_id(payload.category, existing_ids)
        now = datetime.now(timezone.utc).isoformat()
        record = PersonRecord(
            person_id=person_id,
            name=payload.name,
            category=payload.category,
            severity=payload.severity,
            dob=payload.dob,
            gender=payload.gender,
            last_seen_city=payload.city,
            last_seen_country=payload.country,
            notes=payload.notes,
            lifecycle_state=PersonLifecycleStatus.draft,
            lifecycle_reason="awaiting_images",
            created_at=now,
            updated_at=now,
        )
        store.add_or_update_person(record)
        # Pre-create the image directory
        person_image_dir(settings, person_id)
        return {
            "person_id": person_id,
            "name": payload.name,
            "category": payload.category.value,
            "lifecycle_state": "draft",
            "created_at": now,
        }

    # ── GET /api/v1/persons/{person_id} ───────────────────────────────

    @router.get("/persons/{person_id}")
    def get_person(person_id: str) -> dict[str, Any]:
        """Get single person with image count and quality summary."""
        p = store.get_person(person_id)
        if not p:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        img_dir = person_image_dir(settings, person_id)
        image_count = len([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]) if img_dir.exists() else 0
        return {
            "person_id": p["person_id"],
            "name": p.get("name", ""),
            "category": p.get("category", ""),
            "severity": p.get("severity", ""),
            "dob": p.get("dob", ""),
            "gender": p.get("gender", ""),
            "last_seen_city": p.get("last_seen_city", ""),
            "last_seen_country": p.get("last_seen_country", ""),
            "notes": p.get("notes", ""),
            "lifecycle_state": p.get("lifecycle_state", "draft"),
            "lifecycle_reason": p.get("lifecycle_reason", ""),
            "enrollment_score": float(p.get("enrollment_score", 0.0)),
            "valid_embeddings": int(p.get("valid_embeddings", 0)),
            "image_count_on_disk": image_count,
            "created_at": p.get("created_at", ""),
            "updated_at": p.get("updated_at", ""),
        }

    # ── GET /api/v1/persons/{person_id}/enroll-debug ───────────────────

    @router.get("/persons/{person_id}/enroll-debug")
    def enroll_debug(person_id: str) -> dict[str, Any]:
        """Run face-detection diagnosis on every uploaded image for this person.

        Returns per-image detail: whether a face was found, detector score,
        quality components, and the exact reason(s) for rejection.
        Use this to diagnose BLOCKED enrollment without reading log files.
        """
        import cv2 as _cv2
        from trace_aml.recognizers.arcface import ArcFaceRecognizer
        from trace_aml.quality.scoring import build_assessment as _build_assessment

        p = store.get_person(person_id)
        if not p:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

        img_dir = person_image_dir(settings, person_id)
        if not img_dir.exists():
            return {"person_id": person_id, "error": "no image directory", "images": []}

        files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        recognizer = ArcFaceRecognizer(settings)
        results = []
        for f in files:
            img = _cv2.imread(str(f))
            if img is None:
                results.append({"file": f.name, "status": "corrupt", "reason": "cv2.imread returned None"})
                continue

            h, w = img.shape[:2]
            candidate = recognizer.primary_face_from_image(img)
            if candidate is None:
                results.append({
                    "file": f.name,
                    "size": f"{w}x{h}",
                    "status": "no_face",
                    "reason": "SCRFD face detector returned no detections",
                })
                continue

            assessment = _build_assessment(
                settings=settings, person_id=person_id,
                source_path=str(f), frame_bgr=img, bbox=candidate.bbox
            )
            results.append({
                "file": f.name,
                "size": f"{w}x{h}",
                "status": "passed" if assessment.passed else "quality_fail",
                "detector_score": round(candidate.detector_score, 3),
                "face_ratio": round(assessment.face_ratio, 4),
                "sharpness": round(assessment.sharpness, 1),
                "brightness": round(assessment.brightness, 1),
                "pose_score": round(assessment.pose_score, 3),
                "quality_score": round(assessment.quality_score, 3),
                "reasons": assessment.reasons,
                "thresholds": {
                    "min_face_ratio": settings.quality.min_face_ratio,
                    "min_sharpness": settings.quality.min_sharpness,
                    "min_brightness": settings.quality.min_brightness,
                    "max_brightness": settings.quality.max_brightness,
                    "min_pose_score": settings.quality.min_pose_score,
                    "min_quality_score": settings.quality.min_quality_score,
                },
            })

        passed = sum(1 for r in results if r.get("status") == "passed")
        return {
            "person_id": person_id,
            "name": p.get("name", ""),
            "total_images": len(files),
            "passed": passed,
            "blocked_reason": p.get("lifecycle_reason", ""),
            "images": results,
        }


    @router.patch("/persons/{person_id}")
    def update_person(person_id: str, payload: PersonUpdatePayload) -> dict[str, Any]:
        """Update person metadata."""
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        updated = PersonRecord(
            person_id=current["person_id"],
            name=payload.name if payload.name is not None else current.get("name", ""),
            category=payload.category if payload.category is not None else PersonCategory(current.get("category", "criminal")),
            severity=payload.severity if payload.severity is not None else current.get("severity", ""),
            dob=payload.dob if payload.dob is not None else current.get("dob", ""),
            gender=payload.gender if payload.gender is not None else current.get("gender", ""),
            last_seen_city=payload.city if payload.city is not None else current.get("last_seen_city", ""),
            last_seen_country=payload.country if payload.country is not None else current.get("last_seen_country", ""),
            notes=payload.notes if payload.notes is not None else current.get("notes", ""),
            lifecycle_state=PersonLifecycleStatus(current.get("lifecycle_state", "draft")),
            lifecycle_reason=current.get("lifecycle_reason", ""),
            enrollment_score=float(current.get("enrollment_score", 0.0)),
            valid_embeddings=int(current.get("valid_embeddings", 0)),
            created_at=current.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        # Update the persons table (biographical data only) without resetting image counts.
        # add_or_update_person always resets valid_images/total_images to 0, so we bypass it:
        escaped_id = updated.person_id.replace("'", "''")
        store.persons.delete(f"person_id = '{escaped_id}'")
        store.persons.add([store._filtered_row(store.persons, updated.model_dump())])
        # Restore the lifecycle state with the preserved image counts
        store.set_person_state(
            person_id=updated.person_id,
            lifecycle_state=updated.lifecycle_state,
            lifecycle_reason=updated.lifecycle_reason,
            enrollment_score=updated.enrollment_score,
            valid_embeddings=updated.valid_embeddings,
            valid_images=int(current.get("valid_images", 0)),
            total_images=int(current.get("total_images", 0)),
        )
        return {"person_id": person_id, "status": "updated"}

    # ── DELETE /api/v1/persons/{person_id} ────────────────────────────

    @router.delete("/persons/{person_id}")
    def delete_person(person_id: str) -> dict[str, Any]:
        """Delete person and all linked embeddings, images, and detections."""
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        store.delete_person(person_id, delete_detections=True)
        # Remove vault enrollment images
        _vault = DataVault(settings)
        _vault.delete_person_enrollment(person_id)
        # Remove legacy filesystem images if they still exist
        img_dir = person_image_dir(settings, person_id)
        if img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)
        return {"person_id": person_id, "status": "deleted"}

    # ── POST /api/v1/persons/{person_id}/images ────────────────────────

    @router.post("/persons/{person_id}/images")
    async def upload_images(
        person_id: str,
        files: Annotated[list[UploadFile], File(description="Image files to upload")],
    ) -> dict[str, Any]:
        """Upload one or more images for a person.

        After saving the files, the person is automatically queued for
        incremental embedding computation.  The response returns immediately;
        the embedding job runs in the background.  Poll
        ``GET /api/v1/enroll/status/{person_id}`` to track completion.
        """
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

        _vault = DataVault(settings)
        saved: list[str] = []
        existing_count = len(_vault.get_enrollment_image_bytes(person_id))

        for upload in files:
            if not upload.content_type or not upload.content_type.startswith("image/"):
                continue
            content = await upload.read()
            # Encrypt and store in vault — no plaintext JPEG ever written to disk
            _vault.put_enrollment_image(person_id, content)
            saved.append(upload.filename or "image")

        if saved:
            store.set_person_state(
                person_id=person_id,
                lifecycle_state=PersonLifecycleStatus.draft,
                lifecycle_reason="images_uploaded_awaiting_training",
                enrollment_score=float(current.get("enrollment_score", 0.0)),
                valid_embeddings=int(current.get("valid_embeddings", 0)),
                valid_images=int(current.get("valid_images", 0)),
                total_images=existing_count + len(saved),
            )
            queued = _queue_enrollment(person_id)
            enrollment_note = (
                "enrollment queued" if queued else "enrollment already in progress"
            )
        else:
            enrollment_note = "no images saved"

        return {
            "person_id": person_id,
            "uploaded": len(saved),
            "total_images": existing_count + len(saved),
            "enrollment": enrollment_note,
        }

    # ── GET /api/v1/enroll/status ──────────────────────────────────────

    @router.get("/enroll/status")
    def enrollment_status_all() -> dict[str, Any]:
        """Return per-person enrollment status for all persons that
        have been processed since service startup."""
        with _enrollment_lock:
            return {
                "queue_depth": _enrollment_queue.qsize(),
                "persons": dict(_enrollment_per_person),
            }

    @router.get("/enroll/status/{person_id}")
    def enrollment_status_person(person_id: str) -> dict[str, Any]:
        """Return enrollment status for a single person."""
        with _enrollment_lock:
            status = _enrollment_per_person.get(person_id, "never_queued")
        return {"person_id": person_id, "status": status}

    # ── POST /api/v1/train/rebuild ─────────────────────────────────────

    @router.post("/train/rebuild", status_code=202)
    def train_rebuild(payload: Optional[TrainRebuildPayload] = None) -> dict[str, Any]:
        """Trigger embedding rebuild in a background thread.

        scope values:
          ``"all"``              — rebuild every person (full refresh)
          ``"new_only"``         — only persons currently in ``draft`` state
          ``"person_id=<id>"``   — a single specific person
        """
        with _train_lock:
            if _train_status["running"]:
                return {"status": "already_running", "started_at": _train_status["last_started_at"]}

        scope = payload.scope if payload else "all"

        # Resolve scope → person_ids_filter
        person_ids_filter: set[str] | None = None
        if scope == "new_only":
            draft_persons = [
                p for p in store.list_persons()
                if str(p.get("lifecycle_state", "")) == PersonLifecycleStatus.draft.value
            ]
            person_ids_filter = {p["person_id"] for p in draft_persons}
            if not person_ids_filter:
                return {"status": "nothing_to_do", "scope": scope, "reason": "no draft persons found"}
        elif scope.startswith("person_id="):
            person_ids_filter = {scope.split("=", 1)[1]}

        def _worker() -> None:
            with _train_lock:
                _train_status["running"] = True
                _train_status["last_started_at"] = datetime.now(timezone.utc).isoformat()
            try:
                from trace_aml.liveness.base import MiniFASNetStub
                from trace_aml.pipeline.train import rebuild_embeddings
                from trace_aml.recognizers.arcface import ArcFaceRecognizer

                recognizer = ArcFaceRecognizer(settings)
                if settings.liveness.enabled:
                    recognizer.set_liveness_checker(
                        MiniFASNetStub(
                            model_path=settings.liveness.model_path,
                            threshold=settings.liveness.threshold,
                        )
                    )
                stats = rebuild_embeddings(
                    settings,
                    store,
                    recognizer,
                    person_ids_filter=person_ids_filter,
                )
                with _train_lock:
                    _train_status["last_result"] = {
                        "persons_total": stats.persons_total,
                        "persons_processed": stats.persons_processed,
                        "embeddings_created": stats.embeddings_created,
                        "skipped_images": stats.skipped_images,
                        "active_persons": stats.active_persons,
                        "ready_persons": stats.ready_persons,
                        "blocked_persons": stats.blocked_persons,
                    }
            except Exception as exc:
                with _train_lock:
                    _train_status["last_result"] = {"error": str(exc)}
            finally:
                with _train_lock:
                    _train_status["running"] = False
                    _train_status["last_completed_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_worker, name="trace-aml-train-rebuild", daemon=True).start()
        return {"status": "started", "scope": scope, "filter_count": len(person_ids_filter) if person_ids_filter else "all"}

    # ── GET /api/v1/train/status ───────────────────────────────────────

    @router.get("/train/status")
    def train_status() -> dict[str, Any]:
        """Check training progress / last result."""
        with _train_lock:
            return {
                "running": _train_status["running"],
                "last_result": _train_status["last_result"],
                "last_started_at": _train_status["last_started_at"],
                "last_completed_at": _train_status["last_completed_at"],
            }

    return router
