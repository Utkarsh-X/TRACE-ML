"""Person management REST API router.

Endpoints for person CRUD, image upload, and embedding training —
the service-layer equivalent of CLI ``person add``, ``person capture``,
``train rebuild``, etc.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from trace_aml.core.config import Settings
from trace_aml.core.ids import next_person_id
from trace_aml.core.models import PersonCategory, PersonLifecycleStatus, PersonRecord
from trace_aml.pipeline.collect import person_image_dir
from trace_aml.store.vector_store import VectorStore


# ── Request / response models ──────────────────────────────────────

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
    name: str | None = None
    severity: str | None = None
    dob: str | None = None
    gender: str | None = None
    city: str | None = None
    country: str | None = None
    notes: str | None = None


class TrainRebuildPayload(BaseModel):
    scope: str = Field(default="all", pattern=r"^(all|new_only|person_id=.+)$")


# ── Shared training state ─────────────────────────────────────────

_train_lock = threading.Lock()
_train_status: dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_started_at": None,
    "last_completed_at": None,
}


def create_person_router(settings: Settings, store: VectorStore) -> Any:
    """Build and return a FastAPI APIRouter with all person management routes."""

    try:
        from fastapi import APIRouter, File, HTTPException, UploadFile
    except ImportError as exc:
        raise RuntimeError("Person API needs FastAPI: pip install fastapi") from exc

    router = APIRouter(prefix="/api/v1", tags=["persons"])

    # ── GET /api/v1/persons ────────────────────────────────────────

    @router.get("/persons")
    def list_persons() -> list[dict[str, Any]]:
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

    # ── POST /api/v1/persons ───────────────────────────────────────

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

    # ── GET /api/v1/persons/{person_id} ────────────────────────────

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

    # ── PATCH /api/v1/persons/{person_id} ──────────────────────────

    @router.patch("/persons/{person_id}")
    def update_person(person_id: str, payload: PersonUpdatePayload) -> dict[str, Any]:
        """Update person metadata."""
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        updated = PersonRecord(
            person_id=current["person_id"],
            name=payload.name if payload.name is not None else current.get("name", ""),
            category=PersonCategory(current.get("category", "criminal")),
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
        store.add_or_update_person(updated)
        return {"person_id": person_id, "status": "updated"}

    # ── DELETE /api/v1/persons/{person_id} ─────────────────────────

    @router.delete("/persons/{person_id}")
    def delete_person(person_id: str) -> dict[str, Any]:
        """Delete person and all linked embeddings, images, and detections."""
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        store.delete_person(person_id, delete_detections=True)
        img_dir = person_image_dir(settings, person_id)
        if img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)
        return {"person_id": person_id, "status": "deleted"}

    # ── POST /api/v1/persons/{person_id}/images ────────────────────

    @router.post("/persons/{person_id}/images")
    async def upload_images(
        person_id: str,
        files: Annotated[list[UploadFile], File(description="Image files to upload")],
    ) -> dict[str, Any]:
        """Upload one or more images for a person (multipart/form-data)."""
        current = store.get_person(person_id)
        if not current:
            raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")
        img_dir = person_image_dir(settings, person_id)

        saved: list[str] = []
        existing_count = len([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]) if img_dir.exists() else 0

        for i, upload in enumerate(files):
            if not upload.content_type or not upload.content_type.startswith("image/"):
                continue
            ext = Path(upload.filename or "image.jpg").suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
                ext = ".jpg"
            out_path = img_dir / f"upload_{existing_count + i + 1:03d}{ext}"
            content = await upload.read()
            out_path.write_bytes(content)
            saved.append(str(out_path))

        # Update person state to indicate images exist
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

        return {
            "person_id": person_id,
            "uploaded": len(saved),
            "total_images": existing_count + len(saved),
        }

    # ── POST /api/v1/train/rebuild ─────────────────────────────────

    @router.post("/train/rebuild", status_code=202)
    def train_rebuild(payload: TrainRebuildPayload | None = None) -> dict[str, Any]:
        """Trigger embedding rebuild in background thread."""
        with _train_lock:
            if _train_status["running"]:
                return {"status": "already_running", "started_at": _train_status["last_started_at"]}

        scope = payload.scope if payload else "all"

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
                stats = rebuild_embeddings(settings, store, recognizer)
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
        return {"status": "started", "scope": scope}

    # ── GET /api/v1/train/status ───────────────────────────────────

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
