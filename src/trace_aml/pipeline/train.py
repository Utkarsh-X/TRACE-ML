"""Embedding enrollment and rebuild pipeline.

Key concepts
------------
``enroll_person``
    Processes *one* person's images through the frozen ArcFace model and
    upserts their 512-d embeddings into the store (and in-memory cache).
    This is the atomic unit of enrollment — it never touches other persons.

``rebuild_embeddings``
    A thin loop over ``enroll_person`` for batch / full-rebuild scenarios.
    Accepts an optional ``person_ids_filter`` so callers can limit the scope
    to a single person or a set of newly-added persons without re-processing
    the entire gallery.

ArcFace model weights are completely frozen.  The 512-d vector for a given
image is always identical regardless of how many other persons exist in the
database.  Therefore there is *never* a reason to recompute existing
person's embeddings when a new person is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from loguru import logger

from trace_aml.core.config import Settings
from trace_aml.core.ids import new_embedding_id
from trace_aml.core.models import EmbeddingRecord, PersonLifecycleStatus
from trace_aml.quality.gating import decide_person_lifecycle
from trace_aml.quality.scoring import build_assessment
from trace_aml.recognizers.arcface import ArcFaceRecognizer
from trace_aml.store.vector_store import VectorStore


@dataclass
class TrainStats:
    persons_total: int
    persons_processed: int
    embeddings_created: int
    skipped_images: int
    active_persons: int
    ready_persons: int
    blocked_persons: int


def enroll_person(
    person_id: str,
    settings: Settings,
    store: VectorStore,
    recognizer: ArcFaceRecognizer,
) -> tuple[int, int]:
    """Process one person's images and upsert their embeddings.

    This is the *only* function that runs ArcFace inference for a person.
    It is called:
      - Automatically after image upload (incremental enrollment)
      - By ``rebuild_embeddings`` for batch / full-rebuild scenarios

    All existing persons' embeddings are left completely untouched.

    Args:
        person_id:  The person to process.
        settings:   System configuration.
        store:      VectorStore instance (writes embeddings + state).
        recognizer: Shared ArcFaceRecognizer (lazy-loaded, frozen weights).

    Returns:
        ``(embeddings_created, skipped_images)`` counts for this person.
    """
    image_root = Path(settings.store.root) / "person_images"
    image_root.mkdir(parents=True, exist_ok=True)
    person_dir = image_root / person_id

    if not person_dir.exists():
        logger.warning("No image directory for person {}", person_id)
        store.set_person_state(
            person_id=person_id,
            lifecycle_state=PersonLifecycleStatus.blocked,
            lifecycle_reason="missing_image_directory",
            enrollment_score=0.0,
            valid_embeddings=0,
            valid_images=0,
            total_images=0,
        )
        return 0, 0

    image_files = [
        file
        for file in sorted(person_dir.iterdir())
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    records: list[EmbeddingRecord] = []
    store.clear_image_quality(person_id)
    valid_images = 0
    quality_scores: list[float] = []
    skipped_images = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("  [{}] {} — cv2.imread returned None (corrupt or wrong format)",
                           person_id, img_path.name)
            skipped_images += 1
            continue

        candidate = recognizer.primary_face_from_image(image)
        assessment = build_assessment(
            settings=settings,
            person_id=person_id,
            source_path=str(img_path),
            frame_bgr=image,
            bbox=candidate.bbox if candidate else None,
        )
        store.add_image_quality(assessment)

        if candidate is None:
            h, w = image.shape[:2]
            logger.debug("  [{}] {} ({}x{}) — no face detected by SCRFD",
                         person_id, img_path.name, w, h)
            skipped_images += 1
            continue

        if not assessment.passed:
            h, w = image.shape[:2]
            logger.debug(
                "  [{}] {} ({}x{}) — face found (det={:.2f}), "
                "but quality failed: {} | ratio={:.3f} sharp={:.1f} bright={:.1f} pose={:.2f} score={:.3f}",
                person_id, img_path.name, w, h,
                candidate.detector_score,
                assessment.reasons,
                assessment.face_ratio,
                assessment.sharpness,
                assessment.brightness,
                assessment.pose_score,
                assessment.quality_score,
            )
            skipped_images += 1
            continue

        valid_images += 1
        quality_scores.append(float(assessment.quality_score))
        records.append(
            EmbeddingRecord(
                embedding_id=new_embedding_id(person_id),
                person_id=person_id,
                source_path=str(img_path),
                embedding=candidate.embedding,
                quality_score=float(assessment.quality_score),
                quality_flags=[],
            )
        )

    # Upsert embeddings into both LanceDB and the in-memory cache.
    # This is an incremental operation — other persons are untouched.
    store.replace_person_embeddings(person_id, records)

    avg_quality = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0
    lifecycle = decide_person_lifecycle(
        settings=settings,
        total_images=len(image_files),
        valid_images=valid_images,
        embeddings_count=len(records),
        avg_quality=avg_quality,
    )
    store.set_person_state(
        person_id=person_id,
        lifecycle_state=lifecycle.state,
        lifecycle_reason=lifecycle.reason,
        enrollment_score=lifecycle.enrollment_score,
        valid_embeddings=len(records),
        valid_images=valid_images,
        total_images=len(image_files),
    )

    logger.info(
        "Enrolled {}: {} embeddings, state={}, quality={:.2f}",
        person_id,
        len(records),
        lifecycle.state,
        avg_quality,
    )
    return len(records), skipped_images


def rebuild_embeddings(
    settings: Settings,
    store: VectorStore,
    recognizer: ArcFaceRecognizer,
    person_ids_filter: set[str] | None = None,
) -> TrainStats:
    """Batch enrollment loop.

    Args:
        person_ids_filter: When provided, only persons in this set are
            processed.  Pass ``None`` to rebuild the entire gallery.
            Typical values:
              * ``None``                      — full rebuild (e.g. on first run)
              * ``{single_id}``              — one newly-enrolled person
              * ``{ids with state="draft"}``  — scope="new_only"
    """
    all_persons = store.list_persons()
    persons = (
        [p for p in all_persons if p["person_id"] in person_ids_filter]
        if person_ids_filter is not None
        else all_persons
    )

    persons_total = len(persons)
    embeddings_created = 0
    skipped_images = 0
    persons_processed = 0
    active_persons = 0
    ready_persons = 0
    blocked_persons = 0

    for person in persons:
        person_id = person["person_id"]
        created, skipped = enroll_person(person_id, settings, store, recognizer)
        if created > 0:
            persons_processed += 1
        embeddings_created += created
        skipped_images += skipped

        state = store.get_person_state(person_id)
        if state:
            ls = str(state.get("lifecycle_state", ""))
            if ls == PersonLifecycleStatus.active:
                active_persons += 1
            elif ls == PersonLifecycleStatus.ready:
                ready_persons += 1
            elif ls == PersonLifecycleStatus.blocked:
                blocked_persons += 1

    return TrainStats(
        persons_total=persons_total,
        persons_processed=persons_processed,
        embeddings_created=embeddings_created,
        skipped_images=skipped_images,
        active_persons=active_persons,
        ready_persons=ready_persons,
        blocked_persons=blocked_persons,
    )
