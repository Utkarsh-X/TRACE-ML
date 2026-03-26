"""Embedding rebuild and enrollment quality gating pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from loguru import logger

from trace_ml.core.config import Settings
from trace_ml.core.ids import new_embedding_id
from trace_ml.core.models import EmbeddingRecord, PersonLifecycleStatus
from trace_ml.quality.gating import decide_person_lifecycle
from trace_ml.quality.scoring import build_assessment
from trace_ml.recognizers.arcface import ArcFaceRecognizer
from trace_ml.store.vector_store import VectorStore


@dataclass
class TrainStats:
    persons_total: int
    persons_processed: int
    embeddings_created: int
    skipped_images: int
    active_persons: int
    ready_persons: int
    blocked_persons: int


def rebuild_embeddings(settings: Settings, store: VectorStore, recognizer: ArcFaceRecognizer) -> TrainStats:
    persons = store.list_persons()
    persons_total = len(persons)
    embeddings_created = 0
    skipped_images = 0
    persons_processed = 0
    active_persons = 0
    ready_persons = 0
    blocked_persons = 0

    image_root = Path(settings.store.root) / "person_images"
    image_root.mkdir(parents=True, exist_ok=True)

    for person in persons:
        person_id = person["person_id"]
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
            blocked_persons += 1
            continue

        image_files = [
            file
            for file in sorted(person_dir.iterdir())
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        records: list[EmbeddingRecord] = []
        store.clear_image_quality(person_id)
        valid_images = 0
        quality_scores: list[float] = []
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
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

            if not assessment.passed or candidate is None:
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

        store.replace_person_embeddings(person_id, records)
        if records:
            persons_processed += 1
            embeddings_created += len(records)

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
        if lifecycle.state == PersonLifecycleStatus.active:
            active_persons += 1
        elif lifecycle.state == PersonLifecycleStatus.ready:
            ready_persons += 1
        elif lifecycle.state == PersonLifecycleStatus.blocked:
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
