from pathlib import Path

from trace_ml.core.config import load_settings
from trace_ml.core.models import DecisionState, DetectionEvent, HistoryQuery, PersonCategory, PersonLifecycleStatus, PersonRecord
from trace_ml.store.analytics import AnalyticsStore
from trace_ml.store.vector_store import VectorStore


def _settings(tmp_path: Path):
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
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_history_summary_export(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    analytics = AnalyticsStore(store)

    store.add_or_update_person(
        PersonRecord(
            person_id="PRM001",
            name="Bob",
            category=PersonCategory.missing,
        )
    )
    store.set_person_state(
        person_id="PRM001",
        lifecycle_state=PersonLifecycleStatus.active,
        lifecycle_reason="test",
        enrollment_score=0.8,
        valid_embeddings=10,
        valid_images=10,
        total_images=10,
    )

    store.add_detection(
        DetectionEvent(
            detection_id="DET-1",
            person_id="PRM001",
            name="Bob",
            category="missing",
            confidence=88.0,
            similarity=0.88,
            smoothed_confidence=90.0,
            bbox=(1, 2, 3, 4),
            track_id="T0001",
            decision_state=DecisionState.accept,
            decision_reason="stable_accept",
            quality_flags=[],
            liveness_provider="none",
            liveness_score=1.0,
        ),
        embedding=[0.0] * 512,
    )
    store.add_detection_decision(
        detection_id="DET-1",
        track_id="T0001",
        decision_state="accept",
        decision_reason="stable_accept",
        smoothed_confidence=90.0,
        quality_flags=[],
        top_candidates=[{"person_id": "PRM001"}],
        liveness_provider="none",
        liveness_score=1.0,
    )
    store.add_detection(
        DetectionEvent(
            detection_id="DET-2",
            person_id="PRM001",
            name="Bob",
            category="missing",
            confidence=91.0,
            similarity=0.91,
            smoothed_confidence=74.0,
            bbox=(2, 3, 4, 5),
            track_id="T0001",
            decision_state=DecisionState.review,
            decision_reason="review_candidate",
            quality_flags=[],
            liveness_provider="none",
            liveness_score=1.0,
        ),
        embedding=[0.0] * 512,
    )
    store.add_detection_decision(
        detection_id="DET-2",
        track_id="T0001",
        decision_state="review",
        decision_reason="review_candidate",
        smoothed_confidence=74.0,
        quality_flags=[],
        top_candidates=[{"person_id": "PRM001"}],
        liveness_provider="none",
        liveness_score=1.0,
    )

    rows = analytics.history(HistoryQuery(limit=10))
    assert len(rows) == 2
    assert rows[0]["decision_state"] in {"accept", "review"}

    summary = analytics.summary()
    assert summary.total_detections == 2
    assert summary.unique_persons == 1
    assert summary.decision_distribution["accept"] == 1
    assert summary.decision_distribution["review"] == 1

    out = analytics.export_csv(HistoryQuery(limit=10), tmp_path / "out.csv")
    assert out.exists()
