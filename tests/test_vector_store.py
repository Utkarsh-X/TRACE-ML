from pathlib import Path

import lancedb
import pyarrow as pa

from trace_ml.core.config import load_settings
from trace_ml.core.models import EmbeddingRecord, PersonCategory, PersonLifecycleStatus, PersonRecord, QualityAssessment
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


def test_person_and_embedding_crud(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)

    person = PersonRecord(
        person_id="PRC001",
        name="Alice",
        category=PersonCategory.criminal,
    )
    store.add_or_update_person(person)
    store.set_person_state(
        person_id="PRC001",
        lifecycle_state=PersonLifecycleStatus.active,
        lifecycle_reason="test",
        enrollment_score=0.9,
        valid_embeddings=1,
        valid_images=10,
        total_images=10,
    )
    found = store.get_person("PRC001")
    assert found is not None
    assert found["name"] == "Alice"
    assert found["lifecycle_state"] == "active"

    emb = [0.0] * 512
    emb[0] = 1.0
    records = [
        EmbeddingRecord(
            embedding_id="EMB-1",
            person_id="PRC001",
            source_path="fake.jpg",
            embedding=emb,
        )
    ]
    store.replace_person_embeddings("PRC001", records)
    assert store.count_embeddings("PRC001") == 1
    assert "PRC001" in store.active_person_ids()

    store.add_image_quality(
        QualityAssessment(
            person_id="PRC001",
            source_path="fake.jpg",
            passed=True,
            quality_score=0.8,
            sharpness=120.0,
            face_ratio=0.08,
            brightness=110.0,
            pose_score=0.9,
            reasons=[],
        )
    )
    q_rows = store.list_image_quality("PRC001")
    assert len(q_rows) == 1
    assert q_rows[0]["passed"] is True

    result = store.search_embeddings(emb, top_k=1)
    assert len(result) == 1
    assert result[0]["person_id"] == "PRC001"
    active_result = store.search_embeddings_for_person_ids(emb, {"PRC001"}, top_k=1)
    assert len(active_result) == 1
    assert active_result[0]["person_id"] == "PRC001"

    store.delete_person("PRC001", delete_detections=True)
    assert store.get_person("PRC001") is None
    assert store.count_embeddings("PRC001") == 0


def test_query_rows_fallback_filters_generic_equality(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)

    class _QueryFailTable:
        def query(self):
            raise RuntimeError("force fallback path")

        @staticmethod
        def to_arrow():
            return pa.Table.from_pylist(
                [
                    {"incident_id": "INC-1", "entity_id": "PRC001"},
                    {"incident_id": "INC-2", "entity_id": "UNK001"},
                ]
            )

    rows = store._query_rows(_QueryFailTable(), where="incident_id = 'INC-2'", limit=10)
    assert len(rows) == 1
    assert rows[0]["incident_id"] == "INC-2"


def test_incident_schema_migrates_and_persists_severity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    db = lancedb.connect(settings.store.vectors_dir)
    db.create_table(
        "incidents",
        data=[
            {
                "incident_id": "INC-OLD-1",
                "entity_id": "PRC001",
                "status": "open",
                "start_time": "2026-03-27T00:00:00+00:00",
                "last_seen_time": "2026-03-27T00:00:05+00:00",
                "alert_ids": "[]",
                "alert_count": 0,
            }
        ],
        schema=pa.schema(
            [
                pa.field("incident_id", pa.string()),
                pa.field("entity_id", pa.string()),
                pa.field("status", pa.string()),
                pa.field("start_time", pa.string()),
                pa.field("last_seen_time", pa.string()),
                pa.field("alert_ids", pa.string()),
                pa.field("alert_count", pa.int32()),
            ]
        ),
    )

    store = VectorStore(settings)
    migrated = store.get_incident("INC-OLD-1")
    assert migrated is not None
    assert migrated["severity"] == "low"
    assert migrated["last_action_at"] == ""

    assert store.set_incident_severity("INC-OLD-1", "high") is True
    updated = store.get_incident("INC-OLD-1")
    assert updated is not None
    assert updated["severity"] == "high"
