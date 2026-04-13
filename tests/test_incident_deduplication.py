"""Test incident deduplication to prevent duplicate records in database."""

from pathlib import Path

from trace_aml.core.config import load_settings
from trace_aml.core.models import IncidentRecord, IncidentStatus, AlertSeverity
from trace_aml.store.vector_store import VectorStore


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


def test_deduplicate_incidents_removes_duplicates(tmp_path: Path) -> None:
    """Verify that deduplication removes duplicate incident records."""
    settings = _settings(tmp_path)
    store = VectorStore(settings)

    # Create an incident
    inc1 = IncidentRecord(
        incident_id="INC-TEST001",
        entity_id="UNK001",
        status=IncidentStatus.open,
        start_time="2025-01-01T00:00:00Z",
        last_seen_time="2025-01-01T00:00:00Z",
        alert_ids=["ALT-1"],
        alert_count=1,
        severity=AlertSeverity.high,
        summary="Test incident",
    )
    store.create_incident(inc1)

    # Verify it exists
    rows = store._query_rows(store.incidents, where="incident_id = 'INC-TEST001'", limit=100)
    assert len(rows) == 1, "Should have 1 incident after creation"

    # Simulate duplication by directly adding another row with the same ID
    # This can happen due to race conditions in the delete+add pattern
    import json
    import pyarrow as pa

    dup_row = {
        "incident_id": "INC-TEST001",
        "entity_id": "UNK001",
        "status": "open",
        "start_time": "2025-01-01T00:00:00Z",
        "last_seen_time": "2025-01-01T00:00:00Z",
        "alert_ids": json.dumps(["ALT-1"]),
        "alert_count": 1,
        "severity": "high",
        "summary": "Test incident",
        "last_action_at": "",
    }
    store.incidents.add([dup_row])

    # Verify duplication happened
    rows = store._query_rows(store.incidents, where="incident_id = 'INC-TEST001'", limit=100)
    assert len(rows) == 2, "Should have 2 duplicate incidents"

    # Run deduplication
    removed = store.deduplicate_incidents()
    
    # Verify duplicates are removed
    assert removed == 1, "Should have removed 1 duplicate record"
    rows = store._query_rows(store.incidents, where="incident_id = 'INC-TEST001'", limit=100)
    assert len(rows) == 1, "Should have 1 incident after deduplication"


def test_deduplicate_incidents_preserves_latest(tmp_path: Path) -> None:
    """Verify that deduplication keeps the most recently updated record."""
    settings = _settings(tmp_path)
    store = VectorStore(settings)

    # Create initial incident
    inc1 = IncidentRecord(
        incident_id="INC-TEST002",
        entity_id="UNK002",
        status=IncidentStatus.open,
        start_time="2025-01-01T00:00:00Z",
        last_seen_time="2025-01-01T01:00:00Z",  # Earlier time
        alert_ids=["ALT-1"],
        alert_count=1,
        severity=AlertSeverity.high,
        summary="Old incident",
    )
    store.create_incident(inc1)

    # Simulate duplication with updated record
    import json
    newer_row = {
        "incident_id": "INC-TEST002",
        "entity_id": "UNK002",
        "status": "open",
        "start_time": "2025-01-01T00:00:00Z",
        "last_seen_time": "2025-01-01T02:00:00Z",  # Later time
        "alert_ids": json.dumps(["ALT-1", "ALT-2"]),
        "alert_count": 2,
        "severity": "critical",
        "summary": "Updated incident",
        "last_action_at": "",
    }
    store.incidents.add([newer_row])

    # Verify duplication
    rows = store._query_rows(store.incidents, where="incident_id = 'INC-TEST002'", limit=100)
    assert len(rows) == 2, "Should have 2 duplicates"

    # Run deduplication
    removed = store.deduplicate_incidents()
    assert removed == 1, "Should remove 1 duplicate"

    # Verify the newer record is preserved
    rows = store._query_rows(store.incidents, where="incident_id = 'INC-TEST002'", limit=100)
    assert len(rows) == 1, "Should have 1 incident after deduplication"
    
    # Check that the latest record was kept (contains ALT-2)
    record = rows[0]
    alert_ids = json.loads(record.get("alert_ids", "[]"))
    assert "ALT-2" in alert_ids, "Should preserve the newer record with ALT-2"


def test_deduplicate_incidents_no_action_when_no_duplicates(tmp_path: Path) -> None:
    """Verify that deduplication returns 0 when there are no duplicates."""
    settings = _settings(tmp_path)
    store = VectorStore(settings)

    # Create a clean incident
    inc1 = IncidentRecord(
        incident_id="INC-TEST003",
        entity_id="UNK003",
        status=IncidentStatus.open,
        start_time="2025-01-01T00:00:00Z",
        last_seen_time="2025-01-01T00:00:00Z",
        alert_ids=["ALT-1"],
        alert_count=1,
        severity=AlertSeverity.low,
        summary="Clean incident",
    )
    store.create_incident(inc1)

    # Run deduplication
    removed = store.deduplicate_incidents()
    
    # Should return 0 since there are no duplicates
    assert removed == 0, "Should remove 0 records when there are no duplicates"
