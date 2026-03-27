from pathlib import Path

from trace_ml.core.config import load_settings
from trace_ml.core.models import AlertRecord, AlertSeverity, AlertType
from trace_ml.pipeline.incident_manager import IncidentManager
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


def test_incident_manager_creates_and_updates_single_active_incident(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    manager = IncidentManager(store)

    a1 = AlertRecord(
        alert_id="ALT-1",
        entity_id="UNK001",
        type=AlertType.unknown_recurrence,
        severity=AlertSeverity.high,
        reason="unknown recurrence",
    )
    inc1, trigger1 = manager.handle_alert(a1)
    assert trigger1 == "on_create"
    assert inc1.entity_id == "UNK001"
    assert inc1.alert_count == 1

    a2 = AlertRecord(
        alert_id="ALT-2",
        entity_id="UNK001",
        type=AlertType.reappearance,
        severity=AlertSeverity.medium,
        reason="reappearance",
    )
    inc2, trigger2 = manager.handle_alert(a2)
    assert trigger2 == "on_update"
    assert inc2.incident_id == inc1.incident_id
    assert inc2.alert_count == 2
    assert set(inc2.alert_ids) == {"ALT-1", "ALT-2"}

    open_rows = store.list_incidents(status="open")
    assert len([row for row in open_rows if row.get("entity_id") == "UNK001"]) == 1


def test_incident_manager_creates_new_incident_after_close(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    manager = IncidentManager(store)

    first, _ = manager.handle_alert(
        AlertRecord(
            alert_id="ALT-A",
            entity_id="PRC001",
            type=AlertType.reappearance,
            severity=AlertSeverity.low,
            reason="first",
        )
    )
    assert store.close_incident(first.incident_id) is True

    second, trigger = manager.handle_alert(
        AlertRecord(
            alert_id="ALT-B",
            entity_id="PRC001",
            type=AlertType.reappearance,
            severity=AlertSeverity.low,
            reason="second",
        )
    )
    assert trigger == "on_create"
    assert second.incident_id != first.incident_id
    assert len(store.list_incidents(status="open")) == 1
    assert len(store.list_incidents(status="closed")) == 1
