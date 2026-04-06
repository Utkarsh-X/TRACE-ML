from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from trace_aml.core.config import load_settings
from trace_aml.core.models import (
    ActionRecord,
    ActionStatus,
    ActionTrigger,
    ActionType,
    AlertRecord,
    AlertSeverity,
    AlertType,
    DecisionState,
    EntityType,
    EventRecord,
    IncidentRecord,
)
from trace_aml.core.streaming import InMemoryEventStreamPublisher
from trace_aml.service.app import create_service_app
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


def _seed(store: VectorStore) -> None:
    store.ensure_entity(
        entity_id="PRC001",
        entity_type=EntityType.known,
        source_person_id="PRC001",
    )
    store.add_event(
        EventRecord(
            event_id="EVT-1",
            entity_id="PRC001",
            timestamp_utc="2026-04-04T10:00:00+00:00",
            confidence=88.0,
            decision=DecisionState.accept,
            track_id="T0001",
            is_unknown=False,
            source="cam:alpha",
        )
    )
    store.add_alert(
        AlertRecord(
            alert_id="ALT-1",
            entity_id="PRC001",
            type=AlertType.reappearance,
            severity=AlertSeverity.medium,
            reason="entity reappeared",
            timestamp_utc="2026-04-04T10:00:05+00:00",
            first_seen_at="2026-04-04T10:00:05+00:00",
            last_seen_at="2026-04-04T10:00:05+00:00",
            event_count=3,
        )
    )
    store.create_incident(
        IncidentRecord(
            incident_id="INC-1",
            entity_id="PRC001",
            start_time="2026-04-04T10:00:06+00:00",
            last_seen_time="2026-04-04T10:00:08+00:00",
            alert_ids=["ALT-1"],
            alert_count=1,
            severity=AlertSeverity.low,
            summary="REAPPEARANCE summary",
        )
    )
    store.insert_action(
        ActionRecord(
            action_id="ACT-1",
            incident_id="INC-1",
            action_type=ActionType.log,
            trigger=ActionTrigger.on_create,
            status=ActionStatus.success,
            reason="logged",
            context={"trigger": "on_create"},
            timestamp_utc="2026-04-04T10:00:07+00:00",
        )
    )


def test_service_endpoints_cover_snapshot_entity_incident_and_timeline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    _seed(store)
    publisher = InMemoryEventStreamPublisher()
    publisher.publish("session.state", {"fps": 25.0})

    app = create_service_app(settings=settings, store=store, stream_publisher=publisher)
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] == "ok"

    snap = client.get("/api/v1/live/snapshot")
    assert snap.status_code == 200
    assert snap.json()["active_incidents"][0]["incident_id"] == "INC-1"

    entities = client.get("/api/v1/entities")
    assert entities.status_code == 200
    assert entities.json()[0]["entity_id"] == "PRC001"

    entity = client.get("/api/v1/entities/PRC001")
    assert entity.status_code == 200
    assert entity.json()["entity_id"] == "PRC001"

    timeline = client.get("/api/v1/timeline", params={"kinds": "alert"})
    assert timeline.status_code == 200
    assert timeline.json()[0]["kind"] == "alert"

    detail = client.get("/api/v1/incidents/INC-1")
    assert detail.status_code == 200
    assert detail.json()["incident"]["severity"] == "low"

    severity = client.patch("/api/v1/incidents/INC-1/severity", json={"severity": "high"})
    assert severity.status_code == 200
    assert severity.json()["incident"]["severity"] == "high"

    missing = client.get("/api/v1/incidents/INC-DOES-NOT-EXIST")
    assert missing.status_code == 404


def test_service_sse_stream_endpoint_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    _seed(store)
    publisher = InMemoryEventStreamPublisher()
    publisher.publish("event.created", {"event_id": "EVT-1"})

    app = create_service_app(settings=settings, store=store, stream_publisher=publisher)
    client = TestClient(app)

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    stream_params = openapi.json()["paths"]["/api/v1/events/stream"]["get"]["parameters"]
    assert all(param["name"] != "request" for param in stream_params)

    stream_route = openapi.json()["paths"]["/api/v1/events/stream"]["get"]
    assert stream_route["responses"]["200"]["description"] == "Successful Response"
