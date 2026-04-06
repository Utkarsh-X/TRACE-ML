from pathlib import Path

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
    DetectionEvent,
    EntityType,
    EventLocation,
    EventRecord,
    IncidentRecord,
    PersonCategory,
    PersonLifecycleStatus,
    PersonRecord,
    TimelineItemKind,
)
from trace_aml.core.streaming import InMemoryEventStreamPublisher
from trace_aml.query.read_models import IntelligenceReadModelService
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


def test_read_models_aggregate_entity_incident_and_timeline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    publisher = InMemoryEventStreamPublisher()

    store.add_or_update_person(
        PersonRecord(
            person_id="PRC001",
            name="Alice",
            category=PersonCategory.criminal,
        )
    )
    store.set_person_state(
        person_id="PRC001",
        lifecycle_state=PersonLifecycleStatus.active,
        lifecycle_reason="ready",
        enrollment_score=0.92,
        valid_embeddings=5,
        valid_images=5,
        total_images=5,
    )
    store.ensure_entity(
        entity_id="PRC001",
        entity_type=EntityType.known,
        source_person_id="PRC001",
    )

    store.add_detection(
        DetectionEvent(
            detection_id="DET-1",
            timestamp_utc="2026-04-04T10:00:00+00:00",
            source="cam:alpha",
            person_id="PRC001",
            name="Alice",
            category="criminal",
            confidence=88.0,
            similarity=0.88,
            smoothed_confidence=89.5,
            bbox=(1, 2, 10, 12),
            track_id="T0001",
            decision_state=DecisionState.accept,
            decision_reason="stable_accept",
            screenshot_path="D:/shots/DET-1.jpg",
        ),
        embedding=[0.0] * 512,
    )
    store.add_detection_decision(
        detection_id="DET-1",
        track_id="T0001",
        decision_state="accept",
        decision_reason="stable_accept",
        smoothed_confidence=89.5,
        quality_flags=[],
        top_candidates=[{"person_id": "PRC001"}],
        liveness_provider="none",
        liveness_score=1.0,
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
            detection_id="DET-1",
            source="cam:alpha",
            location=EventLocation(lat=12.34, lon=56.78, source="cam:alpha"),
        )
    )
    store.add_alert(
        AlertRecord(
            alert_id="ALT-1",
            entity_id="PRC001",
            type=AlertType.reappearance,
            severity=AlertSeverity.medium,
            reason="entity reappeared in sector 7",
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
            severity=AlertSeverity.high,
            summary="REAPPEARANCE: entity reappeared in sector 7",
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
            context={"trigger": "on_create", "explanation": "policy matched"},
            timestamp_utc="2026-04-04T10:00:07+00:00",
        )
    )

    publisher.publish(
        "session.state",
        {
            "fps": 27.5,
            "active_tracks": 1,
            "latency_ms": 42.0,
        },
    )

    service = IntelligenceReadModelService(store, publisher)

    recent_alerts = service.get_recent_alerts(limit=5)
    assert recent_alerts[0].reason == "entity reappeared in sector 7"

    entity = service.get_entity("PRC001")
    assert entity.name == "Alice"
    assert entity.open_incident_count == 1

    incidents = service.get_entity_incidents("PRC001")
    assert len(incidents) == 1
    assert incidents[0].summary.startswith("REAPPEARANCE")

    entity_timeline = service.get_entity_timeline("PRC001")
    assert [item.kind for item in entity_timeline] == [
        TimelineItemKind.event,
        TimelineItemKind.alert,
        TimelineItemKind.incident,
        TimelineItemKind.action,
    ]

    profile = service.get_entity_profile("PRC001")
    assert profile.entity.entity_id == "PRC001"
    assert profile.linked_person is not None
    assert profile.screenshot_paths == ["D:/shots/DET-1.jpg"]
    assert profile.recent_detections[0]["bbox"] == [1, 2, 10, 12]

    incident_detail = service.get_incident_detail("INC-1")
    assert incident_detail.incident.summary.startswith("REAPPEARANCE")
    assert incident_detail.actions[0].context["trigger"] == "on_create"
    assert incident_detail.timeline[-1].kind == TimelineItemKind.action

    global_timeline = service.get_global_timeline(
        start="2026-04-04T09:59:59+00:00",
        end="2026-04-04T10:00:10+00:00",
    )
    assert len(global_timeline) == 4
    assert global_timeline[0].location.source == "cam:alpha"

    snapshot = service.get_live_ops_snapshot()
    assert snapshot.active_entities[0].entity_id == "PRC001"
    assert snapshot.active_incidents[0].incident_id == "INC-1"
    assert snapshot.system_health.open_incident_count == 1
    assert snapshot.system_health.runtime["fps"] == 27.5


def test_in_memory_stream_publisher_supports_subscriptions() -> None:
    publisher = InMemoryEventStreamPublisher()
    received: list[str] = []

    unsubscribe = publisher.subscribe(lambda event: received.append(event.topic))
    publisher.publish("event", {"id": 1})
    publisher.publish("alert", {"id": 2})
    unsubscribe()
    publisher.publish("incident", {"id": 3})

    assert received == ["event", "alert"]
    assert publisher.subscriber_count() == 0
    assert publisher.latest("alert") is not None
    assert publisher.latest("alert").payload["id"] == 2


def test_global_timeline_supports_filters_and_stable_ordering(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    service = IntelligenceReadModelService(store)

    store.ensure_entity(entity_id="PRC001", entity_type=EntityType.known, source_person_id="PRC001")
    store.add_event(
        EventRecord(
            event_id="EVT-2",
            entity_id="PRC001",
            timestamp_utc="2026-04-04T10:00:00+00:00",
            confidence=70.0,
            decision=DecisionState.review,
            detection_id="DET-2",
            source="cam:beta",
        )
    )
    store.add_alert(
        AlertRecord(
            alert_id="ALT-2",
            entity_id="PRC001",
            type=AlertType.reappearance,
            severity=AlertSeverity.low,
            reason="REAPPEARANCE detected with 3 events",
            timestamp_utc="2026-04-04T10:00:00+00:00",
            event_count=3,
        )
    )
    store.create_incident(
        IncidentRecord(
            incident_id="INC-2",
            entity_id="PRC001",
            start_time="2026-04-04T10:00:00+00:00",
            last_seen_time="2026-04-04T10:00:05+00:00",
            alert_ids=["ALT-2"],
            alert_count=1,
            severity=AlertSeverity.medium,
            summary="REAPPEARANCE summary",
        )
    )
    store.insert_action(
        ActionRecord(
            action_id="ACT-2",
            incident_id="INC-2",
            action_type=ActionType.log,
            trigger=ActionTrigger.on_create,
            status=ActionStatus.success,
            reason="logged",
            context={"trigger": "on_create"},
            timestamp_utc="2026-04-04T10:00:00+00:00",
        )
    )

    ordered = service.get_global_timeline(
        start="2026-04-04T09:59:59+00:00",
        end="2026-04-04T10:00:01+00:00",
    )
    assert [item.kind for item in ordered] == [
        TimelineItemKind.event,
        TimelineItemKind.alert,
        TimelineItemKind.incident,
        TimelineItemKind.action,
    ]

    only_alerts = service.get_global_timeline(
        start="2026-04-04T09:59:59+00:00",
        end="2026-04-04T10:00:01+00:00",
        kinds=["alert"],
    )
    assert len(only_alerts) == 1
    assert only_alerts[0].kind == TimelineItemKind.alert

    incident_view = service.get_global_timeline(incident_id="INC-2")
    assert len(incident_view) == 4
    assert incident_view[0].kind == TimelineItemKind.event
