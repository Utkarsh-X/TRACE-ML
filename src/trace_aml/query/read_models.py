"""UI-ready read models and timeline aggregation for TRACE-AML."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any

from trace_aml.core.models import (
    ActionRecord,
    AlertRecord,
    EntityProfile,
    EntitySummary,
    EventLocation,
    IncidentDetail,
    IncidentSummary,
    LiveOpsSnapshot,
    SystemHealthSnapshot,
    TimelineItem,
    TimelineItemKind,
)
from trace_aml.core.streaming import EventStreamPublisher, NullEventStreamPublisher
from trace_aml.store.vector_store import VectorStore


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _within_range(timestamp_utc: str, start: str | None, end: str | None) -> bool:
    ts = _parse_iso(timestamp_utc)
    if ts is None:
        return False
    if start:
        start_ts = _parse_iso(start)
        if start_ts and ts < start_ts:
            return False
    if end:
        end_ts = _parse_iso(end)
        if end_ts and ts > end_ts:
            return False
    return True


def _coerce_bbox(raw: Any) -> list[int]:
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            import json

            parsed = json.loads(text)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        values = parsed
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


class IntelligenceReadModelService:
    def __init__(
        self,
        store: VectorStore,
        stream_publisher: EventStreamPublisher | None = None,
        snapshot_ttl_seconds: float = 1.5,
    ) -> None:
        self.store = store
        self.stream_publisher = stream_publisher or NullEventStreamPublisher()
        self.snapshot_ttl_seconds = max(0.0, float(snapshot_ttl_seconds))
        self._snapshot_cache: dict[tuple[int, int, int], tuple[float, LiveOpsSnapshot]] = {}

    @staticmethod
    def _location_from(raw: Any, source: str = "") -> EventLocation:
        if isinstance(raw, EventLocation):
            return raw
        payload = raw if isinstance(raw, dict) else {}
        lat = payload.get("lat")
        lon = payload.get("lon")
        try:
            lat_value = float(lat) if lat is not None else None
        except Exception:
            lat_value = None
        try:
            lon_value = float(lon) if lon is not None else None
        except Exception:
            lon_value = None
        return EventLocation(
            lat=lat_value,
            lon=lon_value,
            source=str(payload.get("source", source or "")),
        )

    @staticmethod
    def _alert_from_row(row: dict[str, Any]) -> AlertRecord:
        return AlertRecord(**row)

    @staticmethod
    def _action_from_row(row: dict[str, Any]) -> ActionRecord:
        return ActionRecord(**row)

    def _entity_person(self, entity_row: dict[str, Any]) -> dict[str, Any] | None:
        entity_id = str(entity_row.get("entity_id", ""))
        person_id = str(entity_row.get("source_person_id", "")) or entity_id
        if str(entity_row.get("type", "")) != "known":
            return None
        return self.store.get_person(person_id)

    def _entity_summary_from_row(self, row: dict[str, Any]) -> EntitySummary:
        entity_id = str(row.get("entity_id", ""))
        person = self._entity_person(row)
        incidents = [r for r in self.store.list_incidents(limit=10_000, status="open") if str(r.get("entity_id", "")) == entity_id]
        recent_alert_count = len(self.store.list_alerts(limit=10_000, entity_id=entity_id))
        return EntitySummary(
            entity_id=entity_id,
            type=str(row.get("type", "unknown")),
            status=str(row.get("status", "active")),
            name=str((person or {}).get("name", entity_id if entity_id.startswith("UNK") else "Unknown")),
            category=str((person or {}).get("category", "unknown")),
            person_id=str((person or {}).get("person_id", "")),
            created_at=str(row.get("created_at", "")),
            last_seen_at=str(row.get("last_seen_at", "")),
            open_incident_count=len(incidents),
            recent_alert_count=recent_alert_count,
        )

    @staticmethod
    def _incident_summary_from_row(row: dict[str, Any]) -> IncidentSummary:
        return IncidentSummary(
            incident_id=str(row.get("incident_id", "")),
            entity_id=str(row.get("entity_id", "")),
            status=str(row.get("status", "open")),
            severity=str(row.get("severity", "low")),
            summary=str(row.get("summary", "")),
            start_time=str(row.get("start_time", "")),
            last_seen_time=str(row.get("last_seen_time", "")),
            alert_count=int(row.get("alert_count", 0)),
            last_action_at=str(row.get("last_action_at", "")),
        )

    def _detection_map(self) -> dict[str, dict[str, Any]]:
        rows = self.store.list_detections(limit=100_000)
        normalized: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            item["bbox"] = _coerce_bbox(item.get("bbox", []))
            normalized[str(item.get("detection_id", ""))] = item
        return normalized

    def _entity_detections(
        self,
        entity_id: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        entity_row = self.store.get_entity(entity_id)
        person = self._entity_person(entity_row or {})
        person_id = str((person or {}).get("person_id", ""))
        detections = self.store.list_detections(limit=100_000)
        events = self.store.list_events(limit=100_000, entity_id=entity_id)
        event_detection_ids = {str(row.get("detection_id", "")) for row in events if str(row.get("detection_id", ""))}

        kept: list[dict[str, Any]] = []
        for row in detections:
            det_id = str(row.get("detection_id", ""))
            if person_id and str(row.get("person_id", "")) == person_id:
                pass
            elif det_id in event_detection_ids:
                pass
            else:
                continue
            if not _within_range(str(row.get("timestamp_utc", "")), start, end):
                continue
            item = dict(row)
            item["bbox"] = _coerce_bbox(item.get("bbox", []))
            kept.append(item)
        kept.sort(key=lambda r: str(r.get("timestamp_utc", "")), reverse=True)
        return kept[:limit]

    def _timeline_items_for_events(
        self,
        rows: list[dict[str, Any]],
        detection_map: dict[str, dict[str, Any]],
        incident_id: str = "",
    ) -> list[TimelineItem]:
        items: list[TimelineItem] = []
        for row in rows:
            det = detection_map.get(str(row.get("detection_id", "")), {})
            decision = str(row.get("decision", "")).upper()
            confidence = float(row.get("confidence", 0.0))
            source = str(row.get("source", ""))
            items.append(
                TimelineItem(
                    item_id=str(row.get("event_id", "")),
                    kind=TimelineItemKind.event,
                    timestamp_utc=str(row.get("timestamp_utc", "")),
                    entity_id=str(row.get("entity_id", "")),
                    incident_id=incident_id,
                    title=f"EVENT {decision or 'UNKNOWN'}",
                    summary=f"{decision.lower() or 'event'} at {confidence:.2f}% on {source}",
                    source=source,
                    screenshot_path=str(det.get("screenshot_path", "")),
                    location=self._location_from(row.get("location", {}), source=source),
                    metadata={
                        "track_id": str(row.get("track_id", "")),
                        "is_unknown": bool(row.get("is_unknown", False)),
                        "detection_id": str(row.get("detection_id", "")),
                    },
                )
            )
        return items

    @staticmethod
    def _timeline_items_for_alerts(rows: list[dict[str, Any]], incident_id: str = "") -> list[TimelineItem]:
        items: list[TimelineItem] = []
        for row in rows:
            items.append(
                TimelineItem(
                    item_id=str(row.get("alert_id", "")),
                    kind=TimelineItemKind.alert,
                    timestamp_utc=str(row.get("timestamp_utc", "")),
                    entity_id=str(row.get("entity_id", "")),
                    incident_id=incident_id,
                    severity=str(row.get("severity", "")),
                    title=f"ALERT {str(row.get('type', '')).upper()}",
                    summary=str(row.get("reason", "")),
                    source="rules_engine",
                    metadata={"event_count": int(row.get("event_count", 1))},
                )
            )
        return items

    @staticmethod
    def _timeline_items_for_incidents(rows: list[dict[str, Any]]) -> list[TimelineItem]:
        items: list[TimelineItem] = []
        for row in rows:
            items.append(
                TimelineItem(
                    item_id=str(row.get("incident_id", "")),
                    kind=TimelineItemKind.incident,
                    timestamp_utc=str(row.get("start_time", "")),
                    entity_id=str(row.get("entity_id", "")),
                    incident_id=str(row.get("incident_id", "")),
                    severity=str(row.get("severity", "")),
                    title=f"INCIDENT {str(row.get('status', 'open')).upper()}",
                    summary=str(row.get("summary", "")),
                    source="incident_manager",
                    metadata={"alert_count": int(row.get("alert_count", 0))},
                )
            )
        return items

    @staticmethod
    def _timeline_items_for_actions(rows: list[dict[str, Any]], incidents_by_id: dict[str, dict[str, Any]]) -> list[TimelineItem]:
        items: list[TimelineItem] = []
        for row in rows:
            incident_id = str(row.get("incident_id", ""))
            incident = incidents_by_id.get(incident_id, {})
            items.append(
                TimelineItem(
                    item_id=str(row.get("action_id", "")),
                    kind=TimelineItemKind.action,
                    timestamp_utc=str(row.get("timestamp_utc", "")),
                    entity_id=str(incident.get("entity_id", "")),
                    incident_id=incident_id,
                    severity=str(incident.get("severity", "")),
                    title=f"ACTION {str(row.get('action_type', '')).upper()}",
                    summary=str(row.get("reason", "")),
                    source="policy_engine",
                    metadata={
                        "trigger": str(row.get("trigger", "")),
                        "status": str(row.get("status", "")),
                        "context": dict(row.get("context", {})),
                    },
                )
            )
        return items

    @staticmethod
    def _sort_timeline(items: list[TimelineItem]) -> list[TimelineItem]:
        kind_order = {
            TimelineItemKind.event: 0,
            TimelineItemKind.alert: 1,
            TimelineItemKind.incident: 2,
            TimelineItemKind.action: 3,
        }

        def _timestamp_value(item: TimelineItem) -> float:
            parsed = _parse_iso(item.timestamp_utc)
            return parsed.timestamp() if parsed else float("-inf")

        return sorted(
            items,
            key=lambda item: (
                _timestamp_value(item),
                kind_order.get(item.kind, 99),
                item.item_id,
            ),
        )

    @staticmethod
    def _normalize_kind_filters(kinds: list[str] | None) -> set[str]:
        if not kinds:
            return set()
        normalized: set[str] = set()
        for kind in kinds:
            value = str(kind).strip().lower()
            if not value:
                continue
            try:
                normalized.add(TimelineItemKind(value).value)
            except Exception:
                continue
        return normalized

    def get_recent_alerts(self, limit: int = 20) -> list[AlertRecord]:
        return [self._alert_from_row(row) for row in self.store.list_alerts(limit=limit)]

    def list_entities(
        self,
        limit: int = 200,
        type_filter: str | None = None,
        status: str | None = None,
    ) -> list[EntitySummary]:
        rows = self.store.list_entities(limit=max(limit, 10_000), type_filter=type_filter or None)
        if status:
            rows = [row for row in rows if str(row.get("status", "")).lower() == status.lower()]
        rows.sort(key=lambda row: str(row.get("last_seen_at", "")), reverse=True)
        return [self._entity_summary_from_row(row) for row in rows[:limit]]

    def list_incidents(
        self,
        limit: int = 200,
        status: str | None = None,
        entity_id: str | None = None,
    ) -> list[IncidentSummary]:
        rows = self.store.list_incidents(limit=max(limit, 10_000), status=status or None)
        if entity_id:
            rows = [row for row in rows if str(row.get("entity_id", "")) == entity_id]
        rows.sort(key=lambda row: str(row.get("last_seen_time", "")), reverse=True)
        return [self._incident_summary_from_row(row) for row in rows[:limit]]

    def list_actions(
        self,
        limit: int = 200,
        incident_id: str | None = None,
    ) -> list[ActionRecord]:
        rows = self.store.list_actions(limit=max(limit, 10_000), incident_id=incident_id or None)
        rows.sort(key=lambda row: str(row.get("timestamp_utc", "")), reverse=True)
        return [self._action_from_row(row) for row in rows[:limit]]

    def get_entity(self, entity_id: str) -> EntitySummary:
        row = self.store.get_entity(entity_id)
        if not row:
            raise ValueError(f"Entity not found: {entity_id}")
        return self._entity_summary_from_row(row)

    def get_entity_incidents(self, entity_id: str) -> list[IncidentSummary]:
        rows = [row for row in self.store.list_incidents(limit=10_000) if str(row.get("entity_id", "")) == entity_id]
        rows.sort(key=lambda row: str(row.get("last_seen_time", "")), reverse=True)
        return [self._incident_summary_from_row(row) for row in rows]

    def get_entity_timeline(
        self,
        entity_id: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[TimelineItem]:
        events = [
            row for row in self.store.list_events(limit=100_000, entity_id=entity_id) if _within_range(str(row.get("timestamp_utc", "")), start, end)
        ]
        alerts = [
            row for row in self.store.list_alerts(limit=100_000, entity_id=entity_id) if _within_range(str(row.get("timestamp_utc", "")), start, end)
        ]
        incidents = [
            row
            for row in self.store.list_incidents(limit=10_000)
            if str(row.get("entity_id", "")) == entity_id and _within_range(str(row.get("start_time", "")), start, end)
        ]
        incident_ids = {str(row.get("incident_id", "")) for row in incidents}
        actions = [
            row
            for row in self.store.list_actions(limit=100_000)
            if str(row.get("incident_id", "")) in incident_ids and _within_range(str(row.get("timestamp_utc", "")), start, end)
        ]
        detection_map = self._detection_map()
        incidents_by_id = {str(row.get("incident_id", "")): row for row in incidents}
        items = []
        items.extend(self._timeline_items_for_events(events, detection_map))
        items.extend(self._timeline_items_for_alerts(alerts))
        items.extend(self._timeline_items_for_incidents(incidents))
        items.extend(self._timeline_items_for_actions(actions, incidents_by_id))
        timeline = self._sort_timeline(items)
        return timeline[-limit:] if limit > 0 else timeline

    def get_entity_profile(self, entity_id: str) -> EntityProfile:
        entity_summary = self.get_entity(entity_id)
        entity_row = self.store.get_entity(entity_id)
        linked_person = self._entity_person(entity_row or {})
        incidents = self.get_entity_incidents(entity_id)
        recent_alerts = [self._alert_from_row(row) for row in self.store.list_alerts(limit=20, entity_id=entity_id)]
        recent_detections = self._entity_detections(entity_id, limit=20)
        timeline = self.get_entity_timeline(entity_id, limit=200)
        screenshots = []
        for row in recent_detections:
            path = str(row.get("screenshot_path", ""))
            if path and path not in screenshots:
                screenshots.append(path)
        return EntityProfile(
            entity=entity_summary,
            linked_person=linked_person,
            incidents=incidents,
            recent_alerts=recent_alerts,
            recent_detections=recent_detections,
            timeline=timeline,
            screenshot_paths=screenshots,
            stats={
                "timeline_items": len(timeline),
                "incident_count": len(incidents),
                "detection_count": len(recent_detections),
                "recent_alert_count": len(recent_alerts),
            },
        )

    def get_incident_detail(self, incident_id: str) -> IncidentDetail:
        incident_row = self.store.get_incident(incident_id)
        if not incident_row:
            raise ValueError(f"Incident not found: {incident_id}")
        incident_summary = self._incident_summary_from_row(incident_row)
        entity_summary = self.get_entity(str(incident_row.get("entity_id", "")))
        alert_ids = {str(value) for value in incident_row.get("alert_ids", [])}
        alerts = [
            row
            for row in self.store.list_alerts(limit=100_000, entity_id=incident_summary.entity_id)
            if str(row.get("alert_id", "")) in alert_ids
        ]
        actions = self.store.get_actions(incident_id, limit=10_000)
        events = [
            row
            for row in self.store.list_events(limit=100_000, entity_id=incident_summary.entity_id)
            if _within_range(str(row.get("timestamp_utc", "")), incident_summary.start_time, incident_summary.last_seen_time)
        ]
        detection_map = self._detection_map()
        incidents_by_id = {incident_id: incident_row}
        timeline = []
        timeline.extend(self._timeline_items_for_events(events, detection_map, incident_id=incident_id))
        timeline.extend(self._timeline_items_for_alerts(alerts, incident_id=incident_id))
        timeline.extend(self._timeline_items_for_incidents([incident_row]))
        timeline.extend(self._timeline_items_for_actions(actions, incidents_by_id))
        recent_detections = self._entity_detections(
            incident_summary.entity_id,
            start=incident_summary.start_time,
            end=incident_summary.last_seen_time,
            limit=20,
        )
        return IncidentDetail(
            incident=incident_summary,
            entity=entity_summary,
            alerts=[self._alert_from_row(row) for row in alerts],
            actions=[self._action_from_row(row) for row in actions],
            recent_detections=recent_detections,
            timeline=self._sort_timeline(timeline),
        )

    def get_global_timeline(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1_000,
        entity_id: str | None = None,
        incident_id: str | None = None,
        kinds: list[str] | None = None,
    ) -> list[TimelineItem]:
        if incident_id:
            detail = self.get_incident_detail(incident_id)
            timeline = detail.timeline
            if start or end:
                timeline = [
                    item for item in timeline if _within_range(item.timestamp_utc, start=start, end=end)
                ]
            kind_filters = self._normalize_kind_filters(kinds)
            if kind_filters:
                timeline = [item for item in timeline if item.kind.value in kind_filters]
            return timeline[-limit:] if limit > 0 else timeline

        events = [row for row in self.store.list_events(limit=100_000) if _within_range(str(row.get("timestamp_utc", "")), start, end)]
        alerts = [row for row in self.store.list_alerts(limit=100_000) if _within_range(str(row.get("timestamp_utc", "")), start, end)]
        incidents = [row for row in self.store.list_incidents(limit=10_000) if _within_range(str(row.get("start_time", "")), start, end)]
        actions = [row for row in self.store.list_actions(limit=100_000) if _within_range(str(row.get("timestamp_utc", "")), start, end)]
        detection_map = self._detection_map()
        incidents_by_id = {str(row.get("incident_id", "")): row for row in self.store.list_incidents(limit=10_000)}
        items = []
        items.extend(self._timeline_items_for_events(events, detection_map))
        items.extend(self._timeline_items_for_alerts(alerts))
        items.extend(self._timeline_items_for_incidents(incidents))
        items.extend(self._timeline_items_for_actions(actions, incidents_by_id))
        kind_filters = self._normalize_kind_filters(kinds)
        if entity_id:
            items = [item for item in items if item.entity_id == entity_id]
        if kind_filters:
            items = [item for item in items if item.kind.value in kind_filters]
        timeline = self._sort_timeline(items)
        return timeline[-limit:] if limit > 0 else timeline

    def get_live_ops_snapshot(
        self,
        entity_limit: int = 12,
        incident_limit: int = 12,
        alert_limit: int = 12,
    ) -> LiveOpsSnapshot:
        cache_key = (entity_limit, incident_limit, alert_limit)
        if self.snapshot_ttl_seconds > 0:
            cached = self._snapshot_cache.get(cache_key)
            if cached:
                cached_at, snapshot = cached
                if (time.time() - cached_at) <= self.snapshot_ttl_seconds:
                    return snapshot

        entity_rows = [row for row in self.store.list_entities(limit=100_000) if str(row.get("status", "")) == "active"]
        entity_rows.sort(key=lambda row: str(row.get("last_seen_at", "")), reverse=True)
        active_entities = [self._entity_summary_from_row(row) for row in entity_rows[:entity_limit]]

        incident_rows = self.store.list_incidents(limit=incident_limit, status="open")
        active_incidents = [self._incident_summary_from_row(row) for row in incident_rows]

        recent_alerts = self.get_recent_alerts(alert_limit)
        latest_event = self.store.list_events(limit=1)
        latest_alert = self.store.list_alerts(limit=1)
        runtime_event = self.stream_publisher.latest("session.state")

        # Use lightweight counting: avoid loading 100K+ rows just for a count.
        # list_incidents is already called above (reuse its result).
        # For detections, use a bounded query instead of fetching everything.
        try:
            detection_sample = self.store.list_detections(limit=1)
            # If the store has a count method, use it; otherwise estimate from the sample.
            total_det = getattr(self.store, '_detection_count', len(detection_sample))
        except Exception:
            total_det = 0

        system_health = SystemHealthSnapshot(
            active_entity_count=len(entity_rows),
            open_incident_count=len(incident_rows),
            recent_alert_count=len(recent_alerts),
            total_detection_count=total_det,
            latest_event_at=str(latest_event[0].get("timestamp_utc", "")) if latest_event else "",
            latest_alert_at=str(latest_alert[0].get("timestamp_utc", "")) if latest_alert else "",
            publisher_subscribers=self.stream_publisher.subscriber_count(),
            last_published_at=self.stream_publisher.last_published_at(),
            runtime=dict(runtime_event.payload) if runtime_event else {},
        )
        snapshot = LiveOpsSnapshot(
            active_entities=active_entities,
            active_incidents=active_incidents,
            recent_alerts=recent_alerts,
            system_health=system_health,
        )
        if self.snapshot_ttl_seconds > 0:
            self._snapshot_cache[cache_key] = (time.time(), snapshot)
        return snapshot


def get_live_ops_snapshot(store: VectorStore, stream_publisher: EventStreamPublisher | None = None) -> LiveOpsSnapshot:
    return IntelligenceReadModelService(store, stream_publisher).get_live_ops_snapshot()


def get_entity(
    store: VectorStore,
    entity_id: str,
    stream_publisher: EventStreamPublisher | None = None,
) -> EntitySummary:
    return IntelligenceReadModelService(store, stream_publisher).get_entity(entity_id)


def get_entity_timeline(
    store: VectorStore,
    entity_id: str,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[TimelineItem]:
    return IntelligenceReadModelService(store, stream_publisher).get_entity_timeline(entity_id)


def get_entity_incidents(
    store: VectorStore,
    entity_id: str,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[IncidentSummary]:
    return IntelligenceReadModelService(store, stream_publisher).get_entity_incidents(entity_id)


def get_incident_detail(
    store: VectorStore,
    incident_id: str,
    stream_publisher: EventStreamPublisher | None = None,
) -> IncidentDetail:
    return IntelligenceReadModelService(store, stream_publisher).get_incident_detail(incident_id)


def get_entity_profile(
    store: VectorStore,
    entity_id: str,
    stream_publisher: EventStreamPublisher | None = None,
) -> EntityProfile:
    return IntelligenceReadModelService(store, stream_publisher).get_entity_profile(entity_id)


def get_global_timeline(
    store: VectorStore,
    start: str | None = None,
    end: str | None = None,
    stream_publisher: EventStreamPublisher | None = None,
    *,
    limit: int = 1_000,
    entity_id: str | None = None,
    incident_id: str | None = None,
    kinds: list[str] | None = None,
) -> list[TimelineItem]:
    return IntelligenceReadModelService(store, stream_publisher).get_global_timeline(
        start=start,
        end=end,
        limit=limit,
        entity_id=entity_id,
        incident_id=incident_id,
        kinds=kinds,
    )


def get_recent_alerts(
    store: VectorStore,
    limit: int = 20,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[AlertRecord]:
    return IntelligenceReadModelService(store, stream_publisher).get_recent_alerts(limit=limit)


def list_entities(
    store: VectorStore,
    limit: int = 200,
    type_filter: str | None = None,
    status: str | None = None,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[EntitySummary]:
    return IntelligenceReadModelService(store, stream_publisher).list_entities(
        limit=limit,
        type_filter=type_filter,
        status=status,
    )


def list_incidents(
    store: VectorStore,
    limit: int = 200,
    status: str | None = None,
    entity_id: str | None = None,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[IncidentSummary]:
    return IntelligenceReadModelService(store, stream_publisher).list_incidents(
        limit=limit,
        status=status,
        entity_id=entity_id,
    )


def list_actions(
    store: VectorStore,
    limit: int = 200,
    incident_id: str | None = None,
    stream_publisher: EventStreamPublisher | None = None,
) -> list[ActionRecord]:
    return IntelligenceReadModelService(store, stream_publisher).list_actions(
        limit=limit,
        incident_id=incident_id,
    )
