"""ID generation helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from trace_ml.core.models import PersonCategory


def build_person_prefix(category: PersonCategory) -> str:
    return "PRC" if category == PersonCategory.criminal else "PRM"


def next_person_id(category: PersonCategory, existing_ids: list[str]) -> str:
    prefix = build_person_prefix(category)
    max_num = 0
    for person_id in existing_ids:
        if person_id.startswith(prefix):
            try:
                max_num = max(max_num, int(person_id[3:]))
            except ValueError:
                continue
    return f"{prefix}{max_num + 1:03d}"


def new_detection_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"DET-{ts}-{uuid.uuid4().hex[:8]}"


def new_event_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"EVT-{ts}-{uuid.uuid4().hex[:8]}"


def new_alert_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ALT-{ts}-{uuid.uuid4().hex[:8]}"


def new_incident_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"INC-{ts}-{uuid.uuid4().hex[:8]}"


def new_action_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ACT-{ts}-{uuid.uuid4().hex[:8]}"


def new_embedding_id(person_id: str) -> str:
    return f"EMB-{person_id}-{uuid.uuid4().hex[:10]}"


def next_unknown_entity_id(existing_ids: list[str]) -> str:
    max_num = 0
    for entity_id in existing_ids:
        if entity_id.startswith("UNK"):
            try:
                max_num = max(max_num, int(entity_id[3:]))
            except ValueError:
                continue
    return f"UNK{max_num + 1:03d}"
