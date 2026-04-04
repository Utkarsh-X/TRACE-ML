from pathlib import Path

from trace_ml.core.config import load_settings
from trace_ml.core.models import ActionTrigger, ActionType, AlertSeverity, IncidentRecord
from trace_ml.pipeline.action_engine import ActionEngine
from trace_ml.pipeline.policy_engine import PolicyEngine
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
actions:
  enabled: true
  on_create:
    low: []
    medium: [log]
    high: [log, email, alarm]
  on_update:
    low: []
    medium: [log]
    high: [log]
  cooldown_sec: 60
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_policy_engine_returns_actions_by_trigger_and_severity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    policy = PolicyEngine(settings)
    incident = IncidentRecord(
        incident_id="INC-1",
        entity_id="PRC001",
        severity=AlertSeverity.high,
    )

    on_create = policy.evaluate(incident, ActionTrigger.on_create)
    on_update = policy.evaluate(incident, ActionTrigger.on_update)

    assert [a.value for a in on_create] == ["log", "email", "alarm"]
    assert [a.value for a in on_update] == ["log"]


def test_action_engine_executes_and_respects_cooldown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    engine = ActionEngine(store, settings)
    incident = IncidentRecord(
        incident_id="INC-2",
        entity_id="UNK002",
        severity=AlertSeverity.high,
    )
    store.create_incident(incident)

    first = engine.execute(
        incident=incident,
        actions=[ActionType.log, ActionType.email],
        trigger=ActionTrigger.on_create,
    )
    assert len(first) == 2
    stored_actions = store.get_actions("INC-2")
    assert len(stored_actions) == 2
    assert stored_actions[0]["context"]["trigger"] == "on_create"
    assert stored_actions[0]["context"]["incident_severity"] == "high"

    refreshed = IncidentRecord(**(store.get_incident("INC-2") or {}))
    second = engine.execute(
        incident=refreshed,
        actions=[ActionType.log],
        trigger=ActionTrigger.on_update,
    )
    assert second == []
    assert len(store.get_actions("INC-2")) == 2
