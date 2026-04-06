from pathlib import Path

from trace_aml.core.config import load_settings
from trace_aml.core.models import AlertType, DecisionState, EventRecord
from trace_aml.pipeline.rules_engine import RulesEngine
from trace_aml.store.vector_store import VectorStore


def _settings(tmp_path: Path, text: str = ""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        (
            f"""
camera:
  device_index: 0
store:
  root: {tmp_path.as_posix()}/data
  vectors_dir: {tmp_path.as_posix()}/data/vectors
  screenshots_dir: {tmp_path.as_posix()}/data/screens
  exports_dir: {tmp_path.as_posix()}/data/exports
rules:
  cooldown_sec: 60
  reappearance:
    window_sec: 20
    min_events: 3
  unknown:
    window_sec: 20
    min_events: 3
  instability:
    window_sec: 20
    std_threshold: 0.15
"""
            + text
        ).strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_rules_reappearance_respects_cooldown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    engine = RulesEngine(settings, store)

    emitted = []
    for idx in range(3):
        event = EventRecord(
            event_id=f"EVT-{idx}",
            entity_id="PRC001",
            confidence=82.0,
            decision=DecisionState.accept,
            track_id="T1",
            is_unknown=False,
        )
        store.add_event(event)
        emitted.extend(engine.process_event(event))

    assert len([a for a in emitted if a.type == AlertType.reappearance]) == 1

    # Next event arrives inside cooldown; reappearance should not re-trigger.
    event4 = EventRecord(
        event_id="EVT-4",
        entity_id="PRC001",
        confidence=81.5,
        decision=DecisionState.accept,
        track_id="T1",
        is_unknown=False,
    )
    store.add_event(event4)
    alerts4 = engine.process_event(event4)
    assert not any(a.type == AlertType.reappearance for a in alerts4)


def test_rules_unknown_recurrence_is_high(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    engine = RulesEngine(settings, store)

    final_alerts = []
    for idx in range(3):
        event = EventRecord(
            event_id=f"EVT-U-{idx}",
            entity_id="UNK001",
            confidence=35.0,
            decision=DecisionState.review,
            track_id="T9",
            is_unknown=True,
        )
        store.add_event(event)
        final_alerts = engine.process_event(event)

    unknown = [a for a in final_alerts if a.type == AlertType.unknown_recurrence]
    assert unknown
    assert unknown[0].severity.value == "high"


def test_rules_instability_uses_std_threshold(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        text="""
rules:
  cooldown_sec: 0
  reappearance:
    window_sec: 20
    min_events: 99
  unknown:
    window_sec: 20
    min_events: 99
  instability:
    window_sec: 20
    std_threshold: 0.10
""",
    )
    store = VectorStore(settings)
    engine = RulesEngine(settings, store)

    confidences = [10.0, 95.0, 20.0]
    produced = []
    for idx, conf in enumerate(confidences):
        event = EventRecord(
            event_id=f"EVT-I-{idx}",
            entity_id="PRC777",
            confidence=conf,
            decision=DecisionState.review,
            track_id="T5",
            is_unknown=False,
        )
        store.add_event(event)
        produced = engine.process_event(event)

    assert any(a.type == AlertType.instability for a in produced)
