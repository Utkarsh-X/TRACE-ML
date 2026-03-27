from pathlib import Path

from trace_ml.core.config import load_settings
from trace_ml.core.models import DecisionState, RecognitionMatch
from trace_ml.pipeline.entity_resolver import EntityResolver
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
recognition:
  unknown_reuse_threshold: 0.55
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_entity_resolver_known_and_unknown_reuse(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    resolver = EntityResolver(settings, store)

    known_match = RecognitionMatch(
        person_id="PRC001",
        name="Known",
        category="criminal",
        similarity=0.9,
        confidence=90.0,
        bbox=(0, 0, 10, 10),
        decision_state=DecisionState.accept,
    )
    known = resolver.resolve(known_match, embedding=[0.0] * 512)
    assert known.entity_id == "PRC001"
    assert known.is_unknown is False

    emb = [0.0] * 512
    emb[0] = 1.0
    unknown_match = RecognitionMatch(
        person_id=None,
        name="Unknown",
        category="unknown",
        similarity=0.0,
        confidence=0.0,
        bbox=(0, 0, 10, 10),
        decision_state=DecisionState.review,
    )
    u1 = resolver.resolve(unknown_match, embedding=emb)
    u2 = resolver.resolve(unknown_match, embedding=emb)
    assert u1.is_unknown is True
    assert u2.is_unknown is True
    assert u1.entity_id.startswith("UNK")
    assert u1.entity_id == u2.entity_id
