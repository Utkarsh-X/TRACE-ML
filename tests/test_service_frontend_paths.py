from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from trace_aml.core.config import load_settings
from trace_aml.core.streaming import InMemoryEventStreamPublisher
from trace_aml.service.app import _resolve_frontend_dir, create_service_app
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


def test_resolve_frontend_dir_prefers_frozen_bundle(monkeypatch, tmp_path: Path) -> None:
    bundle_frontend = tmp_path / "src" / "frontend"
    bundle_frontend.mkdir(parents=True)

    monkeypatch.setattr("trace_aml.service.app.sys.frozen", True, raising=False)
    monkeypatch.setattr("trace_aml.service.app.sys._MEIPASS", str(tmp_path), raising=False)

    assert _resolve_frontend_dir() == bundle_frontend


def test_frontend_static_mount_disables_http_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(settings)
    app = create_service_app(
        settings=settings,
        store=store,
        stream_publisher=InMemoryEventStreamPublisher(),
    )
    client = TestClient(app)

    redirect = client.get("/ui", follow_redirects=False)
    assert redirect.status_code in {302, 307}
    assert redirect.headers["cache-control"] == "no-store, no-cache, must-revalidate"

    index = client.get("/ui/live_ops/index.html")
    assert index.status_code == 200
    assert index.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    assert index.headers["pragma"] == "no-cache"
    assert index.headers["expires"] == "0"

    client_js = client.get("/ui/shared/trace_client.js")
    assert client_js.status_code == 200
    assert client_js.headers["cache-control"] == "no-store, no-cache, must-revalidate"
