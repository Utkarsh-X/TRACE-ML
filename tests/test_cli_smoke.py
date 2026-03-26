from pathlib import Path

from typer.testing import CliRunner

from trace_ml.cli import app
from trace_ml.core.config import load_settings
from trace_ml.core.models import PersonCategory, PersonLifecycleStatus, PersonRecord
from trace_ml.store.vector_store import VectorStore


def _write_config(tmp_path: Path) -> Path:
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
logging:
  file_path: {tmp_path.as_posix()}/data/logs/trace_ml.log
""".strip(),
        encoding="utf-8",
    )
    return cfg


def test_cli_person_add_list_and_report(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runner = CliRunner()

    add = runner.invoke(
        app,
        ["--config", str(cfg), "person", "add", "--name", "Alice", "--category", "criminal"],
    )
    assert add.exit_code == 0

    listing = runner.invoke(app, ["--config", str(cfg), "person", "list"])
    assert listing.exit_code == 0
    assert "PRC001" in listing.stdout

    summary = runner.invoke(app, ["--config", str(cfg), "report", "summary"])
    assert summary.exit_code == 0

    export = runner.invoke(app, ["--config", str(cfg), "export", "csv"])
    assert export.exit_code == 0


def test_cli_person_help_includes_capture(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg), "person", "--help"])
    assert result.exit_code == 0
    assert "capture" in result.stdout


def test_cli_person_audit_apply_and_quality_report(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    settings = load_settings(cfg)
    store = VectorStore(settings)
    store.add_or_update_person(
        PersonRecord(
            person_id="PRC001",
            name="Needs Audit",
            category=PersonCategory.criminal,
        )
    )
    # Intentionally set an invalid active state for a person with no images/embeddings.
    store.set_person_state(
        person_id="PRC001",
        lifecycle_state=PersonLifecycleStatus.active,
        lifecycle_reason="forced_bad_state",
        enrollment_score=0.95,
        valid_embeddings=20,
        valid_images=20,
        total_images=20,
    )

    runner = CliRunner()
    audit = runner.invoke(app, ["--config", str(cfg), "person", "audit", "--apply"])
    assert audit.exit_code == 0
    assert "Applied state updates:" in audit.stdout

    listing = runner.invoke(app, ["--config", str(cfg), "person", "list"])
    assert listing.exit_code == 0
    assert "draft" in listing.stdout

    quality = runner.invoke(app, ["--config", str(cfg), "report", "quality"])
    assert quality.exit_code == 0
    assert "Threshold impact bands" in quality.stdout
