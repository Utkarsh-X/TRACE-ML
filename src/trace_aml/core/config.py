"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from trace_aml.core.errors import ConfigError


class AppSettings(BaseModel):
    name: str = "TRACE-AML"
    environment: str = "demo"
    timezone: str = "UTC"


class CameraSettings(BaseModel):
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    @field_validator("device_index")
    @classmethod
    def validate_device_index(cls, value: int) -> int:
        if value != 0:
            raise ValueError("For v3 MVP, only built-in webcam index 0 is supported.")
        return value


class GpuSettings(BaseModel):
    """Controls GPU / execution-provider selection for ONNX Runtime.

    When ``enabled=True`` the system calls ``trace_aml.core.gpu.detect_providers``
    at startup and selects the best available accelerator automatically,
    falling back to CPU if no GPU provider is functional.
    """

    enabled: bool = True
    """Master switch.  Set to False to always use CPU (no GPU probing at all)."""

    preferred_provider: str = ""
    """Override the auto-detected order.

    Leave empty for auto-detection.  Set to a specific provider string to force
    it to the top of the priority list, e.g. ``"CUDAExecutionProvider"`` or
    ``"DmlExecutionProvider"``.
    """

    cuda_device_id: int = 0
    """CUDA device ordinal when multiple GPUs are present."""

    log_provider_selection: bool = True
    """Emit a startup log line describing the selected provider."""


class RecognitionSettings(BaseModel):
    model_name: str = "buffalo_sc"
    # ``provider`` is superseded by GpuSettings when gpu.enabled=True.
    # Kept for backwards-compatibility; a deprecation warning is logged
    # if it is set to a non-default value while gpu.enabled=True.
    provider: str = "CPUExecutionProvider"
    det_size: tuple[int, int] = (640, 640)
    similarity_threshold: float = 0.45
    accept_threshold: float = 0.72
    review_threshold: float = 0.58
    top_k: int = 5
    active_gallery_search_k: int = 96
    log_cooldown_seconds: int = 5
    persist_unknown: bool = False
    persist_review: bool = True
    robust_matching: bool = True
    enable_preprocess_fallback: bool = True
    low_quality_threshold: float = 0.40
    threshold_relaxation: float = 0.12
    unknown_reuse_threshold: float = 0.55
    # Minimum smoothed confidence (0-100) for an unknown to become a surfaced entity.
    # Below this, the face is logged internally but no UNK record is created.
    # NOTE: This is a legacy threshold now only used by the known-person path.
    min_unknown_surface_threshold: float = 35.0
    # Minimum InsightFace detector_score (0.0-1.0) for an UNKNOWN entity to be
    # committed and stored.  This is purely a face-detection confidence —
    # completely independent of who is enrolled in the gallery.
    # A new person with perfect lighting will score 0.85-0.99 even if their
    # gallery similarity is only 5%.  Set to 0.0 to disable (accept all faces).
    min_unknown_detector_score: float = 0.65


class PipelineSettings(BaseModel):
    frame_queue_size: int = 2
    result_queue_size: int = 2
    show_hud: bool = True
    # Startup ghost-entity cleanup: removes UNK entities with fewer than
    # ghost_entity_min_events detection events (warmup artifacts).
    # Threshold of 3 is safe: committed entities reach 3 events in <1s;
    # ghost entities from warmup frames never exceed 1 event.
    purge_ghost_entities_on_start: bool = True
    ghost_entity_min_events: int = 3


class QualitySettings(BaseModel):
    min_face_ratio: float = 0.03
    min_sharpness: float = 55.0
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    min_pose_score: float = 0.28
    min_quality_score: float = 0.38
    min_valid_images: int = 6
    min_embeddings_active: int = 6
    min_embeddings_ready: int = 2
    # Runtime face quality gate — faces with detector_score below this are discarded
    # before ArcFace embedding runs (prevents low-quality faces from becoming entities).
    min_detector_score: float = 0.55


class TemporalSettings(BaseModel):
    decision_window: int = 6
    smoothing_alpha: float = 0.6
    min_accept_votes: int = 2
    track_ttl_seconds: float = 1.8
    max_track_distance_px: int = 120
    min_track_iou: float = 0.08
    track_reuse_min_score: float = 0.35
    # Entity commitment gate: a track must reach this smoothed confidence (0-100)
    # before any DB write occurs. Prevents warmup-phase ghost entities.
    # Note: unknown persons have near-zero gallery similarity so their
    # smoothed_confidence is typically 10-25%; keep this low enough to let
    # genuine unknowns through while still blocking pure noise frames.
    min_commit_confidence: float = 20.0
    min_commit_votes: int = 2


class RuleWindowSettings(BaseModel):
    window_sec: int = 10
    min_events: int = 3


class RuleInstabilitySettings(BaseModel):
    window_sec: int = 10
    std_threshold: float = 0.15


class RulesSettings(BaseModel):
    cooldown_sec: int = 15
    reappearance: RuleWindowSettings = Field(default_factory=RuleWindowSettings)
    unknown: RuleWindowSettings = Field(default_factory=RuleWindowSettings)
    instability: RuleInstabilitySettings = Field(default_factory=RuleInstabilitySettings)


class ActionPolicyBySeverity(BaseModel):
    low: list[str] = Field(default_factory=list)
    medium: list[str] = Field(default_factory=lambda: ["log"])
    high: list[str] = Field(default_factory=lambda: ["log", "email", "alarm"])


class ActionsSettings(BaseModel):
    enabled: bool = True
    on_create: ActionPolicyBySeverity = Field(default_factory=ActionPolicyBySeverity)
    on_update: ActionPolicyBySeverity = Field(
        default_factory=lambda: ActionPolicyBySeverity(
            low=[],
            medium=["log"],
            high=["log"],
        )
    )
    cooldown_sec: int = 20


class StoreSettings(BaseModel):
    root: str = "data"
    vectors_dir: str = "data/vectors"
    screenshots_dir: str = "data/screenshots"
    exports_dir: str = "data/exports"
    portraits_dir: str = "data/portraits"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file_path: str = "data/logs/trace_aml.log"
    rotation: str = "10 MB"
    retention: str = "14 days"


class LivenessSettings(BaseModel):
    enabled: bool = False
    provider: str = "none"
    threshold: float = 0.6
    model_path: str = "models/2.7_80x80_MiniFASNetV2.onnx"
    strict_reject: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRACE_AML_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    rules: RulesSettings = Field(default_factory=RulesSettings)
    actions: ActionsSettings = Field(default_factory=ActionsSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    liveness: LivenessSettings = Field(default_factory=LivenessSettings)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    runtime_config_path: str = ""


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("Config file must be a YAML object.")
    return loaded


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path or os.getenv("TRACE_AML_CONFIG", "config/config.yaml"))
    raw = load_yaml(path)
    try:
        settings = Settings(**raw)
    except Exception as exc:  # pragma: no cover - pydantic supplies details.
        raise ConfigError(str(exc)) from exc
    settings.runtime_config_path = str(path.resolve())
    return settings
