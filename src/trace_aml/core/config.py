"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from trace_aml.core.errors import ConfigError

# ── Portable data root ────────────────────────────────────────────────────────
# Browser / dev mode : TRACE_DATA_ROOT is unset → resolves to "data" (relative)
# Electron packaged  : Electron main sets TRACE_DATA_ROOT=<app.getPath('userData')>/TRACE-AML
#                      before spawning the Python backend, so all vault/store
#                      paths automatically land in the right OS-specific location.
#
# Nothing else in the codebase needs to change for Electron packaging.
_DATA_ROOT: str = os.environ.get("TRACE_DATA_ROOT", "data").rstrip("/\\")



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
    model_name: str = "buffalo_l"
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

class UnknownClusteringSettings(BaseModel):
    """Controls the background daemon that retroactively merges duplicate unknown entities.

    The clusterer runs every ``interval_minutes`` and performs a global pairwise
    max-similarity check across all unknown entity embeddings.  Any two entities
    whose best-matching embedding pair exceeds ``merge_threshold`` are considered
    the same person and are merged into the older entity ID.  Events, alerts,
    incidents and portrait files are all re-pointed to the surviving entity.
    """

    enabled: bool = True
    """Master switch — set to False to disable background clustering entirely."""

    interval_minutes: float = 3.0
    """How often the clusterer runs (minutes).  Lower = faster duplicate resolution
    but more CPU usage.  Values below 1.0 are allowed for testing."""

    merge_threshold: float = 0.28
    """Pairwise max-similarity threshold for merging.  Two entities are merged if
    ANY embedding from entity A scores >= this value against ANY embedding from B.
    Slightly lower than the per-frame reuse threshold (0.25) so the clusterer can
    catch cases the real-time gate missed."""

    min_embeddings_to_cluster: int = 1
    """Entities with fewer stored embeddings than this are skipped as not yet
    stable enough for reliable cross-entity comparison."""


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

    # ── CPU offload controls ──────────────────────────────────────────────
    # inference_skip_frames: run ML inference on 1-in-(N+1) frames.
    #   0 = every frame (30 FPS inference — high CPU)
    #   1 = every other frame (15 FPS inference — ~50% CPU reduction)
    #   2 = every 3rd frame (10 FPS inference — ~66% CPU reduction)
    # Camera capture and display still run at full camera FPS; only the
    # ONNX inference + gallery search is throttled.
    inference_skip_frames: int = 2

    # inference_resolution_scale: resize the frame fed into InsightFace.
    #   1.0 = full resolution (1280×720 default)
    #   0.5 = half (640×360) — 4× fewer pixels → faster GPU + CPU decode
    # The original full-res frame is kept in the result packet for display.
    inference_resolution_scale: float = 0.75

    # live_state_publish_hz: max rate (Hz) at which session.state SSE events
    # are emitted.  At 30 FPS every frame triggered a JSON serialise + push;
    # 5 Hz is plenty for the dashboard latency widgets.
    live_state_publish_hz: float = 5.0




class QualitySettings(BaseModel):
    min_face_ratio: float = 0.03
    min_sharpness: float = 55.0
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    min_pose_score: float = 0.28
    min_quality_score: float = 0.38
    min_valid_images: int = 3
    min_embeddings_active: int = 3
    min_embeddings_ready: int = 1
    # Runtime face quality gate — faces with detector_score below this are discarded
    # before ArcFace embedding runs (prevents low-quality faces from becoming entities).
    min_detector_score: float = 0.55
    # ── Composite quality gate ────────────────────────────────────────────────
    # Pre-embedding gate combining detector confidence, sharpness, and pose.
    # composite = 0.50*det + 0.30*blur + 0.20*pose  (all factors in [0,1])
    # blur_factor  = min(1, laplacian_variance / blur_lap_saturation)
    # pose_factor  = max(0, 1 - yaw_degrees / 60)
    # When composite < min_composite_score the frame is dropped before ArcFace runs.
    composite_gate_enabled: bool = True
    min_composite_score: float = 0.42
    blur_lap_saturation: float = 500.0


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
    high: list[str] = Field(default_factory=lambda: ["log", "pdf_report", "email", "whatsapp"])


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
    root: str = _DATA_ROOT
    vectors_dir: str = f"{_DATA_ROOT}/vectors"
    screenshots_dir: str = f"{_DATA_ROOT}/screenshots"
    exports_dir: str = f"{_DATA_ROOT}/exports"
    portraits_dir: str = f"{_DATA_ROOT}/portraits"


class VaultSettings(BaseModel):
    """DataVault — encrypted binary store for face image assets.

    All sensitive images (portraits, detection screenshots, enrollment photos)
    are stored as XChaCha20-Poly1305 encrypted blobs with SHA-256 content-
    addressed names.  No blob filename ever contains an entity or person ID.

    The encryption key is loaded from the ``TRACE_VAULT_KEY`` environment
    variable (64 hex chars = 32 bytes).  If unset, vault runs in passthrough
    mode (dev use only — images stored unencrypted as plain .bin files).
    """

    enabled: bool = True
    """Master switch. False keeps legacy data/portraits/ layout untouched."""

    portraits_dir: str = f"{_DATA_ROOT}/vault/portraits"
    evidence_dir: str = f"{_DATA_ROOT}/vault/evidence"
    enrollment_dir: str = f"{_DATA_ROOT}/vault/enrollment"
    index_dir: str = f"{_DATA_ROOT}/index"


# ── Notification Channel Settings ──────────────────────────────────────────────

class EmailSettings(BaseModel):
    """SMTP email delivery configuration."""
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587              # 587 = STARTTLS, 465 = SSL
    smtp_user: str = ""
    smtp_password: str = ""           # Insecure fallback. Prefer env var TRACE_AML_SMTP_PASSWORD
    sender_address: str = ""
    sender_name: str = "TRACE-AML Security"
    recipient_addresses: list[str] = Field(default_factory=list)
    use_tls: bool = True
    attach_pdf: bool = True           # Attach generated PDF report to email


class WhatsAppSettings(BaseModel):
    """WhatsApp delivery via local whatsapp-web.js bridge (Node.js).

    No Docker, no cloud providers, no monthly fees. Runs directly on the
    user's machine. First scan shows a QR code; session is saved locally.
    """
    enabled: bool = False
    bridge_url: str = "http://localhost:3001"  # Local Node.js bridge
    recipient_numbers: list[str] = Field(default_factory=list)  # E.164 format
    send_pdf: bool = True      # Send PDF report as WhatsApp document
    send_text: bool = True     # Send alert caption text


class PdfReportSettings(BaseModel):
    """PDF/HTML incident report generation configuration."""
    enabled: bool = True
    library: str = "fpdf2"            # fpdf2 = pure Python, no OS deps
    output_dir: str = "data/exports"
    include_screenshots: bool = True
    include_entity_portrait: bool = True
    max_detection_rows: int = 20
    max_alert_rows: int = 50


class NotificationsSettings(BaseModel):
    """Container for all notification channel configurations."""
    email: EmailSettings = Field(default_factory=EmailSettings)
    whatsapp: WhatsAppSettings = Field(default_factory=WhatsAppSettings)
    pdf_report: PdfReportSettings = Field(default_factory=PdfReportSettings)


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
    notifications: NotificationsSettings = Field(default_factory=NotificationsSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    liveness: LivenessSettings = Field(default_factory=LivenessSettings)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    unknown_clustering: UnknownClusteringSettings = Field(default_factory=UnknownClusteringSettings)
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

    # Decrypt ENC:-prefixed SMTP password stored by the settings persist endpoint.
    # This makes the plaintext available to email handlers without any code changes.
    stored_pw: str = settings.notifications.email.smtp_password or ""
    if stored_pw.startswith("ENC:"):
        try:
            import base64
            from trace_aml.store.data_vault import _load_key, _decrypt
            key = _load_key()
            if key:
                blob = base64.urlsafe_b64decode(stored_pw[4:])
                result = _decrypt(blob, key)
                if result:
                    settings.notifications.email.smtp_password = result.decode("utf-8")
        except Exception:
            pass  # leave as ENC: token; will be caught by email handler

    return settings
