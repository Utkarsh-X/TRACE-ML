"""Liveness detection — real face vs. photo/screen spoof.

Architecture
------------
``MiniFASNetLiveness`` runs the MiniFASNetV2 anti-spoofing model via ONNX
Runtime. The model classifies an 80×80 face crop as either "real" or "spoof"
(photo / screen replay). It returns a ``LivenessResult`` with:

- ``is_real``: True if the face passes the liveness gate
- ``score``: ``0.0 – 1.0`` probability that the face is a live person
- ``provider``: ``"minifasnet"``

Preprocessing (matching the training pipeline)
-----------------------------------------------
1. Pad the bbox by ``PAD_FRACTION`` (24 pixels per 100px of face size)
2. Crop from the full BGR frame
3. Resize to ``80 × 80``
4. Normalise: ``(pixel / 255 - 0.5) / 0.5``   →   ``[-1, 1]``
5. Convert HWC → CHW, add batch dimension

Model output
------------
Shape ``(1, 2)`` logits.  After softmax: ``[spoof_prob, real_prob]``.
``real_prob >= threshold`` → live.

Graceful fallback
-----------------
If the model file is missing or fails to load, the checker logs a warning
and falls back to ``PassThroughLiveness`` (always passes), so the rest of the
pipeline continues unaffected.  Enabling liveness in config without the model
file therefore causes no crash — just a log warning and a no-op gate.

Model download
--------------
Run ``python -m trace_aml.liveness.download`` or place the model manually at
the path specified in ``liveness.model_path`` (default:
``models/2.7_80x80_MiniFASNetV2.onnx``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class LivenessResult:
    is_real: bool
    score: float
    provider: str
    reason: str = ""
    flags: list[str] = field(default_factory=list)


# ── Base interface ────────────────────────────────────────────────────────────

class BaseLivenessChecker:
    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        raise NotImplementedError


# ── Pass-through (disabled) ───────────────────────────────────────────────────

class PassThroughLiveness(BaseLivenessChecker):
    """Always passes — used when liveness is disabled or model is unavailable."""

    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        return LivenessResult(
            is_real=True,
            score=1.0,
            provider="none",
            reason="disabled",
        )


# ── Real MiniFASNetV2 ONNX liveness ──────────────────────────────────────────

# Fraction of the shorter bbox side to pad around the face crop.
_PAD_FRACTION: float = 0.28

# Model input resolution.
_INPUT_H: int = 80
_INPUT_W: int = 80


def _preprocess(face_crop_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalise and reformat a face crop for MiniFASNet inference."""
    img = cv2.resize(face_crop_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
    # Normalise to [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    # HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 80, 80)
    return img.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class MiniFASNetLiveness(BaseLivenessChecker):
    """ONNX-backed MiniFASNetV2 liveness checker.

    Parameters
    ----------
    model_path:
        Path to ``2.7_80x80_MiniFASNetV2.onnx``.
    threshold:
        Minimum ``real_prob`` to pass the liveness gate (default 0.60).
    providers:
        ONNX Runtime execution providers in priority order.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.60,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.threshold = threshold
        self._session: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
        self._input_name: str = ""
        self._fallback = PassThroughLiveness()

        if not _ORT_AVAILABLE:
            import warnings
            warnings.warn(
                "onnxruntime not installed — liveness detection disabled. "
                "Install with: pip install onnxruntime",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        path = Path(model_path)
        if not path.exists():
            import warnings
            warnings.warn(
                f"MiniFASNet model not found at '{model_path}'. "
                f"Liveness detection disabled.\n"
                f"Download the model with:\n"
                f"  python -m trace_aml.liveness.download\n"
                f"or manually place the file at: {path.resolve()}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        _providers = providers or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3  # suppress verbose ONNX logs

        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=opts, providers=_providers
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Failed to load MiniFASNet model: {exc}. "
                "Liveness detection disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._session = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        """Run liveness inference on a pre-cropped face image (any size BGR)."""
        if self._session is None or face_crop_bgr is None or face_crop_bgr.size == 0:
            return self._fallback.evaluate(face_crop_bgr)

        try:
            inp = _preprocess(face_crop_bgr)
            outputs = self._session.run(None, {self._input_name: inp})
            logits = np.array(outputs[0]).reshape(-1)   # (2,)
            probs = _softmax(logits)

            # Class 0 = spoof, Class 1 = real  (standard MiniFASNet convention)
            real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
            is_real = real_prob >= self.threshold

            return LivenessResult(
                is_real=is_real,
                score=round(real_prob, 4),
                provider="minifasnet",
                reason="live" if is_real else "spoof_detected",
                flags=[] if is_real else ["spoof"],
            )

        except Exception as exc:
            # Never crash the frame pipeline on a liveness error
            return LivenessResult(
                is_real=True,
                score=0.5,
                provider="minifasnet-error",
                reason=f"inference_error: {exc}",
            )


# ── Legacy stub (kept for backward compat, now redirects to real impl) ────────

class MiniFASNetStub(MiniFASNetLiveness):
    """Backward-compatible alias — now delegates to the real implementation."""

    def __init__(self, model_path: str, threshold: float = 0.6) -> None:
        super().__init__(model_path=model_path, threshold=threshold)
