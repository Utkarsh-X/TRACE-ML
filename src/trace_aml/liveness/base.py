"""Liveness interfaces (scaffold for future anti-spoofing)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LivenessResult:
    is_real: bool
    score: float
    provider: str
    reason: str = ""
    flags: list[str] = None

    def __post_init__(self) -> None:
        if self.flags is None:
            self.flags = []


class BaseLivenessChecker:
    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        raise NotImplementedError


class PassThroughLiveness(BaseLivenessChecker):
    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        return LivenessResult(is_real=True, score=1.0, provider="none", reason="disabled")


class MiniFASNetStub(BaseLivenessChecker):
    """Placeholder for future ONNX MiniFASNet integration."""

    def __init__(self, model_path: str, threshold: float = 0.6) -> None:
        self.model_path = model_path
        self.threshold = threshold

    def evaluate(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        # Stub: deterministic scaffold with explicit telemetry hooks.
        return LivenessResult(
            is_real=True,
            score=0.99,
            provider="minifasnet-stub",
            reason="stub_pass",
            flags=[],
        )
