"""Image quality scoring for enrollment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from trace_ml.core.config import Settings
from trace_ml.core.models import QualityAssessment


@dataclass
class QualityComponents:
    sharpness: float
    face_ratio: float
    brightness: float
    pose_score: float
    quality_score: float
    reasons: list[str]
    passed: bool


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def score_face_image(
    settings: Settings,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> QualityComponents:
    reasons: list[str] = []
    if bbox is None:
        return QualityComponents(
            sharpness=0.0,
            face_ratio=0.0,
            brightness=0.0,
            pose_score=0.0,
            quality_score=0.0,
            reasons=["no_face_detected"],
            passed=False,
        )

    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return QualityComponents(
            sharpness=0.0,
            face_ratio=0.0,
            brightness=0.0,
            pose_score=0.0,
            quality_score=0.0,
            reasons=["invalid_bbox"],
            passed=False,
        )

    crop = frame_bgr[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    face_area = float((x2 - x1) * (y2 - y1))
    frame_area = float(w * h) if h and w else 1.0
    face_ratio = face_area / frame_area
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    brightness = float(gray_crop.mean())

    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    frame_center_x = w / 2.0
    frame_center_y = h / 2.0
    center_dx = abs(face_center_x - frame_center_x) / max(1.0, frame_center_x)
    center_dy = abs(face_center_y - frame_center_y) / max(1.0, frame_center_y)
    centering = 1.0 - _clamp((center_dx + center_dy) / 1.6, 0.0, 1.0)
    aspect = (x2 - x1) / max(1.0, (y2 - y1))
    aspect_score = 1.0 - _clamp(abs(aspect - 0.8), 0.0, 1.0)
    pose_score = _clamp(0.65 * centering + 0.35 * aspect_score, 0.0, 1.0)

    quality_cfg = settings.quality
    if face_ratio < quality_cfg.min_face_ratio:
        reasons.append("face_too_small")
    if sharpness < quality_cfg.min_sharpness:
        reasons.append("image_blurry")
    if brightness < quality_cfg.min_brightness:
        reasons.append("too_dark")
    if brightness > quality_cfg.max_brightness:
        reasons.append("too_bright")
    if pose_score < quality_cfg.min_pose_score:
        reasons.append("face_off_axis")

    face_score = _normalize(face_ratio, quality_cfg.min_face_ratio, quality_cfg.min_face_ratio * 2.5)
    sharpness_score = _normalize(sharpness, quality_cfg.min_sharpness, quality_cfg.min_sharpness * 2.5)
    brightness_mid = (quality_cfg.min_brightness + quality_cfg.max_brightness) / 2.0
    brightness_range = max(1.0, (quality_cfg.max_brightness - quality_cfg.min_brightness) / 2.0)
    brightness_score = 1.0 - _clamp(abs(brightness - brightness_mid) / brightness_range, 0.0, 1.0)
    quality_score = _clamp(
        0.30 * face_score + 0.30 * sharpness_score + 0.20 * brightness_score + 0.20 * pose_score,
        0.0,
        1.0,
    )
    if quality_score < quality_cfg.min_quality_score:
        reasons.append("quality_below_threshold")

    passed = len(reasons) == 0
    return QualityComponents(
        sharpness=sharpness,
        face_ratio=face_ratio,
        brightness=brightness,
        pose_score=pose_score,
        quality_score=quality_score,
        reasons=reasons,
        passed=passed,
    )


def build_assessment(
    settings: Settings,
    person_id: str,
    source_path: str,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> QualityAssessment:
    score = score_face_image(settings, frame_bgr, bbox)
    return QualityAssessment(
        person_id=person_id,
        source_path=source_path,
        passed=score.passed,
        quality_score=score.quality_score,
        sharpness=score.sharpness,
        face_ratio=score.face_ratio,
        brightness=score.brightness,
        pose_score=score.pose_score,
        reasons=score.reasons,
    )
