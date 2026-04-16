"""Best face capture management for automatic profile picture updates.

When a person is recognized with a high confidence score, the best face
image is extracted and stored as their profile picture. This module handles:
- Face image extraction from detections
- Confidence-based profile picture updates
- Image storage and cleanup
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from trace_aml.core.config import Settings
from trace_aml.core.models import DetectionEvent, RecognitionMatch


class BestCaptureManager:
    """Manage best face captures for persons."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.person_images_dir = Path(settings.store.root) / "person_images"
        self.person_images_dir.mkdir(parents=True, exist_ok=True)
        # Hysteresis: only update if new confidence is 5% better
        self.confidence_threshold = 0.05
        # Minimum quality score to accept a face capture
        self.min_quality = 0.50
        # Only capture if confidence is above this
        self.min_confidence = 0.60
        # Max captures to keep per person (delete oldest beyond this)
        self.max_captures_per_person = 3

    def should_update_profile_picture(
        self, new_confidence: float, current_profile_confidence: float
    ) -> bool:
        """Determine if profile picture should be updated with new detection.

        Args:
            new_confidence: Smoothed confidence of new detection
            current_profile_confidence: Confidence of current profile picture

        Returns:
            True if new_confidence is significantly better (with hysteresis)
        """
        if new_confidence < self.min_confidence:
            return False

        # If no current profile, accept first good capture
        if current_profile_confidence == 0.0:
            return True

        # Otherwise require improvement by hysteresis threshold
        return new_confidence > current_profile_confidence + self.confidence_threshold

    def extract_and_save_face(
        self,
        person_id: str,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float,
        quality_score: float,
    ) -> Optional[str]:
        """Extract face from frame and save as best capture.

        Args:
            person_id: Person ID
            frame: Full frame image (numpy array)
            bbox: Bounding box (x, y, w, h)
            confidence: Detection confidence
            quality_score: Face quality assessment (0.0-1.0)

        Returns:
            Path to saved face image, or None if extraction failed
        """
        try:
            # Quality check
            if quality_score < self.min_quality:
                logger.debug(
                    f"Face quality too low ({quality_score:.2f}) for {person_id}, skipping"
                )
                return None

            # Extract face bbox
            x, y, w, h = bbox
            # Add small padding (10% on each side)
            pad = min(int(w * 0.1), int(h * 0.1))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                logger.warning(f"Failed to extract face crop for {person_id}")
                return None

            # Ensure directory exists
            person_dir = self.person_images_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp and confidence
            timestamp = datetime.now(timezone.utc).timestamp()
            conf_int = int(confidence * 100)
            filename = f"best_detection_{timestamp:.0f}_{conf_int}.jpg"
            filepath = person_dir / filename

            # Save image
            success = cv2.imwrite(str(filepath), face_crop)
            if not success:
                logger.error(f"Failed to save face image to {filepath}")
                return None

            logger.debug(
                f"Saved best capture for {person_id}: {filename} (conf={confidence:.2f}, quality={quality_score:.2f})"
            )

            # Cleanup old captures (keep only 3 most recent)
            self._cleanup_old_captures(person_dir)

            return str(filepath)

        except Exception as exc:
            logger.error(f"Error extracting face for {person_id}: {exc}")
            return None

    def _cleanup_old_captures(self, person_dir: Path) -> None:
        """Keep only the 3 most recent capture files."""
        try:
            captures = sorted(
                person_dir.glob("best_detection_*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_capture in captures[self.max_captures_per_person :]:
                old_capture.unlink()
                logger.debug(f"Deleted old capture: {old_capture.name}")
        except Exception as exc:
            logger.warning(f"Error cleaning up old captures for {person_dir}: {exc}")

    def get_best_capture_path(self, person_id: str) -> Optional[str]:
        """Get the most recent best capture for a person.

        Returns:
            Path to best capture image, or None if none exists
        """
        try:
            person_dir = self.person_images_dir / person_id
            if not person_dir.exists():
                return None

            captures = sorted(
                person_dir.glob("best_detection_*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if captures:
                return str(captures[0])
            return None
        except Exception as exc:
            logger.warning(f"Error getting best capture for {person_id}: {exc}")
            return None

    def delete_person_captures(self, person_id: str) -> None:
        """Delete all captures for a person (e.g., when person is deleted)."""
        try:
            person_dir = self.person_images_dir / person_id
            if person_dir.exists():
                shutil.rmtree(person_dir)
                logger.debug(f"Deleted all captures for {person_id}")
        except Exception as exc:
            logger.warning(f"Error deleting captures for {person_id}: {exc}")
