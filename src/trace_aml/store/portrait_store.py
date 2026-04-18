"""Portrait Store — persists the best-quality face crop for each entity.

Design
------
For every committed *accepted* detection the session crops the face region from
the live frame and calls ``try_update_portrait()``.  A portrait is only updated
when the new detection's cosine similarity is meaningfully better than the one
already stored (gap > ``MIN_IMPROVEMENT``), preventing low-quality jitter from
overwriting a good capture.

Storage layout on disk ::

    {portraits_dir}/
        {entity_id}.jpg          ← 256×256 JPEG portrait
        {entity_id}.meta.json    ← {"score": 0.87, "updated_at": "..."}

The JSON sidecar keeps the best score in an O(1) lookup without any DB query.

Thread safety
-------------
A per-entity lock prevents concurrent updates from racing between sessions.
The lock is fine-grained (keyed by entity_id) so different entities never block
each other.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from trace_aml.core.config import Settings

# Only replace a portrait when the new similarity is this much better.
MIN_IMPROVEMENT: float = 0.03

# Output JPEG compression quality (0–100). 92 is visually lossless.
JPEG_QUALITY: int = 92

# Face crop output resolution in pixels (square).
PORTRAIT_SIZE: int = 256

# Padding around the raw bbox as a fraction of the shorter bbox side.
BBOX_PAD_FRACTION: float = 0.28


class PortraitStore:
    """Filesystem-backed store for per-entity best-match face portraits."""

    def __init__(self, settings: Settings) -> None:
        self._dir = Path(settings.store.portraits_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Per-entity write lock — avoids races from concurrent recognition threads.
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # protects self._locks dict itself

    # ── Public API ──────────────────────────────────────────────────────────

    def try_update_portrait(
        self,
        entity_id: str,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        score: float,
    ) -> bool:
        """Attempt to update the portrait for *entity_id*.

        Parameters
        ----------
        entity_id:
            The resolved entity identifier (e.g. ``"ENT-00042"``).
        frame_bgr:
            Full BGR frame from the camera (H×W×3 numpy array).
        bbox:
            ``(x, y, w, h)`` bounding box of the face inside *frame_bgr*.
        score:
            Cosine similarity of this detection (higher = better quality).

        Returns
        -------
        bool
            ``True`` if the portrait was saved / updated, ``False`` if the
            existing portrait is already at least as good.
        """
        lock = self._entity_lock(entity_id)
        with lock:
            stored_score = self._read_score(entity_id)
            if stored_score is not None and score <= stored_score + MIN_IMPROVEMENT:
                return False  # existing portrait is already good enough

            crop = self._extract_crop(frame_bgr, bbox)
            if crop is None:
                return False

            portrait_path = self._portrait_path(entity_id)
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                ok, buf = cv2.imencode(".jpg", crop, encode_params)
                if not ok:
                    logger.warning("PortraitStore: imencode failed for {}", entity_id)
                    return False
                portrait_path.write_bytes(buf.tobytes())
                self._write_meta(entity_id, score)
                logger.debug(
                    "PortraitStore: updated portrait {} (score {:.3f} → {:.3f})",
                    entity_id,
                    stored_score or 0.0,
                    score,
                )
                return True
            except Exception as exc:
                logger.error("PortraitStore: failed to save portrait for {}: {}", entity_id, exc)
                return False

    def get_portrait_path(self, entity_id: str) -> Optional[Path]:
        """Return the portrait path if it exists, else ``None``."""
        p = self._portrait_path(entity_id)
        return p if p.exists() else None

    def delete_portrait(self, entity_id: str) -> None:
        """Delete the portrait image and metadata for *entity_id*.

        Called when an entity is reset or replaced so that a stale high-score
        portrait from a previous session cannot block future updates.
        """
        lock = self._entity_lock(entity_id)
        with lock:
            for path in (self._portrait_path(entity_id), self._meta_path(entity_id)):
                try:
                    if path.exists():
                        path.unlink()
                        logger.debug("PortraitStore: deleted {} for {}", path.name, entity_id)
                except Exception as exc:
                    logger.warning("PortraitStore: could not delete {} — {}", path, exc)


    def get_best_score(self, entity_id: str) -> Optional[float]:
        """Return the stored similarity score for this entity, or ``None``."""
        return self._read_score(entity_id)

    def has_portrait(self, entity_id: str) -> bool:
        """Fast existence check (no file read)."""
        return self._portrait_path(entity_id).exists()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _entity_lock(self, entity_id: str) -> threading.Lock:
        with self._meta_lock:
            if entity_id not in self._locks:
                self._locks[entity_id] = threading.Lock()
            return self._locks[entity_id]

    def _portrait_path(self, entity_id: str) -> Path:
        safe = entity_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.jpg"

    def _meta_path(self, entity_id: str) -> Path:
        safe = entity_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.meta.json"

    def _read_score(self, entity_id: str) -> Optional[float]:
        path = self._meta_path(entity_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return float(data.get("score", -1.0))
        except Exception:
            return None

    def _write_meta(self, entity_id: str, score: float) -> None:
        path = self._meta_path(entity_id)
        payload = {
            "score": round(float(score), 6),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_crop(
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Crop and resize the face region from the full frame.

        Adds padding proportional to the bounding-box size so the portrait
        shows a bit of context around the face (not just a tight mask).
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None

        fh, fw = frame_bgr.shape[:2]
        pad = int(min(w, h) * BBOX_PAD_FRACTION)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (PORTRAIT_SIZE, PORTRAIT_SIZE), interpolation=cv2.INTER_LANCZOS4)
