"""Portrait Store — persists the best-quality face crop for each entity.

Design
------
All image I/O is delegated to :class:`~trace_aml.store.data_vault.DataVault`.
The vault stores portraits as XChaCha20-Poly1305 encrypted blobs whose
filenames are SHA-256 content hashes — no entity ID is ever visible on disk.

This class retains exactly its original public API so that no callers
(``session.py``, ``app.py``, etc.) need to change signatures.

Score gating
~~~~~~~~~~~~
A portrait is only replaced when the new detection's cosine similarity is
meaningfully better than the one already stored (gap > ``MIN_IMPROVEMENT``),
preventing low-quality jitter from overwriting a good capture.

Thread safety
~~~~~~~~~~~~~
Delegated to the DataVault's per-entity locks.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from trace_aml.core.config import Settings
from trace_aml.store.data_vault import DataVault

# Only replace a portrait when the new similarity is this much better.
MIN_IMPROVEMENT: float = 0.03

# Output JPEG compression quality (0–100). 92 is visually lossless.
JPEG_QUALITY: int = 92

# Face crop output resolution in pixels (square).
PORTRAIT_SIZE: int = 256

# Padding around the raw bbox as a fraction of the shorter bbox side.
BBOX_PAD_FRACTION: float = 0.28


class PortraitStore:
    """Manages the best-match face portrait for each entity via DataVault.

    Public API is identical to the pre-vault version so all callers are
    unaffected.  Internally, no JPEG is ever written to a plain file path —
    everything goes through the encrypted vault.
    """

    def __init__(self, settings: Settings) -> None:
        self._vault = DataVault(settings)
        # Legacy directory kept only for migration-script compatibility.
        # New code never writes here.
        self._legacy_dir = Path(settings.store.portraits_dir)

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
        stored_score = self._vault.get_portrait_score(entity_id)
        if stored_score is not None and score <= stored_score + MIN_IMPROVEMENT:
            return False  # existing portrait is already good enough

        crop = self._extract_crop(frame_bgr, bbox)
        if crop is None:
            return False

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ok, buf = cv2.imencode(".jpg", crop, encode_params)
            if not ok:
                logger.warning("PortraitStore: imencode failed for {}", entity_id)
                return False
            self._vault.put_portrait(entity_id, buf.tobytes(), score)
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
        """Legacy compatibility shim — returns None (use get_portrait_bytes instead).

        This method is kept so that any code checking ``portrait_store.get_portrait_path()``
        still compiles, but the vault design no longer exposes filesystem paths.
        Callers should use :meth:`get_portrait_bytes` directly.
        """
        # Check vault first
        if self._vault.has_portrait(entity_id):
            return None  # Vault manages storage — no path to expose
        # Fallback: check legacy directory for pre-migration portraits
        safe = entity_id.replace("/", "_").replace("\\", "_")
        legacy = self._legacy_dir / f"{safe}.jpg"
        return legacy if legacy.exists() else None

    def get_portrait_bytes(self, entity_id: str) -> Optional[bytes]:
        """Return the decrypted JPEG bytes for the entity's portrait, or None."""
        return self._vault.get_portrait_bytes(entity_id)

    def delete_portrait(self, entity_id: str) -> None:
        """Delete the portrait for *entity_id* from the vault.

        Called when an entity is reset or replaced so that a stale high-score
        portrait from a previous session cannot block future updates.
        """
        self._vault.delete_portrait(entity_id)
        # Also clean up any legacy file that might exist
        safe = entity_id.replace("/", "_").replace("\\", "_")
        for path in (
            self._legacy_dir / f"{safe}.jpg",
            self._legacy_dir / f"{safe}.meta.json",
        ):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    def get_best_score(self, entity_id: str) -> Optional[float]:
        """Return the stored similarity score for this entity, or ``None``."""
        return self._vault.get_portrait_score(entity_id)

    def has_portrait(self, entity_id: str) -> bool:
        """Fast existence check."""
        if self._vault.has_portrait(entity_id):
            return True
        # Legacy fallback
        safe = entity_id.replace("/", "_").replace("\\", "_")
        return (self._legacy_dir / f"{safe}.jpg").exists()

    # ── DataVault pass-through (for app.py endpoints) ───────────────────────

    @property
    def vault(self) -> DataVault:
        """Expose vault for callers that need direct access (e.g. app.py)."""
        return self._vault

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_crop(
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Crop and resize the face region from the full frame."""
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
