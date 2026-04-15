"""InsightFace ArcFace recognizer with SCRFD detector."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from trace_aml.core.config import Settings
from trace_aml.core.errors import DependencyError, RecognitionError
from trace_aml.core.models import DecisionState, FaceCandidate, RecognitionMatch
from trace_aml.liveness.base import BaseLivenessChecker, LivenessResult, PassThroughLiveness
from trace_aml.store.vector_store import VectorStore


class ArcFaceRecognizer:
    """Loads `buffalo_sc` and exposes embedding extraction APIs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._app: Any | None = None
        self._liveness: BaseLivenessChecker = PassThroughLiveness()

    def set_liveness_checker(self, checker: BaseLivenessChecker | None) -> None:
        self._liveness = checker or PassThroughLiveness()

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        try:
            import insightface
        except Exception as exc:
            raise DependencyError(
                "InsightFace not available. Install dependencies from requirements.txt"
            ) from exc

        providers = [self.settings.recognition.provider]
        try:
            self._app = insightface.app.FaceAnalysis(
                name=self.settings.recognition.model_name,
                providers=providers,
            )
            self._app.prepare(
                ctx_id=0,
                det_size=tuple(self.settings.recognition.det_size),
            )
        except Exception as exc:
            raise RecognitionError(f"Failed to initialize InsightFace: {exc}") from exc

    @staticmethod
    def _bbox_tuple(face_obj: Any) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = face_obj.bbox.astype(int).tolist()
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return int(x1), int(y1), int(w), int(h)

    @staticmethod
    def _normalize_embedding(embedding: Any) -> list[float]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != 512:
            raise RecognitionError(f"Expected 512-d embedding, got {vector.shape[0]}")
        norm = np.linalg.norm(vector) + 1e-9
        return (vector / norm).astype(np.float32).tolist()

    def _enhance_low_light(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Improve low-light frames before fallback detection."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

        mean_brightness = float(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).mean())
        if mean_brightness < 85.0:
            gamma = float(np.clip(np.interp(mean_brightness, [20.0, 85.0], [0.45, 0.95]), 0.40, 0.95))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, lut)
        return enhanced

    @staticmethod
    def _face_quality(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], detector_score: float) -> float:
        h, w = frame_bgr.shape[:2]
        _, _, bw, bh = bbox
        face_ratio = float((bw * bh) / max(1.0, float(w * h)))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        ratio_score = float(np.clip(face_ratio / 0.10, 0.0, 1.0))
        brightness_score = float(np.clip(1.0 - abs(brightness - 120.0) / 120.0, 0.0, 1.0))
        det_score = float(np.clip(detector_score, 0.0, 1.0))
        return float(np.clip(0.45 * det_score + 0.30 * ratio_score + 0.25 * brightness_score, 0.0, 1.0))

    def _effective_thresholds(self, face_quality: float) -> tuple[float, float]:
        if not self.settings.recognition.robust_matching:
            return self.settings.recognition.accept_threshold, self.settings.recognition.review_threshold
        relax = (1.0 - float(np.clip(face_quality, 0.0, 1.0))) * self.settings.recognition.threshold_relaxation
        accept = max(0.50, self.settings.recognition.accept_threshold - relax)
        review = max(0.35, self.settings.recognition.review_threshold - (relax * 1.1))
        return accept, min(accept - 0.02, review) if review >= accept else review

    def _aggregate_person_scores(
        self,
        rows: list[dict[str, Any]],
        vector_store: VectorStore,
        active_ids: set[str],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pid = str(row.get("person_id", "")).strip()
            if pid:
                grouped[pid].append(row)

        candidates: list[dict[str, Any]] = []
        for pid, person_rows in grouped.items():
            sims = sorted([float(r.get("similarity", 0.0)) for r in person_rows], reverse=True)
            if not sims:
                continue
            top_n = sims[: min(4, len(sims))]
            best = top_n[0]
            mean_top = float(np.mean(top_n))
            support = len(top_n)
            support_bonus = min(1.0, support / 4.0) * 0.06
            robust_similarity = float(np.clip(0.70 * best + 0.30 * mean_top + support_bonus, 0.0, 1.0))

            person = vector_store.get_person(pid)
            candidates.append(
                {
                    "person_id": pid,
                    "name": person.get("name", "") if person else "",
                    "category": person.get("category", "unknown") if person else "unknown",
                    "lifecycle_state": person.get("lifecycle_state", "draft") if person else "draft",
                    "is_active": pid in active_ids,
                    "similarity": robust_similarity,
                    "best_similarity": best,
                    "mean_top_similarity": mean_top,
                    "support": support,
                }
            )

        candidates.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        return candidates

    def detect_faces(self, frame_bgr: np.ndarray) -> list[FaceCandidate]:
        self._ensure_app()
        faces = self._app.get(frame_bgr)
        if not faces and self.settings.recognition.enable_preprocess_fallback:
            faces = self._app.get(self._enhance_low_light(frame_bgr))
        candidates: list[FaceCandidate] = []
        for face in faces:
            try:
                candidates.append(
                    FaceCandidate(
                        bbox=self._bbox_tuple(face),
                        embedding=self._normalize_embedding(face.embedding),
                        detector_score=float(getattr(face, "det_score", 0.0)),
                    )
                )
            except Exception:
                continue
        return candidates

    def primary_face_from_image(self, image_bgr: np.ndarray) -> FaceCandidate | None:
        candidates = self.detect_faces(image_bgr)
        if not candidates:
            return None
        candidates.sort(key=lambda c: c.bbox[2] * c.bbox[3], reverse=True)
        return candidates[0]

    def match(
        self,
        frame_bgr: np.ndarray,
        vector_store: VectorStore,
    ) -> list[tuple[RecognitionMatch, list[float], LivenessResult]]:
        candidates = self.detect_faces(frame_bgr)
        # ── Layer 1: Runtime face quality gate ────────────────────────────────
        # Discard faces whose detector_score is below the minimum threshold.
        # These are too uncertain to produce reliable embeddings; skipping them
        # prevents warmup-phase ghost entities from being created downstream.
        min_det = self.settings.quality.min_detector_score
        candidates = [c for c in candidates if c.detector_score >= min_det]
        # ─────────────────────────────────────────────────────────────────────
        active_ids = vector_store.active_person_ids()  # served from cache — no DB query
        outputs: list[tuple[RecognitionMatch, list[float], LivenessResult]] = []
        for candidate in candidates:

            x, y, w, h = candidate.bbox
            crop = frame_bgr[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]
            liveness = self._liveness.evaluate(crop)

            # Primary path: BLAS in-memory search (O(1) per-person, no Python loop)
            search_top_k = max(
                self.settings.recognition.active_gallery_search_k,
                self.settings.recognition.top_k,
            )
            active_rows = vector_store.search_active_gallery(
                candidate.embedding,
                top_k=search_top_k,
            )

            # If the cache returned nothing at all, fall back to LanceDB ANN
            best_rows = active_rows
            if not best_rows:
                best_rows = vector_store.search_embeddings(
                    candidate.embedding,
                    top_k=max(self.settings.recognition.top_k * 8, 32),
                )
            candidate_scores = self._aggregate_person_scores(best_rows, vector_store, active_ids)
            best = candidate_scores[0] if candidate_scores else None
            similarity = float(best.get("similarity", 0.0)) if best else 0.0
            confidence = round(similarity * 100, 2)
            person_id = str(best.get("person_id", "")) if best else ""
            person = vector_store.get_person(person_id) if person_id else None

            face_quality = self._face_quality(frame_bgr, candidate.bbox, candidate.detector_score)
            dyn_accept_thr, dyn_review_thr = self._effective_thresholds(face_quality)
            quality_flags: list[str] = []
            if not liveness.is_real or (
                self.settings.liveness.enabled
                and self.settings.liveness.strict_reject
                and liveness.score < self.settings.liveness.threshold
            ):
                quality_flags.append("liveness_fail")
            if face_quality < self.settings.recognition.low_quality_threshold:
                quality_flags.append("low_face_quality")
            if person_id and person_id not in active_ids:
                quality_flags.append("person_not_active")
            if person and str(person.get("lifecycle_state", "draft")) != "active":
                quality_flags.append("person_state_not_active")

            hard_block_flags = {"liveness_fail", "person_not_active", "person_state_not_active"}
            is_match = bool(
                person
                and person_id in active_ids
                and not any(flag in hard_block_flags for flag in quality_flags)
                and similarity >= dyn_review_thr
            )
            outputs.append(
                (
                    RecognitionMatch(
                        person_id=person_id if person_id else None,
                        name=person["name"] if person else "Unknown",
                        category=person["category"] if person else "unknown",
                        similarity=similarity,
                        confidence=confidence,
                        bbox=candidate.bbox,
                        is_match=is_match,
                        decision_state=DecisionState.reject,
                        decision_reason="temporal_pending",
                        smoothed_confidence=confidence,
                        quality_flags=quality_flags,
                        candidate_scores=candidate_scores,
                        metadata={
                            "detector_score": candidate.detector_score,
                            "face_quality": round(face_quality, 3),
                            "dynamic_accept_threshold": round(dyn_accept_thr, 3),
                            "dynamic_review_threshold": round(dyn_review_thr, 3),
                            "liveness_score": liveness.score,
                            "liveness_provider": liveness.provider,
                            "liveness_reason": liveness.reason,
                            "active_candidates": len([c for c in candidate_scores if c["person_id"] in active_ids]),
                        },
                    ),
                    candidate.embedding,
                    liveness,
                )
            )
        return outputs
