"""Temporal decision engine for stable live recognition."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from trace_ml.core.config import Settings
from trace_ml.core.models import DecisionState, RecognitionMatch


@dataclass
class TrackState:
    track_id: str
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    last_seen: float
    smoothed_confidence: float = 0.0
    confidences: deque[float] = field(default_factory=deque)
    identities: deque[str] = field(default_factory=deque)


@dataclass
class TemporalDecision:
    track_id: str
    decision_state: DecisionState
    decision_reason: str
    smoothed_confidence: float
    resolved_person_id: str | None
    vote_count: int


class TemporalDecisionEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tracks: dict[str, TrackState] = {}
        self._next_track = 1

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = bbox
        return x + (w / 2.0), y + (h / 2.0)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        union = float((aw * ah) + (bw * bh) - inter)
        if union <= 0.0:
            return 0.0
        return inter / union

    def _purge_stale(self, now: float) -> None:
        ttl = self.settings.temporal.track_ttl_seconds
        stale = [tid for tid, st in self._tracks.items() if (now - st.last_seen) > ttl]
        for tid in stale:
            self._tracks.pop(tid, None)

    def _assign_track_id(self, bbox: tuple[int, int, int, int], now: float) -> str:
        self._purge_stale(now)
        center = self._center(bbox)
        best_id = ""
        best_score = -1.0
        ttl = max(0.1, float(self.settings.temporal.track_ttl_seconds))
        max_dist = max(1.0, float(self.settings.temporal.max_track_distance_px))
        for tid, state in self._tracks.items():
            dist = self._distance(center, state.center)
            iou = self._iou(bbox, state.bbox)
            if dist > max_dist and iou < self.settings.temporal.min_track_iou:
                continue
            distance_score = max(0.0, 1.0 - (dist / max_dist))
            recency_score = max(0.0, 1.0 - ((now - state.last_seen) / ttl))
            score = (0.55 * distance_score) + (0.35 * iou) + (0.10 * recency_score)
            if score > best_score:
                best_score = score
                best_id = tid
        if best_id and best_score >= self.settings.temporal.track_reuse_min_score:
            state = self._tracks[best_id]
            state.center = center
            state.bbox = bbox
            state.last_seen = now
            return best_id

        new_id = f"T{self._next_track:04d}"
        self._next_track += 1
        self._tracks[new_id] = TrackState(track_id=new_id, center=center, bbox=bbox, last_seen=now)
        return new_id

    @staticmethod
    def _identity_key(match: RecognitionMatch) -> str:
        if match.person_id:
            return match.person_id
        if match.candidate_scores:
            top = match.candidate_scores[0]
            person_id = str(top.get("person_id", "")).strip()
            if person_id:
                return person_id
        return "unknown"

    def evaluate(self, match: RecognitionMatch, now_ts: float | None = None) -> TemporalDecision:
        now = now_ts if now_ts is not None else time.time()
        track_id = self._assign_track_id(match.bbox, now)
        state = self._tracks[track_id]

        inactive_person = any(
            flag in {"person_not_active", "person_state_not_active"} for flag in match.quality_flags
        )
        identity = self._identity_key(match)
        state.identities.append(identity)
        state.confidences.append(float(match.confidence))

        max_window = max(1, self.settings.temporal.decision_window)
        while len(state.identities) > max_window:
            state.identities.popleft()
        while len(state.confidences) > max_window:
            state.confidences.popleft()

        alpha = self.settings.temporal.smoothing_alpha
        if state.smoothed_confidence <= 0.0:
            state.smoothed_confidence = float(match.confidence)
        else:
            state.smoothed_confidence = (
                alpha * float(match.confidence) + (1.0 - alpha) * state.smoothed_confidence
            )

        non_unknown = [x for x in state.identities if x != "unknown"]
        counter = Counter(non_unknown)
        top_id = counter.most_common(1)[0][0] if counter else None
        votes = counter.most_common(1)[0][1] if counter else 0

        dynamic_accept = match.metadata.get("dynamic_accept_threshold")
        dynamic_review = match.metadata.get("dynamic_review_threshold")
        accept_thr = float(dynamic_accept) * 100.0 if dynamic_accept is not None else self.settings.recognition.accept_threshold * 100.0
        review_thr = float(dynamic_review) * 100.0 if dynamic_review is not None else self.settings.recognition.review_threshold * 100.0

        decision = DecisionState.reject
        reason = "confidence_below_review"
        if inactive_person:
            decision = DecisionState.reject
            reason = "person_not_active"
            top_id = None
            votes = 0
        elif top_id and state.smoothed_confidence >= accept_thr and votes >= self.settings.temporal.min_accept_votes:
            decision = DecisionState.accept
            reason = f"stable_accept votes={votes}"
        elif top_id and state.smoothed_confidence >= review_thr:
            decision = DecisionState.review
            reason = f"review_candidate votes={votes}"

        if "liveness_fail" in match.quality_flags:
            decision = DecisionState.reject
            reason = "liveness_reject"

        return TemporalDecision(
            track_id=track_id,
            decision_state=decision,
            decision_reason=reason,
            smoothed_confidence=round(state.smoothed_confidence, 3),
            resolved_person_id=top_id,
            vote_count=votes,
        )

    def active_track_count(self) -> int:
        self._purge_stale(time.time())
        return len(self._tracks)
