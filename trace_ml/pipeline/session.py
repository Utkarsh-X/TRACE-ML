"""Live recognition session orchestration with temporal decisions."""

from __future__ import annotations

import json
import math
import queue
import time
from collections import deque
from pathlib import Path

import cv2

from trace_ml.core.config import Settings
from trace_ml.core.ids import new_detection_id
from trace_ml.core.models import ActionTrigger, DecisionState, DetectionEvent, RecognitionMatch
from trace_ml.core.streaming import EventStreamPublisher, NullEventStreamPublisher
from trace_ml.liveness.base import LivenessResult
from trace_ml.pipeline.action_engine import ActionEngine
from trace_ml.pipeline.capture import CameraCapture
from trace_ml.pipeline.entity_resolver import EntityResolver
from trace_ml.pipeline.incident_manager import IncidentManager
from trace_ml.pipeline.inference import InferencePacket, InferenceWorker
from trace_ml.pipeline.policy_engine import PolicyEngine
from trace_ml.pipeline.rules_engine import RulesEngine
from trace_ml.pipeline.temporal import TemporalDecisionEngine
from trace_ml.recognizers.arcface import ArcFaceRecognizer
from trace_ml.store.vector_store import VectorStore


def draw_text(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_box(frame, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    length = max(12, int(min(w, h) * 0.2))
    cv2.line(frame, (x, y), (x + length, y), color, 2)
    cv2.line(frame, (x, y), (x, y + length), color, 2)
    cv2.line(frame, (x + w, y), (x + w - length, y), color, 2)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, 2)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, 2)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, 2)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, 2)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, 2)


def _decision_color(decision: DecisionState) -> tuple[int, int, int]:
    if decision == DecisionState.accept:
        return (0, 0, 255)
    if decision == DecisionState.review:
        return (0, 200, 255)
    return (0, 255, 0)


class RecognitionSession:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        recognizer: ArcFaceRecognizer,
        stream_publisher: EventStreamPublisher | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.recognizer = recognizer
        self.stream_publisher = stream_publisher or NullEventStreamPublisher()
        self.entity_resolver = EntityResolver(settings, store)
        self.rules_engine = RulesEngine(settings, store)
        self.incident_manager = IncidentManager(store)
        self.policy_engine = PolicyEngine(settings)
        self.action_engine = ActionEngine(store, settings)
        self.temporal = TemporalDecisionEngine(settings)
        self.last_logged_at: dict[str, float] = {}
        self.screenshot_dir = Path(settings.store.screenshots_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.event_feed: deque[str] = deque(maxlen=8)
        self.recent_confidences: deque[float] = deque(maxlen=8)
        self.decision_counters = {"accept": 0, "review": 0, "reject": 0}
        self.current_focus = "none"
        self.last_latency_ms = 0.0
        self.last_frame_queue_depth = 0
        self.last_result_queue_depth = 0

    def _should_log(self, key: str) -> bool:
        now = time.time()
        last = self.last_logged_at.get(key, 0.0)
        if now - last >= self.settings.recognition.log_cooldown_seconds:
            self.last_logged_at[key] = now
            return True
        return False

    def _track_event(self, text: str) -> None:
        self.event_feed.appendleft(text)

    def _publish(self, topic: str, payload: dict) -> None:
        self.stream_publisher.publish(topic, payload)

    def _publish_live_state(self, fps: float) -> None:
        self._publish(
            "session.state",
            {
                "fps": round(float(fps), 2),
                "active_tracks": int(self.temporal.active_track_count()),
                "latency_ms": round(float(self.last_latency_ms), 2),
                "frame_queue_depth": int(self.last_frame_queue_depth),
                "result_queue_depth": int(self.last_result_queue_depth),
                "decision_counters": dict(self.decision_counters),
                "current_focus": self.current_focus,
                "confidence_trend": [float(v) for v in list(self.recent_confidences)],
                "event_feed": list(self.event_feed),
            },
        )

    def _save_detection(
        self,
        frame,
        match: RecognitionMatch,
        embedding: list[float],
        liveness: LivenessResult,
        source: str = "webcam:0",
    ) -> None:
        decision = match.decision_state
        resolution = self.entity_resolver.resolve(match, embedding)
        persist_detection = True
        if decision == DecisionState.reject and not self.settings.recognition.persist_unknown:
            persist_detection = False
        if decision == DecisionState.review and not self.settings.recognition.persist_review:
            persist_detection = False

        detection_id = ""
        if persist_detection:
            if match.person_id:
                person_key = f"pid:{match.person_id}:{decision.value}"
            else:
                person_key = f"trk:{match.track_id or 'unknown'}:{decision.value}"
            if self._should_log(person_key):
                detection_id = new_detection_id()
                screenshot_path = self.screenshot_dir / f"{detection_id}.jpg"
                cv2.imwrite(str(screenshot_path), frame)

                event = DetectionEvent(
                    detection_id=detection_id,
                    source=source,
                    person_id=match.person_id,
                    name=match.name if decision != DecisionState.reject else "Unknown",
                    category=match.category if decision != DecisionState.reject else "unknown",
                    confidence=match.confidence,
                    similarity=match.similarity,
                    smoothed_confidence=match.smoothed_confidence,
                    bbox=match.bbox,
                    track_id=match.track_id,
                    decision_state=decision,
                    decision_reason=match.decision_reason,
                    quality_flags=match.quality_flags,
                    liveness_provider=liveness.provider,
                    liveness_score=liveness.score,
                    screenshot_path=str(screenshot_path),
                    metadata=match.metadata,
                )
                self.store.add_detection(event, embedding=embedding)
                self._publish("detection", event.model_dump(mode="json"))
                self.store.add_detection_decision(
                    detection_id=detection_id,
                    track_id=match.track_id,
                    decision_state=decision.value,
                    decision_reason=match.decision_reason,
                    smoothed_confidence=match.smoothed_confidence,
                    quality_flags=match.quality_flags,
                    top_candidates=match.candidate_scores,
                    liveness_provider=liveness.provider,
                    liveness_score=liveness.score,
                )
                self._track_event(
                    f"{decision.value.upper()} {event.name} {event.smoothed_confidence:.1f}% [{match.track_id}]"
                )

        core_event = self.entity_resolver.create_event_record(
            resolution=resolution,
            match=match,
            detection_id=detection_id,
            source=source,
        )
        self.store.add_event(core_event)
        self._publish("event", core_event.model_dump(mode="json"))
        alerts = self.rules_engine.process_event(core_event)
        for alert in alerts:
            self.store.add_alert(alert)
            self._publish("alert", alert.model_dump(mode="json"))
            incident, trigger_label = self.incident_manager.handle_alert(alert)
            self._publish("incident", incident.model_dump(mode="json"))
            trigger = ActionTrigger(trigger_label)
            planned_actions = self.policy_engine.evaluate(incident, trigger)
            executed = self.action_engine.execute(incident, planned_actions, trigger)
            self._track_event(
                f"ALERT {alert.severity.value.upper()} {alert.type.value} {alert.entity_id} -> {incident.incident_id}"
            )
            if planned_actions:
                action_names = ",".join([a.value for a in planned_actions])
                self._track_event(
                    f"POLICY {incident.incident_id} {trigger.value} sev={incident.severity.value} -> [{action_names}]"
                )
            for action in executed:
                self._publish("action", action.model_dump(mode="json"))
                self._track_event(
                    f"ACTION {action.action_type.value.upper()} {incident.incident_id} ({action.status.value})"
                )

    def _apply_temporal_decision(self, match: RecognitionMatch, now: float) -> RecognitionMatch:
        temporal = self.temporal.evaluate(match, now_ts=now)
        match.track_id = temporal.track_id
        match.smoothed_confidence = temporal.smoothed_confidence
        match.decision_state = temporal.decision_state
        match.decision_reason = temporal.decision_reason

        resolved_person = self.store.get_person(temporal.resolved_person_id) if temporal.resolved_person_id else None
        if temporal.decision_state in {DecisionState.accept, DecisionState.review} and resolved_person:
            match.person_id = resolved_person["person_id"]
            match.name = resolved_person["name"]
            match.category = resolved_person["category"]
            match.is_match = temporal.decision_state == DecisionState.accept
            self.current_focus = f"{match.name} ({temporal.decision_state.value})"
        else:
            match.person_id = None
            match.name = "Unknown"
            match.category = "unknown"
            match.is_match = False
            if temporal.decision_state == DecisionState.reject:
                self.current_focus = "none"

        return match

    def _overlay_panel(self, frame, fps: float) -> None:
        h, w = frame.shape[:2]
        top_h = 122
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (min(w, 720), top_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        pulse = int(6 + (math.sin(time.time() * 4) + 1) * 4)
        cv2.circle(frame, (18, 18), pulse, (0, 255, 200), 1)
        draw_text(frame, "TRACE-ML // OPERATOR MODE", (40, 22), (0, 255, 200))
        draw_text(
            frame,
            f"FPS:{fps:.1f}  TRACKS:{self.temporal.active_track_count()}  LAT:{self.last_latency_ms:.0f}ms  CAMERA:device0",
            (12, 48),
            (190, 190, 190),
            scale=0.5,
        )
        draw_text(
            frame,
            f"ACCEPT:{self.decision_counters['accept']}  REVIEW:{self.decision_counters['review']}  REJECT:{self.decision_counters['reject']}",
            (12, 70),
            (190, 190, 190),
            scale=0.5,
        )
        trend = " ".join([f"{value:02.0f}" for value in list(self.recent_confidences)[-6:]]) or "n/a"
        draw_text(
            frame,
            f"FOCUS:{self.current_focus}  THR A/R:{self.settings.recognition.accept_threshold:.2f}/{self.settings.recognition.review_threshold:.2f}",
            (12, 92),
            (190, 190, 190),
            scale=0.5,
        )
        draw_text(
            frame,
            f"HEALTH fQ:{self.last_frame_queue_depth} rQ:{self.last_result_queue_depth}  TREND:{trend}",
            (12, 114),
            (190, 190, 190),
            scale=0.5,
        )

        # Right-side neon event feed.
        feed_width = min(460, w // 2)
        x0 = w - feed_width
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x0, 0), (w, min(h, 200)), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.45, frame, 0.55, 0, frame)
        draw_text(frame, "EVENT FEED", (x0 + 8, 22), (0, 255, 200), scale=0.52)
        for idx, item in enumerate(list(self.event_feed)[:7]):
            draw_text(frame, f"{idx+1:02d} {item}", (x0 + 8, 44 + idx * 20), (210, 210, 210), scale=0.47)

    def _annotate(self, packet: InferencePacket, fps: float):
        frame = packet.frame.copy()
        now = time.time()

        for i, (match, liveness) in enumerate(zip(packet.matches, packet.liveness, strict=False)):
            match = self._apply_temporal_decision(match, now=now)
            packet.matches[i] = match
            self.decision_counters[match.decision_state.value] += 1
            self.recent_confidences.append(float(match.smoothed_confidence))

            color = _decision_color(match.decision_state)
            draw_box(frame, match.bbox, color)
            if match.decision_state == DecisionState.accept:
                label = f"ACCEPT {match.name} {match.smoothed_confidence:.1f}%"
            elif match.decision_state == DecisionState.review:
                label = f"REVIEW {match.name} {match.smoothed_confidence:.1f}%"
            else:
                label = f"REJECT Unknown {match.smoothed_confidence:.1f}%"

            draw_text(frame, label, (match.bbox[0], max(20, match.bbox[1] - 10)), color)
            draw_text(
                frame,
                f"{match.track_id} | {match.decision_reason}",
                (match.bbox[0], match.bbox[1] + match.bbox[3] + 20),
                (220, 220, 220),
                scale=0.48,
            )
            if match.candidate_scores:
                top = match.candidate_scores[0]
                draw_text(
                    frame,
                    f"top:{top.get('name','?')} sim:{float(top.get('similarity',0.0)):.2f}",
                    (match.bbox[0], match.bbox[1] + match.bbox[3] + 38),
                    (170, 170, 170),
                    scale=0.46,
                )
            if liveness.flags:
                draw_text(
                    frame,
                    f"live:{liveness.score:.2f} {'/'.join(liveness.flags)}",
                    (match.bbox[0], match.bbox[1] + match.bbox[3] + 56),
                    (150, 150, 255),
                    scale=0.45,
                )

        if self.settings.pipeline.show_hud:
            self._overlay_panel(frame, fps=fps)
        self._publish_live_state(fps)
        return frame

    def run(self) -> None:
        frame_q: queue.Queue = queue.Queue(maxsize=self.settings.pipeline.frame_queue_size)
        result_q: queue.Queue = queue.Queue(maxsize=self.settings.pipeline.result_queue_size)
        capture = CameraCapture(self.settings, frame_q)
        inference = InferenceWorker(self.recognizer, self.store, frame_q, result_q)

        capture.start()
        inference.start()

        prev = time.time()
        fps = 0.0
        title = "TRACE-ML v3 | Press q to stop"

        try:
            while True:
                try:
                    packet: InferencePacket = result_q.get(timeout=0.5)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                now = time.time()
                fps = 0.92 * fps + 0.08 * (1.0 / max(1e-6, now - prev))
                prev = now
                self.last_latency_ms = max(0.0, (now - packet.captured_at) * 1000.0)
                self.last_frame_queue_depth = frame_q.qsize()
                self.last_result_queue_depth = result_q.qsize()

                display = self._annotate(packet, fps=fps)
                for match, emb, live in zip(packet.matches, packet.embeddings, packet.liveness, strict=False):
                    self._save_detection(packet.frame, match=match, embedding=emb, liveness=live)

                cv2.imshow(title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            inference.stop()
            capture.stop()
            cv2.destroyAllWindows()
