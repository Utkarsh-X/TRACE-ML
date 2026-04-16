"""Live recognition session orchestration with temporal decisions."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from typing import Any
from collections import deque
from pathlib import Path

import cv2

from trace_aml.core.config import Settings
from trace_aml.core.ids import new_detection_id
from trace_aml.core.models import ActionTrigger, DecisionState, DetectionEvent, RecognitionMatch
from trace_aml.core.streaming import EventStreamPublisher, NullEventStreamPublisher
from trace_aml.liveness.base import LivenessResult
from trace_aml.pipeline.action_engine import ActionEngine
from trace_aml.pipeline.capture import CameraCapture
from trace_aml.pipeline.entity_resolver import EntityResolver
from trace_aml.pipeline.incident_manager import IncidentManager
from trace_aml.pipeline.inference import InferencePacket, InferenceWorker
from trace_aml.pipeline.policy_engine import PolicyEngine
from trace_aml.pipeline.rules_engine import RulesEngine
from trace_aml.pipeline.live_overlay import update_live_overlay
from trace_aml.pipeline.temporal import TemporalDecisionEngine
from trace_aml.recognizers.arcface import ArcFaceRecognizer
from trace_aml.store.vector_store import VectorStore
from trace_aml.store.portrait_store import PortraitStore


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
        self.portrait_store = PortraitStore(settings)
        self.event_feed: deque[str] = deque(maxlen=8)
        self.recent_confidences: deque[float] = deque(maxlen=8)
        self.decision_counters = {"accept": 0, "review": 0, "reject": 0}
        self.current_focus = "none"
        self.last_latency_ms = 0.0
        self.last_frame_queue_depth = 0
        self.last_result_queue_depth = 0
        self._headless_mode = False
        # ── Camera & Recognition Control State ──────────────────────────────────
        # Camera: Controls frame capture (independent)
        # Recognition: Controls inference processing (requires camera on)
        self._camera_enabled = False
        self._recognition_enabled = False
        self._camera_lock = threading.Lock()
        self._capture: CameraCapture | None = None
        self._inference: InferenceWorker | None = None
        self._frame_queue: queue.Queue | None = None
        self._result_queue: queue.Queue | None = None
        # ─────────────────────────────────────────────────────────────────────
        # ── Entity Commitment Protocol ─────────────────────────────────────────
        # Track IDs that have passed the commitment gate and had their first
        # DB write. Prevents warmup-phase ghost entities.
        self._committed_tracks: set[str] = set()
        # ─────────────────────────────────────────────────────────────────────

    # ── Camera Control API (Frontend-driven) ──────────────────────────────────
    def is_camera_enabled(self) -> bool:
        """Check if camera is currently enabled."""
        with self._camera_lock:
            return self._camera_enabled

    def enable_camera(self) -> dict[str, Any]:
        """Enable camera capture. Safe to call multiple times."""
        with self._camera_lock:
            if self._camera_enabled:
                return {"status": "already_enabled", "message": "Camera is already enabled"}
            
            try:
                # Create fresh queues for this session
                self._frame_queue = queue.Queue(maxsize=self.settings.pipeline.frame_queue_size)
                self._result_queue = queue.Queue(maxsize=self.settings.pipeline.result_queue_size)
                
                # Create and start capture + inference
                self._capture = CameraCapture(self.settings, self._frame_queue)
                self._inference = InferenceWorker(
                    self.recognizer,
                    self.store,
                    self._frame_queue,
                    self._result_queue,
                )
                
                self._capture.start()
                self._inference.start()
                self._camera_enabled = True
                
                return {
                    "status": "enabled",
                    "message": "Camera started successfully",
                    "camera_index": self.settings.camera.device_index,
                }
            except Exception as e:
                self._camera_enabled = False
                self._capture = None
                self._inference = None
                self._frame_queue = None
                self._result_queue = None
                return {
                    "status": "error",
                    "message": f"Failed to enable camera: {str(e)}",
                }

    def disable_camera(self) -> dict[str, Any]:
        """Disable camera capture. Safe to call multiple times."""
        with self._camera_lock:
            if not self._camera_enabled:
                return {"status": "already_disabled", "message": "Camera is already disabled"}
            
            try:
                if self._inference is not None:
                    self._inference.stop()
                    self._inference = None
                
                if self._capture is not None:
                    self._capture.stop()
                    self._capture = None
                
                self._frame_queue = None
                self._result_queue = None
                self._camera_enabled = False
                
                return {
                    "status": "disabled",
                    "message": "Camera stopped successfully",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to disable camera: {str(e)}",
                }

    def get_camera_status(self) -> dict[str, Any]:
        """Get current camera status."""
        with self._camera_lock:
            return {
                "enabled": self._camera_enabled,
                "recognition_enabled": self._recognition_enabled,
                "camera_index": self.settings.camera.device_index,
                "resolution": f"{self.settings.camera.width}x{self.settings.camera.height}",
                "fps": self.settings.camera.fps,
            }

    # ── Recognition/Inference Control (requires camera to be enabled first) ──
    def enable_recognition(self) -> dict[str, Any]:
        """Enable face recognition inference. Requires camera to be enabled."""
        with self._camera_lock:
            if not self._camera_enabled:
                return {
                    "status": "error",
                    "message": "Cannot enable recognition: camera is not enabled. Enable camera first.",
                }
            
            if self._recognition_enabled:
                return {"status": "already_enabled", "message": "Recognition is already enabled"}
            
            try:
                if self._inference is None:
                    # Create new inference worker if it doesn't exist
                    assert self._frame_queue is not None
                    self._inference = InferenceWorker(
                        self.recognizer,
                        self.store,
                        self._frame_queue,
                        self._result_queue,
                    )
                
                self._inference.start()
                self._recognition_enabled = True
                
                return {
                    "status": "enabled",
                    "message": "Face recognition started",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to enable recognition: {str(e)}",
                }

    def disable_recognition(self) -> dict[str, Any]:
        """Disable face recognition inference (keep camera running)."""
        with self._camera_lock:
            if not self._recognition_enabled:
                return {"status": "already_disabled", "message": "Recognition is already disabled"}
            
            try:
                if self._inference is not None:
                    self._inference.stop()
                    self._inference = None
                
                self._recognition_enabled = False
                
                return {
                    "status": "disabled",
                    "message": "Face recognition stopped",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to disable recognition: {str(e)}",
                }

    def get_recognition_status(self) -> dict[str, Any]:
        """Get current recognition status."""
        with self._camera_lock:
            return {
                "enabled": self._recognition_enabled,
                "camera_enabled": self._camera_enabled,
                "message": "Can only enable recognition if camera is enabled" if not self._camera_enabled else "Ready",
            }
    # ─────────────────────────────────────────────────────────────────────────

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
        # ── Layer 5: Minimum unknown surface threshold ─────────────────────────
        # Don't create an unknown entity for faces so ambiguous the system has
        # less than min_unknown_surface_threshold% smoothed confidence in any
        # direction. These are noise, not persons of interest.
        if decision == DecisionState.reject and not match.person_id:
            min_surface = self.settings.recognition.min_unknown_surface_threshold
            if match.smoothed_confidence < min_surface:
                return
        # ─────────────────────────────────────────────────────────────────────
        resolution = self.entity_resolver.resolve(match, embedding)

        # ── Best-portrait update ────────────────────────────────────────────
        # Only update for ACCEPT decisions — we never want a blurry "review"
        # or ghost "reject" face to become someone's profile picture.
        if match.decision_state == DecisionState.accept and match.bbox and match.person_id:
            self.portrait_store.try_update_portrait(
                entity_id=resolution.entity_id,
                frame_bgr=frame,
                bbox=match.bbox,
                score=float(match.similarity),
            )
        # ────────────────────────────────────────────────────────────────────

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
        draw_text(frame, "TRACE-AML // OPERATOR MODE", (40, 22), (0, 255, 200))
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
        min_commit_conf = self.settings.temporal.min_commit_confidence
        min_commit_votes = self.settings.temporal.min_commit_votes

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

        if self.settings.pipeline.show_hud and not self._headless_mode:
            self._overlay_panel(frame, fps=fps)
        self._publish_live_state(fps)
        h, w = frame.shape[:2]
        boxes: list[dict[str, Any]] = []
        for match in packet.matches:
            x, y, bw, bh = match.bbox
            # ── Layer 2: Temporal commitment gate ─────────────────────────────
            # Only mark a track as committed (and allow DB writes) once the
            # temporal engine has produced enough stable votes / confidence.
            # Overlay boxes are drawn for all tracks regardless (operator
            # visibility from frame 1), but entity/detection records only
            # persist after commitment.
            tid = match.track_id
            if tid and tid not in self._committed_tracks:
                temporal_result = self.temporal._tracks.get(tid)
                votes = 0
                if temporal_result is not None:
                    from collections import Counter
                    non_unk = [x for x in temporal_result.identities if x != "unknown"]
                    votes = Counter(non_unk).most_common(1)[0][1] if non_unk else 0
                conf_ok = match.smoothed_confidence >= min_commit_conf
                votes_ok = votes >= min_commit_votes or match.decision_state.value != "reject"
                if conf_ok and votes_ok:
                    self._committed_tracks.add(tid)
            # ─────────────────────────────────────────────────────────────────
            boxes.append(
                {
                    "x": float(x) / max(w, 1),
                    "y": float(y) / max(h, 1),
                    "w": float(bw) / max(w, 1),
                    "h": float(bh) / max(h, 1),
                    "label": str(match.name or "Unknown"),
                    "decision": str(match.decision_state.value),
                    "confidence": float(match.smoothed_confidence),
                    "track_id": str(match.track_id or ""),
                }
            )
        update_live_overlay(frame_width=w, frame_height=h, fps=fps, boxes=boxes)
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
        title = "TRACE-AML v3 | Press q to stop"

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
                    # Only persist if track has passed the commitment gate.
                    if match.track_id and match.track_id not in self._committed_tracks:
                        continue
                    self._save_detection(packet.frame, match=match, embedding=emb, liveness=live)

                cv2.imshow(title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            inference.stop()
            capture.stop()
            cv2.destroyAllWindows()

    def run_headless(self) -> None:
        """Run recognition session loop (camera/recognition control is frontend-driven via API).
        
        This method runs indefinitely:
        - If camera disabled: waits (no frame capture)
        - If camera enabled but recognition disabled: captures frames but doesn't process
        - If both enabled: processes frames normally
        """
        prev = time.time()
        fps = 0.0
        self._headless_mode = True
        try:
            while True:
                # Check if camera is enabled
                with self._camera_lock:
                    if not self._camera_enabled or self._result_queue is None:
                        # Camera is disabled, just wait and continue
                        time.sleep(0.1)
                        continue
                    
                    # Camera is enabled - now check if recognition is enabled
                    if not self._recognition_enabled:
                        # Camera on, but recognition off - skip processing, just keep polling
                        time.sleep(0.1)
                        continue
                    
                    result_queue = self._result_queue
                
                # Both camera AND recognition are enabled, try to get a result packet
                try:
                    packet: InferencePacket = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                now = time.time()
                fps = 0.92 * fps + 0.08 * (1.0 / max(1e-6, now - prev))
                prev = now
                self.last_latency_ms = max(0.0, (now - packet.captured_at) * 1000.0)
                
                # Get queue depths (with lock to be safe)
                with self._camera_lock:
                    self.last_frame_queue_depth = self._frame_queue.qsize() if self._frame_queue else 0
                    self.last_result_queue_depth = result_queue.qsize()

                self._annotate(packet, fps=fps)
                for match, emb, live in zip(packet.matches, packet.embeddings, packet.liveness, strict=False):
                    # Only persist if track has passed the commitment gate.
                    if match.track_id and match.track_id not in self._committed_tracks:
                        continue
                    self._save_detection(packet.frame, match=match, embedding=emb, liveness=live)
        finally:
            self._headless_mode = False
            # Cleanup any active state on shutdown
            with self._camera_lock:
                if self._inference is not None:
                    try:
                        self._inference.stop()
                    except Exception:
                        pass
                if self._capture is not None:
                    try:
                        self._capture.stop()
                    except Exception:
                        pass
                self._camera_enabled = False
                self._recognition_enabled = False

