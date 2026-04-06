"""Thread-safe latest frame overlay for UI (normalized bounding boxes)."""

from __future__ import annotations

import threading
from typing import Any

from trace_aml.core.models import utc_now_iso

_lock = threading.Lock()
_state: dict[str, Any] = {
    "active": False,
    "frame_width": 0,
    "frame_height": 0,
    "fps": 0.0,
    "boxes": [],
    "updated_at": "",
}


def update_live_overlay(
    *,
    frame_width: int,
    frame_height: int,
    fps: float,
    boxes: list[dict[str, Any]],
) -> None:
    with _lock:
        _state["active"] = True
        _state["frame_width"] = int(frame_width)
        _state["frame_height"] = int(frame_height)
        _state["fps"] = float(fps)
        _state["boxes"] = boxes
        _state["updated_at"] = utc_now_iso()


def clear_live_overlay() -> None:
    with _lock:
        _state["active"] = False
        _state["boxes"] = []
        _state["updated_at"] = utc_now_iso()


def get_live_overlay() -> dict[str, Any]:
    with _lock:
        return {
            "active": bool(_state.get("active")),
            "frame_width": int(_state.get("frame_width") or 0),
            "frame_height": int(_state.get("frame_height") or 0),
            "fps": float(_state.get("fps") or 0.0),
            "boxes": list(_state.get("boxes") or []),
            "updated_at": str(_state.get("updated_at") or ""),
        }
