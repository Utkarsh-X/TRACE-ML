"""GPU / execution-provider detection for ONNX Runtime.

Algorithm
---------
At startup we probe which ONNX Runtime execution providers are installed in this
Python environment and return a **ranked priority list**.  ONNX Runtime picks the
first provider in the list that is functional on the current machine, then falls
back to the next one automatically.

Priority order (when GPU is allowed):
    1. User-forced ``preferred_provider`` (if specified and available)
    2. CUDAExecutionProvider        — NVIDIA CUDA  (requires onnxruntime-gpu)
    3. DmlExecutionProvider         — Windows DirectML (NVIDIA / AMD without CUDA Toolkit)
    4. ROCmExecutionProvider        — AMD ROCm
    5. OpenVINOExecutionProvider    — Intel iGPU / NPU
    6. CoreMLExecutionProvider      — Apple Silicon (macOS)
    7. CPUExecutionProvider         — always the guaranteed fallback

When ``allow_gpu=False`` the list collapses to ``["CPUExecutionProvider"]``
immediately, skipping all probes.

Usage::

    from trace_aml.core.gpu import detect_providers, get_gpu_info

    providers = detect_providers(preferred="", allow_gpu=True, cuda_device_id=0)
    # e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]

    info = get_gpu_info()
    # {"active_provider": "CUDAExecutionProvider", "available": [...], ...}
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Candidate providers in preference order (GPU candidates only).
# CPUExecutionProvider is always appended at the end.
# ---------------------------------------------------------------------------
_GPU_PROVIDER_PRIORITY: list[str] = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",       # Windows DirectML — works without CUDA Toolkit
    "ROCmExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "TensorrtExecutionProvider",  # inserted after CUDA if present
]

_CPU_PROVIDER = "CPUExecutionProvider"

# Module-level cache so detection only runs once per process.
_cached_result: "_ProviderResult | None" = None


@dataclass
class _ProviderResult:
    """Internal cache entry produced by a single detection run."""

    providers: list[str]                  # ranked list to pass to ONNX Runtime
    active_provider: str                  # best GPU provider chosen, or CPU
    all_available: list[str]              # all providers reported by onnxruntime
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_onnxruntime_providers() -> list[str]:
    """Return all providers installed in the current onnxruntime package."""
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return [_CPU_PROVIDER]


def _probe_cuda() -> tuple[bool, int, list[str]]:
    """Probe CUDA availability via nvidia-smi (does not need onnxruntime-gpu).

    Returns:
        (cuda_available, device_count, device_names)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
            return bool(names), len(names), names
    except Exception:
        pass
    return False, 0, []


def _build_provider_list(
    preferred: str,
    allow_gpu: bool,
    cuda_device_id: int,
    available_providers: list[str],
) -> list[str]:
    """Build the final ranked provider list."""
    if not allow_gpu:
        return [_CPU_PROVIDER]

    ranked: list[str] = []

    # 1. User-forced preferred provider (only if available in this install)
    if preferred and preferred in available_providers:
        ranked.append(preferred)

    # 2. Walk GPU priority list
    for candidate in _GPU_PROVIDER_PRIORITY:
        if candidate in available_providers and candidate not in ranked:
            ranked.append(candidate)

    # 3. CPU always last
    if _CPU_PROVIDER not in ranked:
        ranked.append(_CPU_PROVIDER)

    return ranked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_providers(
    preferred: str = "",
    allow_gpu: bool = True,
    cuda_device_id: int = 0,
) -> list[str]:
    """Detect the best ONNX execution provider(s) and return a ranked list.

    The function caches its result after the first call; subsequent calls return
    the cached list instantly.

    Args:
        preferred:      If non-empty, this provider is tried first (before the
                        auto-detected order).  Ignored if not in the installed
                        providers list.
        allow_gpu:      When ``False`` the function returns
                        ``["CPUExecutionProvider"]`` immediately.
        cuda_device_id: CUDA device ordinal to use for CUDA providers.

    Returns:
        Ranked list of ONNX execution provider strings ready to pass directly to
        ``onnxruntime.InferenceSession`` or ``insightface.app.FaceAnalysis``.
    """
    global _cached_result

    if _cached_result is not None:
        return list(_cached_result.providers)

    available = _query_onnxruntime_providers()
    cuda_available, cuda_count, cuda_names = _probe_cuda()

    providers = _build_provider_list(
        preferred=preferred,
        allow_gpu=allow_gpu,
        cuda_device_id=cuda_device_id,
        available_providers=available,
    )

    # Determine the "active" provider — first non-CPU entry if any
    active_provider = next(
        (p for p in providers if p != _CPU_PROVIDER),
        _CPU_PROVIDER,
    )

    _cached_result = _ProviderResult(
        providers=providers,
        active_provider=active_provider,
        all_available=available,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
    )

    _log_selection(_cached_result, cuda_device_id)
    return list(_cached_result.providers)


def get_gpu_info() -> dict[str, Any]:
    """Return a JSON-serialisable summary of the GPU detection result.

    Call ``detect_providers`` first; if not called yet this function triggers
    detection with default settings.
    """
    if _cached_result is None:
        detect_providers()  # trigger detection with defaults

    assert _cached_result is not None
    return {
        "active_provider": _cached_result.active_provider,
        "provider_list": _cached_result.providers,
        "all_onnxruntime_providers": _cached_result.all_available,
        "cuda_available": _cached_result.cuda_available,
        "cuda_device_count": _cached_result.cuda_device_count,
        "cuda_device_names": _cached_result.cuda_device_names,
        "gpu_acceleration": _cached_result.active_provider != _CPU_PROVIDER,
    }


def reset_cache() -> None:
    """Clear the cached detection result (useful in tests)."""
    global _cached_result
    _cached_result = None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_selection(result: _ProviderResult, cuda_device_id: int) -> None:
    """Emit a single structured startup banner about the selected providers."""
    gpu_on = result.active_provider != _CPU_PROVIDER
    status = "GPU" if gpu_on else "CPU"

    if gpu_on:
        logger.success(
            "[GPU] Acceleration ENABLED  ·  provider={provider}  ·  "
            "devices={devices}  ·  ranked_list={ranked}",
            provider=result.active_provider,
            devices=result.cuda_device_names or ["(non-CUDA)"],
            ranked=result.providers,
        )
    else:
        # Explain WHY we are on CPU so the user can act on it.
        reasons: list[str] = []
        if not result.cuda_available:
            reasons.append("nvidia-smi not found or no NVIDIA GPU present")
        gpu_providers = [p for p in result.all_available if p != _CPU_PROVIDER]
        if not gpu_providers:
            reasons.append(
                "no GPU execution providers installed in onnxruntime "
                "(hint: pip install onnxruntime-gpu)"
            )
        reason_str = "; ".join(reasons) if reasons else "GPU disabled in config"
        logger.info(
            "[GPU] Running on CPU  ·  reason: {reason}  ·  "
            "available_providers={available}",
            reason=reason_str,
            available=result.all_available,
        )
