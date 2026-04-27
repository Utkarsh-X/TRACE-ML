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

import io
import os
from pathlib import Path
import subprocess
import sys
import textwrap
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
    # TensorrtExecutionProvider intentionally omitted: requires a full TensorRT SDK
    # install separate from onnxruntime-gpu. Without it, ONNX Runtime prints a
    # verbose fail/retry block for every model load. Users who have TensorRT can
    # force it via the preferred_provider config field.
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


_CUDA_REQUIRED_DLLS = [
    "cublasLt64_12.dll",   # required by onnxruntime_providers_cuda.dll
    "cublas64_12.dll",     # required by onnxruntime_providers_tensorrt.dll
]

_PROVIDER_REQUIRED_DLLS: dict[str, list[str]] = {
    "CUDAExecutionProvider": _CUDA_REQUIRED_DLLS,
    "TensorrtExecutionProvider": _CUDA_REQUIRED_DLLS,
}


def _windows_cuda_bin_dirs() -> list[Path]:
    """Return existing CUDA bin directories likely to contain runtime DLLs."""
    if os.name != "nt":
        return []

    candidates: list[Path] = []
    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH") and value:
            base = Path(value)
            candidates.append(base / "bin")
            candidates.append(base / "bin" / "x64")

    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if root.exists():
        for version_dir in sorted(root.glob("v*"), reverse=True):
            candidates.append(version_dir / "bin")
            candidates.append(version_dir / "bin" / "x64")

    # Pip-installed NVIDIA runtime wheels (nvidia-*-cu12) place DLLs here.
    site_nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if site_nvidia_root.exists():
        for package_dir in site_nvidia_root.glob("*"):
            candidates.append(package_dir / "bin")

    seen: set[str] = set()
    resolved: list[Path] = []
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            resolved.append(path)
    return resolved


def _load_windows_dll(dll_name: str) -> bool:
    """Try loading a DLL by name, then via known CUDA directories."""
    if os.name != "nt":
        return True

    import ctypes

    try:
        ctypes.WinDLL(dll_name)
        return True
    except OSError:
        pass

    for directory in _windows_cuda_bin_dirs():
        full_path = directory / dll_name
        if not full_path.exists():
            continue
        try:
            dll_dir_handle = os.add_dll_directory(str(directory))
            try:
                ctypes.WinDLL(str(full_path))
                return True
            finally:
                dll_dir_handle.close()
        except Exception:
            continue
    return False


def _probe_cuda_with_real_model(timeout_s: int = 25) -> bool:
    """Verify InsightFace actually binds CUDA in an isolated subprocess.

    Returns True only if the subprocess completes and reports CUDA provider
    usage. This avoids enabling CUDA when ORT can be imported but InsightFace
    still falls back to CPU or triggers native instability.
    """
    script = textwrap.dedent(
        r"""
        import sys
        from pathlib import Path

        import insightface
        import numpy as np

        candidates = [
            Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx",
            Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
        ]
        model = next((p for p in candidates if p.exists()), None)
        if model is None:
            # No local model cache yet - don't block startup on first download.
            sys.exit(0)

        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        app.get(np.zeros((96, 96, 3), dtype=np.uint8))
        """
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )
        combined = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        if completed.returncode != 0:
            return False
        return "CUDAExecutionProvider" in combined
    except Exception:
        return False


def _verify_provider(provider: str, cuda_device_id: int = 0) -> bool:
    """Confirm the given EP is usable before handing the list to ONNX Runtime."""
    if os.name == "nt" and provider in _PROVIDER_REQUIRED_DLLS:
        for dll in _PROVIDER_REQUIRED_DLLS[provider]:
            if not _load_windows_dll(dll):
                return False

        try:
            import ctypes
            import onnxruntime as ort

            capi_dir = Path(ort.__file__).resolve().parent / "capi"
            provider_dll = {
                "CUDAExecutionProvider": "onnxruntime_providers_cuda.dll",
                "TensorrtExecutionProvider": "onnxruntime_providers_tensorrt.dll",
            }.get(provider)
            if not provider_dll:
                return True

            target = capi_dir / provider_dll
            if not target.exists():
                return False

            handles = [os.add_dll_directory(str(d)) for d in _windows_cuda_bin_dirs()]
            try:
                ctypes.WinDLL(str(target))
            finally:
                for h in handles:
                    h.close()

            if provider == "CUDAExecutionProvider" and not _probe_cuda_with_real_model():
                return False
            return True
        except Exception:
            return False

    # Non-Windows or providers without explicit runtime DLL requirements.
    # If ORT reports them as available, defer final execution checks to runtime.
    return True

def _build_provider_list(
    preferred: str,
    allow_gpu: bool,
    cuda_device_id: int,
    available_providers: list[str],
) -> list[str]:
    """Build the final ranked provider list, verifying each GPU EP actually loads."""
    if not allow_gpu:
        return [_CPU_PROVIDER]

    ranked: list[str] = []

    # 1. User-forced preferred provider (only if available in this install)
    if preferred and preferred in available_providers:
        if _verify_provider(preferred, cuda_device_id):
            ranked.append(preferred)
        else:
            logger.warning(
                "[GPU] preferred_provider '{}' is listed by onnxruntime but failed "
                "to load (missing DLL?). Falling back to auto-detection.",
                preferred,
            )

    # 2. Walk GPU priority list — probe each candidate before accepting it
    for candidate in _GPU_PROVIDER_PRIORITY:
        if candidate in available_providers and candidate not in ranked:
            if _verify_provider(candidate, cuda_device_id):
                ranked.append(candidate)
            else:
                logger.warning(
                    "[GPU] {} is installed but its DLL failed to load "
                    "(likely missing CUDA/cuBLAS runtime). Skipping.",
                    candidate,
                )

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
        reason_str = "; ".join(reasons) if reasons else (
            "no functional GPU execution provider could be initialized"
        )
        logger.info(
            "[GPU] Running on CPU  ·  reason: {reason}  ·  "
            "available_providers={available}",
            reason=reason_str,
            available=result.all_available,
        )
