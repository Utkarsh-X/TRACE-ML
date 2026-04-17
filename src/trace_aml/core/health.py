"""Health checks for doctor command."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import cv2

from trace_aml.core.config import Settings


@dataclass
class HealthCheck:
    name: str
    status: str
    detail: str


def check_dependency(module_name: str) -> HealthCheck:
    try:
        importlib.import_module(module_name)
        return HealthCheck(module_name, "OK", "import succeeded")
    except Exception as exc:
        return HealthCheck(module_name, "FAIL", str(exc))


def check_camera(device_index: int = 0) -> HealthCheck:
    cap = cv2.VideoCapture(device_index)
    ok = cap.isOpened()
    cap.release()
    if ok:
        return HealthCheck("camera", "OK", f"device {device_index} available")
    return HealthCheck("camera", "FAIL", f"device {device_index} unavailable")


def check_paths(settings: Settings) -> list[HealthCheck]:
    checks = []
    for p in [
        settings.store.root,
        settings.store.vectors_dir,
        settings.store.screenshots_dir,
        settings.store.exports_dir,
        str(Path(settings.logging.file_path).parent),
    ]:
        path = Path(p)
        try:
            path.mkdir(parents=True, exist_ok=True)
            checks.append(HealthCheck(f"path:{p}", "OK", "ready"))
        except Exception as exc:
            checks.append(HealthCheck(f"path:{p}", "FAIL", str(exc)))
    return checks


def check_gpu(settings: Settings) -> HealthCheck:
    """Probe GPU / ONNX execution-provider availability."""
    try:
        from trace_aml.core.gpu import detect_providers, get_gpu_info

        detect_providers(
            preferred=settings.gpu.preferred_provider,
            allow_gpu=settings.gpu.enabled,
            cuda_device_id=settings.gpu.cuda_device_id,
        )
        info = get_gpu_info()
        active = info["active_provider"]
        gpu_on = info["gpu_acceleration"]

        if not settings.gpu.enabled:
            detail = f"GPU disabled in config — using {active}"
            return HealthCheck("gpu", "WARN", detail)

        if gpu_on:
            devices = info.get("cuda_device_names") or ["(non-CUDA GPU)"]
            detail = f"{active} — devices: {', '.join(devices)}"
            return HealthCheck("gpu", "OK", detail)
        else:
            detail = (
                f"No GPU provider available — falling back to {active}. "
                "Install onnxruntime-gpu for CUDA support."
            )
            return HealthCheck("gpu", "WARN", detail)
    except Exception as exc:
        return HealthCheck("gpu", "FAIL", f"GPU probe error: {exc}")


def run_health_checks(settings: Settings) -> list[HealthCheck]:
    deps = [
        "numpy",
        "lancedb",
        "duckdb",
        "onnxruntime",
        "rich",
        "typer",
        "insightface",
    ]
    checks = [check_dependency(dep) for dep in deps]
    checks.extend(check_paths(settings))
    checks.append(check_camera(settings.camera.device_index))
    checks.append(check_gpu(settings))
    return checks
