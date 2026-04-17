"""Unit tests for GPU detection logic (trace_aml.core.gpu).

All tests mock onnxruntime.get_available_providers() and nvidia-smi so they
run correctly on CPU-only CI machines.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Reset the module-level cache before each test.
from trace_aml.core import gpu as gpu_module


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the detection cache so each test starts fresh."""
    gpu_module.reset_cache()
    yield
    gpu_module.reset_cache()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_providers(providers: list[str]):
    """Patch onnxruntime.get_available_providers to return `providers`."""
    return patch.object(gpu_module, "_query_onnxruntime_providers", return_value=providers)


def _mock_cuda(available: bool, count: int = 0, names: list[str] | None = None):
    """Patch the nvidia-smi probe."""
    names = names or []
    return patch.object(
        gpu_module,
        "_probe_cuda",
        return_value=(available, count, names),
    )


# ---------------------------------------------------------------------------
# Core: provider priority algorithm
# ---------------------------------------------------------------------------

class TestBuildProviderList:
    """Test _build_provider_list directly (pure function, no I/O)."""

    def test_cpu_only_when_gpu_disabled(self):
        result = gpu_module._build_provider_list(
            preferred="",
            allow_gpu=False,
            cuda_device_id=0,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert result == ["CPUExecutionProvider"]

    def test_cuda_first_when_available(self):
        available = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        result = gpu_module._build_provider_list("", True, 0, available)
        assert result[0] == "CUDAExecutionProvider"
        assert result[-1] == "CPUExecutionProvider"

    def test_dml_first_when_cuda_absent(self):
        available = ["DmlExecutionProvider", "CPUExecutionProvider"]
        result = gpu_module._build_provider_list("", True, 0, available)
        assert result[0] == "DmlExecutionProvider"
        assert result[-1] == "CPUExecutionProvider"

    def test_cpu_fallback_always_present(self):
        available = ["CPUExecutionProvider"]
        result = gpu_module._build_provider_list("", True, 0, available)
        assert "CPUExecutionProvider" in result

    def test_preferred_provider_wins(self):
        available = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        result = gpu_module._build_provider_list("DmlExecutionProvider", True, 0, available)
        assert result[0] == "DmlExecutionProvider"
        assert "CUDAExecutionProvider" in result     # still included, just lower
        assert result[-1] == "CPUExecutionProvider"

    def test_preferred_provider_skipped_if_not_available(self):
        available = ["CPUExecutionProvider"]
        result = gpu_module._build_provider_list("CUDAExecutionProvider", True, 0, available)
        assert "CUDAExecutionProvider" not in result
        assert result == ["CPUExecutionProvider"]

    def test_no_duplicates(self):
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        result = gpu_module._build_provider_list("CUDAExecutionProvider", True, 0, available)
        assert result.count("CUDAExecutionProvider") == 1
        assert result.count("CPUExecutionProvider") == 1


# ---------------------------------------------------------------------------
# Public API: detect_providers()
# ---------------------------------------------------------------------------

class TestDetectProviders:

    def test_returns_cpu_when_gpu_disabled(self):
        with _mock_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with _mock_cuda(True, 1, ["NVIDIA RTX 4090"]):
                result = gpu_module.detect_providers(allow_gpu=False)
        assert result == ["CPUExecutionProvider"]

    def test_returns_cuda_when_available(self):
        with _mock_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with _mock_cuda(True, 1, ["NVIDIA RTX 4090"]):
                result = gpu_module.detect_providers(allow_gpu=True)
        assert result[0] == "CUDAExecutionProvider"
        assert "CPUExecutionProvider" in result

    def test_falls_back_to_cpu_without_gpu_providers(self):
        with _mock_providers(["CPUExecutionProvider"]):
            with _mock_cuda(False):
                result = gpu_module.detect_providers(allow_gpu=True)
        assert result == ["CPUExecutionProvider"]

    def test_result_is_cached(self):
        call_count = 0

        def counting_providers():
            nonlocal call_count
            call_count += 1
            return ["CPUExecutionProvider"]

        with patch.object(gpu_module, "_query_onnxruntime_providers", side_effect=counting_providers):
            with _mock_cuda(False):
                gpu_module.detect_providers()
                gpu_module.detect_providers()
                gpu_module.detect_providers()

        assert call_count == 1, "Provider detection should only run once (cached)"

    def test_preferred_provider_honoured(self):
        with _mock_providers(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]):
            with _mock_cuda(True, 1, ["RTX 3080"]):
                result = gpu_module.detect_providers(preferred="DmlExecutionProvider", allow_gpu=True)
        assert result[0] == "DmlExecutionProvider"


# ---------------------------------------------------------------------------
# Public API: get_gpu_info()
# ---------------------------------------------------------------------------

class TestGetGpuInfo:

    def test_gpu_info_structure_on_cpu_only(self):
        with _mock_providers(["CPUExecutionProvider"]):
            with _mock_cuda(False):
                gpu_module.detect_providers()
                info = gpu_module.get_gpu_info()

        assert info["gpu_acceleration"] is False
        assert info["active_provider"] == "CPUExecutionProvider"
        assert "CPUExecutionProvider" in info["provider_list"]
        assert isinstance(info["all_onnxruntime_providers"], list)

    def test_gpu_info_structure_with_cuda(self):
        with _mock_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with _mock_cuda(True, 1, ["Tesla V100"]):
                gpu_module.detect_providers()
                info = gpu_module.get_gpu_info()

        assert info["gpu_acceleration"] is True
        assert info["active_provider"] == "CUDAExecutionProvider"
        assert info["cuda_available"] is True
        assert info["cuda_device_count"] == 1
        assert "Tesla V100" in info["cuda_device_names"]

    def test_get_gpu_info_triggers_detection_if_not_called(self):
        """get_gpu_info() should not crash if detect_providers wasn't called first."""
        with _mock_providers(["CPUExecutionProvider"]):
            with _mock_cuda(False):
                info = gpu_module.get_gpu_info()
        assert "active_provider" in info
