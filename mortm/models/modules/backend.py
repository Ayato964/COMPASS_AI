"""
Hardware Backend Strategy & Factory (SOLID / Strategy Pattern).
Supports NVIDIA CUDA GPUs, Apple Silicon MPS (Unified Memory), and Multi-Core CPUs
without requiring `flash_attn` or hardcoded `bfloat16`/`cuda` calls.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
import os
import threading
from typing import ContextManager, Dict, Any
import torch


# PyTorch native `F.scaled_dot_product_attention` is used exclusively across CUDA, MPS, and CPU;
# external `flash_attn` research package is never imported (`[R5-LOW-02]`).
HAS_FLASH_ATTN: bool = False


class IHardwareBackend(ABC):
    """Abstract Strategy interface for hardware-specific execution policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def compute_dtype(self) -> torch.dtype:
        pass

    @property
    @abstractmethod
    def supports_flash_attn(self) -> bool:
        pass

    @property
    @abstractmethod
    def supports_grouped_mm(self) -> bool:
        pass

    @abstractmethod
    def autocast_context(self) -> ContextManager:
        pass

    @abstractmethod
    def configure_environment(self) -> None:
        pass

    @abstractmethod
    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


    def summary(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "compute_dtype": str(self.compute_dtype),
            "supports_flash_attn": self.supports_flash_attn,
            "supports_grouped_mm": self.supports_grouped_mm,
        }


class CudaHardwareBackend(IHardwareBackend):
    """Strategy for NVIDIA CUDA GPUs (Ampere/Ada/Hopper BF16 or Turing/Volta FP16)."""

    def __init__(self, device_index: int = 0):
        self._device = torch.device(f"cuda:{device_index}")
        self._bf16_supported = bool(torch.cuda.is_bf16_supported())
        self._dtype = torch.bfloat16 if self._bf16_supported else torch.float16
        self._has_grouped_mm = hasattr(torch, "_grouped_mm") and self._bf16_supported
        self.configure_environment()

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def supports_flash_attn(self) -> bool:
        return HAS_FLASH_ATTN

    @property
    def supports_grouped_mm(self) -> bool:
        return self._has_grouped_mm

    def autocast_context(self) -> ContextManager:
        return torch.autocast(device_type="cuda", dtype=self._dtype)

    def configure_environment(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self._device)


class MpsHardwareBackend(IHardwareBackend):
    """Strategy for Apple Silicon (M1 / M2 / M3 / M4) Metal Performance Shaders."""

    def __init__(self, use_fp16: bool = True):
        self._device = torch.device("mps") if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else torch.device("cpu")
        self._dtype = torch.float16 if use_fp16 else torch.float32
        self.configure_environment()

    @property
    def name(self) -> str:
        return "mps"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def supports_flash_attn(self) -> bool:
        return False

    @property
    def supports_grouped_mm(self) -> bool:
        return False

    def autocast_context(self) -> ContextManager:
        if self._device.type == "mps":
            return torch.autocast(device_type="mps", dtype=self._dtype)
        return nullcontext()

    def configure_environment(self) -> None:
        pass

    def synchronize(self) -> None:
        if self._device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


class CpuHardwareBackend(IHardwareBackend):
    """Strategy for Multi-Core x86_64 / ARM64 CPUs without dedicated GPU."""

    def __init__(self, use_bf16: bool = False):
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.configure_environment()

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def supports_flash_attn(self) -> bool:
        return False

    @property
    def supports_grouped_mm(self) -> bool:
        return False

    def autocast_context(self) -> ContextManager:
        if self._dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def configure_environment(self) -> None:
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(1, min(cpu_count // 2, 8))
        try:
            torch.set_num_threads(optimal_threads)
        except Exception:
            pass

    def synchronize(self) -> None:
        pass


class HardwareBackendFactory:
    """Factory for instantiating the optimal IHardwareBackend across CUDA, MPS, and CPU."""

    _instances: Dict[str, IHardwareBackend] = {}
    _lock = threading.Lock()

    @classmethod
    def create(cls, preference: str = "auto") -> IHardwareBackend:
        pref = (preference or os.environ.get("COMPASS_BACKEND", "auto")).strip().lower()
        with cls._lock:
            if pref in cls._instances:
                return cls._instances[pref]

            if pref == "cuda":
                if torch.cuda.is_available():
                    backend = CudaHardwareBackend()
                else:
                    backend = CpuHardwareBackend()
            elif pref == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    backend = MpsHardwareBackend()
                else:
                    backend = CpuHardwareBackend()
            elif pref == "cpu":
                backend = CpuHardwareBackend()
            else:
                # "auto" detection: CUDA -> MPS -> CPU
                if torch.cuda.is_available():
                    backend = CudaHardwareBackend()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    backend = MpsHardwareBackend()
                else:
                    backend = CpuHardwareBackend()

            cls._instances[pref] = backend
            cls._instances[backend.name] = backend
            return backend
