from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig


LOW_MEMORY_CUDA_GIB = 16.0
CUDA_RESERVE_GIB = 1.25


@dataclass(slots=True)
class RuntimePlan:
    backend: str
    execution_device: str
    device_map: str | None
    quantization: str
    low_memory: bool
    max_memory: dict[Any, str] | None = None
    offload_folder: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["max_memory"] is not None:
            payload["max_memory"] = {
                str(key): value for key, value in payload["max_memory"].items()
            }
        return payload


def _load_torch_npu() -> bool:
    if importlib.util.find_spec("torch_npu") is None:
        return False
    try:
        __import__("torch_npu")
    except (ImportError, RuntimeError):
        return False
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def _bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def _cuda_total_memory_gib(index: int) -> float:
    return float(torch.cuda.get_device_properties(index).total_memory) / 1024**3


def _cuda_available_memory_gib(index: int) -> float:
    try:
        free, _total = torch.cuda.mem_get_info(index)
        return float(free) / 1024**3
    except (AttributeError, RuntimeError):
        return _cuda_total_memory_gib(index)


def _cpu_available_memory_gib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024**2
    except OSError:
        pass
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return float(page_size * available_pages) / 1024**3


def _memory_string(gib: float) -> str:
    return f"{max(int(gib), 1)}GiB"


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid accelerator device {device!r}") from exc


def _resolve_backend(requested: str) -> tuple[str, str]:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if _load_torch_npu():
            return "npu", "npu:0"
        if torch.cuda.is_available():
            return "cuda", "cuda:0"
        return "cpu", "cpu"
    if normalized == "npu" or normalized.startswith("npu:"):
        if not _load_torch_npu():
            raise RuntimeError(
                "NPU was requested but torch_npu is unavailable or torch.npu.is_available() "
                "is false. Install the torch_npu build matching PyTorch and CANN."
            )
        return "npu", "npu:0" if normalized == "npu" else normalized
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda", "cuda:0" if normalized == "cuda" else normalized
    if normalized == "cpu":
        return "cpu", "cpu"
    raise ValueError("device must be auto, cpu, cuda[:index], or npu[:index]")


def resolve_runtime(config: ModelConfig) -> RuntimePlan:
    backend, execution_device = _resolve_backend(config.device)
    quantization = config.quantization
    notes: list[str] = []
    if quantization not in {"auto", "none", "4bit", "8bit"}:
        raise ValueError("quantization must be auto, none, 4bit, or 8bit")

    if backend == "cuda":
        index = _device_index(execution_device)
        total_gib = _cuda_total_memory_gib(index)
        low_memory = total_gib < LOW_MEMORY_CUDA_GIB
        if quantization == "auto":
            if low_memory and _bitsandbytes_available():
                quantization = "4bit"
                notes.append(
                    "Low-memory CUDA mode selected NF4 4-bit weights with CPU/disk offload."
                )
            else:
                quantization = "none"
                if low_memory:
                    notes.append(
                        "bitsandbytes is not installed; using native weights with CPU/disk "
                        "offload. Install the cuda-low-memory extra for much lower RAM use."
                    )
        device_map = config.device_map
        if device_map is None:
            max_memory = None
        else:
            accelerator_memory = config.accelerator_memory or _memory_string(
                _cuda_available_memory_gib(index) - CUDA_RESERVE_GIB
            )
            cpu_memory = config.cpu_memory or _memory_string(_cpu_available_memory_gib() * 0.85)
            max_memory = {index: accelerator_memory, "cpu": cpu_memory}
        offload_folder = str(
            Path(config.offload_folder).expanduser()
            / hashlib.sha256(
                f"{config.path}@{config.revision or 'main'}".encode("utf-8")
            ).hexdigest()[:12]
        )
        return RuntimePlan(
            backend=backend,
            execution_device=execution_device,
            device_map=device_map,
            quantization=quantization,
            low_memory=low_memory,
            max_memory=max_memory,
            offload_folder=offload_folder,
            notes=notes,
        )

    if quantization in {"4bit", "8bit"}:
        raise ValueError("bitsandbytes 4-bit/8-bit loading is supported only on CUDA")
    if quantization == "auto":
        quantization = "none"
    if backend == "npu":
        notes.append("Ascend NPU detected; using native torch_npu execution.")
    return RuntimePlan(
        backend=backend,
        execution_device=execution_device,
        device_map=None,
        quantization=quantization,
        low_memory=False,
        notes=notes,
    )
