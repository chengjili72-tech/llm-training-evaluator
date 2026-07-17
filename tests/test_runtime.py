from __future__ import annotations

import pytest

from llm_training_evaluator.config import ModelConfig
from llm_training_evaluator.runtime import resolve_runtime


def test_auto_device_prefers_npu(monkeypatch) -> None:
    monkeypatch.setattr("llm_training_evaluator.runtime._load_torch_npu", lambda: True)
    monkeypatch.setattr("llm_training_evaluator.runtime.torch.cuda.is_available", lambda: True)

    plan = resolve_runtime(ModelConfig(path="demo/model"))

    assert plan.backend == "npu"
    assert plan.execution_device == "npu:0"
    assert plan.device_map is None
    assert plan.quantization == "none"


def test_eight_gib_cuda_automatically_uses_4bit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._resolve_backend", lambda _device: ("cuda", "cuda:0")
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cuda_total_memory_gib", lambda _i: 8.0)
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._cuda_available_memory_gib", lambda _i: 7.8
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cpu_available_memory_gib", lambda: 32.0)
    monkeypatch.setattr("llm_training_evaluator.runtime._bitsandbytes_available", lambda: True)

    plan = resolve_runtime(
        ModelConfig(path="deepseek-ai/DeepSeek-V2-Lite", offload_folder=str(tmp_path))
    )

    assert plan.backend == "cuda"
    assert plan.low_memory is True
    assert plan.quantization == "4bit"
    assert plan.device_map == "auto"
    assert plan.max_memory == {0: "6GiB", "cpu": "27GiB"}
    assert plan.offload_folder is not None
    assert plan.offload_folder.startswith(str(tmp_path))


def test_low_memory_cuda_falls_back_to_native_offload_without_bitsandbytes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._resolve_backend", lambda _device: ("cuda", "cuda:0")
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cuda_total_memory_gib", lambda _i: 8.0)
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._cuda_available_memory_gib", lambda _i: 7.5
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cpu_available_memory_gib", lambda: 16.0)
    monkeypatch.setattr("llm_training_evaluator.runtime._bitsandbytes_available", lambda: False)

    plan = resolve_runtime(ModelConfig(path="demo/model", offload_folder=str(tmp_path)))

    assert plan.quantization == "none"
    assert plan.max_memory == {0: "6GiB", "cpu": "13GiB"}
    assert "bitsandbytes is not installed" in plan.notes[0]


def test_npu_rejects_cuda_only_quantization(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._resolve_backend", lambda _device: ("npu", "npu:0")
    )

    with pytest.raises(ValueError, match="only on CUDA"):
        resolve_runtime(ModelConfig(path="demo/model", quantization="4bit"))


def test_runtime_metadata_uses_json_safe_memory_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._resolve_backend", lambda _device: ("cuda", "cuda:1")
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cuda_total_memory_gib", lambda _i: 24.0)
    monkeypatch.setattr(
        "llm_training_evaluator.runtime._cuda_available_memory_gib", lambda _i: 20.0
    )
    monkeypatch.setattr("llm_training_evaluator.runtime._cpu_available_memory_gib", lambda: 64.0)

    payload = resolve_runtime(ModelConfig(path="demo/model", device="cuda:1")).to_dict()

    assert payload["max_memory"] == {"1": "18GiB", "cpu": "54GiB"}
