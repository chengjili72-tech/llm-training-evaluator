from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PositionMode = Literal["last", "all", "selected"]
QuantizationMode = Literal["auto", "none", "4bit", "8bit"]


@dataclass(slots=True)
class ModelConfig:
    path: str
    revision: str | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device: str = "auto"
    device_map: str | None = "auto"
    quantization: QuantizationMode = "auto"
    accelerator_memory: str | None = None
    cpu_memory: str | None = None
    offload_folder: str = "~/.cache/llm-training-evaluator/offload"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    layers_path: str | None = None
    norm_path: str | None = None


@dataclass(slots=True)
class AnalysisConfig:
    top_k: int = 10
    similarity_top_k: int = 10
    position_mode: PositionMode = "last"
    positions: list[int] = field(default_factory=list)
    capture_moe: bool = True
    moe_top_k: int | None = None
    moe_detail: bool = False
    similarity_chunk_size: int = 16_384
    include_hidden_states: bool = False
    native_only: bool = False
    score_targets: bool = True
    max_length: int = 2048
    scoring_stride: int = 512
    calibration_bins: int = 15
    projection_chunk_size: int = 32

    def validate(self) -> None:
        if self.max_length < 2 or not 1 <= self.scoring_stride < self.max_length:
            raise ValueError("Require max_length >= 2 and 1 <= scoring_stride < max_length")
        if self.calibration_bins < 1 or self.projection_chunk_size < 1:
            raise ValueError("calibration_bins and projection_chunk_size must be positive")
        if self.similarity_chunk_size < 1:
            raise ValueError("similarity_chunk_size must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.similarity_top_k < 1:
            raise ValueError("similarity_top_k must be at least 1")
        if self.moe_top_k is not None and self.moe_top_k < 1:
            raise ValueError("moe_top_k must be at least 1")
        if self.position_mode == "selected" and not self.positions:
            raise ValueError("positions are required when position_mode='selected'")
