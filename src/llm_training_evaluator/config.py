from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PositionMode = Literal["last", "all", "selected"]


@dataclass(slots=True)
class ModelConfig:
    path: str
    revision: str | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device_map: str | None = "auto"
    trust_remote_code: bool = False


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

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.similarity_top_k < 1:
            raise ValueError("similarity_top_k must be at least 1")
        if self.moe_top_k is not None and self.moe_top_k < 1:
            raise ValueError("moe_top_k must be at least 1")
        if self.position_mode == "selected" and not self.positions:
            raise ValueError("positions are required when position_mode='selected'")
