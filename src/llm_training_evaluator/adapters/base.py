from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class RouterModule:
    layer: int
    name: str
    module: nn.Module


class ModelAdapter(ABC):
    """Normalizes architecture-specific module locations for the probe engine."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def layers(self) -> list[nn.Module]: ...

    @abstractmethod
    def final_norm(self) -> nn.Module: ...

    @abstractmethod
    def input_embeddings(self) -> nn.Module: ...

    @abstractmethod
    def output_embeddings(self) -> nn.Module: ...

    @abstractmethod
    def router_modules(self) -> Iterable[RouterModule]: ...

    @staticmethod
    def hidden_tensor(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        raise TypeError(f"Cannot extract a hidden-state tensor from {type(output)!r}")

