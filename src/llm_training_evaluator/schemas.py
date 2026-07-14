from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Sample:
    id: str
    prompt: str
    target: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenScore:
    token_id: int
    token: str
    text: str
    probability: float
    rank: int


@dataclass(slots=True)
class LayerPrediction:
    layer: int
    position: int
    input_token_id: int
    input_token: str
    entropy: float
    top_tokens: list[TokenScore]
    hidden_state: list[float] | None = None


@dataclass(slots=True)
class ExpertScore:
    expert_id: int
    probability: float
    rank: int


@dataclass(slots=True)
class MoeLayerResult:
    layer: int
    module: str
    routing_entropy: float
    expert_load: dict[int, int]
    selected_routes: list[dict[str, Any]]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value

