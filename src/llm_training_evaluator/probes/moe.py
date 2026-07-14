from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import Tensor

from ..config import AnalysisConfig
from ..schemas import MoeLayerResult
from .logit_lens import position_indices


def _tensor_from_mapping(output: dict[str, Any], keys: tuple[str, ...]) -> Tensor | None:
    for key in keys:
        value = output.get(key)
        if isinstance(value, Tensor):
            return value
    return None


class MoeRoutingProbe:
    """Extracts selected experts and routing summaries from common MoE router outputs."""

    def __init__(self, tokenizer: Any, config: AnalysisConfig, top_k: int) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.top_k = top_k

    def analyze(
        self,
        output: Any,
        layer: int,
        module_name: str,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> MoeLayerResult | None:
        logits, explicit_indices, explicit_weights = self._extract(output)
        batch_size, sequence_length = input_ids.shape

        if explicit_indices is not None:
            indices = self._reshape(explicit_indices, batch_size, sequence_length)
            if explicit_weights is None:
                weights = torch.ones_like(indices, dtype=torch.float32)
            else:
                weights = self._reshape(explicit_weights, batch_size, sequence_length).float()
            probabilities = None
        elif logits is not None:
            logits = self._reshape(logits, batch_size, sequence_length).float()
            if logits.shape[-1] < 2:
                return None
            row_sums = logits.sum(dim=-1)
            looks_like_probability = bool(
                logits.min().item() >= 0
                and torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3)
            )
            probabilities = logits if looks_like_probability else torch.softmax(logits, dim=-1)
            weights, indices = torch.topk(
                probabilities, k=min(self.top_k, probabilities.shape[-1]), dim=-1
            )
        else:
            return None

        valid = attention_mask.bool().to(indices.device)
        valid_indices = indices[valid]
        load = Counter(int(expert) for expert in valid_indices.flatten().tolist())

        if probabilities is not None:
            entropy_values = -(
                probabilities * probabilities.clamp_min(1e-30).log()
            ).sum(dim=-1)
            routing_entropy = float(entropy_values[valid].mean().item())
        else:
            normalized = weights.float() / weights.float().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy_values = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)
            routing_entropy = float(entropy_values[valid].mean().item())

        selected_positions = position_indices(
            input_ids, attention_mask, self.config.position_mode, self.config.positions
        )
        if self.config.moe_detail:
            selected_positions = list(range(int(attention_mask[0].sum().item())))

        routes: list[dict[str, object]] = []
        for position in selected_positions:
            experts = []
            for rank, (expert_id, probability) in enumerate(
                zip(indices[0, position].tolist(), weights[0, position].tolist()), start=1
            ):
                experts.append(
                    {
                        "expert_id": int(expert_id),
                        "probability": float(probability),
                        "rank": rank,
                    }
                )
            token_id = int(input_ids[0, position].item())
            routes.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "token": str(self.tokenizer.convert_ids_to_tokens(token_id)),
                    "experts": experts,
                }
            )

        return MoeLayerResult(
            layer=layer,
            module=module_name,
            routing_entropy=routing_entropy,
            expert_load=dict(sorted(load.items())),
            selected_routes=routes,
        )

    @staticmethod
    def _reshape(tensor: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        if tensor.ndim == 2 and tensor.shape[0] == batch_size * sequence_length:
            return tensor.reshape(batch_size, sequence_length, tensor.shape[-1])
        if tensor.ndim >= 3 and tensor.shape[0] == batch_size:
            return tensor
        raise ValueError(
            f"Router tensor shape {tuple(tensor.shape)} cannot be aligned with "
            f"input shape {(batch_size, sequence_length)}"
        )

    @staticmethod
    def _extract(output: Any) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if isinstance(output, Tensor):
            return output, None, None

        if isinstance(output, dict):
            indices = _tensor_from_mapping(
                output, ("topk_idx", "topk_indices", "expert_indices", "selected_experts")
            )
            weights = _tensor_from_mapping(
                output, ("topk_weight", "topk_weights", "expert_weights", "routing_weights")
            )
            logits = _tensor_from_mapping(output, ("router_logits", "logits", "scores"))
            return logits, indices, weights

        if isinstance(output, (tuple, list)):
            tensors = [value for value in output if isinstance(value, Tensor)]
            indices = next(
                (value for value in tensors if not value.dtype.is_floating_point), None
            )
            weights = next(
                (
                    value
                    for value in tensors
                    if value.dtype.is_floating_point
                    and indices is not None
                    and value.shape == indices.shape
                ),
                None,
            )
            logits = next(
                (
                    value
                    for value in tensors
                    if value.dtype.is_floating_point and value is not weights
                ),
                None,
            )
            return logits, indices, weights

        return None, None, None

