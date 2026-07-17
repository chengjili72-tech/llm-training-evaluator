from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..adapters.base import ModelAdapter
from ..config import AnalysisConfig
from ..schemas import LayerPrediction, TokenScore


def module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in module.buffers():
        if buffer.device.type != "meta":
            return buffer.device
    return torch.device("cpu")


def position_indices(
    input_ids: Tensor,
    attention_mask: Tensor,
    mode: str,
    configured: list[int],
) -> list[int]:
    valid_length = int(attention_mask[0].sum().item())
    if mode == "last":
        return [max(valid_length - 1, 0)]
    if mode == "all":
        return list(range(valid_length))

    resolved: list[int] = []
    for position in configured:
        normalized = position if position >= 0 else valid_length + position
        if normalized < 0 or normalized >= valid_length:
            raise IndexError(
                f"Position {position} resolves to {normalized}, outside valid sequence length "
                f"{valid_length}"
            )
        resolved.append(normalized)
    return resolved


class LogitLensProbe:
    def __init__(
        self,
        adapter: ModelAdapter,
        tokenizer: Any,
        config: AnalysisConfig,
    ) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.config = config
        self.norm = adapter.final_norm()
        self.lm_head = adapter.output_embeddings()
        self.device = module_device(self.lm_head)
        self._last_layer_hidden: Tensor | None = None
        self._last_positions: list[int] = []

    def analyze(
        self,
        hidden: Tensor,
        layer: int,
        input_ids: Tensor,
        attention_mask: Tensor,
        is_last_layer: bool = False,
    ) -> list[LayerPrediction]:
        positions = position_indices(
            input_ids, attention_mask, self.config.position_mode, self.config.positions
        )
        selected = hidden[0, positions, :].detach()
        logits = self._project(selected)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_k = min(self.config.top_k, probabilities.shape[-1])
        values, token_ids = torch.topk(probabilities, k=top_k, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)

        if is_last_layer:
            self._last_layer_hidden = selected.detach().cpu()
            self._last_positions = positions

        rows: list[LayerPrediction] = []
        cpu_ids = input_ids[0].detach().cpu()
        for row_index, position in enumerate(positions):
            scores: list[TokenScore] = []
            for rank, (token_id, probability) in enumerate(
                zip(token_ids[row_index].tolist(), values[row_index].tolist()), start=1
            ):
                scores.append(
                    TokenScore(
                        token_id=int(token_id),
                        token=self._token(int(token_id)),
                        text=self._decode(int(token_id)),
                        probability=float(probability),
                        rank=rank,
                    )
                )

            input_token_id = int(cpu_ids[position].item())
            rows.append(
                LayerPrediction(
                    layer=layer,
                    position=position,
                    input_token_id=input_token_id,
                    input_token=self._token(input_token_id),
                    entropy=float(entropy[row_index].item()),
                    top_tokens=scores,
                    hidden_state=(
                        selected[row_index].float().cpu().tolist()
                        if self.config.include_hidden_states
                        else None
                    ),
                )
            )
        return rows

    def validate_final_logits(
        self,
        native_logits: Tensor,
        rtol: float = 2e-3,
        atol: float = 2e-3,
    ) -> dict[str, float]:
        if self._last_layer_hidden is None:
            raise RuntimeError("No final-layer hidden state was captured")
        lens_logits = self._project(self._last_layer_hidden)
        native = native_logits[0, self._last_positions, :].detach().to(
            device=lens_logits.device, dtype=lens_logits.dtype
        )
        difference = (lens_logits - native).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                lens_logits.float().flatten(), native.float().flatten(), dim=0
            ).item()
        )
        if not torch.allclose(lens_logits, native, rtol=rtol, atol=atol):
            raise ValueError(
                "Final-layer Logit Lens validation failed: "
                f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, cosine={cosine:.8f}. "
                "This architecture needs a specialized model adapter."
            )
        return {"max_abs": max_abs, "mean_abs": mean_abs, "cosine": cosine}

    def _project(self, hidden: Tensor) -> Tensor:
        norm_device = module_device(self.norm)
        hidden = hidden.to(norm_device)
        normalized = self.norm(hidden)
        normalized = normalized.to(self.device)
        logits = self.lm_head(normalized)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, Tensor):
            raise TypeError(f"lm_head returned unsupported type {type(logits)!r}")
        return logits

    def _token(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return str(token)

    def _decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
