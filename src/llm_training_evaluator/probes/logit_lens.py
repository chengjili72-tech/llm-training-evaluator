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
    valid = attention_mask[0].nonzero(as_tuple=False).flatten().cpu().tolist()
    if not valid:
        raise ValueError("No non-padding token positions")
    if mode == "last":
        return [valid[-1]]
    if mode == "all":
        return valid
    resolved = []
    for position in configured:
        index = position if position >= 0 else len(valid) + position
        if not 0 <= index < len(valid):
            raise IndexError(f"Position {position} outside valid sequence length {len(valid)}")
        resolved.append(valid[index])
    return list(dict.fromkeys(resolved))


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
        self._previous_hidden: list[Tensor] = []
        self._chunk_index = 0

    def analyze(self, hidden: Tensor, layer: int, input_ids: Tensor,
                attention_mask: Tensor, is_last_layer: bool = False) -> list[LayerPrediction]:
        positions = position_indices(input_ids, attention_mask, self.config.position_mode,
                                     self.config.positions)
        rows = []
        chunks = []
        for index, start in enumerate(range(0, len(positions), self.config.projection_chunk_size)):
            self._chunk_index = index
            self._active_positions = positions[start:start + self.config.projection_chunk_size]
            rows.extend(self._analyze_chunk(hidden, layer, input_ids, attention_mask, is_last_layer))
            if is_last_layer:
                chunks.append(self._last_layer_hidden)
        if is_last_layer:
            self._last_layer_hidden = torch.cat(chunks, dim=0)
            self._last_positions = positions
        return rows

    def _analyze_chunk(
        self,
        hidden: Tensor,
        layer: int,
        input_ids: Tensor,
        attention_mask: Tensor,
        is_last_layer: bool = False,
    ) -> list[LayerPrediction]:
        positions = self._active_positions
        selected = hidden[0, positions, :].detach()
        logits = self._project(selected)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_k = min(self.config.top_k, probabilities.shape[-1])
        values, token_ids = torch.topk(probabilities, k=top_k, dim=-1)
        previous = (self._project(self._previous_hidden[self._chunk_index]).float().softmax(-1)
                    if self._chunk_index < len(self._previous_hidden) else None)
        js = None
        if previous is not None:
            midpoint = (previous + probabilities) * .5
            js = .5 * (previous * (previous.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())
                       + probabilities * (probabilities.clamp_min(1e-30).log()
                                          - midpoint.clamp_min(1e-30).log())).sum(-1)
            js = js.clamp_min(0)
        if previous is None:
            self._previous_hidden.append(selected.detach().cpu())
        else:
            self._previous_hidden[self._chunk_index] = selected.detach().cpu()
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)

        if is_last_layer:
            self._last_layer_hidden = selected.detach().cpu()
            self._last_positions = positions

        rows: list[LayerPrediction] = []
        cpu_ids = input_ids[0].detach().cpu()
        cpu_token_ids, cpu_values = token_ids.cpu().tolist(), values.cpu().tolist()
        cpu_entropy = entropy.cpu().tolist()
        cpu_js = js.cpu().tolist() if js is not None else None
        cpu_hidden = selected.float().cpu().tolist() if self.config.include_hidden_states else None
        for row_index, position in enumerate(positions):
            scores: list[TokenScore] = []
            for rank, (token_id, probability) in enumerate(
                zip(cpu_token_ids[row_index], cpu_values[row_index]), start=1
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
                    entropy=float(cpu_entropy[row_index]),
                    adjacent_js_divergence=float(cpu_js[row_index]) if cpu_js is not None else None,
                    top_tokens=scores,
                    hidden_state=(
                        cpu_hidden[row_index]
                        if cpu_hidden is not None
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
        max_abs, error_sum, count = 0.0, 0.0, 0
        dot, lens_sq, native_sq = 0.0, 0.0, 0.0
        passed = True
        with torch.inference_mode():
            for start in range(0, len(self._last_positions), self.config.projection_chunk_size):
                positions = self._last_positions[start:start + self.config.projection_chunk_size]
                lens = self._project(self._last_layer_hidden[start:start + len(positions)]).float()
                native = native_logits[0, positions].detach().to(lens.device).float()
                lens, native = lens.cpu(), native.cpu()
                difference = (lens - native).abs()
                max_abs = max(max_abs, float(difference.max().item()))
                error_sum += float(difference.double().sum().item())
                count += difference.numel()
                dot += float((lens.double() * native.double()).sum().item())
                lens_sq += float(lens.double().square().sum().item())
                native_sq += float(native.double().square().sum().item())
                passed = passed and bool(torch.allclose(lens, native, rtol=rtol, atol=atol))
        mean_abs = error_sum / count
        cosine = dot / max((lens_sq * native_sq) ** .5, 1e-30)
        if not passed:
            raise ValueError(
                "Final-layer Logit Lens validation failed: "
                f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, cosine={cosine:.8f}. "
                "Use --native-only for target scoring, or add a specialized model adapter."
            )
        return {"max_abs": max_abs, "mean_abs": mean_abs, "cosine": cosine,
                "passed": True, "rtol": rtol, "atol": atol}

    def _project(self, hidden: Tensor) -> Tensor:
        if hasattr(self.adapter, "project"):
            return self.adapter.project(hidden)
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
        return str(token) if token is not None else f"<id:{token_id}>"

    def _decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
