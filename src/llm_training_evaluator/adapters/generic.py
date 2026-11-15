from __future__ import annotations

import re
from collections.abc import Iterable
from functools import reduce
from typing import Any

from torch import nn

from .base import ModelAdapter, RouterModule


LAYER_PATHS = (
    "model.layers",
    "model.decoder.layers",
    "transformer.h",
    "transformer.blocks",
    "model.transformer.blocks",
    "transformer.layers",
    "model.decoder.block",
    "gpt_neox.layers",
    "language_model.model.layers",
    "language_model.layers",
    "backbone.layers",
)

NORM_PATHS = (
    "model.norm",
    "model.decoder.final_layer_norm",
    "transformer.ln_f",
    "transformer.norm_f",
    "model.transformer.norm_f",
    "transformer.norm",
    "model.final_layernorm",
    "model.final_layer_norm",
    "gpt_neox.final_layer_norm",
    "language_model.model.norm",
    "language_model.norm",
    "backbone.norm",
)

ROUTER_SEGMENTS = {"gate", "router", "router_layer"}
LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")


def _resolve(root: Any, path: str) -> Any:
    return reduce(getattr, path.split("."), root)


class GenericDecoderAdapter(ModelAdapter):
    """Adapter for common decoder-only Hugging Face architectures.

    New architectures should subclass this adapter when their transformer blocks,
    final normalization, or router outputs do not follow the common conventions.
    """

    def __init__(self, model: nn.Module, layers_path: str | None = None,
                 norm_path: str | None = None) -> None:
        super().__init__(model)
        self.layers_path = layers_path
        self.norm_path = norm_path

    def _first_path(self, candidates: tuple[str, ...], kind: str) -> Any:
        for path in candidates:
            try:
                value = _resolve(self.model, path)
            except AttributeError:
                continue
            if value is not None:
                return value
        raise ValueError(
            f"Could not locate {kind}. Add a model adapter; tried: {', '.join(candidates)}"
        )

    def layers(self) -> list[nn.Module]:
        layers = self._first_path((self.layers_path,) if self.layers_path else LAYER_PATHS, "decoder layers")
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError(f"Located layers object has unsupported type: {type(layers)!r}")
        return list(layers)

    def final_norm(self) -> nn.Module:
        if self.norm_path == "identity":
            return nn.Identity()
        try:
            norm = self._first_path((self.norm_path,) if self.norm_path else NORM_PATHS,
                                    "final normalization")
        except ValueError:
            if getattr(self.model.config, "model_type", None) == "opt":
                return nn.Identity()
            raise
        if not isinstance(norm, nn.Module):
            raise TypeError("Final normalization is not a torch module")
        return norm

    def input_embeddings(self) -> nn.Module:
        embeddings = self.model.get_input_embeddings()
        if embeddings is None:
            raise ValueError("model.get_input_embeddings() returned None")
        return embeddings

    def output_embeddings(self) -> nn.Module:
        embeddings = self.model.get_output_embeddings()
        if embeddings is None:
            embeddings = getattr(self.model, "lm_head", None)
        if embeddings is None:
            raise ValueError("Could not locate the model output embeddings/lm_head")
        return embeddings

    def project(self, hidden):
        """Replay the architecture's post-block readout, including documented logit transforms."""
        from ..probes.logit_lens import module_device
        norm = self.final_norm()
        device = module_device(norm) if any(True for _ in norm.parameters()) else hidden.device
        hidden = norm(hidden.to(device))
        decoder = getattr(getattr(self.model, "model", None), "decoder", None)
        projection = getattr(decoder, "project_out", None)
        if projection is not None:
            hidden = projection(hidden.to(module_device(projection)))
        head = self.output_embeddings()
        logits = head(hidden.to(module_device(head)))
        config = getattr(self.model, "config", None)
        cap = getattr(config, "final_logit_softcapping", None)
        if cap is not None:
            logits = (logits / cap).tanh() * cap
        divisor = getattr(config, "logits_scaling", None)
        if divisor is not None:
            logits = logits / divisor
        scale = getattr(config, "logit_scale", None)
        if scale is not None:
            logits = logits * scale
        return logits

    def router_modules(self) -> Iterable[RouterModule]:
        layer_modules = self.layers()
        layer_ids = {id(module): index for index, module in enumerate(layer_modules)}
        seen: set[int] = set()

        for name, module in self.model.named_modules():
            if id(module) in seen:
                continue
            final_segment = name.rsplit(".", 1)[-1]
            if final_segment not in ROUTER_SEGMENTS:
                continue
            if not isinstance(module, nn.Module) or not any(True for _ in module.parameters()):
                continue

            layer_index = self._layer_index(name, module, layer_ids)
            if layer_index is None:
                continue
            seen.add(id(module))
            yield RouterModule(layer=layer_index, name=name, module=module)

    def _layer_index(
        self, name: str, module: nn.Module, layer_ids: dict[int, int]
    ) -> int | None:
        match = LAYER_INDEX_RE.search(name)
        if match:
            return int(match.group(1))

        for parent_name, parent in self.model.named_modules():
            if id(parent) in layer_ids and (name == parent_name or name.startswith(parent_name + ".")):
                return layer_ids[id(parent)]
        return None
