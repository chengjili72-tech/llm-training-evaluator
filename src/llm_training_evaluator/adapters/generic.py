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
    "gpt_neox.layers",
    "language_model.model.layers",
    "language_model.layers",
    "backbone.layers",
)

NORM_PATHS = (
    "model.norm",
    "model.decoder.final_layer_norm",
    "transformer.ln_f",
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
        layers = self._first_path(LAYER_PATHS, "decoder layers")
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError(f"Located layers object has unsupported type: {type(layers)!r}")
        return list(layers)

    def final_norm(self) -> nn.Module:
        norm = self._first_path(NORM_PATHS, "final normalization")
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
