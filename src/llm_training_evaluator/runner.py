from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import GenericDecoderAdapter
from .config import AnalysisConfig, ModelConfig
from .probes import LogitLensProbe, MoeRoutingProbe, TokenSimilarityProbe
from .probes.logit_lens import module_device
from .schemas import Sample, to_dict


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class Evaluator:
    def __init__(self, model_config: ModelConfig, analysis_config: AnalysisConfig) -> None:
        analysis_config.validate()
        self.model_config = model_config
        self.analysis_config = analysis_config

    def run(self, samples: list[Sample]) -> dict[str, Any]:
        tokenizer, model = self._load()
        adapter = GenericDecoderAdapter(model)
        layers = adapter.layers()
        router_modules = list(adapter.router_modules()) if self.analysis_config.capture_moe else []
        moe_top_k = self._moe_top_k(model)
        similarity_probe = TokenSimilarityProbe(
            adapter, tokenizer, chunk_size=self.analysis_config.similarity_chunk_size
        )

        sample_results = []
        for sample in samples:
            encoded = tokenizer(sample.prompt, return_tensors="pt")
            input_device = module_device(adapter.input_embeddings())
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(input_device)
            if input_ids.shape[0] != 1:
                raise ValueError("V1 evaluates one sample per forward pass")

            predictions: list[Any] = []
            moe_results: list[Any] = []
            warnings: list[str] = []
            logit_probe = LogitLensProbe(adapter, tokenizer, self.analysis_config)
            moe_probe = MoeRoutingProbe(tokenizer, self.analysis_config, top_k=moe_top_k)

            with ExitStack() as stack:
                for layer_index, layer in enumerate(layers):
                    handle = layer.register_forward_hook(
                        self._layer_hook(
                            adapter,
                            logit_probe,
                            layer_index,
                            len(layers),
                            input_ids,
                            attention_mask,
                            predictions,
                        )
                    )
                    stack.callback(handle.remove)

                for router in router_modules:
                    handle = router.module.register_forward_hook(
                        self._router_hook(
                            moe_probe,
                            router.layer,
                            router.name,
                            input_ids,
                            attention_mask,
                            moe_results,
                            warnings,
                        )
                    )
                    stack.callback(handle.remove)

                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )

            native_logits = output.logits
            validation = logit_probe.validate_final_logits(native_logits)
            similarities = similarity_probe.analyze(
                input_ids, top_k=self.analysis_config.similarity_top_k
            )
            tokens = [
                {
                    "position": position,
                    "token_id": int(token_id),
                    "token": str(tokenizer.convert_ids_to_tokens(int(token_id))),
                    "text": tokenizer.decode(
                        [int(token_id)], clean_up_tokenization_spaces=False
                    ),
                }
                for position, token_id in enumerate(input_ids[0].detach().cpu().tolist())
            ]
            sample_results.append(
                {
                    "id": sample.id,
                    "prompt": sample.prompt,
                    "target": sample.target,
                    "tags": sample.tags,
                    "metadata": sample.metadata,
                    "tokens": tokens,
                    "layer_predictions": [to_dict(item) for item in predictions],
                    "token_similarity": similarities,
                    "moe_routing": [to_dict(item) for item in moe_results],
                    "final_logit_lens_validation": validation,
                    "warnings": warnings,
                }
            )

        config_dict = model.config.to_dict()
        vocabulary = tokenizer.get_vocab()
        metadata = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_config.path,
            "revision": self.model_config.revision,
            "model_type": config_dict.get("model_type"),
            "num_layers": len(layers),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_config_hash": _stable_hash(config_dict),
            "tokenizer_hash": _stable_hash(vocabulary),
            "analysis_config": to_dict(self.analysis_config),
            "router_modules": [
                {"layer": router.layer, "name": router.name} for router in router_modules
            ],
        }
        return {"metadata": metadata, "samples": sample_results}

    def _load(self) -> tuple[Any, Any]:
        common: dict[str, Any] = {
            "revision": self.model_config.revision,
            "trust_remote_code": self.model_config.trust_remote_code,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, **common)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = dict(common)
        if self.model_config.dtype != "auto":
            model_kwargs["torch_dtype"] = DTYPES[self.model_config.dtype]
        else:
            model_kwargs["torch_dtype"] = "auto"
        if self.model_config.device_map is not None:
            model_kwargs["device_map"] = self.model_config.device_map
        model = AutoModelForCausalLM.from_pretrained(self.model_config.path, **model_kwargs)
        model.eval()
        return tokenizer, model

    def _moe_top_k(self, model: Any) -> int:
        if self.analysis_config.moe_top_k is not None:
            return self.analysis_config.moe_top_k
        for name in (
            "num_experts_per_tok",
            "num_selected_experts",
            "num_experts_per_token",
            "moe_top_k",
        ):
            value = getattr(model.config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return 2

    @staticmethod
    def _layer_hook(
        adapter: GenericDecoderAdapter,
        probe: LogitLensProbe,
        layer_index: int,
        layer_count: int,
        input_ids: Tensor,
        attention_mask: Tensor,
        destination: list[Any],
    ):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = adapter.hidden_tensor(output)
            destination.extend(
                probe.analyze(
                    hidden,
                    layer=layer_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    is_last_layer=layer_index == layer_count - 1,
                )
            )

        return hook

    @staticmethod
    def _router_hook(
        probe: MoeRoutingProbe,
        layer: int,
        module_name: str,
        input_ids: Tensor,
        attention_mask: Tensor,
        destination: list[Any],
        warnings: list[str],
    ):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            try:
                result = probe.analyze(
                    output,
                    layer=layer,
                    module_name=module_name,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                warnings.append(f"MoE router {module_name} was not decoded: {exc}")
                return
            if result is not None:
                destination.append(result)

        return hook

