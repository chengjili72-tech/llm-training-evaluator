from __future__ import annotations

import hashlib
import json
import warnings
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import GenericDecoderAdapter
from .config import AnalysisConfig, ModelConfig
from .probes import LogitLensProbe, MoeRoutingProbe, TokenSimilarityProbe
from .probes.logit_lens import module_device
from .runtime import resolve_runtime
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
        self.runtime_plan = resolve_runtime(model_config)

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
            if input_device.type == "meta":
                input_device = torch.device(self.runtime_plan.execution_device)
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
            "runtime": self.runtime_plan.to_dict(),
            "hf_device_map": {
                key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()
            },
            "accelerator_memory": self._accelerator_memory(),
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

        model_kwargs: dict[str, Any] = dict(common)
        model_kwargs["low_cpu_mem_usage"] = True
        if self.model_config.dtype != "auto":
            model_kwargs["torch_dtype"] = DTYPES[self.model_config.dtype]
        else:
            model_kwargs["torch_dtype"] = "auto"
        plan = self.runtime_plan
        if plan.device_map is not None:
            model_kwargs["device_map"] = plan.device_map
        if plan.max_memory is not None:
            model_kwargs["max_memory"] = plan.max_memory
        if plan.offload_folder is not None and plan.device_map is not None:
            Path(plan.offload_folder).mkdir(parents=True, exist_ok=True)
            model_kwargs.update(
                offload_folder=plan.offload_folder,
                offload_state_dict=True,
                offload_buffers=True,
            )
        if plan.quantization in {"4bit", "8bit"}:
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16
            if plan.backend == "cuda" and not torch.cuda.is_bf16_supported():
                compute_dtype = torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=plan.quantization == "4bit",
                load_in_8bit=plan.quantization == "8bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        for note in plan.notes:
            warnings.warn(note, stacklevel=2)
        model = AutoModelForCausalLM.from_pretrained(self.model_config.path, **model_kwargs)
        if plan.device_map is None and plan.backend in {"cuda", "npu"}:
            model = model.to(plan.execution_device)
        model.eval()
        return tokenizer, model

    def _accelerator_memory(self) -> dict[str, int] | None:
        plan = self.runtime_plan
        if plan.backend == "cuda":
            index = int(plan.execution_device.split(":", 1)[1])
            try:
                return {
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                }
            except RuntimeError:
                return None
        if plan.backend == "npu":
            npu = getattr(torch, "npu", None)
            if npu is not None:
                index = int(plan.execution_device.split(":", 1)[1])
                try:
                    return {
                        "allocated_bytes": int(npu.memory_allocated(index)),
                        "reserved_bytes": int(npu.memory_reserved(index)),
                        "max_allocated_bytes": int(npu.max_memory_allocated(index)),
                    }
                except (AttributeError, RuntimeError):
                    return None
        return None

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
