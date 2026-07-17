# LLM Training Evaluator

Inspect how a Hugging Face causal language model's token predictions evolve layer by layer,
find embedding-nearest tokens, inspect MoE expert routing, and compare two trained checkpoints.

## What it measures

- **Layer-wise Top-K predictions:** each decoder block's hidden state is passed through the
  model's final normalization and LM head (Logit Lens), then converted to probabilities.
- **Static token similarity:** cosine-nearest vocabulary tokens from the input embedding matrix.
- **MoE routing:** selected experts, gate weights, routing entropy, and per-sample expert load for
  common `gate` and `router` modules.
- **Checkpoint comparison:** Top-1 agreement, Top-K overlap/Jaccard, probability deltas, entropy
  deltas, optional hidden-state cosine, expert-selection overlap, and expert-load drift.

The evaluator fails fast when the final-layer Logit Lens does not reproduce the model's native
logits. This prevents plausible-looking but incorrectly decoded intermediate predictions.

## Current scope

V1 targets decoder-only `AutoModelForCausalLM` checkpoints in standard Hugging Face format.
Common Llama/Qwen/Mistral/DeepSeek-style module layouts are detected automatically. Models with
custom layer paths, additional output transforms, or nonstandard router outputs need a small
adapter under `src/llm_training_evaluator/adapters/`.

For safety, `trust_remote_code` is disabled by default. Prefer `safetensors` checkpoints and only
enable remote code for model repositories you trust.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Accelerators are detected automatically in this order: Ascend NPU, CUDA, then CPU.
No device flag is required for the common case. Install the optional CUDA low-memory runtime when
the model is larger than the GPU:

```bash
pip install -e ".[cuda-low-memory]"
```

`torch_npu` is intentionally not pinned by this project because its wheel must exactly match the
installed PyTorch and CANN versions. Install it from the Ascend software source for the target
environment before running the same CLI command on NPU.

## Sample input

Samples use JSON Lines. `id` is optional; `prompt` is required.

```json
{"id":"math_001","prompt":"1 + 1 =","target":"2","tags":["math"]}
```

`target` is retained in the result schema. Supervised target probability, rank, NLL, and
perplexity metrics are planned for the next milestone; current V1 focuses on the requested
layer trajectory and checkpoint-difference analysis.

## Analyze one checkpoint

```bash
llm-training-evaluator analyze \
  --model /path/to/hf_checkpoint \
  --samples examples/samples.jsonl \
  --output outputs/model.json \
  --top-k 10
```

This also creates `outputs/model.html`. By default, predictions are measured at the final
non-padding prompt position. Select other positions with:

```bash
--positions 0,10,-1
```

or analyze every prompt position with:

```bash
--position-mode all
```

The latter can produce large reports.

## Automatic GPU/NPU execution

The default `--device auto` behavior is:

1. use `npu:0` when `torch_npu` is installed and an Ascend device is available;
2. otherwise use `cuda:0` when CUDA is available;
3. otherwise run on CPU.

On CUDA devices below 16 GiB, the loader automatically selects nested NF4 4-bit weights when
`bitsandbytes` is installed. It reserves accelerator memory for activations and uses Accelerate to
place remaining modules on CPU or NVMe. Without `bitsandbytes`, it still falls back to native-weight
CPU/disk offload, but that path needs substantially more host RAM and is much slower.

Override the automatic plan only when needed:

```bash
--device cuda:1 \
--quantization 4bit \
--accelerator-memory 6GiB \
--cpu-memory 32GiB \
--offload-dir /fast-nvme/llm-evaluator-offload
```

Every JSON result records the selected backend, execution device, quantization, memory budgets,
offload directory, actual Hugging Face device map, and peak accelerator memory.

## DeepSeek-V2-Lite on one 8 GB GPU

DeepSeek-V2-Lite has about 15.7B total parameters, so its roughly 31 GB BF16 checkpoint cannot fit
on an 8 GB card. The supported low-memory route uses 4-bit CUDA weights plus CPU/NVMe offload:

```bash
pip install -e ".[cuda-low-memory]"

llm-training-evaluator analyze \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --samples examples/deepseek_v2_lite.jsonl \
  --output outputs/deepseek_v2_lite.json \
  --top-k 10 \
  --similarity-top-k 10 \
  --moe-top-k 6 \
  --moe-detail
```

The command needs no GPU-specific flags. For a typical 8 GB card, the automatic plan budgets about
6 GiB for weights and leaves the rest for activations. Host requirements depend on the final device
map; 32 GB RAM and 50-80 GB of free SSD space are practical starting points. The first run also
downloads the original checkpoint.

Four-bit readings are useful for inspecting layer trajectories and routing on constrained hardware,
but quantization changes probabilities slightly. Use BF16 on a 40+ GB accelerator when performing
high-precision checkpoint-to-checkpoint acceptance comparisons.

## Compare two checkpoints

```bash
llm-training-evaluator compare \
  --model-a /path/to/checkpoint_a \
  --model-b /path/to/checkpoint_b \
  --samples examples/samples.jsonl \
  --output-dir outputs/comparison
```

Models must use the same architecture depth and tokenizer vocabulary. Each model is loaded and
evaluated sequentially, so both full checkpoints do not need to fit in accelerator memory at the
same time.

## MoE routing

MoE capture is enabled by default. The number of selected experts is inferred from fields such as
`num_experts_per_tok` or `moe_top_k`; override it when necessary:

```bash
--moe-top-k 6
```

Per-layer output contains routing entropy, expert assignment counts, and routes for the same token
positions used by Logit Lens. Use `--moe-detail` to store routes for every non-padding input token.
If a model's router cannot be decoded generically, the report includes a warning instead of
silently producing incorrect expert IDs.

## Important interpretation notes

- Intermediate probabilities are Logit Lens readings, not native early-exit predictions.
- Static embedding neighbors are not contextual synonyms and tokenizer fragments may look odd.
- Without labeled targets, the tool measures behavior and representation drift; it cannot by
  itself prove that one checkpoint has better task quality.
- Exact full-vocabulary JS divergence and Tuned Lens support are planned extensions.
