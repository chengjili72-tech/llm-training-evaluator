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

