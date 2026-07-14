from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .compare import compare_analyses
from .config import AnalysisConfig, ModelConfig
from .io import load_samples, read_json, write_json
from .report import write_html_report
from .runner import Evaluator


def _positions(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positions must be comma-separated integers") from exc


def _analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    mode = "selected" if args.positions else args.position_mode
    return AnalysisConfig(
        top_k=args.top_k,
        similarity_top_k=args.similarity_top_k,
        position_mode=mode,
        positions=args.positions or [],
        capture_moe=not args.no_moe,
        moe_top_k=args.moe_top_k,
        moe_detail=args.moe_detail,
        include_hidden_states=args.include_hidden_states,
    )


def _model_config(path: str, args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        path=path,
        revision=args.revision,
        dtype=args.dtype,
        device_map=None if args.device_map.lower() == "none" else args.device_map,
        trust_remote_code=args.trust_remote_code,
    )


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--revision")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity-top-k", type=int, default=10)
    parser.add_argument("--position-mode", choices=("last", "all"), default="last")
    parser.add_argument("--positions", type=_positions)
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--moe-top-k", type=int)
    parser.add_argument("--moe-detail", action="store_true")
    parser.add_argument("--include-hidden-states", action="store_true")


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-training-evaluator",
        description="Inspect layer-wise predictions and MoE routing in HF causal LMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one model")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--samples", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--html")
    _add_runtime_options(analyze)

    compare = subparsers.add_parser("compare", help="Run and compare two models")
    compare.add_argument("--model-a", required=True)
    compare.add_argument("--model-b", required=True)
    compare.add_argument("--samples", required=True)
    compare.add_argument("--output-dir", required=True)
    _add_runtime_options(compare)

    compare_results = subparsers.add_parser(
        "compare-results", help="Compare two existing analysis JSON files"
    )
    compare_results.add_argument("--analysis-a", required=True)
    compare_results.add_argument("--analysis-b", required=True)
    compare_results.add_argument("--output", required=True)
    compare_results.add_argument("--html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        samples = load_samples(args.samples)
        result = Evaluator(_model_config(args.model, args), _analysis_config(args)).run(samples)
        write_json(args.output, result)
        html_path = args.html or str(Path(args.output).with_suffix(".html"))
        write_html_report(result, html_path)
        print(f"Analysis JSON: {args.output}")
        print(f"HTML report: {html_path}")
        return 0

    if args.command == "compare":
        samples = load_samples(args.samples)
        config = _analysis_config(args)
        output_dir = Path(args.output_dir)
        model_a = Evaluator(_model_config(args.model_a, args), config).run(samples)
        write_json(output_dir / "model_a.json", model_a)
        _release_accelerator_memory()
        model_b = Evaluator(_model_config(args.model_b, args), config).run(samples)
        write_json(output_dir / "model_b.json", model_b)
        comparison = compare_analyses(model_a, model_b)
        write_json(output_dir / "comparison.json", comparison)
        write_html_report(comparison, output_dir / "comparison.html")
        print(f"Comparison results: {output_dir}")
        return 0

    model_a = read_json(args.analysis_a)
    model_b = read_json(args.analysis_b)
    comparison = compare_analyses(model_a, model_b)
    write_json(args.output, comparison)
    html_path = args.html or str(Path(args.output).with_suffix(".html"))
    write_html_report(comparison, html_path)
    print(f"Comparison JSON: {args.output}")
    print(f"HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
