from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_demo.py ANALYSIS_JSON COMPARISON_JSON")
    analysis = load(sys.argv[1])
    comparison = load(sys.argv[2])
    metadata = analysis["metadata"]
    print("=== CLOUD DEMO ===")
    print(f"model: {metadata['model']}")
    print(f"model_type: {metadata['model_type']}")
    print(f"layers: {metadata['num_layers']}")
    print(f"parameters: {metadata['num_parameters']}")

    for sample in analysis["samples"]:
        validation = sample["final_logit_lens_validation"]
        print(f"\n[sample] {sample['id']}: {sample['prompt']}")
        print(
            "final_logit_validation: "
            f"max_abs={validation['max_abs']:.8g}, "
            f"mean_abs={validation['mean_abs']:.8g}, "
            f"cosine={validation['cosine']:.8f}"
        )
        for prediction in sample["layer_predictions"]:
            top = ", ".join(
                f"{token['text']!r}:{token['probability']:.6f}"
                for token in prediction["top_tokens"][:3]
            )
            print(
                f"layer={prediction['layer']:02d} position={prediction['position']} "
                f"entropy={prediction['entropy']:.5f} top3=[{top}]"
            )
        first_similarity = sample["token_similarity"][0]
        neighbors = ", ".join(
            f"{item['text']!r}:{item['similarity']:.5f}"
            for item in first_similarity["neighbors"][:3]
        )
        print(f"neighbors({first_similarity['text']!r})=[{neighbors}]")

    print("\n=== IDENTICAL CHECKPOINT COMPARISON ===")
    for key, value in comparison["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

