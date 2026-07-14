from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


STYLE = """
body{font-family:Inter,system-ui,sans-serif;margin:0;background:#f6f7fb;color:#18202a}
main{max-width:1200px;margin:auto;padding:32px}.card{background:white;border:1px solid #e4e7ec;
border-radius:12px;padding:20px;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:9px;border-bottom:1px solid #edf0f3;text-align:left}th{background:#f8fafc}
code{background:#f1f3f5;padding:2px 5px;border-radius:4px}.good{color:#137333}.bad{color:#b3261e}
details{margin:12px 0}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;word-break:break-word}
"""


def _analysis_table(sample: dict[str, Any]) -> str:
    rows = []
    for item in sample["layer_predictions"]:
        top = item["top_tokens"][0]
        rows.append(
            "<tr>"
            f"<td>{item['layer']}</td><td>{item['position']}</td>"
            f"<td><code>{html.escape(top['text'])}</code></td>"
            f"<td>{top['probability']:.6f}</td><td>{item['entropy']:.4f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Layer</th><th>Position</th><th>Top-1 token</th>"
        "<th>Probability</th><th>Entropy</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _comparison_table(sample: dict[str, Any]) -> str:
    rows = []
    for item in sample["layer_comparisons"]:
        css = "good" if item["top1_agreement"] else "bad"
        rows.append(
            "<tr>"
            f"<td>{item['layer']}</td><td>{item['position']}</td>"
            f"<td class='{css}'>{item['top1_agreement']}</td>"
            f"<td>{html.escape(item['model_a_top1']['text'])}</td>"
            f"<td>{html.escape(item['model_b_top1']['text'])}</td>"
            f"<td>{item['top_k_overlap']:.4f}</td>"
            f"<td>{item['top_k_probability_l1']:.6f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Layer</th><th>Position</th><th>Top-1 same</th>"
        "<th>Model A</th><th>Model B</th><th>Top-K overlap</th><th>Probability L1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def write_html_report(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    is_comparison = bool(payload.get("samples") and "layer_comparisons" in payload["samples"][0])
    sections = []
    for sample in payload.get("samples", []):
        table = _comparison_table(sample) if is_comparison else _analysis_table(sample)
        sections.append(
            "<div class='card'><details open>"
            f"<summary>{html.escape(sample['id'])}</summary>"
            f"<pre>{html.escape(sample.get('prompt', ''))}</pre>{table}</details></div>"
        )
    title = "Model comparison" if is_comparison else "Layer-wise model analysis"
    metadata = html.escape(json.dumps(payload.get("metadata", {}), ensure_ascii=False, indent=2))
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width'><title>{title}</title>"
        f"<style>{STYLE}</style></head><body><main><h1>{title}</h1>"
        f"<div class='card'><h2>Metadata</h2><pre>{metadata}</pre></div>"
        + "".join(sections)
        + "</main></body></html>"
    )
    destination.write_text(document, encoding="utf-8")
    return destination

