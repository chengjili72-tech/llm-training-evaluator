from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


STYLE = """
:root {
  color-scheme: light;
  --bg: #f4f7fb;
  --surface: rgba(255,255,255,.94);
  --surface-2: #f7f9fc;
  --surface-3: #eef3f8;
  --text: #152235;
  --muted: #64748b;
  --border: #dce4ee;
  --primary: #3157d5;
  --primary-soft: #e9efff;
  --teal: #0f8b8d;
  --teal-soft: #e2f7f4;
  --good: #16835c;
  --good-soft: #e6f7ef;
  --warn: #b36a05;
  --warn-soft: #fff3d8;
  --bad: #c33b4a;
  --bad-soft: #ffeaed;
  --shadow: 0 14px 42px rgba(31, 51, 81, .08);
}
body.dark {
  color-scheme: dark;
  --bg: #0d1420;
  --surface: rgba(22,31,45,.96);
  --surface-2: #192435;
  --surface-3: #223047;
  --text: #e7edf7;
  --muted: #9eabc0;
  --border: #2e3d53;
  --primary: #8da5ff;
  --primary-soft: #26355d;
  --teal: #5fd2cc;
  --teal-soft: #173f42;
  --good: #5bd49e;
  --good-soft: #173e32;
  --warn: #ffc765;
  --warn-soft: #49391b;
  --bad: #ff8792;
  --bad-soft: #4c252c;
  --shadow: 0 16px 46px rgba(0,0,0,.24);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 8% 0%, rgba(49,87,213,.11), transparent 32rem),
    radial-gradient(circle at 94% 8%, rgba(15,139,141,.10), transparent 28rem), var(--bg);
  color: var(--text);
  font: 14px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
button, input { font: inherit; }
.shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; }
.hero { padding: 38px 0 20px; }
.eyebrow { color: var(--primary); font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }
h1 { margin: 8px 0 5px; font-size: clamp(27px, 4vw, 44px); letter-spacing: -.035em; line-height: 1.1; }
.subtitle { max-width: 900px; margin: 0; color: var(--muted); font-size: 15px; }
.hero-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
.status-badge, .badge {
  display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
  padding: 6px 10px; font-size: 12px; font-weight: 750; white-space: nowrap;
}
.status-badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); }
.badge.good, .status-badge.good { background: var(--good-soft); }
.badge.warn, .status-badge.warn { background: var(--warn-soft); }
.badge.bad, .status-badge.bad { background: var(--bad-soft); }
.badge.neutral { color: var(--primary); background: var(--primary-soft); }
.toolbar {
  position: sticky; top: 0; z-index: 20; display: flex; gap: 10px; align-items: center;
  padding: 10px 0; background: color-mix(in srgb, var(--bg) 86%, transparent);
  backdrop-filter: blur(14px);
}
.search { flex: 1; min-width: 180px; border: 1px solid var(--border); border-radius: 11px;
  padding: 10px 13px; color: var(--text); background: var(--surface); outline: none; }
.search:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); }
.button { border: 1px solid var(--border); border-radius: 11px; padding: 9px 12px; cursor: pointer;
  color: var(--text); background: var(--surface); }
.button:hover { border-color: var(--primary); }
main { padding: 5px 0 60px; }
.section-title { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin: 24px 0 10px; }
.section-title h2 { margin: 0; font-size: 20px; letter-spacing: -.015em; }
.section-title p { margin: 0; color: var(--muted); }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
.metric, .panel, .sample-card {
  border: 1px solid var(--border); background: var(--surface); box-shadow: var(--shadow);
  border-radius: 15px;
}
.metric { position: relative; overflow: hidden; padding: 16px 17px; min-height: 113px; }
.metric::after { content: ""; position: absolute; inset: auto -30px -44px auto; width: 100px; height: 100px;
  border-radius: 50%; background: var(--metric-soft, var(--primary-soft)); opacity: .85; }
.metric-label { color: var(--muted); font-size: 12px; font-weight: 700; }
.metric-value { margin: 5px 0 2px; font-size: 25px; font-weight: 800; letter-spacing: -.025em; }
.metric-hint { color: var(--muted); font-size: 11px; }
.metric.good { --metric-soft: var(--good-soft); } .metric.warn { --metric-soft: var(--warn-soft); }
.metric.bad { --metric-soft: var(--bad-soft); }
.panel { padding: 18px; margin: 12px 0; }
.panel h3 { margin: 0 0 4px; font-size: 16px; }
.panel-subtitle { color: var(--muted); margin: 0 0 14px; font-size: 12px; }
.sample-card { margin: 16px 0; overflow: hidden; scroll-margin-top: 72px; }
.sample-head { padding: 18px 20px; display: flex; justify-content: space-between; gap: 18px;
  border-bottom: 1px solid var(--border); background: linear-gradient(115deg, var(--surface), var(--surface-2)); }
.sample-head h2 { margin: 0 0 3px; font-size: 18px; }
.sample-index { color: var(--primary); font-size: 11px; font-weight: 800; letter-spacing: .08em; }
.prompt { margin: 0; max-width: 980px; color: var(--text); white-space: pre-wrap; word-break: break-word; }
.sample-body { padding: 18px 20px 22px; }
.charts { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.chart-card { border: 1px solid var(--border); border-radius: 13px; padding: 14px; background: var(--surface-2); }
.chart-title { font-weight: 750; }
.chart-note { color: var(--muted); font-size: 11px; margin-bottom: 8px; }
.chart-svg { width: 100%; height: auto; overflow: visible; }
.chart-grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 5; }
.chart-line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.chart-dot { stroke: var(--surface); stroke-width: 2; }
.chart-axis { fill: var(--muted); font-size: 10px; }
.table-wrap { width: 100%; overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px; }
th { position: sticky; top: 0; z-index: 2; color: var(--muted); background: var(--surface-2);
  text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
th, td { padding: 10px 11px; border-bottom: 1px solid var(--border); vertical-align: middle; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; } tbody tr:hover td { background: var(--surface-2); }
.layer-cell { font-weight: 800; color: var(--primary); }
.token { display: inline-flex; align-items: center; gap: 5px; border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 7px; margin: 2px 3px 2px 0; background: var(--surface); }
.token b { color: var(--primary); font-size: 10px; }
.token small { color: var(--muted); }
.top-token { font-weight: 750; padding: 3px 7px; border-radius: 7px; background: var(--primary-soft); }
.bar { position: relative; width: 115px; height: 8px; border-radius: 99px; overflow: hidden; background: var(--surface-3); }
.bar > span { display: block; height: 100%; min-width: 2px; border-radius: inherit;
  background: linear-gradient(90deg, var(--primary), var(--teal)); }
.bar.bad > span { background: linear-gradient(90deg, #ed7c87, var(--bad)); }
.bar.good > span { background: linear-gradient(90deg, var(--teal), var(--good)); }
.bar-label { margin-top: 3px; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 10px; }
details.block { border-top: 1px solid var(--border); margin-top: 16px; }
details.block > summary { cursor: pointer; list-style: none; padding: 14px 2px 4px; font-weight: 750; }
details.block > summary::-webkit-details-marker { display:none; }
details.block > summary::after { content: "+"; float: right; color: var(--primary); font-size: 18px; }
details.block[open] > summary::after { content: "−"; }
.similarity-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 9px; margin-top: 10px; }
.similarity-row { border: 1px solid var(--border); background: var(--surface-2); border-radius: 10px; padding: 10px; }
.query-token { display: block; margin-bottom: 6px; font-weight: 800; color: var(--teal); }
.expert-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 10px; }
.expert-card { border: 1px solid var(--border); border-radius: 11px; padding: 11px; background: var(--surface-2); }
.expert-row { display: grid; grid-template-columns: 52px 1fr 30px; align-items: center; gap: 7px; margin: 6px 0; font-size: 11px; }
.strip { display: flex; gap: 4px; flex-wrap: wrap; }
.layer-dot { min-width: 27px; height: 27px; display: inline-grid; place-items: center; border-radius: 7px;
  font-size: 10px; font-weight: 800; background: var(--good-soft); color: var(--good); }
.layer-dot.bad { background: var(--bad-soft); color: var(--bad); }
.hotspot td:first-child { border-left: 3px solid var(--bad); }
.meta-json { padding: 13px; max-height: 340px; overflow: auto; border-radius: 10px;
  color: var(--muted); background: var(--surface-2); white-space: pre-wrap; word-break: break-word; font: 11px/1.55 ui-monospace, monospace; }
.empty { padding: 24px; color: var(--muted); text-align: center; border: 1px dashed var(--border); border-radius: 12px; }
.hidden-by-filter { display: none; }
footer { padding: 18px 0 36px; color: var(--muted); text-align: center; font-size: 11px; }
@media (max-width: 980px) { .metrics { grid-template-columns: repeat(2,minmax(0,1fr)); } .charts { grid-template-columns: 1fr; } }
@media (max-width: 650px) { .shell { width: min(100% - 20px,1480px); } .hero-row { display:block; }
  .hero-row .status-badge { margin-top: 14px; } .metrics { grid-template-columns: 1fr; }
  .sample-head { display:block; } .similarity-grid { grid-template-columns: 1fr; } .sample-body { padding: 14px; } }
@media print { body { background:white; } .toolbar,.theme-button { display:none; } .sample-card,.panel,.metric { box-shadow:none; break-inside:avoid; } }
"""


SCRIPT = """
const search = document.getElementById('sample-search');
if (search) {
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    document.querySelectorAll('.sample-card').forEach(card => {
      card.classList.toggle('hidden-by-filter', q && !card.dataset.filter.includes(q));
    });
  });
}
const themeButton = document.getElementById('theme-button');
if (themeButton) {
  const preferred = localStorage.getItem('llm-eval-theme');
  if (preferred === 'dark') document.body.classList.add('dark');
  themeButton.addEventListener('click', () => {
    document.body.classList.toggle('dark');
    localStorage.setItem('llm-eval-theme', document.body.classList.contains('dark') ? 'dark' : 'light');
  });
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number != 0 and (abs(number) < 1e-4 or abs(number) >= 1e5):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _metric(label: str, value: str, hint: str, tone: str = "") -> str:
    return (
        f"<div class='metric {_esc(tone)}'><div class='metric-label'>{_esc(label)}</div>"
        f"<div class='metric-value'>{_esc(value)}</div><div class='metric-hint'>{_esc(hint)}</div></div>"
    )


def _tone_lower(value: float, good: float, warn: float) -> str:
    if value <= good:
        return "good"
    if value <= warn:
        return "warn"
    return "bad"


def _tone_higher(value: float, good: float, warn: float) -> str:
    if value >= good:
        return "good"
    if value >= warn:
        return "warn"
    return "bad"


def _bar(value: float, maximum: float = 1.0, tone: str = "") -> str:
    width = max(0.0, min(100.0, 100.0 * value / max(maximum, 1e-12)))
    return (
        f"<div class='bar {_esc(tone)}'><span style='width:{width:.2f}%'></span></div>"
        f"<div class='bar-label'>{_num(value, 6)}</div>"
    )


def _tokens(tokens: list[dict[str, Any]], limit: int | None = None) -> str:
    visible = tokens[:limit] if limit is not None else tokens
    return "".join(
        "<span class='token' title='token_id="
        + _esc(token.get("token_id"))
        + "'><b>#"
        + _esc(token.get("rank", rank))
        + "</b><span>"
        + _esc(token.get("text") or token.get("token"))
        + "</span><small>"
        + _num(token.get("probability", token.get("similarity")), 5)
        + "</small></span>"
        for rank, token in enumerate(visible, start=1)
    )


def _sparkline(
    values: list[float],
    labels: list[str],
    color: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> str:
    if not values:
        return "<div class='empty'>暂无趋势数据</div>"
    width, height = 720, 190
    left, right, top, bottom = 48, 18, 16, 34
    lower = min(values) if minimum is None else minimum
    upper = max(values) if maximum is None else maximum
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.08, 1e-6)
        lower, upper = lower - padding, upper + padding
    plot_width, plot_height = width - left - right, height - top - bottom

    def x_at(index: int) -> float:
        return left + (plot_width / max(len(values) - 1, 1)) * index

    def y_at(value: float) -> float:
        return top + plot_height * (upper - value) / (upper - lower)

    points = " ".join(f"{x_at(i):.2f},{y_at(value):.2f}" for i, value in enumerate(values))
    grid = []
    for step in range(4):
        y = top + plot_height * step / 3
        grid_value = upper - (upper - lower) * step / 3
        grid.append(
            f"<line class='chart-grid' x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}'/>"
            f"<text class='chart-axis' x='{left-5}' y='{y+3:.2f}' text-anchor='end'>{_num(grid_value,3)}</text>"
        )
    label_indices = sorted({0, len(values) // 2, len(values) - 1})
    x_labels = "".join(
        f"<text class='chart-axis' x='{x_at(i):.2f}' y='{height-7}' text-anchor='middle'>{_esc(labels[i])}</text>"
        for i in label_indices
    )
    dots = "".join(
        f"<circle class='chart-dot' cx='{x_at(i):.2f}' cy='{y_at(value):.2f}' r='4' fill='{color}'>"
        f"<title>{_esc(labels[i])}: {_num(value,6)}</title></circle>"
        for i, value in enumerate(values)
    )
    return (
        f"<svg class='chart-svg' viewBox='0 0 {width} {height}' role='img'>"
        + "".join(grid)
        + f"<polyline class='chart-line' stroke='{color}' points='{points}'/>"
        + dots
        + x_labels
        + "</svg>"
    )


def _analysis_summary(payload: dict[str, Any]) -> str:
    samples = payload.get("samples", [])
    predictions = [item for sample in samples for item in sample.get("layer_predictions", [])]
    validations = [
        sample.get("final_logit_lens_validation", {}).get("max_abs", 0.0) for sample in samples
    ]
    max_error = max(validations, default=0.0)
    top1_probabilities = [
        row["top_tokens"][0]["probability"]
        for row in predictions
        if row.get("top_tokens")
    ]
    entropies = [float(row.get("entropy", 0.0)) for row in predictions]
    switches = 0
    for sample in samples:
        by_position: dict[int, list[dict[str, Any]]] = {}
        for row in sample.get("layer_predictions", []):
            by_position.setdefault(int(row["position"]), []).append(row)
        for rows in by_position.values():
            ordered = sorted(rows, key=lambda item: item["layer"])
            ids = [item["top_tokens"][0]["token_id"] for item in ordered if item["top_tokens"]]
            switches += sum(left != right for left, right in zip(ids, ids[1:]))
    warnings = sum(len(sample.get("warnings", [])) for sample in samples)
    return (
        "<div class='metrics'>"
        + _metric(
            "Logit Lens 最大误差",
            _num(max_error, 3),
            "最终层与模型原生 logits 的一致性",
            _tone_lower(max_error, 1e-5, 2e-3),
        )
        + _metric("平均 Top-1 概率", _num(_mean(top1_probabilities), 6), "所有样例与层的平均置信度")
        + _metric("平均分布熵", _num(_mean(entropies), 4), "越低通常表示分布越集中")
        + _metric(
            "层间 Top-1 切换",
            str(switches),
            f"共 {warnings} 条采集警告",
            "good" if warnings == 0 else "warn",
        )
        + "</div>"
    )


def _analysis_table(predictions: list[dict[str, Any]]) -> str:
    rows = []
    max_probability = max(
        (item["top_tokens"][0]["probability"] for item in predictions if item.get("top_tokens")),
        default=1.0,
    )
    for item in predictions:
        if not item.get("top_tokens"):
            continue
        top = item["top_tokens"][0]
        rows.append(
            "<tr>"
            f"<td class='layer-cell'>L{int(item['layer']):02d}</td>"
            f"<td>{item['position']}</td>"
            f"<td><span class='top-token'>{_esc(top.get('text') or top.get('token'))}</span></td>"
            f"<td>{_bar(float(top['probability']), max_probability)}</td>"
            f"<td>{_num(item.get('entropy'),4)}</td>"
            f"<td>{_tokens(item['top_tokens'])}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>层</th><th>位置</th><th>Top-1</th>"
        "<th>概率（相对刻度）</th><th>熵</th><th>完整 Top-K</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _similarities(sample: dict[str, Any]) -> str:
    results = sample.get("token_similarity", [])
    if not results:
        return ""
    rows = []
    for result in results:
        query = result.get("text") or result.get("token")
        rows.append(
            "<div class='similarity-row'><span class='query-token'>查询 Token："
            + _esc(query)
            + "</span>"
            + _tokens(result.get("neighbors", []))
            + "</div>"
        )
    return (
        "<details class='block'><summary>Token Embedding 相似度</summary>"
        "<p class='panel-subtitle'>每个输入 Token 在输入词嵌入矩阵中的余弦近邻。</p>"
        "<div class='similarity-grid'>"
        + "".join(rows)
        + "</div></details>"
    )


def _moe_analysis(sample: dict[str, Any]) -> str:
    results = sample.get("moe_routing", [])
    if not results:
        return ""
    cards = []
    for result in results:
        load = {int(key): int(value) for key, value in result.get("expert_load", {}).items()}
        ordered = sorted(load.items(), key=lambda item: (-item[1], item[0]))
        maximum = max(load.values(), default=1)
        expert_rows = "".join(
            f"<div class='expert-row'><b>E{expert}</b>{_bar(count,maximum,'good')}<span>{count}</span></div>"
            for expert, count in ordered[:12]
        )
        routes = []
        for route in result.get("selected_routes", []):
            route_tokens = [
                {
                    "rank": expert.get("rank"),
                    "text": f"E{expert.get('expert_id')}",
                    "probability": expert.get("probability"),
                    "token_id": expert.get("expert_id"),
                }
                for expert in route.get("experts", [])
            ]
            routes.append(
                f"<div><span class='badge neutral'>P{route.get('position')} {_esc(route.get('token'))}</span>"
                + _tokens(route_tokens)
                + "</div>"
            )
        cards.append(
            "<div class='expert-card'>"
            f"<b>Layer {result.get('layer')}</b> · <span class='good'>路由熵 {_num(result.get('routing_entropy'),4)}</span>"
            f"<div class='panel-subtitle'>{_esc(result.get('module'))}</div>{expert_rows}"
            + "".join(routes)
            + "</div>"
        )
    return (
        "<details class='block' open><summary>MoE 专家路由</summary>"
        "<p class='panel-subtitle'>优先展示专家负载、路由熵和被选专家。仅显示负载最高的前 12 个专家。</p>"
        "<div class='expert-grid'>"
        + "".join(cards)
        + "</div></details>"
    )


def _analysis_sample(sample: dict[str, Any], index: int) -> str:
    predictions = sorted(
        sample.get("layer_predictions", []), key=lambda item: (item["position"], item["layer"])
    )
    labels = [f"L{item['layer']}" for item in predictions]
    probabilities = [float(item["top_tokens"][0]["probability"]) for item in predictions if item.get("top_tokens")]
    entropy = [float(item.get("entropy", 0.0)) for item in predictions]
    tags = "".join(f"<span class='badge neutral'>{_esc(tag)}</span>" for tag in sample.get("tags", []))
    validation = sample.get("final_logit_lens_validation", {})
    max_error = float(validation.get("max_abs", 0.0))
    status = "校验通过" if max_error <= 2e-3 else "校验异常"
    status_tone = _tone_lower(max_error, 1e-5, 2e-3)
    filter_text = " ".join(
        [str(sample.get("id", "")), str(sample.get("prompt", "")), *sample.get("tags", [])]
    ).lower()
    warnings = sample.get("warnings", [])
    warning_html = ""
    if warnings:
        warning_html = (
            "<details class='block' open><summary class='warn'>采集警告</summary><ul>"
            + "".join(f"<li>{_esc(item)}</li>" for item in warnings)
            + "</ul></details>"
        )
    return (
        f"<article class='sample-card' id='sample-{index}' data-filter='{_esc(filter_text)}'>"
        "<div class='sample-head'><div>"
        f"<div class='sample-index'>SAMPLE {index:02d}</div><h2>{_esc(sample.get('id'))}</h2>"
        f"<p class='prompt'>{_esc(sample.get('prompt'))}</p><div>{tags}</div></div>"
        f"<span class='status-badge {status_tone}'>{status}</span></div>"
        "<div class='sample-body'><div class='charts'>"
        "<div class='chart-card'><div class='chart-title'>逐层 Top-1 概率</div>"
        "<div class='chart-note'>观察模型置信度如何随层数演化；悬停节点查看精确值。</div>"
        + _sparkline(probabilities, labels[: len(probabilities)], "#3157d5", minimum=0.0)
        + "</div><div class='chart-card'><div class='chart-title'>逐层分布熵</div>"
        "<div class='chart-note'>分布越集中，熵通常越低；重点关注突变层。</div>"
        + _sparkline(entropy, labels[: len(entropy)], "#0f8b8d")
        + "</div></div>"
        "<div class='section-title'><h2>逐层 Token 轨迹</h2><p>Top-K 完整概率按层展开</p></div>"
        + _analysis_table(predictions)
        + _moe_analysis(sample)
        + _similarities(sample)
        + warning_html
        + "</div></article>"
    )


def _comparison_summary(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    agreement = float(summary.get("top1_agreement", 0.0))
    overlap = float(summary.get("mean_top_k_overlap", 0.0))
    l1 = float(summary.get("mean_top_k_probability_l1", 0.0))
    expert_tv = summary.get("mean_expert_load_total_variation")
    return (
        "<div class='metrics'>"
        + _metric("Top-1 一致率", f"{agreement:.2%}", "最优 Token 是否保持一致", _tone_higher(agreement, .95, .8))
        + _metric("Top-K 平均重合", f"{overlap:.2%}", "候选 Token 集合的一致程度", _tone_higher(overlap, .9, .7))
        + _metric("概率 L1 漂移", _num(l1, 6), "Top-K 联集上的概率差异，越低越好", _tone_lower(l1, .01, .1))
        + _metric(
            "专家负载 TV",
            _num(expert_tv, 5),
            "MoE 专家负载分布偏移，越低越好",
            "" if expert_tv is None else _tone_lower(float(expert_tv), .05, .2),
        )
        + "</div>"
    )


def _hotspots(payload: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        for item in sample.get("layer_comparisons", []):
            rows.append({"sample_id": sample.get("id"), **item})
    rows.sort(key=lambda item: (item.get("top1_agreement", True), -float(item.get("top_k_probability_l1", 0.0))))
    rows = rows[:12]
    if not rows:
        return ""
    body = []
    for item in rows:
        changed = not item.get("top1_agreement", False)
        status = (
            '<span class="badge bad">变化</span>'
            if changed
            else '<span class="badge good">一致</span>'
        )
        body.append(
            f"<tr class='{'hotspot' if changed else ''}'><td>{_esc(item['sample_id'])}</td>"
            f"<td class='layer-cell'>L{int(item['layer']):02d}</td><td>{item['position']}</td>"
            f"<td>{status}</td>"
            f"<td>{_esc(item['model_a_top1'].get('text'))} → {_esc(item['model_b_top1'].get('text'))}</td>"
            f"<td>{_num(item.get('top_k_overlap'),4)}</td><td>{_num(item.get('top_k_probability_l1'),6)}</td></tr>"
        )
    return (
        "<div class='panel'><h3>优先关注：差异热点</h3>"
        "<p class='panel-subtitle'>Top-1 发生变化的层优先，其次按概率 L1 漂移从大到小排序。</p>"
        "<div class='table-wrap'><table><thead><tr><th>样例</th><th>层</th><th>位置</th>"
        "<th>状态</th><th>Top-1 变化</th><th>Top-K 重合</th><th>概率 L1</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div></div>"
    )


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    body = []
    max_l1 = max((float(item.get("top_k_probability_l1", 0.0)) for item in rows), default=1.0)
    for item in rows:
        agreement = bool(item.get("top1_agreement"))
        body.append(
            "<tr>"
            f"<td class='layer-cell'>L{int(item['layer']):02d}</td><td>{item['position']}</td>"
            f"<td><span class='badge {'good' if agreement else 'bad'}'>{'一致' if agreement else '变化'}</span></td>"
            f"<td>{_esc(item['model_a_top1'].get('text'))}</td><td>{_esc(item['model_b_top1'].get('text'))}</td>"
            f"<td>{_bar(float(item.get('top_k_overlap',0.0)),1.0,'good')}</td>"
            f"<td>{_num(item.get('top_k_jaccard'),4)}</td>"
            f"<td>{_bar(float(item.get('top_k_probability_l1',0.0)),max_l1,'bad')}</td>"
            f"<td>{_num(item.get('entropy_delta'),5)}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>层</th><th>位置</th><th>Top-1</th>"
        "<th>模型 A</th><th>模型 B</th><th>Top-K 重合</th><th>Jaccard</th>"
        "<th>概率 L1（相对刻度）</th><th>熵变化 B-A</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _moe_comparison(sample: dict[str, Any]) -> str:
    rows = sample.get("moe_comparisons", [])
    if not rows:
        return ""
    body = "".join(
        f"<tr><td class='layer-cell'>L{int(item['layer']):02d}</td>"
        f"<td>{_num(item.get('selected_expert_overlap'),4)}</td>"
        f"<td>{_num(item.get('expert_load_total_variation'),5)}</td>"
        f"<td>{_num(item.get('routing_entropy_delta'),5)}</td>"
        f"<td>{_esc(item.get('model_a_module'))}</td></tr>"
        for item in rows
    )
    return (
        "<details class='block' open><summary>MoE 路由漂移</summary>"
        "<div class='table-wrap'><table><thead><tr><th>层</th><th>专家选择重合</th>"
        "<th>负载 TV</th><th>路由熵变化</th><th>Router</th></tr></thead><tbody>"
        + body
        + "</tbody></table></div></details>"
    )


def _comparison_sample(sample: dict[str, Any], index: int) -> str:
    rows = sorted(sample.get("layer_comparisons", []), key=lambda item: (item["position"], item["layer"]))
    labels = [f"L{item['layer']}" for item in rows]
    overlap = [float(item.get("top_k_overlap", 0.0)) for item in rows]
    l1 = [float(item.get("top_k_probability_l1", 0.0)) for item in rows]
    dots = "".join(
        f"<span class='layer-dot {'bad' if not item.get('top1_agreement') else ''}' title='Layer {item['layer']}'>"
        f"{item['layer']}</span>" for item in rows
    )
    filter_text = f"{sample.get('id','')} {sample.get('prompt','')}".lower()
    return (
        f"<article class='sample-card' id='sample-{index}' data-filter='{_esc(filter_text)}'>"
        "<div class='sample-head'><div>"
        f"<div class='sample-index'>SAMPLE {index:02d}</div><h2>{_esc(sample.get('id'))}</h2>"
        f"<p class='prompt'>{_esc(sample.get('prompt'))}</p></div><div class='strip'>{dots}</div></div>"
        "<div class='sample-body'><div class='charts'>"
        "<div class='chart-card'><div class='chart-title'>逐层 Top-K 重合率</div>"
        "<div class='chart-note'>越接近 1，两个权重的候选 Token 越一致。</div>"
        + _sparkline(overlap, labels, "#16835c", minimum=0.0, maximum=1.0)
        + "</div><div class='chart-card'><div class='chart-title'>逐层概率 L1 漂移</div>"
        "<div class='chart-note'>峰值对应最值得优先检查的差异层。</div>"
        + _sparkline(l1, labels, "#c33b4a", minimum=0.0)
        + "</div></div>"
        "<div class='section-title'><h2>逐层差异</h2><p>红色标记表示 Top-1 Token 发生变化</p></div>"
        + _comparison_table(rows)
        + _moe_comparison(sample)
        + "</div></article>"
    )


def _metadata(payload: dict[str, Any]) -> str:
    encoded = html.escape(json.dumps(payload.get("metadata", {}), ensure_ascii=False, indent=2))
    return (
        "<details class='panel'><summary><b>运行元数据</b></summary>"
        f"<pre class='meta-json'>{encoded}</pre></details>"
    )


def _page(payload: dict[str, Any], is_comparison: bool) -> str:
    metadata = payload.get("metadata", {})
    samples = payload.get("samples", [])
    title = "双权重逐层对比" if is_comparison else "模型逐层训练效果分析"
    subtitle = (
        "优先定位 Top-1 变化、Top-K 候选漂移、概率差异峰值和 MoE 路由偏移。"
        if is_comparison
        else "从 Logit Lens 一致性开始，观察逐层置信度、Token 演化和 MoE 专家路由。"
    )
    if is_comparison:
        summary = payload.get("summary", {})
        overall = float(summary.get("top1_agreement", 0.0))
        status = "整体稳定" if overall >= .95 else "存在明显漂移"
        tone = _tone_higher(overall, .95, .8)
        overview = _comparison_summary(payload) + _hotspots(payload)
        sections = "".join(_comparison_sample(sample, index) for index, sample in enumerate(samples, 1))
        model_line = f"{metadata.get('model_a','Model A')} ↔ {metadata.get('model_b','Model B')}"
    else:
        errors = [
            float(sample.get("final_logit_lens_validation", {}).get("max_abs", 0.0))
            for sample in samples
        ]
        maximum_error = max(errors, default=0.0)
        status = "Logit Lens 校验通过" if maximum_error <= 2e-3 else "Logit Lens 校验异常"
        tone = _tone_lower(maximum_error, 1e-5, 2e-3)
        overview = _analysis_summary(payload)
        sections = "".join(_analysis_sample(sample, index) for index, sample in enumerate(samples, 1))
        model_line = str(metadata.get("model", "Unknown model"))
    empty = "<div class='empty'>没有可展示的样例。</div>" if not samples else ""
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{STYLE}</style></head><body>"
        "<header class='hero'><div class='shell hero-row'><div>"
        "<div class='eyebrow'>LLM Training Evaluator</div>"
        f"<h1>{_esc(title)}</h1><p class='subtitle'>{_esc(subtitle)}</p>"
        f"<p class='subtitle'><b>{_esc(model_line)}</b> · {_esc(metadata.get('model_type',''))} · "
        f"{_esc(metadata.get('num_layers','?'))} 层 · {len(samples)} 个样例</p></div>"
        f"<span class='status-badge {tone}'>{_esc(status)}</span></div></header>"
        "<div class='shell toolbar'><input id='sample-search' class='search' "
        "placeholder='筛选样例 ID、Prompt 或标签…' aria-label='筛选样例'>"
        "<button id='theme-button' class='button theme-button' type='button'>明暗主题</button>"
        "<button class='button' type='button' onclick='window.print()'>打印 / PDF</button></div>"
        f"<main class='shell'><div class='section-title'><h2>核心指标总览</h2>"
        "<p>按关注优先级从左到右排列</p></div>"
        + overview
        + f"<div class='section-title'><h2>样例明细</h2><p>共 {len(samples)} 个样例</p></div>"
        + sections
        + empty
        + _metadata(payload)
        + "</main><footer class='shell'>由 LLM Training Evaluator 生成 · 自包含离线报告</footer>"
        f"<script>{SCRIPT}</script></body></html>"
    )


def write_html_report(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = payload.get("samples", [])
    is_comparison = bool(samples and "layer_comparisons" in samples[0])
    destination.write_text(_page(payload, is_comparison), encoding="utf-8")
    return destination
