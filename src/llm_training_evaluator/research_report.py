"""Offline, escaped scientific report panels for native likelihood and uncertainty."""
from __future__ import annotations

import html
from typing import Any


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def num(value: Any) -> str:
    return '—' if value is None else f'{float(value):.5g}'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return ("<div class='table-wrap'><table><thead><tr>" + ''.join(
        '<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in r) + '</tr>' for r in rows)
        + '</tbody></table></div>')


def evaluation_panel(payload: dict) -> str:
    evaluation = payload.get('evaluation', {})
    if not evaluation.get('token_count'):
        return ("<section class='panel'><h3>目标续写评估</h3><p>未评分：请提供非空 target，"
                "并启用目标评分。无标签的层轨迹不能证明模型质量更好。</p></section>")
    cards = [('条件 PPL', evaluation.get('perplexity'), 'exp(总 NLL / 总目标 token 数)'),
             ('NLL · nats/token', evaluation.get('mean_nll'), 'token 加权；只评分 target'),
             ('下一 token 准确率', evaluation.get('accuracy'), 'teacher forcing，非生成任务正确率'),
             ('ECE · 校准误差', evaluation.get('ece'), '等宽置信度分桶；依赖桶数与样本量')]
    metrics = "<div class='metrics'>" + ''.join(
        "<div class='metric'><div class='metric-label'>" + esc(label)
        + "</div><div class='metric-value'>" + num(value)
        + "</div><div class='metric-hint'>" + esc(hint) + '</div></div>'
        for label, value, hint in cards) + '</div>'
    groups = evaluation.get('by_tag', {})
    group_table = _table(['切片', '样本数', '目标 token 数', 'NLL', 'PPL', '准确率', 'MRR', 'Brier'], [
        [tag, g['sample_count'], g['token_count'], num(g.get('mean_nll')),
         num(g.get('perplexity')), num(g.get('accuracy')), num(g.get('mrr')), num(g.get('brier'))]
        for tag, g in groups.items()])
    bins = evaluation.get('calibration_bins', [])
    dots = ''.join(f"<circle cx='{40+float(b['confidence'])*280:.2f}' "
                   f"cy='{310-float(b['accuracy'])*280:.2f}' r='5' fill='#0f8b8d'>"
                   f"<title>n={b['count']}; confidence={num(b['confidence'])}; "
                   f"accuracy={num(b['accuracy'])}</title></circle>" for b in bins if b['count'])
    diagram = ("<svg viewBox='0 0 350 350' role='img' aria-label='可靠性图：横轴置信度，纵轴准确率' "
               "style='width:min(350px,100%)'><path d='M40 30V310H320' fill='none' stroke='#64748b'/>"
               "<path d='M40 310L320 30' stroke='#94a3b8' stroke-dasharray='4 4'/>"
               "<text x='140' y='342' fill='currentColor'>预测置信度 [0,1]</text>"
               "<text x='40' y='20' fill='currentColor'>准确率 [0,1]</text>" + dots + '</svg>')
    bin_table = _table(['置信区间', 'token 数', '平均置信度', '实际准确率'], [
        [f"[{b['lower']:.3f}, {b['upper']:.3f}{']' if i == len(bins)-1 else ')'}",
         b['count'], num(b['confidence']), num(b['accuracy'])] for i, b in enumerate(bins)])
    worst = sorted((s for s in payload.get('samples', [])
                    if (s.get('target_metrics') or {}).get('status') == 'scored'),
                   key=lambda s: s['target_metrics']['mean_nll'], reverse=True)[:10]
    worst_table = _table(['高 NLL 样本', '目标 token 数', 'NLL', 'PPL'], [
        [s['id'], s['target_metrics']['token_count'], num(s['target_metrics']['mean_nll']),
         num(s['target_metrics']['perplexity'])] for s in worst])
    return ("<section class='panel'><h2>目标续写：原生概率评估</h2><p class='panel-subtitle'>"
            f"{evaluation['sample_count']} 个样本 / {evaluation['token_count']} 个目标 token。"
            "条件似然不是生成准确率；不同 tokenizer 或上下文协议的 PPL 不可直接比较。</p>"
            + metrics + '<h3>按标签切片</h3><p>多标签样本会出现在多个切片中。</p>'
            + group_table + "<h3>可靠性图与分桶明细</h3><p>虚线为理想校准；点的悬停信息显示桶内样本数。"
            "相关 token 不能视为独立观测，少量样本不支持总体校准结论。</p>" + "<div class='charts'><div>" + diagram
            + "</div><div><h3>高损失样本</h3>" + worst_table + "</div></div>"
            + '<details><summary>完整分桶明细（含空桶）</summary>' + bin_table
            + '</details></section>')


def target_panel(sample: dict) -> str:
    metrics = sample.get('target_metrics')
    if not metrics:
        return ''
    if metrics.get('status') != 'scored':
        return "<section class='panel'><h3>目标评分</h3><p>空目标，不计入指标分母。</p></section>"
    token_html = ''.join(
        "<span style='display:inline-block;padding:6px;margin:3px;border-radius:6px;"
        f"background:rgba(195,59,74,{min(.65, .07 + t['nll']/20):.3f})' "
        f"title='NLL={num(t['nll'])}; rank={t['rank']}; p={num(t['probability'])}'>"
        + esc(t['text']) + '</span>' for t in metrics['tokens'])
    rows = [[t['position'], t['text'], num(t['probability']), num(t['nll']),
             t['rank'], num(t['confidence']), '✓' if t['correct'] else '×']
            for t in metrics['tokens']]
    return ("<section class='panel'><h3>目标 token 损失热图</h3><p>目标：" + esc(sample.get('target'))
            + f"</p><p>NLL={num(metrics['mean_nll'])} · PPL={num(metrics['perplexity'])} · "
            + f"窗口={metrics['context_window']} · 步长={metrics['stride']}</p>"
            + '<p>边界协议：' + esc(metrics['boundary_policy'])
            + '</p><p>颜色越深，token 损失越大（NLL=11.6 后饱和）。悬停查看数值。</p>'
            + token_html + "<details><summary>逐 token 概率、排名和损失</summary>"
            + _table(['位置', '目标 token', '概率', 'NLL', '排名', 'Top-1 置信度', '预测正确'], rows)
            + '</details></section>')


def comparison_panel(payload: dict) -> str:
    result = payload.get('target_evaluation', {})
    paired = result.get('paired_nll')
    if not paired:
        return ''
    rows = []
    for tag, item in [('全部', paired), *result.get('by_tag', {}).items()]:
        ci = item['ci95']
        rows.append([tag, item['sample_count'], num(item['estimate']),
                     '不估计（不足两个样本）' if ci is None else f'[{num(ci[0])}, {num(ci[1])}]'])
    return ("<section class='panel'><h2>目标似然差异与配对 Bootstrap</h2><p>ΔNLL=B−A，"
            "负值表示 B 在这些目标上的条件似然更高。95% 百分位区间按整个样本配对重采样，"
            "不把相关 token 当作独立观测。默认 2000 次，seed=0。切片未做多重比较校正。</p>"
            + _table(['切片', '配对样本数', 'ΔNLL · nats/token', '95% CI'], rows)
            + '<p>区间只描述当前样本集合的采样不确定性；不覆盖训练随机性、数据污染或分布偏移。</p></section>')


def representation_panel(payload: dict) -> str:
    rows = payload.get('representation_similarity', [])
    if not rows:
        return ''
    return ("<section class='panel'><h2>表征相似性 · Linear CKA</h2><p>中心化线性核 CKA，"
            "参考 Kornblith et al., ICML 2019；对特征正交旋转和各向同性缩放不变。"
            "每层最多取 512 个均匀间隔的配对观测；有偏估计，少样本会高估相似性。"
            "至少四个观测，常量表征不估计。它不是模型质量或因果等价的证明。</p>"
            + _table(['层', '配对观测数', '使用观测数', 'CKA', '说明'], [
                [r['layer'], r['observation_count'], r['used_observations'], num(r['value']),
                 r.get('warning', '')] for r in rows]) + '</section>')
