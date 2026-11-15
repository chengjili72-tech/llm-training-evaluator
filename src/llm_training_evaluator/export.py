"""Tidy CSV companion tables, independent of plotting or spreadsheet packages."""
from __future__ import annotations

import csv
from pathlib import Path


def write_csv_tables(payload: dict, path: str | Path) -> list[Path]:
    base = Path(path).with_suffix('')
    datasets = {}
    tokens, layers, targets = [], [], []
    for sample in payload.get('samples', []):
        score = sample.get('target_metrics') or {}
        for row in score.get('tokens', []):
            tokens.append({'sample_id': sample['id'], 'tags': '|'.join(sample.get('tags', [])), **row})
        for row in sample.get('layer_predictions', []):
            top = row['top_tokens'][0]
            layers.append({'sample_id': sample['id'], 'layer': row['layer'],
                           'position': row['position'], 'entropy': row['entropy'],
                           'adjacent_js_divergence': row.get('adjacent_js_divergence'),
                           'top1_id': top['token_id'], 'top1_probability': top['probability']})
        if sample.get('target_comparison'):
            targets.append({'sample_id': sample['id'], **sample['target_comparison']})
    datasets.update(tokens=tokens, layers=layers, target_deltas=targets)
    paths = []
    for name, rows in datasets.items():
        if not rows:
            continue
        destination = base.with_name(base.name + '.' + name + '.csv')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(destination)
    return paths
