import csv

import pytest

from llm_training_evaluator.export import write_csv_tables
from llm_training_evaluator.io import write_json


def test_csv_retains_unicode_and_embedded_newlines(tmp_path):
    payload = {'samples': [{'id': '中文', 'tags': ['a'], 'target_metrics': {'tokens': [
        {'token_id': 4, 'text': 'a,"\n字', 'nll': .25}]}}]}
    paths = write_csv_tables(payload, tmp_path / 'report.json')
    with paths[0].open(encoding='utf-8-sig', newline='') as f:
        row = next(csv.DictReader(f))
    assert row['sample_id'] == '中文'
    assert row['text'] == 'a,"\n字'


def test_nonfinite_json_does_not_replace_previous_result(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('old')
    with pytest.raises(ValueError):
        write_json(path, {'bad': float('nan')})
    assert path.read_text() == 'old'
