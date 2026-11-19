"""Plot provenance checks run without importing matplotlib or an accelerator."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataset_contract import sha
import plot_compute_reference as plot


def source(root):
    root.mkdir()
    pair = dict(status='measured', full_model_certificate=True, arms={'fixture': True})
    for name in ('status.json', 'matched-reference-pair.json'):
        (root / name).write_text(json.dumps(pair))
    (root / 'checksums.json').write_text(json.dumps({p.name: sha(p) for p in root.iterdir()}))
    return pair


def test_plot_consumes_the_sealed_pair_and_status(tmp_path, monkeypatch):
    root = tmp_path / 'reference'
    pair = source(root)
    calls = []
    monkeypatch.setattr(plot, 'draw', lambda result, out: calls.append((result, out)))
    monkeypatch.setattr(sys, 'argv', ['plot', '--run', str(root), '--out', str(tmp_path)])
    plot.main()
    assert calls == [(pair, tmp_path)]


@pytest.mark.parametrize('change', ['unsealed', 'changed', 'inconsistent'])
def test_plot_rejects_changed_or_inconsistent_inputs(tmp_path, monkeypatch, change):
    root = tmp_path / 'reference'
    source(root)
    seal = json.loads((root / 'checksums.json').read_bytes())
    if change == 'unsealed':
        seal.pop('matched-reference-pair.json')
    else:
        path = root / 'matched-reference-pair.json'
        path.write_text(json.dumps(dict(status='measured', arms={})))
        if change == 'inconsistent':
            seal['matched-reference-pair.json'] = sha(path)
    (root / 'checksums.json').write_text(json.dumps(seal))
    monkeypatch.setattr(plot, 'draw', lambda *args: pytest.fail('Invalid data must not reach the plotter'))
    monkeypatch.setattr(sys, 'argv', ['plot', '--run', str(root), '--out', str(tmp_path)])
    with pytest.raises(ValueError):
        plot.main()
