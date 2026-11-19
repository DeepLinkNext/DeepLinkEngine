"""Validate complete fixed data and rebuild actual evaluator plans with networking disabled."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from dataset_bundle import prepare


def verify(cache, archives=None):
    assets = prepare(cache, ['mmlu', 'hle', 'frontier', 'swe', 'terminal'], archives=archives, context=True)
    metadata = assets / 'mmlu/inputs/model-metadata'
    from dataset_contract import prepare as prepare_plan, client
    from hle_scope import verify_selected
    result = {}
    # After fetching repository archives, every actual original loader must work offline.
    with tempfile.TemporaryDirectory(prefix='dataset-roundtrip-', dir=Path(cache)) as temporary:
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected upstream network access')):
            for name, count in [('hle', 200), ('frontier', 160), ('swe', 64), ('terminal', 48)]:
                source = assets / ('public-agent-traces' if name in ('swe', 'terminal') else name)
                target = Path(temporary) / name
                prepared = prepare_plan(name, source, target, metadata)
                module = client(name)
                contract, tasks = module.load_plan(target)
                if name == 'hle':
                    verify_selected(target, contract, tasks)
                if len(tasks) != count:
                    raise ValueError('Frozen request population changed: ' + name)
                result[name] = dict(requests=len(tasks), network_requests=0,
                    contract_sha256=prepared['contract_sha256'])
            # Re-read cached data under the same network ban.
            prepare(cache, ['mmlu', 'hle', 'frontier', 'swe', 'terminal'], archives=archives, context=True)
    mmlu = json.loads((ROOT / 'datasets/mmlu-fixed200.json').read_bytes())
    for name, expected in mmlu['frozen_files_sha256'].items():
        if name.startswith('inputs/'):
            from dataset_bundle import digest
            if digest(assets / 'mmlu' / name) != expected:
                raise ValueError('MMLU original differs: ' + name)
    result['mmlu'] = dict(requests=200, original_input_files_verified=True)
    print(json.dumps(dict(passed=True, datasets=result), ensure_ascii=False, indent=2))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--archives', type=Path)
    a = parser.parse_args()
    verify(a.cache, a.archives)
