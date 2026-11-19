#!/usr/bin/env python3
"""Exercise public source/config/download routes against Topic4 V2 budgets."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
from evaluation_contract import probe, require, load

ROOT = Path(__file__).resolve().parents[1]


def verify(root, bootstrap=None):
    document = json.loads((root / 'protocols/compute-capture-profiles-v2.json').read_bytes())
    expected = dict(max_tokens=256, formal_repetitions=3, ignore_eos=False,
                    requests_per_repetition=200)
    require(document['default'] == 'submission-v2' and document['profiles']['submission-v2'] == expected,
            'Topic4 V2 fixed200/out256/three-repeat contract changed')
    module = load(root / 'scripts/reference_capture_profile.py', 'checked_compute_capture')
    require(module.profile() == dict(id='submission-v2', **expected), 'Runtime capture policy differs')
    rows = []
    with tempfile.TemporaryDirectory(prefix='compute-capture-contract-') as folder:
        folder = Path(folder)
        for mode in ('full', 'quick'):
            config = dict(engine_profile='r8', test_mode=mode, model='model', deployment='sdk',
                          physical_devices=[0, 1], out='run')
            path = folder / 'evaluation.json'
            path.write_text(json.dumps(config))
            for command in ('prepare', 'run'):
                args = probe(root, [command, '--config', str(path)], bootstrap)
                require(args.reference_profile == 'submission-v2', 'Configuration changed the default reference profile')
                require(not args.performance_only, 'Reference was disabled')
                rows.append(dict(command=command, mode=mode, capture=args.reference_profile))
        args = probe(root, ['--engine-profile', 'r8', '--model', str(folder / 'model'),
                           '--out', str(folder / 'run')], bootstrap)
        require(args.reference_profile == 'submission-v2', 'Direct CLI reference profile differs')
        rows.append(dict(command='direct', capture=args.reference_profile))
    return dict(passed=True, cases=rows, capture=expected, model_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.bootstrap)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPUTE_CAPTURE_CONTRACT_OK: fixed200/out256/three repeats; five public routes')


if __name__ == '__main__':
    main()
