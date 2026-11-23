import json
from pathlib import Path

import pytest

from benchmark import native_timing, native_timing_contract as contract


def test_native_only_reuses_the_original_timed_session():
    scripts = Path(native_timing.__file__).parent / 'one-click-kit/scripts'
    source = {f'scripts/{name}': (scripts / name).read_bytes() for name in
              ('run_native_reference.py', 'native_timing_prepare.py', 'native_timing_session.py')}
    updates = native_timing.derive(source)
    assert set(updates) == {'scripts/run_native_reference.py', 'scripts/native_timing_prepare.py',
                            'scripts/native_timing_contract.py'}
    assert b'qualify(args.qualified_continuous, deployment)' not in updates['scripts/run_native_reference.py']
    assert b'model_weights(model, args.model_lock)' in updates['scripts/run_native_reference.py']
    assert b'current_http_token_hashes_matched' not in updates['scripts/native_timing_prepare.py']
    assert b'native_timing_session.py' in updates['scripts/native_timing_prepare.py']
    for name, data in updates.items():
        compile(data, name, 'exec')


def test_native_workload_checks_order_and_warmup(tmp_path, monkeypatch):
    prompts = [[i] * (1411 + (i < 65)) for i in range(200)]
    warmup = [[[1]] * width for width in contract.WARMUP_WIDTHS]
    monkeypatch.setattr(contract, 'FORMAL_SHA', contract.sha(contract.compact(prompts)))
    monkeypatch.setattr(contract, 'WARMUP_SHA', contract.sha(contract.compact(warmup)))
    path = tmp_path / 'token-workload.json'
    path.write_text(json.dumps(dict(request_token_ids=prompts, warmup_cohorts=warmup)))
    assert contract.workload(path)[1]['requests'] == 200
    prompts[0], prompts[1] = prompts[1], prompts[0]
    path.write_text(json.dumps(dict(request_token_ids=prompts, warmup_cohorts=warmup)))
    with pytest.raises(ValueError, match='input IDs or synthetic warmup differ'):
        contract.workload(path)


def test_selected_npu_process_column_is_not_memory_usage():
    sample = '| 0       0 | 966515        | VLLMWorker_TP | 57278 | 9999 |\n' \
             '| 3       0 | 965515        | VLLMWorker_TP | 57278 | 9999 |\n' \
             '| 1       0 | 910B2         | 100000         | 57278 | 9999 |'
    assert list(contract.selected_processes(sample, [0, 1])) == [966515]
