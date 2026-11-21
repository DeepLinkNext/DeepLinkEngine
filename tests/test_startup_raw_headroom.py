"""The controller reserve must not consume the driver's raw host headroom."""
import ast
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from startup_memory_recovery import release_startup_headroom

G = 2**30


class Reserve:
    def __init__(self, current, working=40*G):
        self.current, self.working = current, working
        self.buffers = [None] * 256
        self.record = {'released': False}
        self.calls = []

    def check(self):
        self.calls.append('original-check')
        return dict(limit=64*G, current=self.current, working_estimate=self.working,
                    events=dict(oom=0, oom_kill=0))

    def release_chunks(self, count, reason):
        self.calls.append(('release', count, reason))
        del self.buffers[-count:]
        self.current -= count * 64 * 2**20
        self.working -= count * 64 * 2**20
        self.record['released'] = not self.buffers

    def save(self):
        self.calls.append('save')


def test_raw_limit_returns_owned_pages_while_working_set_gate_passes():
    r = Reserve(64*G - 2**20)
    assert r.check()['working_estimate'] < 60*G
    release_startup_headroom(r, r.check)
    assert r.record['released'] and not r.buffers
    event = r.record['raw_headroom_releases'][0]
    assert event['released_bytes'] == 16*G
    assert event['allocated_bytes'] == 0
    assert event['after']['current'] == 48*G - 2**20


def test_releases_only_enough_optional_pages_when_raw_margin_recovers():
    r = Reserve(43*G)
    release_startup_headroom(r, r.check)
    assert len(r.buffers) == 128 and r.current == 35*G
    assert r.record['raw_headroom_releases'][0]['released_bytes'] == 8*G


def test_sufficient_margin_and_already_released_reserve_do_nothing():
    r = Reserve(39*G)
    release_startup_headroom(r, r.check)
    assert len(r.buffers) == 256 and 'raw_headroom_releases' not in r.record
    r.record['released'] = True
    release_startup_headroom(r, lambda: pytest.fail('No remaining optional pages'))


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize('message', ['New cgroup oom event', '4GiB conservative working-set memory reserve reached', 'counter read failed'])
def test_original_guard_errors_are_never_suppressed(when, message):
    r = Reserve(55*G)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == (1 if when == 'before' else 2):
            raise RuntimeError(message)
        return r.check()
    with pytest.raises(RuntimeError, match=message):
        release_startup_headroom(r, check)
    assert len(r.buffers) == (256 if when == 'before' else 128)


def test_generated_controller_changes_only_the_owned_reserve_boundary():
    from test_context_baseline_reuse import combined
    files = combined()
    old = ast.parse((ROOT / 'benchmark/one-click-kit/scripts/run_b1_r4.py').read_text())
    new = ast.parse(files['scripts/run_b1_r4.py'])
    a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'monitor_loaded_memory')
    b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == a.name)
    additions = [n for n in b.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Name) and x.id == 'release_startup_headroom' for x in ast.walk(n))]
    assert len(additions) == 1
    b.body.remove(additions[0])
    assert ast.dump(a) == ast.dump(b)
    assert files['scripts/startup_memory_recovery.py'] == (ROOT/'startup_memory_recovery.py').read_bytes()
