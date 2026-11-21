"""Exercise managed shutdown against real Linux process trees."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from managed_shutdown import stop_managed_service
from owned_processes import Owned

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Requires Linux /proc')

WORKER = '''
from pathlib import Path
import os
import signal
import sys
import time

def stop(*_):
    time.sleep(0.2)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready = Path(sys.argv[1])
pending = ready.with_suffix('.pending')
pending.write_text(str(os.getpid()))
pending.replace(ready)
while True:
    signal.pause()
'''

MANAGER = '''
import signal
import subprocess
import sys

child = None

def stop(*_):
    if sys.argv[1] == 'True':
        child.send_signal(signal.SIGTERM)

signal.signal(signal.SIGTERM, stop)
child = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[3]])
raise SystemExit(child.wait())
'''


@pytest.mark.parametrize('forwards_term', [True, False])
def test_manager_first_preserves_fallback_and_quiescence(tmp_path, forwards_term):
    ready = tmp_path / 'ready'
    owned = Owned(tmp_path)
    env = dict(os.environ, VLLM_HUST_TEST_OUTPUT=str(tmp_path))
    manager = subprocess.Popen([sys.executable, '-c', MANAGER, str(forwards_term), WORKER, str(ready)],
                               env=env, start_new_session=True)
    try:
        owned.add(manager.pid)
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        worker = int(ready.read_text())
        actions = stop_managed_service(owned, manager, grace_seconds=2 if forwards_term else 0.1)
        manager.wait(timeout=2)
        assert actions[0]['pid'] == manager.pid
        assert actions[0]['signal'] == signal.SIGTERM
        assert [row['pid'] for row in actions[0]['managed_children']] == [worker]
        if forwards_term:
            assert len(actions) == 1
        else:
            assert any(row['pid'] == worker and row['signal'] == signal.SIGTERM for row in actions[1:])
        assert not owned.scan()
    finally:
        owned.stop()
        manager.wait(timeout=2)
