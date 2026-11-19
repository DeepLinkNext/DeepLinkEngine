import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ast

import pytest

from native_observer_removal import transform


WORKER = '''
class Worker:
    def load_model(self):
        self.guard()
        result = self.model()
        self._install_graph_receipts()
        return result
    def _install_graph_receipts(self):
        self.events.append('pure graph receipt installer')
'''


def test_model_and_guard_remain_identical_while_receipt_installation_is_removed():
    cleaned, proof = transform(WORKER, "worker")
    namespace = {}
    exec(cleaned, namespace)
    worker = namespace["Worker"]()
    effects = []
    worker.guard = lambda: effects.append("guard")
    worker.model = lambda: 42
    assert worker.load_model() == 42
    assert effects == ["guard"]
    assert not hasattr(worker, "_install_graph_receipts")
    original = ast.parse(WORKER).body[0].body[0]
    original.body.pop(2)
    actual = ast.parse(cleaned).body[0].body[0]
    assert ast.dump(original) == ast.dump(actual)
    assert proof["only_declared_observers_removed"] is True


def test_changed_graph_install_call_is_rejected():
    with pytest.raises(ValueError, match="Unexpected graph"):
        transform(WORKER.replace("self._install_graph_receipts()", "self._install_graph_receipts(self.functional_state)"), "worker")


def test_missing_or_duplicate_observer_nodes_are_rejected():
    with pytest.raises(ValueError, match="population"):
        transform(WORKER.replace("        self._install_graph_receipts()\n", ""), "worker")
    with pytest.raises(ValueError, match="population"):
        transform(WORKER.replace("        self._install_graph_receipts()", "        self._install_graph_receipts()\n        self._install_graph_receipts()"), "worker")
