"""Replay the real admission/guard statements with changing cgroup samples."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def guarded_statements():
    path = Path(__file__).resolve().parents[1] / "scripts/run_native_reference.py"
    tree = ast.parse(path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
                and ast.unparse(node.test) == "process.poll() is None")
    admission = next(node for node in loop.body if isinstance(node, ast.If)
                     and "not admitted_cache" in ast.unparse(node.test))
    memory = next(node for node in loop.body if isinstance(node, ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='memory' for t in node.targets))
    reserve = [node for node in loop.body if isinstance(node,ast.If)
               and "memory['limit']" in ast.unparse(node.test)]
    return compile(ast.Module(body=[memory,admission,*reserve], type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("after_current,working,raises", [(17 * 2**30,16 * 2**30,False),(63 * 2**30,43 * 2**30,False),(63 * 2**30,61 * 2**30,True)])
def test_native_uses_working_reserve_and_does_not_reject_clean_cache(tmp_path, after_current, working, raises):
    (tmp_path / "service.log").write_text("Worker_TP0 pid=1 Loading model weights took\nWorker_TP1 pid=2 Loading model weights took\n")
    state = dict(current=64 * 2**30, limit=64 * 2**30,working_estimate=44 * 2**30)
    trace = []
    def release(*args):
        trace.append("release")
        state["current"] = after_current
        state["working_estimate"] = working
    def sample(*args):
        trace.append("sample")
        if state['working_estimate']>=state['limit']-4*2**30:raise RuntimeError('working-set reserve reached')
        return dict(state)
    import re
    namespace = dict(args=SimpleNamespace(preflight=False), admitted_cache=False,
        root=tmp_path, service=tmp_path, receipt={}, model=None, weights={},
        memory_guard=None, initial=None, memory=dict(state), re=re,
        release_loaded_checkpoint_cache=release, check_memory=sample)
    if raises:
        with pytest.raises(RuntimeError, match="working-set reserve"):
            exec(guarded_statements(), namespace)
    else:
        exec(guarded_statements(), namespace)
        assert namespace["admitted_cache"] is True
        assert namespace["memory"]["current"] == after_current
    assert trace == ["sample", "release", "sample"]
