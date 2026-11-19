"""HTTP warmup/evaluation use the admitted SDK without changing service startup."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_b1_r4 import http_client_context


def test_client_context_keeps_the_service_command_and_final_sdk_runtime(tmp_path, monkeypatch):
    import engine_entry

    calls = []
    def environment(*args):
        calls.append(args)
        return dict(VLLMHUST_ENGINE_BOOTSTRAPPED=str(tmp_path),
                    VLLMHUST_ENGINE_LOCKS="owned-lease", PYTHONPATH="/verified/sdk")
    monkeypatch.setattr(engine_entry, "environment", environment)
    profile = dict(python="/verified/python")
    original = [sys.executable, "/tools/baseline_entry.py", "--phase", "serve",
                "--baseline-deployment", "/frozen/018", "--kind", "continuous"]
    child, env = http_client_context(original, tmp_path, profile, tmp_path / "service",
                                     tmp_path / "cache", "0,1")
    child[child.index("--phase") + 1] = "warmup"
    assert original[original.index("--phase") + 1] == "serve"
    assert child[0] == profile["python"] and child[1:] == [
        "/tools/baseline_entry.py", "--phase", "warmup", "--baseline-deployment", "/frozen/018", "--kind", "continuous"]
    assert env["VLLMHUST_ENGINE_LOCKS"] == "owned-lease"
    assert env["VLLM_HUST_TEST_OUTPUT"] == str(tmp_path / "service")
    assert calls == [(tmp_path, profile, tmp_path / "service", tmp_path / "cache", "0,1")]
