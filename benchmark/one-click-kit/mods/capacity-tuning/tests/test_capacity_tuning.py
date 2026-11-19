import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capacity_tuning", ROOT / "capacity_tuning.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def baseline():
    return json.loads((ROOT / "fixtures/baseline-launch.json").read_bytes())


def test_capture_bins_and_baseline_option():
    assert mod.captures(24) == mod.captures(26)
    assert set(mod.captures(32)) - set(mod.captures(24)) == {96}
    for seats in (24, 26, 32):
        source = baseline()
        original = json.loads(json.dumps(source))
        result = mod.configure(source, seats)
        assert source == original
        changes = [
            (a, b) for a, b in zip(source["command"], result["command"]) if a != b
        ]
        assert len(changes) <= 2
        assert result["capacity_tuning"]["quality_qualified"] is False
        assert result["capacity_tuning"]["common_warmup_concurrencies"][-1] == 32
    # Entire launch command remains identical for the all-off/base choice.
    result = mod.configure(baseline(), 24)
    assert result["command"] == baseline()["command"]


@pytest.mark.parametrize("seats", [True, 0, 25, 36])
def test_no_unmeasured_operating_point(seats):
    with pytest.raises(ValueError):
        mod.configure(baseline(), seats)


def test_composed_mtp_guard_and_state_budget():
    for flag, value in [
        ("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'),
        ("--max-num-batched-tokens", "2048"),
        ("--additional-config", "{}"),
    ]:
        config = baseline()
        config["command"][config["command"].index(flag) + 1] = value
        with pytest.raises(ValueError):
            mod.configure(config, 32)


def test_all_repetitions_including_failure_preserved(tmp_path):
    row = dict(id="failed", completed=False, failure="startup")
    obj = dict(
        schema="vllm-hust-capacity-analysis/v2",
        protocol="test",
        planned_trials=4,
        sealed_trials=1,
        rows=[row],
        pairs=[],
    )
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(obj))
    data = mod.load_evidence([path])
    assert data["rows"][0]["failure"] == "startup"
    assert data["sources"][0]["planned_trials"] == 4
    with pytest.raises(ValueError, match="Duplicate"):
        mod.load_evidence([path, path])


def test_native_adapter_requires_warmup_and_preserves_baseline(tmp_path):
    config = ROOT / "execution-24.json"
    b = baseline()
    plan = dict(engine_release=b["engine_release"])
    with pytest.raises(ValueError, match="warmup"):
        mod.apply(b["command"], b["env"], plan, config)
    assert b == baseline()
    plan["serving_warmup"] = dict(
        policy="observed-torch-triton-sealed-task-warmup-v4-capacity32",
        dense_decode_concurrencies=[1, 2, 4, 8, 16, 24, 32],
        formal_clock_excludes_warmup=True,
    )
    mod.apply(b["command"], b["env"], plan, config)
    assert b["command"] == baseline()["command"]
    assert plan["optimization"]["formal_acceptance"] is False
