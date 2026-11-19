"""Reference timing must not accidentally reuse shared or warmed request prefixes."""

import importlib.util
from pathlib import Path
import sys

import pytest

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"
sys.path.insert(0, str(KIT))
spec = importlib.util.spec_from_file_location(
    "isolation_backend", KIT / "vllm_reference_backend.py"
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_shared_prefixes_are_isolated_without_changing_tokens():
    workload = dict(
        warmup_token_ids=[[1, 2, 3]], request_token_ids=[[1, 2, 3], [1, 2, 4]]
    )
    warm, measured = backend.isolated_inputs(workload, "capture-a")
    assert [x["prompt_token_ids"] for x in measured] == workload["request_token_ids"]
    assert warm[0]["prompt_token_ids"] == workload["warmup_token_ids"][0]
    salts = [x["cache_salt"] for x in warm + measured]
    assert len(set(salts)) == len(salts)
    next_warm, next_measured = backend.isolated_inputs(workload, "capture-b")
    assert set(salts).isdisjoint(x["cache_salt"] for x in next_warm + next_measured)
    # Backend mutations must not rewrite the already-frozen token population.
    measured[0]["prompt_token_ids"].append(9)
    assert workload["request_token_ids"][0] == [1, 2, 3]


@pytest.mark.parametrize("bad", [[], [[]], [[True]], [[-1]], [[1.5]]])
def test_malformed_input_population_is_rejected(bad):
    with pytest.raises(ValueError):
        backend.isolated_inputs(
            dict(warmup_token_ids=[[1]], request_token_ids=bad), "capture"
        )
