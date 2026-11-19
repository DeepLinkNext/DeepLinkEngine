import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from public_output_binding import validate_effective_contract


def contract():
    return dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
        stream_interval=1, async_scheduling=True, disable_log_stats=True,
        per_request_spec_decode_metrics="none",
        scheduler_cls="betterscale.models.qwen35.seat_scheduler.LiveStateScheduler",
        speculative_config=dict(method="mtp", num_speculative_tokens=2)), dict(
        engine_release="full-answer-b1-20261117-r8", schema="current-native-m01-timing/v1",
        max_tokens=8192, sampling=dict(max_tokens=8192, temperature=0, seed=0,
            ignore_eos=False, top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False))


def test_fixed_native_publication_contract_is_accepted_without_mutation():
    arguments, plan = contract()
    original = copy.deepcopy((arguments, plan))
    validate_effective_contract(arguments, plan)
    assert (arguments, plan) == original


@pytest.mark.parametrize("change", [
    lambda a, p: a.update(stream_interval=2),
    lambda a, p: a.update(stream_interval=True),
    lambda a, p: a.update(async_scheduling=False),
    lambda a, p: a.update(data_parallel_size=2),
    lambda a, p: a.update(scheduler_cls="other.Scheduler"),
    lambda a, p: a["speculative_config"].update(num_speculative_tokens=2.0),
    lambda a, p: p.update(engine_release="full-answer-b1-20261114-r6"),
    lambda a, p: p["sampling"].update(detokenize=True),
    lambda a, p: p["sampling"].update(seed=False),
    lambda a, p: p["sampling"].update(stop=["END"]),
])
def test_coalesced_foreign_or_modified_publication_contract_is_rejected(change):
    arguments, plan = contract()
    change(arguments, plan)
    with pytest.raises(ValueError):
        validate_effective_contract(arguments, plan)
