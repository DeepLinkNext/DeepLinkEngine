"""Cross-model hidden roots merge by actual request position, excluding tails."""

from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from target_state_closure import close, matrix_components
from test_semantic_work import model
from semantic_work import candidate


@pytest.mark.parametrize("seed", range(12))
def test_compressed_intervals_match_literal_hidden_producer_union(seed):
    rng = random.Random(seed)
    requests, spans, expected = [], [], {}
    for name in ("a", "b", "c"):
        prompt, output = rng.randrange(1, 80), rng.randrange(1, 20)
        length = prompt + output - 1
        requests.append(dict(request=name, input_tokens=prompt, output_tokens=output))
        expected[name] = set(range(prompt - 1, length))
        for _ in range(9):
            start = rng.randrange(length)
            stop = rng.randrange(start, length)
            spans.append(dict(request=name, start=start, stop=stop, step=0))
            expected[name].update(range(start, stop + 1))
        # A second-step draft hidden input comes from the first draft model,
        # not from another target hidden tensor at this position.
        spans.append(dict(request=name, start=length + 7, stop=length + 7, step=1))
    result = close(requests, dict(kv_intervals=spans))
    assert result["necessary_last_full_rows"] == sum(map(len, expected.values()))
    assert result["necessary_last_full_causal_pairs"] == sum(p + 1 for values in expected.values() for p in values)
    observed = {name: set() for name in expected}
    for row in result["full_intervals"]:
        observed[row["request"]].update(range(row["start"], row["stop"] + 1))
    assert observed == expected


def test_prompt_kv_only_rows_do_not_acquire_final_query_moe_or_norm():
    request = dict(request="a", input_tokens=10, output_tokens=1)
    draft = dict(kv_intervals=[], necessary_kv_prefix_rows=0, necessary_full_rows=0,
                 necessary_vocabulary_rows=0, necessary_full_causal_pairs=0)
    target = close([request], draft)
    parts = matrix_components(model(), target)
    # Nine complete attention layers for ten rows, final layer for one output.
    assert parts["target_attention_output"] == 2 * 4096 * 2048 * 91
    assert parts["target_active_expert"] == 6 * 2048 * 512 * 8 * 391
    scalar = candidate(model(), [request], draft, target)
    assert scalar["components"]["target.hidden_residual"]["basic"]["residual_add"] == 782 * 2048
    assert scalar["target_all_layer_valid_row_convention"] is False


def test_unobserved_target_hidden_origin_cannot_be_replaced_by_zero():
    with pytest.raises(ValueError, match="target producer"):
        close([dict(request="a", input_tokens=3, output_tokens=1)],
              dict(kv_intervals=[dict(request="a", start=3, stop=3, step=0)]))
