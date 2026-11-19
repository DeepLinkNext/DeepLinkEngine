"""Check causal boundaries and partial MTP source populations independently."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from semantic_work import candidate, convolution, gdn_preprocess, rms, softmax


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 9])
@pytest.mark.parametrize("initial", [False, True])
def test_convolution_counts_actual_nonzero_history_taps(length, initial):
    multiplications = additions = 0
    for position in range(length):
        taps = [offset for offset in range(4) if initial or position - offset >= 0]
        multiplications += len(taps) * 7
        additions += max(0, len(taps) - 1) * 7
    result = convolution(length, initial=initial, width=4, channels=7)
    assert result["basic"] == dict(multiply=multiplications, add=additions)
    assert result["special_functions"]["silu"] == length * 7


def model():
    return SimpleNamespace(attention_bias=False, attn_output_gate=True, mtp_num_hidden_layers=1,
        hidden_act="silu", num_hidden_layers=40,
        layer_types=["linear_attention"] * 30 + ["full_attention"] * 10,
        rope_parameters=dict(partial_rotary_factor=.25), hidden_size=2048,
        num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        linear_num_value_heads=32, linear_num_key_heads=16, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, moe_intermediate_size=512,
        shared_expert_intermediate_size=512, num_experts_per_tok=8, num_experts=256)


def test_kv_only_draft_ancestors_do_not_acquire_full_layer_scalar_work():
    closure = dict(necessary_kv_prefix_rows=19, necessary_full_rows=2,
                   necessary_vocabulary_rows=1, necessary_full_causal_pairs=20)
    result = candidate(model(), [dict(input_tokens=3, output_tokens=2)], closure)
    parts = result["components"]
    assert parts["draft.prefix_key_norm"] == rms(19 * 2, 256)
    assert parts["draft.full_query_norm"] == rms(2 * 16, 256)
    assert parts["draft.full_attention_gate"]["basic"]["multiply"] == 2 * 4096
    assert parts["draft.full_attention_softmax"] == softmax(2, 20, 16)
    assert parts["target.causal_convolution"]["total_basic"] == 30 * convolution(4, initial=False, width=4, channels=8192)["total_basic"]
    assert result["effective_flops"] is None and not result["full_model_certificate"]


def test_gdn_global_heads_are_not_tp_replicated():
    result = gdn_preprocess(1, 16, 32, 128)
    assert result["total_basic"] == 12384
    assert result["special_functions"]["qk_inverse_sqrt"] == 32
    assert result["comparisons"]["softplus_threshold"] == 32


def test_rms_and_softmax_do_not_assign_special_function_flops():
    assert rms(1, 1)["total_basic"] == 5
    result = softmax(2, 7, 1)
    assert result["total_basic"] == 19
    assert result["special_functions"] == dict(exp=7, divide=7)
    assert result["comparisons"] == dict(maximum=5)


def test_missing_causal_keys_are_rejected():
    with pytest.raises(ValueError, match="causal query"):
        softmax(3, 2, 16)
