"""Observe live MTP ownership and preserve the original callable/result exactly."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit/proc_capture"))
from m01_draft_observer import DraftObservation


class Tensor:
    def __init__(self, values, device="npu"):
        self.values = values
        self.device = SimpleNamespace(type=device)
        self.ndim = 2 if values and isinstance(values[0], list) else 1

    def __getitem__(self, index):
        return Tensor(self.values[index], self.device.type)

    def numel(self):
        return sum(map(len, self.values)) if self.ndim == 2 else len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.copy()


def fixture(tmp_path):
    output = Tensor([[31, 32], [41, 42]])
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        return output
    proposer = SimpleNamespace(method="mtp", num_speculative_tokens=2,
        needs_extra_input_slots=False, pass_hidden_states_to_model=True,
        positions=Tensor([7, 8, 9, 19, 0, 0]), input_ids=Tensor([1, 2, 3, 4, 0, 0]),
        _runnable=original)
    proposer._get_positions = lambda n: proposer.positions[:n]
    runner = SimpleNamespace(drafter=proposer, input_batch=SimpleNamespace(req_ids=["request-a", "request-b"]))
    metadata = SimpleNamespace(_mtp_device_seq_lens=Tensor([10, 20]),
        actual_seq_lengths_q=[3, 4, 6], seq_lens_list=[10, 20, 0])
    kwargs = dict(batch_size=2, num_input_tokens=6, num_tokens=4,
        token_indices_to_sample=Tensor([2, 3]), multi_steps_attn_metadata=[{"layer40": metadata}] * 2)
    owner = DraftObservation(runner, tmp_path, 0, "a" * 32, lambda: 17)
    return owner, proposer, original, output, calls, kwargs, metadata


def test_draft_capture_keeps_padding_out_of_request_ownership_and_restores(tmp_path):
    owner, proposer, original, output, calls, kwargs, metadata = fixture(tmp_path)
    assert proposer._runnable(**kwargs) is output
    owner.stop()
    assert proposer._runnable is original and len(calls) == 1
    record = json.loads((tmp_path / "rank-0-drafts.jsonl").read_bytes())
    assert record["wave"] == 17 and record["actual_input_tokens"] == 4 and record["capacity"] == 6
    assert len(record["requests"]) == 2 and record["draft_token_ids"] == [[31, 32], [41, 42]]
    assert record["sample_indices"] == [2, 3]
    assert record["length_planes_before"][0]["actual_kv_lengths"] == [10, 20]
    assert record["timing_eligible_for_performance_comparison"] is False


def test_cpu_tiling_envelope_cannot_substitute_actual_device_lengths(tmp_path):
    owner, proposer, original, output, calls, kwargs, metadata = fixture(tmp_path)
    metadata._mtp_device_seq_lens.device.type = "cpu"
    try:
        with pytest.raises(ValueError, match="device-authoritative"):
            proposer._runnable(**kwargs)
        assert calls == []
    finally:
        owner.stop()
    assert proposer._runnable is original


def test_mrope_accessor_preserves_all_three_equal_text_axes_without_positions_attribute(tmp_path):
    owner, proposer, original, output, calls, kwargs, metadata = fixture(tmp_path)
    values = proposer.positions.values
    del proposer.positions
    proposer._get_positions = lambda n: Tensor([values[:n], values[:n], values[:n]])
    try:
        assert proposer._runnable(**kwargs) is output
    finally:
        owner.stop()
    record = json.loads((tmp_path / "rank-0-drafts.jsonl").read_bytes())
    assert record["first_pass_positions"] == values
    assert record["first_pass_position_planes"] == [values] * 3
    assert record["text_position_axes_equal"] is True
    assert proposer._runnable is original


def test_nonidentical_mrope_planes_are_not_silently_collapsed(tmp_path):
    owner, proposer, original, output, calls, kwargs, metadata = fixture(tmp_path)
    proposer._get_positions = lambda n: Tensor([[0] * n, [1] * n, [0] * n])
    try:
        with pytest.raises(ValueError, match="equal MRoPE"):
            proposer._runnable(**kwargs)
        assert calls == []
    finally:
        owner.stop()
