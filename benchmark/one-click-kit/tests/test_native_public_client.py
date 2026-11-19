import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from types import SimpleNamespace as NS

import pytest

from native_public_client import timed_public_requests, validate_frames


class Engine:
    def __init__(self):
        self.events = []
        self.submitted = []
        self.frames = iter([[], [NS(request_id="a", finished=False, num_cached_tokens=0,
                                   outputs=[NS(index=0, token_ids=[10, 11], finish_reason=None, stop_reason=None)])],
                            [NS(request_id="a", finished=True, num_cached_tokens=0,
                                outputs=[NS(index=0, token_ids=[12], finish_reason="stop", stop_reason=12)])]])
        self.live = False

    def has_unfinished_requests(self):
        return self.live

    def collective_rpc(self, name, timeout):
        assert name == "hust_runtime_synchronize"
        self.events.append("fence")
        return [dict(rank=0), dict(rank=1)]

    def add_request(self, name, prompt, sampling):
        self.events.append("add")
        self.live = True
        return name

    def step(self):
        self.events.append("step")
        frame = next(self.frames)
        if frame and frame[0].finished:
            self.live = False
        return frame


def capture():
    engine = Engine()
    times = iter([100, 1000])

    def clock():
        engine.events.append("clock")
        return next(times)

    result = timed_public_requests(engine, [dict(prompt_token_ids=[1, 2])], object(), ["a"], clock=clock)
    assert engine.events == ["fence", "clock", "add", "step", "step", "step", "fence", "clock"]
    return result


def test_public_empty_frames_delta_tokens_and_end_fence_are_preserved():
    result = capture()
    assert result["public_output_frames"][0] == ()
    assert validate_frames(result, [2], 8192) == [dict(request_id="a", input_tokens=2,
                                                    output_ids=[10, 11, 12], finish_reason="stop", stop_reason=12)]
    assert result["T_seconds"] == 900 / 1e9


@pytest.mark.parametrize("change", ["duplicate", "late", "missing", "length", "cached", "unknown"])
def test_invalid_public_population_is_rejected(change):
    result = capture()
    frames = list(result["public_output_frames"])
    final = frames[-1][0]
    if change == "duplicate":
        frames[-1] = (final, final)
    elif change == "late":
        frames.append((final,))
    elif change == "missing":
        frames = frames[:-1]
    elif change == "length":
        frames[-1] = (("a", True, 0, ((0, (12,), "length", None),)),)
    elif change == "cached":
        frames[-1] = (("a", True, 1, final[3]),)
    elif change == "unknown":
        frames[-1] = (("other", True, 0, final[3]),)
    result["public_output_frames"] = frames
    with pytest.raises(ValueError):
        validate_frames(result, [2], 8192)


def test_missing_rank_rejects_before_native_submission():
    engine = Engine()
    engine.collective_rpc = lambda *a, **k: [dict(rank=0)]
    with pytest.raises(ValueError, match="rank fences"):
        timed_public_requests(engine, [{}], object(), ["a"])
    assert engine.events == []
