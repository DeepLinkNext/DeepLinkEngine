"""Exercise the complete common warmup/formal protocol with public fake outputs."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_native_entry import reset_prefix_cache
import native_timing_session as session


@pytest.mark.parametrize("role", ["B0", "B1"])
@pytest.mark.parametrize("capture_id,maximum,repetitions", [("full-answer-v1",8192,1), ("submission-v2",256,3)])
def test_same_warmup_sampling_boundary_and_public_clock_for_both_arms(tmp_path, monkeypatch, role, capture_id, maximum, repetitions):
    events = []
    engines = []

    class Sampling:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Engine:
        @classmethod
        def from_engine_args(cls, args):
            engine = cls()
            engine.pending = []
            engine.engine_core = NS(shutdown=lambda: events.append("shutdown"))
            engines.append(engine)
            return engine

        def collective_rpc(self, method, timeout):
            events.append(method)
            return [dict(rank=rank, sealed=method != "hust_compile_prepare", failure=None) for rank in (0, 1)]

        def has_unfinished_requests(self):
            return bool(self.pending)

        def add_request(self, name, prompt, sampling):
            events.append(("request", sampling.max_tokens, sampling.seed, sampling.ignore_eos,
                           sampling.temperature, sampling.detokenize, prompt["cache_salt"]))
            self.pending.append((name, sampling.max_tokens))
            return name

        def step(self):
            pending, self.pending = self.pending, []
            return [NS(request_id=name, finished=True, num_cached_tokens=0,
                       outputs=[NS(index=0, token_ids=[8] * (32 if maximum == 32 else 2),
                                   finish_reason="length" if maximum == 32 else "stop", stop_reason=None)])
                    for name, maximum in pending]

    for name in ("vllm", "vllm.sampling_params", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.llm_engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm"].SamplingParams = Sampling
    sys.modules["vllm.sampling_params"].RequestOutputKind = NS(DELTA="delta")
    sys.modules["vllm.v1.engine.llm_engine"].LLMEngine = Engine
    monkeypatch.setattr(session, "wait_checkpoint_admission", lambda path: events.append("checkpoint-admitted"))
    monkeypatch.setattr(session, "uuid4", lambda: NS(hex=str(len(events))))
    tokens = [[7] * 1476] + [[7] * 1411 for _ in range(199)]
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(dict(request_token_ids=tokens, warmup_cohorts=[[[1], [2]], [[3]]])))
    output = tmp_path / "client"
    output.mkdir()
    config = dict(schema="baseline-native-m01-timing/v1" if role == "B0" else "current-native-m01-timing/v1", role=role, output=str(output), admission_root=str(tmp_path), workload=str(workload),
                  reference_capture_profile=capture_id, formal_repetitions=repetitions, max_tokens=maximum, sampling=dict(max_tokens=maximum, temperature=0, seed=0, ignore_eos=False,
                      top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False))
    session.run(config, object(), lambda engine, path: events.append("reset-cache"))
    requests = [event for event in events if isinstance(event, tuple)]
    assert [event[1:6] for event in requests] == [(32, 17, True, 0, False)] * 3 + [(maximum, 0, False, 0, False)] * (200 * repetitions)
    assert len({event[-1] for event in requests}) == 3 + 200 * repetitions
    seal = events.index("hust_compile_seal")
    formal = next(i for i, event in enumerate(events) if isinstance(event, tuple) and event[1] == maximum)
    assert seal < formal
    assert events[formal - 1] == "hust_runtime_synchronize"
    assert events[-1] == "shutdown"
    result = json.loads((output / "timings.json").read_bytes())
    assert len(result["repetitions"]) == repetitions
    assert all(row["output_tokens"] == 400 for row in result["repetitions"])
    for repetition in range(1, repetitions + 1):
        geometry = json.loads((output / f"formal-{repetition}/public-output-geometry.json").read_bytes())
        assert geometry["maximum_output_tokens"] == maximum
        assert len(geometry["requests"]) == 200
    assert result["efficiency_percent"] is None and not result["necessary_work_certified"]


def test_baseline_reset_requires_idle_and_does_not_call_b1_state_utility(tmp_path):
    engine = NS(has_unfinished_requests=lambda: True, reset_prefix_cache=lambda: True)
    with pytest.raises(ValueError, match="unfinished"):
        reset_prefix_cache(engine, tmp_path / "never.json")
    engine.has_unfinished_requests = lambda: False
    engine.reset_prefix_cache = lambda: False
    with pytest.raises(ValueError, match="reset failed"):
        reset_prefix_cache(engine, tmp_path / "never.json")
    engine.reset_prefix_cache = lambda: True
    reset_prefix_cache(engine, tmp_path / "reset.json")
    assert json.loads((tmp_path / "reset.json").read_bytes())["b1_state_cache_present"] is False
