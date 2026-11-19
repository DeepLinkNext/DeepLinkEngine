"""Production named transport ownership and actual HTTP response identity joins."""

import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third-party-test-kit/proc_capture")]
from m01_http_capture import Control
from reference_observation import configure, response_counts


class Client:
    def __init__(self):
        self.engine_core = self
        self.calls = []
        self.active = False
        self.window = None
        self.unfinished = 0
        self.host = ["owned-warmup-checkpoint"]
        self.completed = []

    async def call_utility_async(self, name, *args):
        self.calls.append((name, args))
        if name == "state_cache":
            if args[0]["kind"] == "snapshot":
                return dict(host=self.host.copy(), pending=[], completed=self.completed.copy(),
                            seats=[dict(owner=None, io_owner=None)], partial_reclaim={})
            if args[0]["kind"] == "drop":
                assert args[0]["key"] == self.host[0]
                self.host.clear()
                self.completed.append(dict(operation="drop-id", kind="drop", cancelled=False, ranks=[0, 1]))
                return "drop-id"
        if name == "hust_m01_core_status":
            return dict(active=self.active, poisoned=False, unfinished_requests=self.unfinished)
        if name == "hust_m01_core_identity":
            return dict(engine_core_mode="proc")
        if name == "hust_m01_core_begin":
            self.window, self.active = args[1], True
            return dict(window=self.window, active=True)
        if name == "hust_m01_core_end":
            self.active = False
            return dict(window=self.window, completed=True, waves=7)
        raise AssertionError(name)

    async def reset_prefix_cache(self):
        assert not self.host
        self.calls.append(("reset_prefix_cache", ()))
        return True

    async def collective_rpc(self, *, method, timeout, args):
        self.calls.append((method, args))
        result = []
        for rank in (0, 1):
            row = dict(rank=rank)
            if method == "hust_m01_status":
                row.update(active=False, poisoned=False, backend_active=False)
            elif method == "hust_m01_start":
                row.update(window=args[2], active=True)
            elif method == "hust_m01_stop":
                row.update(window=args[0], completed=True, waves=7)
            elif method == "hust_compile_status":
                row.update(sealed=True, failure=None)
            result.append(row)
        return result


def test_single_owned_window_and_real_request_drain(tmp_path):
    async def scenario():
        client = Client()
        control = Control(client, tmp_path / "raw")
        with pytest.raises(ValueError, match="No capture"):
            await control.call("end")
        begin = await control.call("begin")
        with pytest.raises(ValueError, match="already"):
            await control.call("begin")
        client.unfinished = 1
        with pytest.raises(ValueError, match="Requests remain"):
            await control.call("end")
        assert client.active and not control.failed
        client.unfinished = 0
        end = await control.call("end")
        assert end["window"] == begin["window"] and end["counts"] == dict(scheduler=7, **{"0": 7, "1": 7})
        assert end["timing_eligible_for_performance_comparison"] is False
        with pytest.raises(ValueError, match="fresh service"):
            await control.call("begin")
        assert [name for name, _ in client.calls].index("hust_runtime_synchronize") < [name for name, _ in client.calls].index("hust_m01_core_begin")
        assert [name for name, _ in client.calls].index("reset_prefix_cache") < [name for name, _ in client.calls].index("hust_runtime_synchronize")
        assert json.loads((tmp_path / "raw/state-cache-drain.json").read_bytes())["completed"]
    asyncio.run(scenario())


def test_observation_only_changes_explicit_extension_and_whitelist(tmp_path):
    command = ["serve", "--worker-extension-cls", "compile_boundary.WorkerExtension"]
    env = dict(PYTHONPATH="/frozen/runtime", VLLM_PLUGINS="ascend,hust_m01_correction")
    configure(command, env, tmp_path)
    assert command[-1] == "m01_proc_worker.M01WorkerExtension"
    assert env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] == "1"
    assert env["VLLMHUST_M01_HTTP_PROFILE"] == "1"
    assert env["VLLM_PLUGINS"] == "ascend,hust_m01_correction,hust_m01_proc_capture"
    with pytest.raises(ValueError, match="foreign"):
        configure(command, env, tmp_path)


def test_profile_flag_reaches_both_workers_in_the_same_owned_window(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLMHUST_M01_HTTP_PROFILE", "1")
    async def scenario():
        client = Client()
        control = Control(client, tmp_path / "raw")
        result = await control.call("begin")
        assert result["profile_requested"] is True
        start, = [args for method, args in client.calls if method == "hust_m01_start"]
        assert start == (str(tmp_path / "raw"), True, result["window"])
        end = await control.call("end")
        assert end["profile_requested"] is True and end["device_trace_validated"] is False
        assert end["timing_eligible_for_performance_comparison"] is False
    asyncio.run(scenario())


def test_incomplete_state_drop_cannot_start_or_reopen_capture(tmp_path):
    class MissingRank(Client):
        async def call_utility_async(self, name, *args):
            result = await super().call_utility_async(name, *args)
            if name == "state_cache" and args[0]["kind"] == "drop":
                self.completed[-1]["ranks"] = [0]
            return result

    async def scenario():
        client = MissingRank()
        control = Control(client, tmp_path / "raw")
        with pytest.raises(ValueError, match="complete matching TP2"):
            await control.call("begin")
        assert not any(name == "hust_m01_core_begin" for name, _ in client.calls)
        assert not client.active and control.failed
        drain = json.loads((tmp_path / "raw/state-cache-drain.json").read_bytes())
        assert not drain["completed"] and not drain["prefix_reset"]
        with pytest.raises(ValueError, match="fresh service"):
            await control.call("begin")
    asyncio.run(scenario())


def response_fixture(root, *, duplicate=False):
    for i in range(200):
        base = root / f"case-{i:03d}"
        base.with_suffix(".json").write_text(json.dumps(dict(success=True,
            usage=dict(prompt_tokens=123 + i, completion_tokens=17))))
        identity = f"chatcmpl-{i if not duplicate else 0}"
        with gzip.open(base.with_suffix(".sse.jsonl.gz"), "wt") as stream:
            for raw in ["data: " + json.dumps(dict(id=identity, choices=[])), "data: [DONE]"]:
                stream.write(json.dumps(dict(raw=raw)) + "\n")


def test_http_counts_use_sse_ids_and_keep_original_population(tmp_path):
    response_fixture(tmp_path)
    counts = response_counts(tmp_path)
    assert len(counts) == 200
    assert counts[hashlib.sha256(b"chatcmpl-4").hexdigest()] == (127, 17)
    (tmp_path / "case-004.json").unlink()
    with pytest.raises(ValueError, match="200 responses"):
        response_counts(tmp_path)


def test_duplicate_response_id_cannot_invent_a_request_join(tmp_path):
    response_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Repeated"):
        response_counts(tmp_path)
