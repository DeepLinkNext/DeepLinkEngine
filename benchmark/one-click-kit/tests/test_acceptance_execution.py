"""Test shared task identities, failed populations, and non-overlapping throughput."""

import copy
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance_client as client
from acceptance_report import metrics
from agent_trace import save, sha

REAL = (
    Path(__file__).resolve().parents[1]
    / "validation/acceptance-real-inputs-v4/contract.json"
)


def fixture(tmp_path):
    inputs = {}
    for low, high in [(4096, 8192), (16384, 24576), (32768, 49152)]:
        inputs[str(low)] = [
            dict(
                task_id=f"task-{i}",
                source_sha256="a" * 64,
                source_body_sha256="b" * 64,
                input_tokens=low + 100 + i,
                token_ids_sha256="c" * 64,
                request=dict(
                    messages=[dict(role="user", content="original")],
                    max_tokens=256,
                    ignore_eos=True,
                    temperature=0,
                    seed=17,
                    stream=True,
                    stream_options=dict(include_usage=True),
                    tool_choice="none",
                    chat_template_kwargs=dict(enable_thinking=False),
                ),
            )
            for i in range(5)
        ]
    save(tmp_path / "inputs.json", inputs)
    c = dict(
        schema=client.SCHEMA,
        input_file="inputs.json",
        input_sha256=sha(tmp_path / "inputs.json"),
        task_count=5,
        phases=[
            dict(
                id=str(n),
                cohort=str(n),
                concurrency=8,
                tenants=1,
                warmup_seconds=60,
                formal_seconds=300,
            )
            for n in [4096, 16384, 32768]
        ]
        + [
            dict(
                id="multi-tenant",
                cohort="4096",
                concurrency=24,
                tenants=4,
                warmup_seconds=60,
                formal_seconds=300,
            )
        ],
        cohorts=[
            dict(min_tokens=lo, max_tokens_exclusive=hi, concurrency=8)
            for lo, hi in [(4096, 8192), (16384, 24576), (32768, 49152)]
        ],
        output_tokens=256,
        temperature=0,
        seed=17,
        ignore_eos=True,
        automatic_retries=0,
    )
    save(tmp_path / "contract.json", c)
    return c, inputs


def test_five_tasks_are_not_eight_concurrent_lanes(tmp_path):
    fixture(tmp_path)
    contract, inputs = client.load(tmp_path / "contract.json")
    assert contract["task_count"] == 5 and {len(c) for c in inputs.values()} == {5}
    if REAL.exists():
        real, cases = client.load(REAL)
        assert real["task_count"] == 5 and {len(v) for v in cases.values()} == {5}


@pytest.mark.parametrize(
    "change",
    [
        lambda c, i: i["32768"].reverse(),
        lambda c, i: i["16384"].__setitem__(0, copy.deepcopy(i["16384"][1])),
        lambda c, i: i["4096"][0]["request"].__setitem__("max_tokens", 32),
        lambda c, i: c["phases"][3].__setitem__("formal_seconds", 10),
        lambda c, i: c["cohorts"][2].__setitem__("min_tokens", 16384),
    ],
)
def test_no_replacement_short_budget_or_band_drift(tmp_path, change):
    c, inputs = fixture(tmp_path)
    change(c, inputs)
    save(tmp_path / "inputs.json", inputs)
    c["input_sha256"] = sha(tmp_path / "inputs.json")
    save(tmp_path / "contract.json", c)
    with pytest.raises(ValueError):
        client.load(tmp_path / "contract.json")


def test_input_bytes_must_match_even_if_contract_is_well_formed(tmp_path):
    fixture(tmp_path)
    (tmp_path / "inputs.json").write_text("{}")
    with pytest.raises(ValueError):
        client.load(tmp_path / "contract.json")


def test_failed_request_is_retained_without_retry(tmp_path, monkeypatch):
    _, inputs = fixture(tmp_path)
    calls = []

    def fail(config, body, directory):
        calls.append(body)
        return dict(status="transport_error", error="test failure")

    monkeypatch.setattr(client, "call", fail)
    spec = dict(cohort="4096", concurrency=1, tenants=1)
    with pytest.raises(RuntimeError):
        client.phase(
            tmp_path, spec, inputs["4096"], "failed-formal", 60, "http://unused"
        )
    p = json.loads((tmp_path / "failed-formal-phase.json").read_bytes())
    assert p["status"] == "failed" and len(p["request_files"]) == len(calls) == 1
    row = json.loads((tmp_path / p["request_files"][0]).read_bytes())
    assert row["success"] is False and row["error"] == "test failure"


def test_four_tenants_receive_identical_case_mix_and_unique_salts(
    tmp_path, monkeypatch
):
    _, inputs = fixture(tmp_path)
    barrier = threading.Barrier(24)
    captured = []
    stop = threading.Event()

    def fake_monotonic():
        return 100 if stop.is_set() else 0

    def once(config, body, directory):
        captured.append((body["user"], body["cache_salt"], body["messages"]))
        barrier.wait(timeout=5)
        stop.set()
        return dict(
            status="ok",
            finish_reason="length",
            usage=dict(
                prompt_tokens=4096 + 100 + int(body["messages"][0]["content"]),
                completion_tokens=256,
            ),
        )

    monkeypatch.setattr(client, "clock", lambda: dict(monotonic=0, wall=0))
    monkeypatch.setattr(client.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(client, "call", once)
    for i, case in enumerate(inputs["4096"]):
        case["request"]["messages"][0]["content"] = str(i)
    result = client.phase(
        tmp_path,
        dict(cohort="4096", concurrency=24, tenants=4),
        inputs["4096"],
        "multi-tenant-formal",
        60,
        "http://unused",
    )
    assert result["status"] == "completed" and len(captured) == 24
    assert len({salt for _, salt, _ in captured}) == 24
    mixes = [
        [messages for who, _, messages in captured if who == f"tenant-{i}"]
        for i in range(4)
    ]
    assert all(sorted(m, key=str) == sorted(mixes[0], key=str) for m in mixes)


def test_throughput_uses_phase_wall_time_and_retains_failed_denominator():
    rows = [
        dict(
            success=True,
            start=dict(monotonic=0),
            first_token=dict(monotonic=1),
            end=dict(monotonic=10),
            usage=dict(completion_tokens=256),
        )
        for _ in range(2)
    ]
    rows.append(dict(success=False))
    result = metrics(rows, 0, 10)
    assert (
        result["requests"] == 3 and result["failed"] == 1 and result["successful"] == 2
    )
    assert result["output_tokens_per_second"] == 51.2
    assert result["ttft_ms"]["mean"] == 1000
    assert result["tpot_ms"]["mean"] == 9000 / 255


def test_original_linear_percentiles_not_nearest_rank():
    from acceptance_report import percentile

    assert percentile([0, 100], 0.95) == 95
    assert percentile([100, 0], 0.99) == 99
    assert percentile([], 0.99) is None


def test_failed_before_model_is_reported_without_fake_zero_scores(tmp_path):
    from acceptance_report import inspect
    import hashlib

    save(
        tmp_path / "receipt.json",
        dict(
            role="B0",
            quiescent=True,
            completed=False,
            model_service_started=False,
            failure="Invalid contract before loading",
        ),
    )
    save(
        tmp_path / "checksums.json",
        {
            "receipt.json": hashlib.sha256(
                (tmp_path / "receipt.json").read_bytes()
            ).hexdigest()
        },
    )
    result = inspect(tmp_path)
    assert result["execution_complete"] is False and result["phases"] == []
    assert result["failure"] == "Invalid contract before loading"
    assert "output_tokens_per_second" not in str(result)


def suite_args(tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        out=tmp_path / "new-suite",
        contract=tmp_path / "contract.json",
        deployment=tmp_path / "deployment",
        baseline_deployment=tmp_path / "baseline",
        model=tmp_path / "model",
        b1_python=Path("/unused/python"),
        cache=tmp_path / "cache",
        devices="0,1",
        physical_devices="1,6",
        report_python=None,
        reuse_b0=None,
        reuse_b1=None,
    )


def test_orchestrator_keeps_failed_b0_and_never_starts_b1(tmp_path, monkeypatch):
    import acceptance_suite as suite
    from subprocess import CompletedProcess

    fixture(tmp_path)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 2)

    monkeypatch.setattr(suite, "run_stage", fail)
    checkpoints = []
    args = suite_args(tmp_path)
    assert suite.run(args, lambda role, root, stage: checkpoints.append(role)) == 2
    ledger = json.loads((args.out / "execution.json").read_bytes())
    assert ledger["status"] == "failed" and len(ledger["stages"]) == 1
    assert (
        ledger["stages"][0]["role"] == "B0"
        and checkpoints == ["B0"]
        and len(calls) == 1
    )
    assert "no automatic retry" in ledger["failure"]


def test_orchestrator_reuses_b0_and_binds_only_new_b1_to_it(tmp_path, monkeypatch):
    import acceptance_suite as suite
    from subprocess import CompletedProcess

    fixture(tmp_path)
    prior = tmp_path / "released-B0"
    prior.mkdir()
    (prior / "checksums.json").write_text("{}")
    args = suite_args(tmp_path)
    args.reuse_b0 = prior
    monkeypatch.setattr(suite, "admit_reuse", lambda *a: (prior, {}, {}))
    monkeypatch.setattr(suite, "inspect", lambda *a: dict(execution_complete=True))
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 2)

    monkeypatch.setattr(suite, "run_stage", fail)
    assert suite.run(args) == 2
    assert len(calls) == 1 and calls[0][calls[0].index("--role") + 1] == "B1"
    assert calls[0][calls[0].index("--acceptance-baseline") + 1] == str(prior)
    ledger = json.loads((args.out / "execution.json").read_bytes())
    assert (
        ledger["stages"][0]["reused"] is True
        and ledger["stages"][1]["status"] == "failed"
    )


def test_completion_warmup_covers_three_bands_and_requires_both_rank_guards(
    monkeypatch,
):
    import completion_warmup as w

    groups = {
        str(n): [dict(case=i) for i in range(8 if n < 32768 else 5)]
        for n in (4096, 16384, 32768)
    }
    dense = dict(dense=True)
    rows = list(w.warmup_cohorts(groups, dense, acceptance=True))
    assert sum(map(len, rows)) == w.request_count(acceptance=True) == 134
    assert rows[7] == [groups["32768"][i % 5] for i in range(8)]

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return json.dumps(
                dict(
                    results=[
                        dict(
                            rank=0,
                            torch_dynamo_guard=True,
                            triton_guard=True,
                            failure=None,
                        ),
                        dict(
                            rank=1,
                            torch_dynamo_guard=True,
                            triton_guard=False,
                            failure=None,
                        ),
                    ]
                )
            ).encode()

    class Opener:
        def open(self, *args, **kwargs):
            return Reply()

    monkeypatch.setattr(w.urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(ValueError, match="both ranks"):
        w.rpc("hust_compile_status")


def test_console_follows_actual_supplement_leaf_and_open_window_counts(tmp_path):
    from live_console import progress

    leaf = tmp_path / "B1"
    d = leaf / "service/acceptance/32768-formal/request-one"
    d.mkdir(parents=True)
    (d / "request.json").write_text("{}")
    save(leaf / "receipt.json", dict(status="running-client"))
    save(
        tmp_path / "execution.json",
        dict(
            schema="vllm-hust-acceptance-supplement-execution/v1",
            status="running",
            stages=[
                dict(role="B0", run=str(tmp_path / "B0"), status="completed"),
                dict(role="B1", run=str(leaf), status="running"),
            ],
        ),
    )
    row = progress(tmp_path)
    assert row["stage"] == "B1" and row["responses"] == 1 and row["planned"] is None
    assert row["stages_done"] == 1 and row["phase"] == "running-client:32768-formal"
