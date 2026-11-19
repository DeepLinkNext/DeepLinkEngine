"""Run all frozen real-context and four-tenant phases on one admitted service."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

from agent_trace import digest, save, sha
from streaming_transport import call, clock

SCHEMA = "vllm-hust-context-tenant-supplement/v1"


def load(path):
    path = Path(path).resolve()
    contract = json.loads(path.read_bytes())
    name = Path(contract["input_file"])
    if name.is_absolute() or len(name.parts) != 1:
        raise ValueError("Acceptance inputs must be a local adjacent file")
    inputs_path = path.parent / name
    if inputs_path.is_symlink():
        raise ValueError("Acceptance inputs must not be a mutable symlink")
    if contract["schema"] != SCHEMA or sha(inputs_path) != contract["input_sha256"]:
        raise ValueError("Frozen acceptance input binding changed")
    inputs = json.loads(inputs_path.read_bytes())
    if type(contract.get("task_count")) is not int or contract["task_count"] != 5:
        raise ValueError("Exactly five distinct original three-band tasks are frozen")
    expected = [(str(n), str(n), 8, 1) for n in (4096, 16384, 32768)] + [
        ("multi-tenant", "4096", 24, 4)
    ]
    if [
        (p["id"], p["cohort"], p["concurrency"], p["tenants"])
        for p in contract["phases"]
    ] != expected:
        raise ValueError("Require all three context bands and the four-tenant phase")
    if set(inputs) != {"4096", "16384", "32768"} or any(
        len(v) != contract["task_count"] for v in inputs.values()
    ):
        raise ValueError(
            "All frozen original requests in each context band are required"
        )
    if [(p["warmup_seconds"], p["formal_seconds"]) for p in contract["phases"]] != [
        (60, 300)
    ] * 4:
        raise ValueError("Preserve the complete warmup/formal measurement budgets")
    if (
        contract["output_tokens"],
        contract["temperature"],
        contract["seed"],
        contract["ignore_eos"],
        contract["automatic_retries"],
    ) != (256, 0, 17, True, 0):
        raise ValueError("Frozen generation policy changed")
    bands = [(4096, 8192), (16384, 24576), (32768, 49152)]
    if [
        (p["min_tokens"], p["max_tokens_exclusive"], p["concurrency"])
        for p in contract["cohorts"]
    ] != [(lo, hi, 8) for lo, hi in bands]:
        raise ValueError("All original context bands are required in order")
    task_order = None
    for spec in contract["cohorts"]:
        group = str(spec["min_tokens"])
        tasks = [case["task_id"] for case in inputs[group]]
        if len(set(tasks)) != contract["task_count"] or (
            task_order is not None and tasks != task_order
        ):
            raise ValueError(
                "All three bands must share the same distinct ordered source tasks"
            )
        task_order = tasks
        for case in inputs[group]:
            if (
                not spec["min_tokens"]
                <= case["input_tokens"]
                < spec["max_tokens_exclusive"]
            ):
                raise ValueError(
                    "Original request is outside its declared context band"
                )
            body = case["request"]
            if (
                body["max_tokens"],
                body["ignore_eos"],
                body["temperature"],
                body["seed"],
                body.get("stream"),
                body.get("stream_options"),
                body.get("tool_choice"),
                body.get("chat_template_kwargs"),
            ) != (
                256,
                True,
                0,
                17,
                True,
                {"include_usage": True},
                "none",
                {"enable_thinking": False},
            ):
                raise ValueError("Per-request generation drift")
            original = {
                k: copy.deepcopy(v)
                for k, v in body.items()
                if k in ("messages", "tools", "tool_choice", "parallel_tool_calls")
            }
            # tool_choice is a declared new serving parameter; original-body hashes remain separately bound.
            if (
                not case.get("source_sha256")
                or not case.get("source_body_sha256")
                or not original["messages"]
            ):
                raise ValueError("Published input provenance missing")
    return contract, inputs


def tokenize(inputs, url, out, baseline=None):
    """Freeze actual service IDs before sending any generated request."""
    checked = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    folder = out / "tokenize"
    folder.mkdir(exist_ok=False)
    for group, cases in inputs.items():
        for index, case in enumerate(cases):
            name = f"{group}-{index}"
            body = dict(
                model="qwen3.5-35b",
                messages=case["request"]["messages"],
                tools=case["request"].get("tools"),
                add_generation_prompt=True,
                continue_final_message=False,
                add_special_tokens=False,
                chat_template_kwargs=dict(enable_thinking=False),
            )
            request = urllib.request.Request(
                url + "/tokenize",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            with opener.open(request, timeout=60) as response:
                result = json.load(response)
            save(folder / (name + ".json"), dict(request=body, response=result))
            ids = result.get("tokens")
            if (
                not isinstance(ids, list)
                or result.get("count") != len(ids)
                or len(ids) != case["input_tokens"]
                or digest(ids) != case["token_ids_sha256"]
            ):
                raise ValueError(
                    "Actual service tokenizer differs from the frozen original request: "
                    + name
                )
            checked[name] = dict(count=len(ids), token_ids_sha256=digest(ids))
    if baseline is not None and checked != baseline:
        raise ValueError(
            "B0/B1 actual request token IDs differ before generated traffic"
        )
    save(
        out / "tokenizer-admission.json",
        dict(complete=True, inputs=checked, generation_requests=0),
    )
    return checked


def phase(out, specification, cases, name, seconds, url):
    start = clock()
    deadline = start["monotonic"] + seconds
    stop = threading.Event()
    rows, errors = [], []
    lock = threading.Lock()
    directory = out / name
    directory.mkdir(exist_ok=False)

    def lane(index):
        sequence = 0
        tenant = index % specification["tenants"]
        case = cases[(index // specification["tenants"]) % len(cases)]
        while time.monotonic() < deadline and not stop.is_set():
            identity = f"{name}-tenant{tenant}-lane{index}-{sequence:05d}"
            body = dict(
                copy.deepcopy(case["request"]),
                cache_salt=uuid.uuid4().hex,
                user=f"tenant-{tenant}",
            )
            request_dir = directory / identity
            request_dir.mkdir()
            row = call(dict(base_url=url, timeout=600), body, request_dir)
            good = (
                row["status"] == "ok"
                and row.get("finish_reason") == "length"
                and row["usage"]["prompt_tokens"] == case["input_tokens"]
                and row["usage"]["completion_tokens"] == 256
            )
            row.update(
                id=identity,
                success=good,
                phase=name,
                tenant=f"tenant-{tenant}",
                lane=index,
                case_index=(index // specification["tenants"]) % len(cases),
                input_tokens=case["input_tokens"],
                source_sha256=case["source_sha256"],
                source_body_sha256=case["source_body_sha256"],
            )
            save(request_dir / "request.json", row)
            with lock:
                rows.append(str((request_dir / "request.json").relative_to(out)))
            sequence += 1
            if not good:
                stop.set()
                raise RuntimeError(
                    "Acceptance request failed; no retries, retained " + identity
                )

    with ThreadPoolExecutor(max_workers=specification["concurrency"]) as pool:
        futures = [pool.submit(lane, i) for i in range(specification["concurrency"])]
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                stop.set()
                errors.append(type(error).__name__ + ": " + str(error))
    result = dict(
        name=name,
        cohort=specification["cohort"],
        concurrency=specification["concurrency"],
        tenants=specification["tenants"],
        start=start,
        submission_deadline_monotonic=deadline,
        seconds=seconds,
        end=clock(),
        request_files=sorted(rows),
        errors=errors,
        status="failed" if errors else "completed",
    )
    save(out / (name + "-phase.json"), result)
    print("ACCEPTANCE_PHASE", name, result["status"], len(rows), flush=True)
    if errors:
        raise RuntimeError(errors[0])
    return result


def run(path, out):
    contract, inputs = load(path)
    out = Path(out)
    folder = out / "acceptance"
    folder.mkdir(exist_ok=False)
    save(folder / "contract.json", contract)
    save(folder / "request-plan.json", inputs)
    plan = json.loads((out / "plan.json").read_bytes())
    baseline = None
    if plan.get("acceptance_suite", {}).get("baseline_run"):
        from qualification import sealed_files

        prior = Path(plan["acceptance_suite"]["baseline_run"])
        sealed_files(prior)
        if not json.loads((prior / "receipt.json").read_bytes()).get("completed"):
            raise ValueError(
                "B1 supplement requires the sealed successful B0 supplement"
            )
        previous = json.loads((prior / "service/acceptance/contract.json").read_bytes())
        if previous != contract:
            raise ValueError("B0 supplement used a different context/tenant contract")
        baseline = json.loads(
            (prior / "service/acceptance/tokenizer-admission.json").read_bytes()
        )["inputs"]
    result = dict(schema=SCHEMA, status="running", started=time.time(), phases=[])
    try:
        tokenize(inputs, "http://127.0.0.1:18390", folder, baseline)
        for spec in contract["phases"]:
            for suffix, seconds in (
                ("warmup", spec["warmup_seconds"]),
                ("formal", spec["formal_seconds"]),
            ):
                name = spec["id"] + "-" + suffix
                try:
                    result["phases"].append(
                        phase(
                            folder,
                            spec,
                            inputs[spec["cohort"]],
                            name,
                            seconds,
                            "http://127.0.0.1:18390",
                        )
                    )
                except BaseException:
                    failed = folder / (name + "-phase.json")
                    if failed.exists():
                        result["phases"].append(json.loads(failed.read_bytes()))
                    raise
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        result["ended"] = time.time()
        save(folder / "execution.json", result)
    return 0
