"""Versioned shared client warmup outside formal clocks and answers."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

POLICY = "observed-torch-triton-sealed-warmup-v2"
TASK_POLICY = "observed-torch-triton-sealed-task-warmup-v3"
DENSE_CONCURRENCIES = (1, 2, 4, 8, 16, 24)
CAPACITY36_CONCURRENCIES = DENSE_CONCURRENCIES + (32, 36)
CAPACITY36_TASK_POLICY = "observed-torch-triton-sealed-task-warmup-v5-capacity36"
DENSE_PROMPT = (
    "Synthetic unscored serving warmup. Continue writing the word warmup "
    "separated by spaces until the generation budget is reached."
)
COHORTS = [(4096, n) for n in (1, 2, 4, 8, 16, 24)] + [(16384, 16)]


def selected(profile):
    return profile["release"] in (
        "full-answer-b1-20261113-r5",
        "full-answer-b1-20261113-r6",
        "full-answer-b1-20261116-r7", "full-answer-b1-20261117-r8",
    )


def protocol(kind="continuous", *, acceptance=False, capacity36=False):
    value = dict(
        policy=POLICY,
        cohorts=[
            dict(minimum_prompt_tokens=p, concurrency=c)
            for p, c in (COHORTS + ([(32768, 8)] if acceptance else []))
        ],
        input_source="Original frozen C8 synthetic chat messages, validated token counts and token-ID hashes on each service",
        transport="/v1/chat/completions; same request-scoped KV control path as formal traffic",
        request_seed=17,
        output_tokens=32,
        ignore_eos=True,
        no_prefix_reuse=True,
        modality_warmup="HLE solver: original synthetic image-interface qualification on both arms",
        grammar_warmup="HLE judge: same bounded-v2 generation schema, synthetic prompt,32-token non-scored response",
        formal_requests_unchanged=True,
        formal_clock_excludes_warmup=True,
        post_boundary_compilation="fail-closed Torch/Dynamo and Triton; CANN not covered",
    )
    if kind != "continuous" or acceptance:
        value.update(
            policy=CAPACITY36_TASK_POLICY if capacity36 else TASK_POLICY,
            dense_decode_concurrencies=list(CAPACITY36_CONCURRENCIES if capacity36 else DENSE_CONCURRENCIES),
            dense_prompt=DENSE_PROMPT,
            dense_prompt_maximum_tokens=128,
            reason="Cover live-batch Triton tile variants before task measurement; short C8 outputs alone did not reach them",
        )
    if acceptance:
        value["policy"] = "observed-torch-triton-sealed-completion-warmup-v2-capacity36" if capacity36 else "observed-torch-triton-sealed-completion-warmup-v1"
    return value


def request_count(kind="continuous", *, acceptance=False, capacity36=False):
    return (
        sum(n for _, n in COHORTS)
        + (8 if acceptance else 0)
        + (sum(CAPACITY36_CONCURRENCIES if capacity36 else DENSE_CONCURRENCIES) if kind != "continuous" or acceptance else 0)
    )


def warmup_cohorts(inputs, dense=None, *, acceptance=False, capacity36=False):
    for tokens, concurrency in COHORTS + ([(32768, 8)] if acceptance else []):
        yield [
            inputs[str(tokens)][i % len(inputs[str(tokens)])]
            for i in range(concurrency)
        ]
    if dense is not None:
        for concurrency in (CAPACITY36_CONCURRENCIES if capacity36 else DENSE_CONCURRENCIES):
            yield [dense] * concurrency


def rpc(method):
    body = json.dumps(dict(method=method, timeout=120)).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:18390/collective_rpc",
        body,
        {"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=150) as response:
        result = json.load(response)["results"]
    if len(result) != 2 or {r["rank"] for r in result} != {0, 1}:
        raise ValueError("Both worker ranks must acknowledge compilation boundary")
    if any(
        not r["torch_dynamo_guard"] or not r["triton_guard"] or r["failure"]
        for r in result
    ):
        raise ValueError("Compiler boundary not admitted on both ranks")
    return result


def warmup_body(case, salt):
    import copy

    body = copy.deepcopy(case["request"])
    if "prompt" in body or not body.get("messages"):
        raise ValueError("Warmup must exercise the frozen chat control path")
    body.update(max_tokens=32, ignore_eos=True, cache_salt=salt)
    return body


def run(out, dataset_spec=None, continuous_inputs=None, qualified_continuous=None):
    root = Path(out) / "serving-warmup"
    root.mkdir(exist_ok=False)
    started = time.time()
    before = rpc("hust_compile_prepare")
    plan = json.loads((Path(out) / "plan.json").read_bytes())
    capacity36 = plan.get("engine_release") == "full-answer-b1-20261117-r8"
    if plan.get("runtime_geometry"):
        from capacity_snapshot import capture

        capture(out, "before-warmup")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    source = (
        Path(continuous_inputs)
        if continuous_inputs is not None
        else Path(qualified_continuous) / "service/request-plan.json"
    )
    inputs_raw = source.read_bytes()
    inputs = json.loads(inputs_raw)
    plan = json.loads((Path(out) / "plan.json").read_bytes())
    acceptance = bool(plan.get("acceptance_suite"))
    if set(inputs) != (
        {"4096", "16384", "32768"} if acceptance else {"4096", "16384"}
    ) or any(
        not cases or (len(cases) != 8 if key != "32768" else len(cases) != 5)
        for key, cases in inputs.items()
    ):
        raise ValueError("Warmup requires both original C8 synthetic chat cohorts")
    (root / "input-plan.json").write_bytes(inputs_raw)
    token_checks = []
    for minimum, cases in inputs.items():
        for case in cases:
            body = dict(
                model="qwen3.5-35b",
                messages=case["request"]["messages"],
                add_generation_prompt=True,
            )
            if minimum == "32768":
                body.update(
                    tools=case["request"].get("tools"),
                    chat_template_kwargs=case["request"]["chat_template_kwargs"],
                    add_special_tokens=False,
                    continue_final_message=False,
                )
            raw = json.dumps(body).encode()
            call = urllib.request.Request(
                "http://127.0.0.1:18390/tokenize",
                raw,
                {"Content-Type": "application/json"},
            )
            with opener.open(call, timeout=60) as response:
                result = json.load(response)
            if (
                result["count"] != case["input_tokens"]
                or result["count"] < int(minimum)
                or hashlib.sha256(
                    json.dumps(
                        result["tokens"],
                        **({"separators": (",", ":")} if minimum == "32768" else {}),
                    ).encode()
                ).hexdigest()
                != case["token_ids_sha256"]
            ):
                raise ValueError("Warmup chat tokenizer differs from original C8 plan")
            token_checks.append(
                dict(
                    input_tokens=result["count"],
                    token_ids_sha256=case["token_ids_sha256"],
                )
            )
    if dataset_spec is not None:
        from dataset_client import checked

        spec, module = checked(dataset_spec)
        if spec["dataset"] == "hle" and spec["stage"] == "solver":
            from image_qualification import qualify

            contract, _ = module.load_plan(spec["plan"])
            result = qualify(contract["solver"], root / "image-qualification")
            (root / "image-qualification.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
        if spec["dataset"] == "hle" and spec["stage"] == "judge":
            from judge_grammar import effective_format

            contract, _ = module.load_plan(spec["plan"])
            body = dict(
                model="qwen3.5-35b",
                messages=[
                    dict(
                        role="user",
                        content="Return a JSON object with reasoning, extracted_final_answer, correct and confidence.",
                    )
                ],
                max_tokens=32,
                temperature=0,
                seed=0,
                response_format=effective_format(
                    spec, contract["judge_response_format"]
                ),
                cache_salt=uuid.uuid4().hex,
            )
            raw = json.dumps(body).encode()
            (root / "grammar.request.json").write_bytes(raw)
            call = urllib.request.Request(
                "http://127.0.0.1:18390/v1/chat/completions",
                raw,
                {"Content-Type": "application/json"},
            )
            with opener.open(call, timeout=900) as response:
                (root / "grammar.response.json").write_bytes(response.read())

    def request(item):
        ordinal, case = item
        body = warmup_body(case, uuid.uuid4().hex)
        raw = json.dumps(body).encode()
        (root / f"{ordinal:04d}.request.json").write_bytes(raw)
        call = urllib.request.Request(
            "http://127.0.0.1:18390/v1/chat/completions",
            raw,
            {"Content-Type": "application/json"},
        )
        begin = time.time()
        with opener.open(call, timeout=900) as response:
            status = response.status
            data = response.read()
        (root / f"{ordinal:04d}.sse").write_bytes(data)
        chunks = [
            json.loads(line[6:])
            for line in data.decode().splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        usage = [c["usage"] for c in chunks if c.get("usage")]
        if (
            status != 200
            or b"data: [DONE]" not in data
            or not usage
            or any(c.get("error") for c in chunks)
        ):
            raise ValueError("Warmup streaming request incomplete")
        if (
            usage[-1]["prompt_tokens"] != case["input_tokens"]
            or usage[-1]["completion_tokens"] != 32
            or usage[-1].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        ):
            raise ValueError("Warmup budget or zero-prefix condition differs")
        return dict(
            ordinal=ordinal,
            started=begin,
            ended=time.time(),
            usage=usage[-1],
            request_sha256=hashlib.sha256(raw).hexdigest(),
            sse_sha256=hashlib.sha256(data).hexdigest(),
        )

    kind = "continuous" if continuous_inputs is not None else "task"
    dense = None
    if kind == "task" or acceptance:
        import copy

        dense = copy.deepcopy(inputs["4096"][0])
        dense["request"]["messages"] = [dict(role="user", content=DENSE_PROMPT)]
        raw = json.dumps(
            dict(
                model="qwen3.5-35b",
                messages=dense["request"]["messages"],
                add_generation_prompt=True,
            )
        ).encode()
        call = urllib.request.Request(
            "http://127.0.0.1:18390/tokenize", raw, {"Content-Type": "application/json"}
        )
        with opener.open(call, timeout=60) as response:
            dense_tokens = json.load(response)
        if not 0 < dense_tokens["count"] <= 128 or dense_tokens["count"] * (36 if capacity36 else 24) > 4096:
            raise ValueError(
                "Dense synthetic warmup must fit the selected live batch in4096 tokens"
            )
        dense["input_tokens"] = dense_tokens["count"]
        dense["token_ids_sha256"] = hashlib.sha256(
            json.dumps(dense_tokens["tokens"]).encode()
        ).hexdigest()
        (root / "dense-input.json").write_text(json.dumps(dense, indent=2) + "\n")
        token_checks.append(
            dict(
                input_tokens=dense["input_tokens"],
                token_ids_sha256=dense["token_ids_sha256"],
                scope="unscored synthetic dense warmup",
            )
        )
    rows = []
    for cases in warmup_cohorts(inputs, dense, acceptance=acceptance, capacity36=capacity36):
        offset = len(rows)
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            rows.extend(
                pool.map(request, [(offset + i, case) for i, case in enumerate(cases)])
            )
    sealed = rpc("hust_compile_seal")
    if not all(r["sealed"] for r in sealed):
        raise ValueError("Cannot start measurement without both sealed ranks")
    value = dict(
        protocol=protocol(kind, acceptance=acceptance, capacity36=capacity36),
        dense_input_sha256=hashlib.sha256(
            (root / "dense-input.json").read_bytes()
        ).hexdigest()
        if dense is not None
        else None,
        started=started,
        ended=time.time(),
        ranks_before=before,
        ranks_ready=sealed,
        input_plan_sha256=hashlib.sha256(inputs_raw).hexdigest(),
        token_checks=token_checks,
        requests=rows,
        formal_answers_reused=False,
        completed=True,
    )
    (root / "receipt.json").write_text(json.dumps(value, indent=2) + "\n")
    print("SERVING_WARMUP_SEALED", len(rows), flush=True)
    return 0


def verify_after(out):
    result = rpc("hust_compile_status")
    plan = json.loads((Path(out) / "plan.json").read_bytes())
    if plan.get("runtime_geometry"):
        from capacity_snapshot import capture

        capture(out, "after-measurement")
    if not all(r["sealed"] for r in result):
        raise ValueError("Compilation guard no longer sealed")
    (Path(out) / "compilation-after-measurement.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit(root):
    """Recheck sealed evidence; successful r5 runs must prove both boundaries."""
    root = Path(root)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if (
        not (root / "deployment.json").exists()
        and not receipt.get("client_started")
        and not receipt.get("completed")
    ):
        return None
    deployment = json.loads((root / "deployment.json").read_bytes())
    if not selected(deployment):
        return None
    receipt = json.loads((root / "receipt.json").read_bytes())
    if not receipt.get("completed"):
        return dict(
            policy=POLICY,
            completed=False,
            reason="Arm failed; no successful warmup/performance claim",
        )
    service = root / "service"
    kind = receipt["kind"]
    capacity36 = deployment["release"] == "full-answer-b1-20261117-r8"
    acceptance = bool(receipt.get("acceptance_suite"))
    value = json.loads((service / "serving-warmup/receipt.json").read_bytes())
    after = json.loads((service / "compilation-after-measurement.json").read_bytes())
    capacity36 = capacity36 or value["protocol"]["policy"] in (CAPACITY36_TASK_POLICY, "observed-torch-triton-sealed-completion-warmup-v2-capacity36")
    if (
        value["protocol"] != protocol(kind, acceptance=acceptance, capacity36=capacity36)
        or value["completed"] is not True
        or len(value["requests"]) != request_count(kind, acceptance=acceptance, capacity36=capacity36)
    ):
        raise ValueError("Sealed warmup protocol or denominator differs")
    if kind != "continuous" or acceptance:
        raw = (service / "serving-warmup/dense-input.json").read_bytes()
        dense = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != value.get("dense_input_sha256")
            or dense["request"]["messages"] != [dict(role="user", content=DENSE_PROMPT)]
            or not 0 < dense["input_tokens"] <= 128
        ):
            raise ValueError("Dense synthetic pre-measurement warmup differs")
    ready = value["ranks_ready"]
    if len(after) != 2 or {r["rank"] for r in after} != {0, 1} or len(ready) != 2:
        raise ValueError("Compiler evidence must cover both ranks")
    if sorted(ready, key=lambda r: r["rank"]) != sorted(after, key=lambda r: r["rank"]):
        raise ValueError("Compiler state changed during measured requests")
    if not all(
        r["sealed"]
        and r["torch_dynamo_guard"]
        and r["triton_guard"]
        and not r["failure"]
        for r in after
    ):
        raise ValueError("Compiler boundary missing after measurement")
    if value["ended"] > receipt["measurement_ready"]:
        raise ValueError("Formal client started before shared warmup drained")
    for row in value["requests"]:
        for suffix, key in (("request.json", "request_sha256"), ("sse", "sse_sha256")):
            raw = (
                service / "serving-warmup" / f"{row['ordinal']:04d}.{suffix}"
            ).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row[key]:
                raise ValueError("Warmup raw transport binding differs")
    if (service / "plan.json").exists() and json.loads((service / "plan.json").read_bytes()).get("runtime_geometry"):
        from capacity_snapshot import audit as audit_capacity

        audit_capacity(root)
    return dict(
        policy=protocol(kind, acceptance=acceptance, capacity36=capacity36)["policy"],
        completed=True,
        requests=len(value["requests"]),
        wall_seconds=value["ended"] - value["started"],
        ranks=after,
        formal_clock_excludes_warmup=True,
        cann_compilation_observed=False,
    )
