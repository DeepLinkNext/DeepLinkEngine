"""Configure and join a new HTTP fixed200 raw-work observation, outside performance runs."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request

from dataset_contract import save

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"


def configure(command, environment, output):
    index = command.index("--worker-extension-cls") + 1
    if command[index] != "compile_boundary.WorkerExtension":
        raise ValueError("Do not replace a foreign worker extension")
    command[index] = "m01_proc_worker.M01WorkerExtension"
    environment.update(VLLMHUST_M01_PROC_CAPTURE="1",
                       VLLMHUST_M01_HTTP_OUTPUT=str(output / "raw-work"),
                       VLLMHUST_M01_HTTP_PROFILE="1",
                       VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    environment["PYTHONPATH"] = ":".join([str(KIT / "proc_capture"), str(KIT), environment["PYTHONPATH"]])
    environment["VLLM_PLUGINS"] += ",hust_m01_proc_capture"


def boundary(action):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request("http://127.0.0.1:18390/hust_reference_observation/" + action,
                                     data=b"{}", headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=2100 if action == "end" else 600) as response:
        result = json.load(response)
    if not result["completed"]:
        raise ValueError("HTTP observation boundary did not close")
    return result


def response_counts(output):
    """Join actual returned SSE IDs and usage. No invented token-ID arrays."""
    requests = {}
    cases = sorted(output.glob("case-*.json"))
    if len(cases) != 200:
        raise ValueError("Reference observation requires all original 200 responses")
    for case in cases:
        data = json.loads(case.read_bytes())
        if data["success"] is not True or data["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0):
            raise ValueError("Incomplete or prefix-cached observation response")
        identities = set()
        with gzip.open(case.with_suffix(".sse.jsonl.gz"), "rt") as stream:
            for line in stream:
                raw = json.loads(line)["raw"]
                if not raw.startswith("data:") or raw[5:].strip() == "[DONE]":
                    continue
                event = json.loads(raw[5:])
                if event.get("id"):
                    identities.add(event["id"])
        if len(identities) != 1:
            raise ValueError("Response SSE does not bind to one unique engine request")
        identity = hashlib.sha256(identities.pop().encode()).hexdigest()
        if identity in requests:
            raise ValueError("Repeated HTTP engine request ID")
        counts = data["usage"]
        prompt, generated = counts["prompt_tokens"], counts["completion_tokens"]
        if type(prompt) is not int or prompt < 1 or type(generated) is not int or not 1 <= generated <= 8192:
            raise ValueError("Invalid current HTTP request token counts")
        requests[identity] = (prompt, generated)
    return requests


def finish(output, model):
    stopped = boundary("end")
    from resolved_http_inputs import prepare, remove_input_aliases

    requests = response_counts(output)
    deployment = json.loads((output.parent / "source-admission.json").read_bytes())["deployment"]
    view = output / "resolved-raw-work"
    prepare(output / "raw-work", deployment, view)
    sys.path[:0] = [str(KIT / "proc_capture"), str(KIT)]
    from m01_observer_work import collect

    config = json.loads((model / "config.json").read_bytes())["text_config"]
    counts = dict(scheduler=stopped["counts"]["scheduler"])
    counts.update({rank: stopped["counts"][str(rank)] for rank in (0, 1)})
    result = collect(view, stopped["window"], counts, {}, [],
                     SimpleNamespace(**config), request_counts=requests,
                     draft_counts={r["rank"]:r["draft_calls"] for r in stopped["workers"]})
    remove_input_aliases(view)
    save(output / "raw-work/http-request-counts.json", requests)
    save(output / "raw-work/raw-partial-work.json", dict(result, frontend="HTTP",
         timing_eligible_for_performance_comparison=False, arithmetic_certified=False,
         efficiency_percent=None))
    return result
