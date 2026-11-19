"""Freeze original sources and the explicitly selected HLE/Frontier task scope."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CLIENT_MANIFEST_SHA256 = (
    "19c269ba9d9ae547fa1ce03b9f415e2fed37100b34ae463d1421628d659cf35d"
)
DATASETS = {
    "swe": {"tasks": 731},
    "terminal": {"tasks": 89},
    "hle": {
        "module": "hust_eval_hle_verified.pipeline",
        "directory": "hle",
        "tasks": 1811,
        "parts": {
            "gold": (
                668,
                "f03c2632400970ffe5167d5ecde169ad38d34ce4dceb2eb8cfb54ece9c5fae00",
            ),
            "revision": (
                1143,
                "a5a1b1d84edfb90844dfd2f506f331e69d58cb8ef9488ef5f4ebd420fb30cec6",
            ),
        },
    },
    "frontier": {
        "module": "hust_eval_frontierscience.pipeline",
        "directory": "frontier",
        "tasks": 160,
        "parts": {
            "olympiad": (
                100,
                "450e7fceb53c8140283a752f0d62a3abc69a85e252d4dbbd5d9dfbfb762db4e4",
            ),
            "research": (
                60,
                "96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46",
            ),
        },
    },
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def client(name):
    if name in ("swe", "terminal"):
        return importlib.import_module("agent_trace")
    source = ROOT / "evaluation-clients"
    manifest = source / "frozen-clients.json"
    if sha(manifest) != CLIENT_MANIFEST_SHA256:
        raise ValueError("Evaluation client manifest differs from external pin")
    files = json.loads(manifest.read_bytes())["files"]
    actual = {str(p.relative_to(ROOT)) for p in source.rglob("*.py")}
    if actual != set(files):
        raise ValueError("Evaluation client source inventory differs")
    for path, expected in files.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen evaluator source changed: " + path)
    sys.path.insert(0, str(source / DATASETS[name]["directory"]))
    return importlib.import_module(DATASETS[name]["module"])


def assets(name, root):
    root.mkdir(parents=True, exist_ok=True)
    for part, (count, expected) in DATASETS[name]["parts"].items():
        target = root / part / "test.jsonl"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "hle":
                filename = {
                    "gold": "Gold_subset.jsonl",
                    "revision": "Revision_subset.jsonl",
                }[part]
                url = (
                    "https://media.githubusercontent.com/media/SKYLENAGE-AI/HLE-Verified/b705e0fb541c025a1532ce0d60d70ae2f53b00e0/data/"
                    + filename
                )
            else:
                url = (
                    "https://huggingface.co/datasets/openai/frontierscience/resolve/25ed67db7da8f4591484e764008ff585544f5a30/"
                    + part
                    + "/test.jsonl"
                )
            partial = target.with_suffix(".partial")
            with (
                urllib.request.urlopen(url, timeout=180) as response,
                partial.open("wb") as stream,
            ):
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha(partial) != expected:
                raise ValueError(
                    "Downloaded dataset differs from frozen source: "
                    + name
                    + "/"
                    + part
                )
            partial.replace(target)
        if sha(target) != expected:
            raise ValueError("Dataset differs from frozen source: " + name + "/" + part)
        with target.open() as stream:
            if sum(bool(line.strip()) for line in stream) != count:
                raise ValueError("Dataset cardinality differs")
    return root


def prepare(name, asset_root, out, model, hle_profile="fixed200-v1"):
    if name in ("swe", "terminal"):
        from agent_trace import prepare as prepare_trace

        return prepare_trace(name, asset_root, out, model)
    module = client(name)
    source = assets(name, asset_root)
    config_root = out.parent / (out.name + "-configs")
    config_root.mkdir(parents=True, exist_ok=False)
    for stage, maximum in (("solver", 16384), ("judge", 8192)):
        config = {
            "base_url": "http://127.0.0.1:18390",
            "model": "qwen3.5-35b",
            "model_artifact": str(model.resolve()),
            "api_key_env": None,
            "concurrency": 16,
            "timeout": 1800,
            "request": {
                "temperature": 0,
                "seed": 17,
                "max_tokens": maximum,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if stage == "judge":
            config["scoring_policy"] = (
                "strict_json" if name == "hle" else "mmlu_extract_first"
            )
        save(config_root / (stage + ".json"), config)
    args = argparse.Namespace(
        assets=str(source),
        output=str(out),
        subset=["gold", "revision"],
        modality="all",
        solver_config=str(config_root / "solver.json"),
        judge_config=str(config_root / "judge.json"),
        solver_runtime_contract=None,
        judge_runtime_contract=None,
    )
    module.prepare(args)
    contract = json.loads((out / "contract.json").read_bytes())
    scope = None
    if name == "hle":
        from hle_scope import apply

        scope = apply(out, contract, hle_profile)
    contract["engine_transport"] = {
        "stream": True,
        "include_usage": True,
        "adapter_sha256": sha(ROOT / "scripts/streaming_transport.py"),
        "reason": "Collect real TTFT/TPOT and raw SSE while retaining frozen prompts, generation limits and original grading rules; same transport for B0/B1",
    }
    contract["image_interface_qualification"] = {
        "requests_before_hle_solver": 2 if name == "hle" else 0,
        "rule": "Fixed synthetic red/blue squares; same checks before both arms; excluded from dataset score and measured solver interval",
    }
    save(out / "contract.json", contract)
    if name == "hle":
        (out / "contract.sha256").write_text(sha(out / "contract.json") + "\n")
    _, tasks = module.load_plan(out)
    expected_tasks = scope["selected_tasks"] if scope else DATASETS[name]["tasks"]
    if len(tasks) != expected_tasks:
        raise ValueError("Frozen named execution scope was not prepared")
    if name == "hle":
        from hle_scope import verify_selected

        verify_selected(out, contract, tasks)
        if sum(bool(t["image"]) for t in tasks) != scope["image_tasks"]:
            raise ValueError("HLE selected image tasks were dropped")
    return {
        "name": name,
        "plan": str(out.resolve()),
        "contract_sha256": sha(out / "contract.json"),
        "tasks": len(tasks),
        "client_manifest_sha256": CLIENT_MANIFEST_SHA256,
        "modality": "all",
        "image_tasks": scope["image_tasks"] if scope else 0,
        "execution_scope": scope,
    }
