"""r8 is the measured C15 runtime, with explicit launch metadata and shared warmup."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_geometry as geometry  # noqa: E402 - bundled scripts use a private source path
import runtime_overlay as overlay  # noqa: E402 - bundled scripts use a private source path
import serving_warmup as warmup  # noqa: E402 - bundled scripts use a private source path
import completion_warmup as completion  # noqa: E402 - bundled scripts use a private source path


def native_command():
    command = [
        "python",
        "serve",
        "--max-num-seqs",
        "24",
        "--max-num-batched-tokens",
        "4096",
        "--speculative-config",
        json.dumps(dict(method="mtp", num_speculative_tokens=2)),
        "--additional-config",
        json.dumps(
            dict(
                using_live_runtime=True,
                state_resident_seats=36,
                state_cache_host_bytes=4 * 2**30,
                state_cache_incremental=True,
                state_cache_streaming=True,
                state_cache_policy=True,
                state_cache_partial_reclaim=True,
            )
        ),
        "--compilation-config",
        json.dumps(
            dict(cudagraph_mode="FULL", cudagraph_capture_sizes=geometry.captures(24))
        ),
    ]
    env = dict(
        MTP_TOKENS="2",
        BETTERSCALE_CONTEXT_PARALLEL="1",
        BETTERSCALE_STATE_CACHE_HOST_BYTES=str(4 * 2**30),
    )
    return command, env


def bound_profile():
    profile = dict(release=geometry.RELEASE)
    geometry.bind_profile(profile)
    return profile


def test_geometry_changes_only_measured_capacity_and_graph_bins():
    command, env = native_command()
    original, original_env = list(command), dict(env)
    record = geometry.apply(bound_profile(), command, env)
    assert record["original_command"] == original
    assert record["effective_command"] == command
    assert env == original_env
    assert command[command.index("--max-num-seqs") + 1] == "36"
    assert (
        json.loads(command[command.index("--additional-config") + 1])[
            "state_resident_seats"
        ]
        == 54
    )
    captures = json.loads(command[command.index("--compilation-config") + 1])[
        "cudagraph_capture_sizes"
    ]
    assert captures == geometry.captures(36) and {96, 108}.issubset(captures)
    changed = [
        original[i - 1] for i in range(len(command)) if original[i] != command[i]
    ]
    assert changed == ["--max-num-seqs", "--additional-config", "--compilation-config"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_seats", 24),
        ("resident_seats", 36),
        ("speculative_tokens", True),
        ("host_bytes_per_rank", 1),
        ("prefill_tokens", 4096.0),
    ],
)
def test_self_declared_mutated_geometry_cannot_be_admitted(field, value):
    profile = bound_profile()
    profile["runtime_configuration"][field] = value
    profile["runtime_configuration_sha256"] = geometry.digest(
        profile["runtime_configuration"]
    )
    with pytest.raises(ValueError, match="geometry"):
        geometry.validate_profile(profile)


@pytest.mark.parametrize(
    "release",
    [
        "full-answer-b1-20261030-r4",
        "full-answer-b1-20261113-r6",
        "full-answer-b1-20261116-r7",
    ],
)
def test_legacy_launch_is_literal_noop_and_cannot_borrow_r8_config(release):
    command, env = native_command()
    before = copy.deepcopy((command, env))
    assert geometry.apply(dict(release=release), command, env) is None
    assert (command, env) == before
    profile = bound_profile()
    profile["release"] = release
    with pytest.raises(ValueError, match="bound"):
        geometry.validate_profile(profile)


@pytest.mark.parametrize(
    "mutation", ["eager", "host-budget", "duplicate-seats", "state-policy"]
)
def test_launch_drift_rejected_before_mutation(mutation):
    command, env = native_command()
    if mutation == "eager":
        command.append("--enforce-eager")
    elif mutation == "host-budget":
        env["BETTERSCALE_STATE_CACHE_HOST_BYTES"] = "0"
    elif mutation == "duplicate-seats":
        command += ["--max-num-seqs", "24"]
    else:
        i = command.index("--additional-config") + 1
        value = json.loads(command[i])
        value["state_cache_incremental"] = False
        command[i] = json.dumps(value)
    before = copy.deepcopy((command, env))
    with pytest.raises(ValueError):
        geometry.apply(bound_profile(), command, env)
    assert (command, env) == before


def test_r8_overlay_contains_exact_ten_frozen_runtime_files():
    manifest = overlay.manifest(geometry.RELEASE)
    profile = json.loads((ROOT / "engines/b1-r8.json").read_bytes())
    assert len(manifest["files"]) == 10
    assert profile["runtime_configuration"] == geometry.CONFIG
    assert profile["combination"]["E"] == 36 and profile["combination"]["R"] == 54
    assert profile["qualification"]["default_promoted"] is False
    assert "restore/RMS" in profile["excluded_exploration"][0]
    for name, pin in manifest["files"].items():
        assert (
            hashlib.sha256(
                (overlay.overlay_root(geometry.RELEASE) / name).read_bytes()
            ).hexdigest()
            == pin["after_sha256"]
        )
    assert (
        manifest["files"][
            "mods/betterscale/betterscale/models/qwen35/prefill_round_robin.py"
        ]["after_sha256"]
        == "1f00bfb11e91d0601c761bdfe1691fcd59fedf69bbea3afa6a252babdf97a957"
    )


@pytest.mark.parametrize("module", [warmup, completion])
def test_shared_r8_dense_warmup_is_194_and_legacy_stays_126(module):
    inputs = {str(n): [dict(case=f"{n}-{i}") for i in range(8)] for n in (4096, 16384)}
    dense = dict(unscored=True)
    cohorts = list(module.warmup_cohorts(inputs, dense, capacity36=True))
    assert sum(map(len, cohorts)) == 194
    assert [len(c) for c in cohorts[-8:]] == [1, 2, 4, 8, 16, 24, 32, 36]
    assert module.request_count("joint", capacity36=True) == 194
    assert module.request_count("joint") == 126
    assert module.request_count("continuous", capacity36=True) == 71
    assert (
        module.protocol("task", capacity36=True)["policy"]
        == "observed-torch-triton-sealed-task-warmup-v5-capacity36"
    )
    assert (
        module.protocol("task")["policy"]
        == "observed-torch-triton-sealed-task-warmup-v3"
    )


def test_long_context_shared_warmup_uses_r8_dense_capacity():
    value = completion.protocol("continuous", acceptance=True, capacity36=True)
    assert (
        value["policy"]
        == "observed-torch-triton-sealed-completion-warmup-v2-capacity36"
    )
    assert (
        completion.request_count("continuous", acceptance=True, capacity36=True) == 202
    )
    assert value["dense_decode_concurrencies"][-2:] == [32, 36]


def test_r8_mod_inventory_has_canonical_ids_and_separates_parameters():
    stack = json.loads((ROOT / "mods/active-stack-r8.json").read_bytes())
    ids = stack["inherited"] + stack["additions"]
    assert len(ids) == len(set(ids)) == 10
    assert [m["id"] for m in stack["enabled_mods"]] == ids
    assert stack["parameters"]["execution_seats"] == 36
    for item in stack["enabled_mods"]:
        mod = json.loads((ROOT / item["manifest"]).read_bytes())
        assert mod["id"] == item["id"] and mod["version"] == item["version"]
        assert item["enabled_in"] == "r8"
        assert (ROOT / item["documentation"]).is_file()
    assert "restore-gated-rms-fusion" not in ids
