import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from qualify_baseline_capacity import (
    PROTOCOL,
    PROTOCOL_SHA256,
    qualify,
    sha,
    validate_artifacts,
)


def inputs():
    protocol = json.loads(PROTOCOL.read_bytes())
    args = copy.deepcopy(protocol["arguments"])
    reset = dict(
        schema="baseline-native-prefix-reset/v1",
        reset=True,
        b1_state_cache_present=False,
        shared_request_cache_salts_unique=True,
    )
    work = dict(request_token_ids=[[1] * 1411 for _ in range(199)] + [[1] * 1476])
    log = "GPU KV cache size: 473,088 tokens\nMaximum concurrency for 65,536 tokens per request: 7x"
    return args, reset, work, log, protocol


def test_original_pool_conversion_is_qualified_separately_from_r8():
    assert sha(PROTOCOL) == PROTOCOL_SHA256
    result = validate_artifacts(*inputs())
    assert result["sufficient_page_capacity"]
    assert not result["no_preemption_certified"]


@pytest.mark.parametrize(
    "change", ["state_reset", "old_budget", "other_cache", "small_pool", "two_pools"]
)
def test_incompatible_baseline_capacity_domain_is_rejected(change):
    args, reset, work, log, protocol = inputs()
    if change == "state_reset":
        reset["b1_state_cache_present"] = True
    elif change == "old_budget":
        work["request_token_ids"].pop()
    elif change == "other_cache":
        args["mamba_cache_mode"] = "all"
    elif change == "small_pool":
        log = "GPU KV cache size: 128 tokens"
    else:
        log += "\nGPU KV cache size: 256 tokens"
    with pytest.raises(ValueError):
        validate_artifacts(args, reset, work, log, protocol)


def test_failed_or_unsealed_native_run_cannot_certify_source_ownership(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(role="B0", completed=False)))
    seal = tmp_path / "checksums.json"
    seal.write_text(json.dumps({"receipt.json": sha(receipt)}))
    with pytest.raises(ValueError, match="completed quiescent"):
        qualify(tmp_path, tmp_path)
    receipt.write_text(json.dumps(dict(role="B0", completed=True)))
    with pytest.raises(ValueError, match="changed or is unsealed"):
        qualify(tmp_path, tmp_path)


def client_plan():
    from public_output_binding import CLIENT_PINS

    root = Path(__file__).resolve().parents[1]
    files = {str(root / "scripts" / name): pin for name, pin in CLIENT_PINS.items()}
    for name in ["baseline_native_entry.py", "reference_capture_profile.py"]:
        path = root / "scripts" / name
        files[str(path)] = sha(path)
    return dict(files=files)


@pytest.mark.parametrize("capture", ["submission-v2", "full-answer-v1"])
def test_reviewed_versioned_entry_preserves_original_reset_for_both_profiles(capture):
    from qualify_baseline_capacity import bind_client

    protocol = json.loads(PROTOCOL.read_bytes())
    binding = bind_client(client_plan(), dict(id=capture), protocol)
    assert binding["baseline_native_entry.py"] in protocol["baseline_entrypoints"]
    assert binding["reference_capture_profile.py"]


def test_legacy_entry_is_accepted_only_for_its_original_single_run_profile(tmp_path):
    from qualify_baseline_capacity import bind_client, RESET_SHA256

    root = Path(__file__).resolve().parents[1]
    frozen = (
        root
        / "third-party-test-kit/native-client-contract-sources/baseline_native_entry.full-answer-v1.py.source"
    )
    assert sha(frozen) == RESET_SHA256
    path = tmp_path / "baseline_native_entry.py"
    path.write_bytes(frozen.read_bytes())
    plan = client_plan()
    plan["files"] = {
        name: pin
        for name, pin in plan["files"].items()
        if Path(name).name != "baseline_native_entry.py"
    }
    plan["files"][str(path)] = sha(path)
    protocol = json.loads(PROTOCOL.read_bytes())
    assert (
        bind_client(plan, dict(id="full-answer-v1"), protocol)[
            "baseline_native_entry.py"
        ]
        == RESET_SHA256
    )
    with pytest.raises(ValueError, match="reviewed capture profile"):
        bind_client(plan, dict(id="submission-v2"), protocol)


@pytest.mark.parametrize(
    "change", ["unreviewed_entry", "missing_validator", "forged_validator_pin"]
)
def test_current_entry_needs_the_exact_profile_validator(tmp_path, change):
    from qualify_baseline_capacity import bind_client

    plan = client_plan()
    if change == "unreviewed_entry":
        path = tmp_path / "baseline_native_entry.py"
        original = next(Path(n) for n in plan["files"] if Path(n).name == path.name)
        path.write_bytes(original.read_bytes() + b"\n# unreviewed source\n")
        del plan["files"][str(original)]
        plan["files"][str(path)] = sha(path)
    else:
        key = next(
            n for n in plan["files"] if Path(n).name == "reference_capture_profile.py"
        )
        if change == "missing_validator":
            del plan["files"][key]
        else:
            plan["files"][key] = "0" * 64
    with pytest.raises(ValueError):
        bind_client(plan, dict(id="submission-v2"), json.loads(PROTOCOL.read_bytes()))
