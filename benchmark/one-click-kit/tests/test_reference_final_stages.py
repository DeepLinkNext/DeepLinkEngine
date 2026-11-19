import copy
import fcntl
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reference_final_stages import (
    dense_request,
    validate_native_timing,
    release_owned_model_artifact_cache,
)
from dataset_contract import sha


def inputs():
    fences = [dict(rank=r, device=r) for r in (0, 1)]
    raw = dict(
        start_ns=0,
        end_ns=10**9,
        T_seconds=1.0,
        start_fences=fences,
        end_fences=copy.deepcopy(fences),
        request_ids=list(map(str, range(200))),
        public_step_output_only=True,
        http_inside_timing=False,
        tokenizer_inside_timing=False,
        per_step_engine_internals_read=False,
        per_step_disk_writes=False,
    )
    timing = dict(
        start_ns=0,
        end_ns=10**9,
        T_seconds=1.0,
        repetition=1,
        ordinary_public_outputs_retained=True,
        boundary=[dict(rank=r, sealed=True, failure=None) for r in (0, 1)],
    )
    audit = [dict(rank=r, native_per_step_observer=False) for r in (0, 1)]
    work = dict(
        requests=200,
        necessary_work_certified=True,
        source_runtime_applicability_certified=True,
    )
    return raw, timing, audit, work


def test_both_rank_fences_and_raw_native_interval_match_own_certified_population():
    assert validate_native_timing(*inputs()) == 1.0


@pytest.mark.parametrize(
    "change", ["http", "clock", "rank", "observer", "population", "compilation", "work"]
)
def test_ineligible_native_interval_cannot_be_used_by_the_reference_pair(change):
    raw, timing, audit, work = inputs()
    if change == "http":
        raw["http_inside_timing"] = True
    elif change == "clock":
        raw["T_seconds"] = timing["T_seconds"] = 2.0
    elif change == "rank":
        raw["end_fences"][1]["rank"] = 0
    elif change == "observer":
        audit[0]["native_per_step_observer"] = True
    elif change == "population":
        raw["request_ids"].pop()
    elif change == "compilation":
        timing["boundary"][1]["sealed"] = False
    else:
        work["necessary_work_certified"] = False
    with pytest.raises(ValueError):
        validate_native_timing(raw, timing, audit, work)


def test_canonical_domain_is_prospective_complete_and_independent_of_performance(
    tmp_path,
):
    files = [
        "runtime/engine-core/vllm/model_executor/models/qwen3_next.py",
        "runtime/engine-core/vllm/model_executor/models/qwen3_5.py",
        "runtime/engine-core/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "runtime/engine-core/vllm/model_executor/layers/linear.py",
        "runtime/ascend/vllm_ascend/ops/linear.py",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    request = dense_request(tmp_path)
    assert len(request["cases"]) == 432
    assert request["cases"][0] == dict(case=0, m=1, n=32, k=2048)
    assert request["cases"][-1] == dict(case=431, m=108, n=6144, k=2048)
    assert all(set(c) == {"case", "m", "n", "k"} for c in request["cases"])


def model_artifacts(root):
    for name in (
        "native-baseline-timing",
        "native-current-timing",
        "http-current-work",
    ):
        folder = root / name
        folder.mkdir()
        (folder / "receipt.json").write_text(
            json.dumps(dict(completed=True, quiescent=True))
        )
        (folder / "data.bin").write_bytes(b"immutable data" * 100000)
        (folder / "checksums.json").write_text(
            json.dumps({p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        )


def test_only_hash_verified_completed_owned_artifact_descriptors_receive_read_only_hints(
    tmp_path, monkeypatch
):
    model_artifacts(tmp_path)
    hints = []

    def advise(fd, offset, length, advice):
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert (offset, length, advice) == (0, 0, os.POSIX_FADV_DONTNEED)
        hints.append(fd)

    monkeypatch.setattr(os, "posix_fadvise", advise)
    result = release_owned_model_artifact_cache(tmp_path)
    assert len(hints) == len(result["files"]) == 3
    assert result["model_services_quiescent"]
    assert not result["file_content_written"] and not result["global_drop_caches"]
    assert (
        tmp_path / "http-current-work/data.bin"
    ).read_bytes() == b"immutable data" * 100000


def test_changed_owned_artifact_never_receives_a_cache_release_hint(
    tmp_path, monkeypatch
):
    model_artifacts(tmp_path)
    (tmp_path / "native-baseline-timing/data.bin").write_bytes(b"changed data" * 100000)
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda *args: pytest.fail("Changed artifact must not receive a hint"),
    )
    with pytest.raises(ValueError, match="changed before cache release"):
        release_owned_model_artifact_cache(tmp_path)


def canonical_program_fixture(root):
    from canonical_protocol import prepare, protocol

    prepared = prepare(
        root / "sampler",
        "dense",
        [dict(case=0, m=1, n=32, k=2048)],
        {"source.py": "a" * 64},
    )
    (root / "checksums.json").write_text(
        json.dumps(
            {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
        )
    )
    row, _ = protocol("dense")
    plan = json.loads((root / "sampler" / row["original_plan"]).read_bytes())
    return prepared, row, plan


def test_canonical_reuse_requires_original_sampler_bytes_and_protocol_identity(
    tmp_path,
):
    from reference_final_stages import validate_canonical_program

    prepared, row, plan = canonical_program_fixture(tmp_path)
    validate_canonical_program(tmp_path, prepared, row, plan)
    program = tmp_path / "sampler" / row["sampler"]
    program.write_bytes(program.read_bytes() + b"\n# substituted program\n")
    seal = json.loads((tmp_path / "checksums.json").read_bytes())
    seal["sampler/" + row["sampler"]] = sha(program)
    (tmp_path / "checksums.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="sampler bytes or protocol identity"):
        validate_canonical_program(tmp_path, prepared, row, plan)


@pytest.mark.parametrize(
    "field",
    [
        "numerical_or_statistical_thresholds_relaxed",
        "historical_counts_or_timings_reused",
        "original_protocol_plan_sha256",
    ],
)
def test_canonical_reuse_rejects_relaxed_or_historical_protocol(tmp_path, field):
    from reference_final_stages import validate_canonical_program

    prepared, row, plan = canonical_program_fixture(tmp_path)
    plan[field] = True if field.endswith(("relaxed", "reused")) else "b" * 64
    with pytest.raises(ValueError, match="sampler bytes or protocol identity"):
        validate_canonical_program(tmp_path, prepared, row, plan)


def matched_fixture(root):
    from dataset_contract import save

    model = root / "model"
    output = root / "certificate"
    output.mkdir()
    save(output / "qualified-dense-calibration.json", dict(fixture=True))
    necessary_arms, graph_arms = {}, {}
    for role, name in [
        ("B0", "native-baseline-timing"),
        ("B1", "native-current-timing"),
    ]:
        native = model / name
        native.mkdir(parents=True)
        raw, timing, audit, work = inputs()
        native_plan = dict(
            max_tokens=8192,
            formal_repetitions=1,
            sampling=dict(
                max_tokens=8192,
                temperature=0,
                seed=0,
                ignore_eos=False,
                top_p=1.0,
                top_k=-1,
                presence_penalty=0.0,
                detokenize=False,
            ),
        )
        files = {
            "receipt.json": dict(formal_repetitions=1),
            "service/native-plan.json": native_plan,
            "service/client/timings.json": dict(repetitions=[timing]),
            "model-weights.json": {"fixture": "same frozen identity"},
            "service/client/formal-1/public-output-geometry.json": dict(
                requests=[dict(input_ids_sha256="a" * 64)] * 200
            ),
            "service/client/formal-1/raw-public-output-frames.json": raw,
            "service/client/formal-1-timing.json": timing,
            "service/client/formal-1-pre-audit.json": audit,
        }
        for name, value in files.items():
            (native / name).parent.mkdir(parents=True, exist_ok=True)
            save(native / name, value)
        save(native / "checksums.json", {name: sha(native / name) for name in files})
        work.update(
            F_interval=[10**12, 2 * 10**12],
            geometry_sha256="b" * 64,
            native_checksums_sha256=sha(native / "checksums.json"),
        )
        save(output / (role + "-necessary-work.json"), work)
        save(output / (role + "-dense-source.json"), dict(fixture=True))
        dense_sha = sha(output / (role + "-dense-source.json"))
        graph = dict(
            geometry_sha256="b" * 64,
            reference_seconds_lower=0.75,
            source_runtime_applicability_certified=True,
            source_proof_sha256=dense_sha,
        )
        save(output / (role + "-reference-lower-graph.json"), graph)
        save(
            output / (role + "-reference-graph-audit.json"),
            dict(graph_integrity_passed=True, reference_model_lower_seconds=0.75),
        )
        necessary_arms[role] = dict(
            necessary_work_sha256=sha(output / (role + "-necessary-work.json")),
            dense_source_sha256=dense_sha,
        )
        graph_arms[role] = dict(
            graph_sha256=sha(output / (role + "-reference-lower-graph.json")),
            independent_audit_sha256=sha(
                output / (role + "-reference-graph-audit.json")
            ),
            reference_seconds_lower=0.75,
        )
    save(
        output / "necessary-work-audit.json", dict(completed=True, arms=necessary_arms)
    )
    save(
        output / "reference-graph-audits.json",
        dict(
            completed=True,
            arms=graph_arms,
            qualified_calibration_sha256=sha(
                output / "qualified-dense-calibration.json"
            ),
        ),
    )
    return model, output


def test_dense_tail_preserves_native_proofs_but_cannot_certify_complete_reference(tmp_path):
    from reference_final_stages import matched_pair

    model, output = matched_fixture(tmp_path)
    result = matched_pair({}, output, model)
    assert result["full_model_certificate"] is False
    assert result["status"] == "incomplete"
    assert result["certificate_kind"] == "partial_dense_projection_diagnostic"
    assert result["efficiency_lower_observed_percent"] is None
    assert result["milestone_met"] is None and result["final_numeric_met"] is None
    for arm in result["arms"].values():
        assert arm["source_and_timing_qualified"] is True
        assert arm["full_model_certificate"] is False
        assert "efficiency_lower_percent" not in arm
        assert "reference_tflops_per_card_upper" not in arm
        assert float(arm["partial_dense_diagnostic"]["efficiency_lower_percent"]) == 75


def test_final_pair_rejects_changed_independent_graph_audit_before_certification(
    tmp_path,
):
    from reference_final_stages import matched_pair

    model, output = matched_fixture(tmp_path)
    path = output / "B1-reference-graph-audit.json"
    value = json.loads(path.read_bytes())
    value["graph_integrity_passed"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source populations differ"):
        matched_pair({}, output, model)
    assert not (output / "matched-reference-pair.json").exists()


def test_cache_release_of_closed_failed_http_requires_its_sealed_independent_recovery(
    tmp_path,
):
    from dataset_contract import save, sha
    from reference_final_stages import release_owned_model_artifact_cache

    origin = tmp_path / "origin"
    output = tmp_path / "new-analysis"
    output.mkdir()
    for name in [
        "native-current-timing",
        "native-baseline-timing",
        "http-current-work",
    ]:
        root = origin / name
        root.mkdir(parents=True)
        save(
            root / "receipt.json",
            dict(
                completed=name != "http-current-work",
                quiescent=True,
                failure=None
                if name != "http-current-work"
                else "closed CPU derivation failure",
            ),
        )
        save(root / "checksums.json", {"receipt.json": sha(root / "receipt.json")})
    with pytest.raises((ValueError, FileNotFoundError)):
        release_owned_model_artifact_cache(output, origin)
    http = origin / "http-current-work"
    independent = output / "http-independent-audit"
    independent.mkdir()
    save(
        independent / "receipt.json",
        dict(
            completed=True,
            original_completed=False,
            original_failure="closed CPU derivation failure",
            all_request_closures_contained=True,
            source=str(http.resolve()),
        ),
    )
    save(
        independent / "source-inputs.json",
        dict(
            source=str(http.resolve()),
            source_checksums_sha256=sha(http / "checksums.json"),
            original_receipt_sha256=sha(http / "receipt.json"),
        ),
    )
    save(
        independent / "checksums.json",
        {n: sha(independent / n) for n in ["receipt.json", "source-inputs.json"]},
    )
    assert (
        release_owned_model_artifact_cache(output, origin)["model_services_quiescent"]
        is True
    )
    (http / "receipt.json").write_text("{}")
    with pytest.raises(ValueError):
        release_owned_model_artifact_cache(output, origin)
