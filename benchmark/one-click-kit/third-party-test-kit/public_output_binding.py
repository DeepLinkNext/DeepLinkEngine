"""Narrow reviewed r8 publication-source binding, without device imports.

Whole-file pins retain the reviewed transport, scheduler and output-processor
branches. This qualifies public emission grouping only. Cache ownership,
MTP producer geometry, arithmetic and reference timing are separate audits.
"""

import hashlib
import json
from pathlib import Path


PINS = {
    "runtime/engine-core/vllm/v1/engine/llm_engine.py": "d4d5fba309db4d3f271ac7479e3ddaa623045499e4c6cfac41c0c7710012413c",
    "runtime/engine-core/vllm/v1/engine/core_client.py": "f9a721c21088ea2f38d74139c78bab2eb6b535b561724ea2cd06ba30497f283a",
    "runtime/engine-core/vllm/v1/engine/output_processor.py": "1a96ed5e41bf3df9f16ccba6e721a0bbda5baa500b3ff3117fe75679292b3bfc",
    "runtime/engine-core/vllm/config/scheduler.py": "01e7efd19ab03512d03656570296f5f392775b1f926745e229209c3c81def9e1",
    "runtime/engine-core/vllm/v1/core/sched/scheduler.py": "50e7cea25eea05fa192bdd99f30e990f45c499eb327c42b247bf61220d7b64b7",
    "runtime/engine-core/vllm/v1/core/sched/async_scheduler.py": "e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508",
    "runtime/engine-core/vllm/v1/engine/core.py": "bc12ad6e1a03f242e322e0506462c456c584dd3346f1a11413a9f51094c2049e",
    "mods/betterscale/betterscale/models/qwen35/seat_scheduler.py": "37c6308e6ec569aa81dbd3f1affa3b64692b48fada7504b51e6e1ecbbcb41400",
}
CLIENT_PINS = {
    "native_timing_session.py": "5cae8a52370c3586b3e688913515eb839d5ac395436c34998dd064f98990d414",
    "native_public_client.py": "f7359cffb34f60cd04ca72372c26f550f1ee4119a5d33d2003146da62f8045f3",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    return left == right


def validate_effective_contract(arguments, plan):
    expected = dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
        stream_interval=1, async_scheduling=True, disable_log_stats=True,
        per_request_spec_decode_metrics="none",
        scheduler_cls="betterscale.models.qwen35.seat_scheduler.LiveStateScheduler",
        speculative_config=dict(method="mtp", num_speculative_tokens=2))
    for key, value in expected.items():
        if key not in arguments or not exact(arguments[key], value):
            raise ValueError("Unreviewed native publication configuration: " + key)
    from reference_capture_profile import validate_native_plan

    validate_native_plan(plan)
    if (plan["engine_release"] != "full-answer-b1-20261117-r8"
            or plan["schema"] != "current-native-m01-timing/v1"
):
        raise ValueError("Unreviewed native input/output contract")


def bind(deployment, native_plan_path, effective_arguments_path):
    deployment = Path(deployment).resolve()
    native_plan_path, effective_arguments_path = Path(native_plan_path), Path(effective_arguments_path)
    plan = json.loads(native_plan_path.read_bytes())
    validate_effective_contract(json.loads(effective_arguments_path.read_bytes()), plan)
    if Path(plan["deployment"]).resolve() != deployment:
        raise ValueError("Native plan belongs to another deployment")
    installed = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    for relative, pin in PINS.items():
        path = deployment / relative
        if installed.get(relative) != pin or digest(path) != pin:
            raise ValueError("Reviewed public publication source differs: " + relative)
    from native_public_client_domain import bind as bind_clients
    from reference_capture_profile import validate_native_plan

    clients = bind_clients(plan, validate_native_plan(plan))
    return dict(schema="reviewed-r8-public-output-source-binding/v1",
        native_plan_sha256=digest(native_plan_path),
        effective_arguments_sha256=digest(effective_arguments_path),
        deployment_source_sha256=PINS, client_source_sha256=clients,
        uncoalesced_public_delivery_source_qualified=True,
        source_claims=[
            "EngineCore.step_with_batch_queue consumes one oldest model result and invokes one scheduler update.",
            "LiveStateScheduler updates its frontier then returns the inherited scheduler outputs without token regrouping.",
            "Scheduler computes raw acceptance before EOS/length trimming and emits that request's trimmed token list once.",
            "EngineCoreProc queues each per-client EngineCoreOutputs object separately; transport decodes one message per queue item.",
            "SyncMPClient.get_output and LLMEngine.step consume one item and process its outputs once.",
            "RequestState uses effective stream_interval=1; DELTA passes that delivery's token list without cumulative slicing.",
            "The pinned native session selects DELTA, one completion, detokenize=False and no stop strings.",
            "Under fixed nonstale MTP2 delivery, nonterminal c has c-1 accepted drafts; terminal retained drafts lie in [c-1,min(2,c)]."],
        additional_conditions=["Fixed nonstale MTP2 producer path", "No necessary-prefix preemption/recomputation",
                               "Fresh request-owned target/draft cache"],
        necessary_work_certified=False, full_model_certificate=False, efficiency_percent=None)
