"""Version and audit the HLE judge's structural JSON-whitespace policy."""

import copy
import json

BOUNDED = "xgrammar-bounded-whitespace-confidence-v2"
WHITESPACE_ONLY = "xgrammar-bounded-whitespace-v1"
LEGACY = "legacy-unrestricted-whitespace-v1"
CONFIG = {"backend": "xgrammar", "disable_any_whitespace": True}


def selected(plan):
    # Historical plans predate this choice and must retain their original policy.
    value = plan.get("hle_judge_whitespace", "unrestricted")
    choices = {
        "bounded-v2": BOUNDED,
        "bounded-v1": WHITESPACE_ONLY,
        "unrestricted": LEGACY,
    }
    if value not in choices:
        raise ValueError("Unknown HLE judge whitespace policy")
    return choices[value]


def requested(spec):
    policy = spec.get("structured_output_policy", LEGACY)
    if policy not in (BOUNDED, WHITESPACE_ONLY, LEGACY):
        raise ValueError("Unknown structured-output policy")
    if "structured_output_policy" in spec and (
        spec["dataset"],
        spec["stage"],
        spec["service_role"],
    ) != ("hle", "judge", "B1"):
        raise ValueError("Structured-output policy belongs to the HLE B1 judge only")
    return policy


def effective_format(spec, original):
    """Mirror the existing strict parser's integer0..100 domain in generation."""
    policy = requested(spec)
    result = copy.deepcopy(original)
    if policy == BOUNDED:
        schema = result["json_schema"]
        confidence = schema["schema"]["properties"]["confidence"]
        if schema["name"] != "ExtractedAnswer" or confidence != {
            "type": "integer",
            "title": "Confidence",
        }:
            raise ValueError("Unexpected original HLE confidence schema")
        confidence.update(minimum=0, maximum=100)
    return result


def configure(spec, command, plan):
    policy = requested(spec)
    if (spec["dataset"], spec["stage"]) != ("hle", "judge"):
        return
    plan["structured_output_policy"] = policy
    if policy in (BOUNDED, WHITESPACE_ONLY):
        if "--structured-outputs-config" in command:
            raise ValueError("Duplicate structured-output configuration")
        command += ["--structured-outputs-config", json.dumps(CONFIG)]
    plan["judge_whitespace_scope"] = (
        "Bound structural JSON whitespace; preserve whitespace inside string values, "
        "the original prompt/seed/budget, string content and strict grade parser."
        if policy in (BOUNDED, WHITESPACE_ONLY)
        else "Explicit original unrestricted JSON-whitespace policy."
    )
    plan["deployment_difference"] = (
        "HLE substitute judge JSON-whitespace policy: "
        + policy
        + "; strict parsing, prompts, budgets and B1 r4 optimization combination preserved."
    )
    if policy == BOUNDED:
        plan["confidence_generation_bounds"] = [0, 100]
        plan["deployment_difference"] += (
            " Generation schema mirrors the existing strict parser's integer0..100 confidence domain."
        )


def audit(spec, service_plan, launch):
    policy = requested(spec)
    if (spec["dataset"], spec["stage"]) != ("hle", "judge"):
        return None
    declared = service_plan.get("structured_output_policy", LEGACY)
    if declared != policy:
        raise ValueError("Actual judge serving plan has a different whitespace policy")
    command = launch["command"]
    flags = [
        i for i, value in enumerate(command) if value == "--structured-outputs-config"
    ]
    configs = [json.loads(command[i + 1]) for i in flags]
    if (policy in (BOUNDED, WHITESPACE_ONLY) and configs != [CONFIG]) or (
        policy == LEGACY and any(c.get("disable_any_whitespace") for c in configs)
    ):
        raise ValueError("Actual judge launch differs from frozen whitespace policy")
    result = {
        "policy": policy,
        "backend_configuration_verified": True,
        "scope": "Serving grammar and versioned confidence generation domain; answer strings and strict grade parsing unchanged",
    }
    if policy == BOUNDED:
        if service_plan.get("confidence_generation_bounds") != [0, 100]:
            raise ValueError("Actual judge plan omits frozen confidence bounds")
        result["confidence_generation_bounds"] = [0, 100]
        result["confidence_scope"] = (
            "Same integer0..100 domain as the unchanged strict parser"
        )
    return result
