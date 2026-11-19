"""Versioned compute-capture budgets; independent of full-answer quality tests."""

import hashlib
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/compute-capture-profiles-v2.json"
PROTOCOL_SHA256 = "4ade0ae036c53b385168e1728f369cb65c36f55aaac45e8009b8aed10ed0d6b1"


def profile(name="submission-v2"):
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Compute capture budget contract changed")
    document = json.loads(PROTOCOL.read_bytes())
    if not isinstance(name, str) or name not in document["profiles"]:
        raise ValueError("Unknown compute reference profile: " + repr(name))
    return dict(id=name, **document["profiles"][name])


def validate_native_plan(plan):
    # Missing identity belongs to the previously frozen full-answer protocol.
    capture = profile(plan.get("reference_capture_profile", "full-answer-v1"))
    expected = dict(max_tokens=capture["max_tokens"], temperature=0, seed=0,
        ignore_eos=capture["ignore_eos"], top_p=1.0, top_k=-1,
        presence_penalty=0.0, detokenize=False)
    if (type(plan["max_tokens"]) is not int or plan["max_tokens"] != capture["max_tokens"]
            or type(plan.get("formal_repetitions", 1)) is not int
            or plan.get("formal_repetitions", 1) != capture["formal_repetitions"]
            or plan["sampling"] != expected
            or any(type(plan["sampling"][key]) is not type(value) for key, value in expected.items())):
        raise ValueError("Native sampling or repetitions differ from the frozen compute profile")
    return capture


def repetitions(plan):
    capture = plan.get("capture_profile") or profile("full-answer-v1")
    if capture != profile(capture["id"]):
        raise ValueError("Automatic reference capture controls differ from their contract")
    return range(1, capture["formal_repetitions"] + 1)
