"""Versioned, exact native client readback; legacy clients only admit full-answer."""

import hashlib
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/native-public-client-source-domain-v1.json"
PROTOCOL_SHA256 = "01d8008020a70b9294391d8f9a8463114c74c424ae418f70bec9f0ae065f2132"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(plan, capture):
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Native public client source-domain protocol changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    result = {}
    for filename, versions in protocol["sources"].items():
        matches = [(Path(name), pin) for name, pin in plan["files"].items()
                   if Path(name).name == filename]
        if len(matches) != 1:
            raise ValueError("Exactly one reviewed native client required: " + filename)
        path, pin = matches[0]
        if capture["id"] not in versions.get(pin, []) or sha(path) != pin:
            raise ValueError("Native public client is outside its reviewed capture profile: " + filename)
        result[filename] = pin
    return result
