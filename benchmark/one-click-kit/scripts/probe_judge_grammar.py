"""CPU acceptance checks for the frozen HLE JSON schema; no model generation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from judge_grammar import effective_format, BOUNDED


def probe(contract_path):
    import xgrammar as xgr

    contract = json.loads(contract_path.read_bytes())
    schema = contract["judge_response_format"]["json_schema"]["schema"]
    bounded_schema = effective_format(
        {
            "dataset": "hle",
            "stage": "judge",
            "service_role": "B1",
            "structured_output_policy": BOUNDED,
        },
        contract["judge_response_format"],
    )["json_schema"]["schema"]
    sample = {}
    for key, field in schema["properties"].items():
        if "const" in field:
            sample[key] = field["const"]
        elif "enum" in field:
            sample[key] = field["enum"][0]
        elif field["type"] == "string":
            sample[key] = "New York and a normal explanation with spaces."
        elif field["type"] == "integer":
            sample[key] = 50
        else:
            raise ValueError("Unrecognized frozen HLE schema property: " + key)
    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    tokenizer = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[256])
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=2)
    compiled = {
        policy: compiler.compile_json_schema(
            json.dumps(bounded_schema if policy == "bounded-v2" else schema),
            any_whitespace=policy == "unrestricted",
        )
        for policy in ("unrestricted", "bounded-v1", "bounded-v2")
    }
    text = json.dumps(sample, ensure_ascii=False)
    marker = '"strict":'
    prefix = text[: text.index(marker) + len(marker)]
    cases = {
        "valid_string_spaces": text,
        "repeated_tabs_after_strict": prefix + " \t\t\t",
        "repeated_spaces_after_strict": prefix + " " * 16,
        "repeated_newlines_after_strict": prefix + "\n" * 16,
        "confidence_zero": text.replace('"confidence": 50', '"confidence": 0'),
        "confidence100": text.replace('"confidence": 50', '"confidence": 100'),
        "confidence101": text.replace('"confidence": 50', '"confidence": 101'),
        "confidence_negative": text.replace('"confidence": 50', '"confidence": -1'),
        "confidence_zero_loop": text.replace(
            '"confidence": 50', '"confidence": 1' + "0" * 16
        ),
    }
    results = {
        name: {
            policy: xgr.GrammarMatcher(grammar).accept_string(value)
            for policy, grammar in compiled.items()
        }
        for name, value in cases.items()
    }
    for name, accepted in results.items():
        expected = {
            "unrestricted": True,
            "bounded-v1": not name.startswith("repeated_"),
            "bounded-v2": name
            in ("valid_string_spaces", "confidence_zero", "confidence100"),
        }
        if accepted != expected:
            raise ValueError("Grammar acceptance check failed: " + name)
    return {
        "schema": "hle-structural-whitespace-grammar-probe/v1",
        "kind": "CPU grammar acceptance only; not model generation or a quality score",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "judge_schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest(),
        "effective_judge_schema_sha256": hashlib.sha256(
            json.dumps(bounded_schema, sort_keys=True).encode()
        ).hexdigest(),
        "xgrammar_version": importlib.metadata.version("xgrammar"),
        "tokenizer": "Synthetic byte vocabulary for grammar acceptance, no model logits",
        "results": results,
        "new_native_inference": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.contract)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
