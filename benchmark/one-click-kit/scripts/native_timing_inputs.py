"""Prepare and hash-check the current HTTP token population for native timing.

All chat formatting and tokenization finish before the model timing client.
Each formal and synthetic prompt must reproduce its frozen HTTP token hash.
"""

import copy
import hashlib
import json
from pathlib import Path

from dataset_contract import save, sha
from serving_warmup import DENSE_PROMPT, warmup_cohorts, request_count


def normalized_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        content = message.get("content")
        if isinstance(content, list):
            if any(set(part) != {"type", "text"} or part["type"] != "text" for part in content):
                raise ValueError("Native M01 input must contain text parts only")
            message["content"] = "\n".join(part["text"] for part in content)
        elif not isinstance(content, str):
            raise ValueError("Native M01 message content must be text")
    return result


def tokenize(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(normalized_messages(messages), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Native tokenization returned an invalid prompt")
    return tokens


def check(tokens, expected):
    digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    count = expected.get("count", expected.get("input_tokens"))
    if count != len(tokens) or digest != expected["token_ids_sha256"]:
        raise ValueError("Native prompt IDs differ from the original HTTP population")


def prepare(deployment, model, qualified, output):
    from transformers import AutoTokenizer

    deployment, model, qualified, output = map(Path, (deployment, model, qualified, output))
    sources = dict(quality=deployment / "inputs/mmlu-quality018-r2/quality-request-plan.json",
                   expected=deployment / "inputs/mmlu-quality018-r2/quality-tokenize-preflight.json",
                   synthetic=qualified / "service/request-plan.json")
    quality, expected, synthetic = (json.loads(sources[key].read_bytes()) for key in ("quality", "expected", "synthetic"))
    requests = quality["requests"]
    if (len(requests) != 200 or len(expected) != 200 or quality["max_completion_tokens"] != 8192
            or quality["temperature"] != 0 or quality["seed"] != 0):
        raise ValueError("Native input preparation requires the fixed200/max8192 source")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    formal = []
    for request, identity in zip(requests, expected, strict=True):
        if (request["max_completion_tokens"] != 8192 or request["temperature"] != 0 or request["seed"] != 0):
            raise ValueError("Per-request native sampling differs")
        tokens = tokenize(tokenizer, request["messages"])
        check(tokens, identity)
        formal.append(tokens)
    if sum(map(len, formal)) != 282265:
        raise ValueError("Formal native prompt-token total differs")
    if set(synthetic) != {"4096", "16384"} or any(len(rows) != 8 for rows in synthetic.values()):
        raise ValueError("Native warmup needs the two original C8 synthetic cohorts")
    prepared = {}
    for minimum, cases in synthetic.items():
        prepared[minimum] = []
        for case in cases:
            tokens = tokenize(tokenizer, case["request"]["messages"])
            check(tokens, case)
            if len(tokens) < int(minimum):
                raise ValueError("Native warmup prompt is below its frozen cohort")
            prepared[minimum].append(tokens)
    dense = tokenize(tokenizer, [dict(role="user", content=DENSE_PROMPT)])
    if not 0 < len(dense) <= 128 or len(dense) * 36 > 4096:
        raise ValueError("Native dense warmup cannot fit the frozen4096-token scheduler budget")
    cohorts = list(warmup_cohorts(prepared, dense, capacity36=True))
    if sum(map(len, cohorts)) != request_count("task", capacity36=True):
        raise ValueError("Native synthetic warmup population differs")
    result = dict(schema="current-native-m01-token-workload/v1", request_token_ids=formal,
                  warmup_cohorts=cohorts, formal_questions_used_for_warmup=False,
                  current_http_token_hashes_matched=True, source_files={str(p.resolve()): sha(p) for p in sources.values()},
                  tokenizer_files={name: sha(model / name) for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")})
    save(output, result)
    return result
