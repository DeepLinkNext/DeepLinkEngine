"""Capture actual SSE timings for the frozen quality clients, without retries."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def clock():
    return {"monotonic": time.monotonic(), "wall": time.time()}


def streamed_body(original, config, messages):
    body = original(config, messages)
    body.update(stream=True, stream_options={"include_usage": True})
    # Distinct request salts prohibit borrowing prior task KV states. These
    # transport identifiers do not change the problem, seed or sampling.
    body["cache_salt"] = uuid.uuid4().hex
    return body


def call(config, body, directory):
    if body.get("stream") is not True or body.get("stream_options") != {
        "include_usage": True
    }:
        raise ValueError("Frozen stream transport differs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    capture = directory / (uuid.uuid4().hex + ".sse.jsonl.gz")
    started = clock()
    record = {
        "request": body,
        "started_unix": started["wall"],
        "start": started,
        "status": "transport_error",
        "stream_capture": str(capture),
        "first_token": None,
        "first_output_token": None,
        "last_token": None,
        "response_reconstructed_from_stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if config.get("api_key_env"):
        token = os.environ.get(config["api_key_env"])
        if not token:
            raise ValueError("Configured API key is unset")
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        config["base_url"] + "/v1/chat/completions", json.dumps(body).encode(), headers
    )
    content, reasoning, usage, finish, done = "", "", None, None, False
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (
            opener.open(request, timeout=config["timeout"]) as response,
            gzip.open(capture, "wt") as wire,
        ):
            record["http_status"] = response.status
            for line in response:
                stamp = clock()
                raw = line.decode("utf-8")
                wire.write(json.dumps({"clock": stamp, "raw": raw}) + "\n")
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError("SSE error in retained stream")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    content += text
                    reasoning += (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if text or delta.get("reasoning_content") or delta.get("reasoning"):
                        record["first_output_token"] = record["first_output_token"] or stamp
                    if text:
                        record["first_token"] = record["first_token"] or stamp
                        record["last_token"] = stamp
                    finish = choice.get("finish_reason") or finish
        if (
            record["http_status"] != 200
            or not done
            or finish not in ("stop", "length")
            or not (content.strip() or reasoning.strip())
            or usage is None
        ):
            raise ValueError("Incomplete stream, empty answer or missing usage")
        if not 0 < usage.get("completion_tokens", 0) <= body["max_tokens"]:
            raise ValueError("Actual completion usage differs from frozen budget")
        result = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }
        record.update(
            status="ok",
            content=content,
            reasoning=reasoning,
            usage=usage,
            finish_reason=finish,
            response=result,
            raw_response=json.dumps(result, ensure_ascii=False),
        )
    except urllib.error.HTTPError as error:
        record.update(
            http_status=error.code,
            raw_response=error.read().decode(errors="replace"),
            error=str(error),
        )
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
    record["end"] = clock()
    record["elapsed_s"] = record["end"]["monotonic"] - started["monotonic"]
    record["stream_sha256"] = (
        hashlib.sha256(capture.read_bytes()).hexdigest() if capture.exists() else None
    )
    return record
