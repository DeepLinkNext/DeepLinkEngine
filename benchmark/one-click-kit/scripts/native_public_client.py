"""Keep ordinary public native outputs in memory inside a synchronized clock.

No scheduler, runner, model or worker per-step fields are read. JSON and disk
serialization happen after the final two-rank fence. Public DELTA output frames
retain engine step boundaries, including empty frames, for a later source audit.
"""

import json
from pathlib import Path
import time


def ranks(rows):
    if (not isinstance(rows, (list, tuple)) or len(rows) != 2
            or any(type(r.get("rank")) is not int for r in rows) or {r["rank"] for r in rows} != {0, 1}):
        raise ValueError("Both unique TP2 rank fences are required")
    return rows


def public_frame(outputs):
    return tuple((row.request_id, bool(row.finished), row.num_cached_tokens,
                  tuple((choice.index, tuple(choice.token_ids), choice.finish_reason, choice.stop_reason)
                        for choice in row.outputs)) for row in outputs)


def timed_public_requests(engine, inputs, sampling, request_ids, *, clock=time.monotonic_ns):
    if not inputs or len(inputs) != len(request_ids) or len(set(request_ids)) != len(inputs):
        raise ValueError("Expected one unique public request ID per input")
    if engine.has_unfinished_requests():
        raise ValueError("Previous native requests are unfinished")
    begin = ranks(engine.collective_rpc("hust_runtime_synchronize", timeout=120))
    frames, admitted = [], []
    started = clock()
    for identity, prompt in zip(request_ids, inputs, strict=True):
        admitted.append(engine.add_request(identity, prompt, sampling))
    while engine.has_unfinished_requests():
        frames.append(public_frame(engine.step()))
    end = ranks(engine.collective_rpc("hust_runtime_synchronize", timeout=120))
    ended = clock()
    # All validation is outside the native model interval.
    if admitted != request_ids or ended <= started:
        raise ValueError("Public native IDs or synchronized timing differ")
    return dict(start_ns=started, end_ns=ended, T_seconds=(ended - started) / 1e9,
                start_fences=begin, end_fences=end, public_output_frames=frames,
                request_ids=admitted, public_step_output_only=True,
                http_inside_timing=False, tokenizer_inside_timing=False,
                per_step_engine_internals_read=False, per_step_disk_writes=False)


def validate_frames(result, prompt_lengths, maximum):
    expected = dict(zip(result["request_ids"], prompt_lengths, strict=True))
    generated = {name: [] for name in expected}
    finished = {}
    for frame in result["public_output_frames"]:
        seen = set()
        for name, done, cached, choices in frame:
            if name not in expected or name in seen or name in finished or cached not in (None, 0):
                raise ValueError("Foreign, duplicate, late or prefix-cached public native output")
            seen.add(name)
            if len(choices) != 1:
                raise ValueError("Native reference requires one public completion")
            index, tokens, reason, stop = choices[0]
            if index != 0 or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("Invalid public native completion tokens")
            generated[name].extend(tokens)
            if not 0 <= len(generated[name]) <= maximum:
                raise ValueError("Native reference completion exceeded the fixed budget")
            if done:
                if reason not in ("stop", "length") or not generated[name]:
                    raise ValueError("Incomplete public native response")
                if reason == "length" and len(generated[name]) != maximum:
                    raise ValueError("Length completion differs from the fixed native budget")
                finished[name] = dict(request_id=name, input_tokens=expected[name],
                                      output_ids=generated[name], finish_reason=reason, stop_reason=stop)
            elif reason is not None:
                raise ValueError("Unfinished native output has a finish reason")
    if set(finished) != set(expected):
        raise ValueError("Missing public native completion")
    return [finished[name] for name in expected]


def save_result(folder, result, prompt_lengths, maximum):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    responses = validate_frames(result, prompt_lengths, maximum)
    for name, value in (("raw-public-output-frames.json", result), ("responses.json", responses)):
        with (folder / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return responses
