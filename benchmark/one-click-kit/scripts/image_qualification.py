"""Two fixed synthetic image-interface checks, separate from dataset scoring."""

import base64
import json
from pathlib import Path
import re
import struct
import zlib

from streaming_transport import call


def image_url(color):
    rgb = {"red": (255, 0, 0), "blue": (0, 0, 255)}[color]

    def chunk(kind, raw):
        return (
            struct.pack(">I", len(raw))
            + kind
            + raw
            + struct.pack(">I", zlib.crc32(kind + raw))
        )

    scanlines = (b"\0" + bytes(rgb) * 64) * 64
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def qualify(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for color in ("red", "blue"):
        body = {
            "model": config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color fills the square in the image? Answer with exactly one English color word.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url(color)}},
                    ],
                }
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 32,
            "temperature": 0,
            "seed": 17,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": "image-interface-" + out.parent.name + "-" + color,
        }
        row = call(config, body, out / "transport")
        row["expected"] = color
        row["passed"] = (
            row["status"] == "ok"
            and re.sub(r"[^a-z]", "", row.get("content", "").lower()) == color
        )
        (out / (color + ".json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n"
        )
        results.append(row["passed"])
    if not all(results):
        raise ValueError(
            "Synthetic red/blue image qualification failed; no HLE tasks submitted"
        )
    return {
        "passed": True,
        "requests": 2,
        "scope": "Synthetic image interface only; excluded from task score and solver performance. Does not establish full HLE quality.",
    }
