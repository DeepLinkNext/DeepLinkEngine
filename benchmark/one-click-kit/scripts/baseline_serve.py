"""Record actual image-package imports, then execute the frozen018 service entry."""

import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    plan_file = Path(sys.argv[1])
    plan = json.loads(plan_file.read_bytes())
    import torch
    import torch_npu
    import vllm
    import vllm_ascend

    packages = {}
    for name, module in (
        ("torch", torch),
        ("torch_npu", torch_npu),
        ("vllm", vllm),
        ("vllm_ascend", vllm_ascend),
    ):
        path = Path(module.__file__).resolve()
        packages[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    record = {
        "packages": packages,
        "python": sys.executable,
        "prefix": sys.prefix,
        "no_site": sys.flags.no_site,
        "baseline_variant": plan["baseline_variant"],
    }
    with (Path(plan["output"]) / "actual-package-imports.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    original = Path(plan["deployment"]) / "runtime/entrypoints/serve_b0_r2.py"
    if (
        hashlib.sha256(original.read_bytes()).hexdigest()
        != plan["frozen_serving_entry_sha256"]
    ):
        raise ValueError("Frozen original018 serving entry changed")
    runpy.run_path(str(original), run_name="__main__")


if __name__ == "__main__":
    main()
