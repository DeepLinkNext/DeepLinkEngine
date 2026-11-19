"""Exercise the original installed SDK manager without a model or device backend."""
import argparse
import contextlib
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import sys
import traceback


def check():
    manager = Path(metadata.distribution("vllm-hust-runtime").locate_file("vllm_hust_runtime/sdk/manager")).resolve()
    if not (manager / "vllm_hust_ext/cli.py").is_file():
        raise ValueError("Installed runtime manager source is missing")
    sys.path.insert(0, str(manager))
    from vllm_hust_ext import cli
    from vllm_hust_ext.attestation import profile
    if not Path(cli.__file__).resolve().is_relative_to(manager):
        raise ValueError("Manager import resolved outside the installed runtime")
    with contextlib.redirect_stdout(io.StringIO()) as output:
        try:
            result = cli.main(["--help"])
        except SystemExit as error:
            result = error.code
    if result not in (None, 0) or "run" not in output.getvalue():
        raise ValueError("Original manager CLI help failed")
    if any(name in sys.modules for name in ("torch", "torch_npu", "vllm", "vllm_ascend")):
        raise ValueError("Manager qualification unexpectedly imported a model framework")
    return dict(verified=True, model_loaded=False, npu_backend_imported=False,
                checks=["original_cli_help", "attestation_import_only"],
                cli_path=str(Path(cli.__file__).resolve()),
                cli_sha256=hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check()
    except Exception:
        result = dict(verified=False, model_loaded=False, error=traceback.format_exc())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["verified"] else 1)
