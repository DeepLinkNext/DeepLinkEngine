"""Install the four frozen distributions from verified repository release assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sources():
    from tools.verify_dependency_index import build_index
    runtime = json.loads((ROOT / "engine/r8-lock.json").read_bytes())
    evaluation = json.loads((ROOT / "benchmark/one-click-lock.json").read_bytes())
    index = build_index(runtime, evaluation)
    if index != json.loads((ROOT / "engine/dependencies.json").read_bytes()):
        raise ValueError("Engine dependency index differs from the frozen locks")
    records = [dict(row, filename=row["name"]) for row in index["artifacts"]
               if row["kind"] == "engine"]
    if len(records) != 4:
        raise ValueError("Expected the four frozen engine distributions")
    return records


def install(cache, *, wheelhouse=None, download_only=False):
    from environment_setup import cached_wheel
    from one_click import verify_kit
    verify_kit()
    cache = Path(cache).resolve()
    links = [str(Path(wheelhouse).resolve())] if wheelhouse else []
    if links and not Path(links[0]).is_dir():
        raise ValueError("Engine wheelhouse is not a directory")
    artifacts = []
    for record in sources():
        local = cached_wheel(record, links, cache / "wheels")
        artifacts.append(dict(name=record["name"], path=str(local),
                              bytes=record["bytes"], sha256=record["sha256"]))
    # All four inputs are explicit local files. pip never scans the expanded
    # GitHub HTML page or searches PyPI for the custom vLLM distributions.
    lines = [Path(row["path"]).as_uri() + " --hash=sha256:" + row["sha256"]
             for row in artifacts]
    content = "\n".join(lines) + "\n"
    key = hashlib.sha256(content.encode()).hexdigest()[:16]
    requirements = cache / "engine-install" / (key + ".txt")
    requirements.parent.mkdir(parents=True, exist_ok=True)
    requirements.write_text(content)
    receipt = requirements.with_suffix(".json")
    result = dict(status="downloaded", artifacts=artifacts, requirements=str(requirements))
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    if not download_only:
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        from child_runner import run_stage
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-deps", "--no-compile", "--require-hashes", "-r", str(requirements)]
        completed = run_stage(command)
        result.update(status="installed" if completed.returncode == 0 else "failed",
                      returncode=completed.returncode)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError("Frozen engine installation failed; see " + str(receipt))
    print("ENGINE_ARTIFACTS", receipt, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    try:
        install(args.cache, wheelhouse=args.wheelhouse, download_only=args.download_only)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
