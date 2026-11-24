"""Launch the official 4.1 controller with the source release and current audited client."""

import json
import os
from pathlib import Path
import sys


def main(entry, arguments):
    entry = Path(entry).resolve()
    sys.path.insert(0, str(entry.parent.parent if entry.parent.name == "benchmark" else entry.parent))
    from benchmark.evaluation_client import prepare
    from benchmark.one_click import option, validate_deployment
    from engine.environment_setup import cached_wheel
    from engine.release_selection import METADATA_CLIENT, SELECTION_SHA, active, metadata

    selection = active()
    if selection is None:
        raise ValueError("4.1 补跑必须绑定来源批次的引擎版本")
    validate_deployment(arguments)
    cache = Path(option(arguments, "--cache"))
    archive = cached_wheel(selection["sdist"], [], cache / "wheels")
    frozen = metadata(archive, selection["version"])
    if frozen != selection["frozen"]:
        raise ValueError("来源批次的引擎元数据与发行包不一致")
    client = prepare(cache, archive, frozen)
    os.environ[METADATA_CLIENT] = json.dumps({
        "selection_sha256": os.environ[SELECTION_SHA], "client": client,
    })
    print(f'4.1 来源引擎 {selection["version"]} · 评测客户端 {client["manifest_sha256"]}', flush=True)
    os.execv(sys.executable, [sys.executable, "-B", str(entry), *arguments])


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2:])
    except (KeyError, OSError, ValueError) as exc:
        print(f"4.1 补跑启动失败：{exc}", file=sys.stderr)
        sys.exit(2)
