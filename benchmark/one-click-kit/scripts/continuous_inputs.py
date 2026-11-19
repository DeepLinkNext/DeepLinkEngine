"""Execute the frozen input-building prefix without launching historical plans."""

import ast
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(deployment, out):
    source = deployment / "runtime/entrypoints/prepare_continuous.py"
    out.mkdir(parents=True, exist_ok=False)
    local = out / source.name
    shutil.copyfile(source, local)
    tree = ast.parse(source.read_bytes())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            break
        nodes.append(node)
    code = ast.Module(body=nodes, type_ignores=[])
    exec(compile(code, str(local), "exec"), {"__file__": str(local)})
    path = out / "continuous-inputs-r1.json"
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "New inputs rebuilt from the frozen original recipe and tokenizer; historical input-file byte identity is not asserted.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    # Runtime libraries can emit warnings to stdout without a newline. Keep
    # the trusted result on a separate file channel and preserve all logging.
    # Claim the result before running any input preparation, rejecting reuse.
    with args.result.open("x") as stream:
        result = prepare(args.deployment, args.out)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
