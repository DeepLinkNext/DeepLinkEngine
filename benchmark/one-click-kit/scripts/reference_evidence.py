"""Scientific reference evidence excludes per-attempt compiler caches.

Old controller envelopes accidentally included those caches, including B0's
text-model symlinks to external checkpoint shards. Primitive owned-stage seals
already excluded compiler-cache. Admit their scientific records unchanged;
do not dereference or require transient cache entries from a legacy envelope.
"""

import os
import ast
import json
import re
from pathlib import Path, PurePosixPath

from dataset_contract import sha

TRANSIENT_CACHE_DIRECTORIES = {"compiler-cache", "http-current-work-compiler-cache"}


def transient_cache(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts
        and any(part in TRANSIENT_CACHE_DIRECTORIES for part in path.parts))


def evidence_manifest(root):
    root = Path(root)
    result = {}
    for directory, children, files in os.walk(root, followlinks=False):
        # Prune before traversal: cache trees can contain many thousands of
        # files and model symlinks; none is a source measurement record.
        children[:] = sorted(name for name in children if name not in TRANSIENT_CACHE_DIRECTORIES)
        for name in sorted(files):
            path = Path(directory) / name
            if path == root / "checksums.json" or not path.is_file():
                continue
            result[str(path.relative_to(root))] = sha(path)
    return result


def sealed_file_matches(root, name, pin, seal, view_context=None):
    """Check a scientific file, including a bound read-only auditor input view.

    Only an exact input alias to this same envelope's completed HTTP source is
    admitted. Caches are handled separately by the caller. Metadata is checked
    once per view while every referenced raw file retains its content check.
    """
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    path = root / name
    context = {} if view_context is None else view_context
    try:
        if not path.is_symlink():
            return path.is_file() and path.resolve().is_relative_to(root) and sha(path) == pin
        if (path.parent.name != "raw-view" or path.parent.parent.name != "http-independent-audit"
                or not re.fullmatch(r"(?:scheduler-waves\.jsonl|rank-[01]-(?:waves\.jsonl|drafts\.jsonl|.+-routes\.npz))", path.name)):
            return False
        binding = path.parent.parent / "source-inputs.json"
        binding_pin = seal.get(str(binding.relative_to(root)))
        key = (str(binding), binding_pin)
        if key not in context:
            if binding.is_symlink() or not binding.resolve().is_relative_to(root) or sha(binding) != binding_pin:
                return False
            record = json.loads(binding.read_bytes())
            if not isinstance(record, dict) or not isinstance(record.get("raw_inputs"), dict):
                return False
            source = Path(record["source"])
            if not source.is_absolute() or not source.resolve().is_relative_to(root):
                return False
            source = source.resolve()
            if source != (root / "http-current-work").resolve():
                return False
            source_seal = source / "checksums.json"
            receipt = source / "receipt.json"
            if (source_seal.is_symlink() or receipt.is_symlink()
                    or sha(source_seal) != record["source_checksums_sha256"]):
                return False
            original = json.loads(source_seal.read_bytes())
            if not isinstance(original, dict):
                return False
            if original.get("receipt.json") != sha(receipt):
                return False
            state = json.loads(receipt.read_bytes())
            if not isinstance(state, dict):
                return False
            if (state.get("completed") is not True or state.get("quiescent") is not True
                    or state.get("role") != "B1" or state.get("planned_joint_requests") != 200):
                return False
            context[key] = (record, source, original)
        record, source, original = context[key]
        raw_name = "service/raw-work/" + path.name
        target = source / raw_name
        return (record["raw_inputs"].get(raw_name) == original.get(raw_name) == pin
            and not target.is_symlink() and target.resolve().is_relative_to(source)
            and path.resolve() == target.resolve() and sha(target) == pin)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def http_scope_fix_only(original, current, pin):
    """Admit exactly removal of the redundant HTTP-local sys import.

    Native timing imports this module's memory/admission helpers. Those and
    every other statement must remain bytecode-equivalent in the source AST;
    the defective HTTP controller function is never called by native timing.
    Both old source bytes and the new plan's source pin are checked by caller.
    """
    if sha(original) != pin:
        return False
    old, new = ast.parse(original.read_text()), ast.parse(current.read_text())
    removed = 0
    for function in old.body:
        if isinstance(function, ast.FunctionDef) and function.name == "admitted_run":
            for node in ast.walk(function):
                for field in ("body", "orelse", "finalbody"):
                    children = getattr(node, field, None)
                    if not isinstance(children, list):
                        continue
                    retained = [child for child in children if not (
                        isinstance(child, ast.Import) and len(child.names) == 1
                        and child.names[0].name == "sys" and child.names[0].asname is None)]
                    removed += len(children) - len(retained)
                    setattr(node, field, retained)
    return removed == 1 and ast.dump(old) == ast.dump(new)


def post_capture_fix_only(original, current, pin):
    """Keep every capture statement exact through the successful stop boundary.

    Only finish's CPU consumer tail can differ. In particular configure,
    boundary, SSE counting, imports and all module-level code stay unchanged.
    Native timing never calls finish. A closed observation needs its separate
    independent sealed recomputation before the continuation can reuse it.
    """
    if sha(original) != pin:
        return False
    old, new = ast.parse(original.read_text()), ast.parse(current.read_text())
    found = []
    for tree in (old, new):
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "finish"]
        if len(functions) != 1 or not functions[0].body:
            return False
        function = functions[0]
        expected = ast.parse('stopped = boundary("end")').body[0]
        if ast.dump(function.body[0]) != ast.dump(expected):
            return False
        found.append(function)
        function.body = function.body[:1]
    return ast.dump(old) == ast.dump(new)

