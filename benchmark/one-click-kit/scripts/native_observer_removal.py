"""Source-pinned, observer-only import transforms for the native M01 clock.

The SDK files are read, never edited. All other AST nodes (including guarded
State copies, causal draft lengths and compiler guards) retain their order.
Install before any frozen serving plugin is imported, including in spawn.
"""

import ast
from collections import Counter
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys


EXPECTED = {
    "kvmat": {"observation": 1, "original_init": 1, "initialize": 1, "request_init": 1},
    "attention": {"event_import": 1, "prefix_event": 1},
    "worker": {"graph_method": 1, "graph_install": 1},
}
LOADED = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(tree):
    return ast.dump(tree, include_attributes=False)


def is_call(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def transform(source, kind):
    if kind not in EXPECTED:
        raise ValueError("Unknown observer-removal family")
    counts = Counter()

    class Remove(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            key = None
            if kind == "kvmat" and is_call(call, "record_observation"):
                if dump(call) != dump(ast.parse("record_observation(observation)").body[0].value):
                    raise ValueError("Unexpected policy observation call")
                key = "observation"
            if kind == "attention" and is_call(call, "event"):
                if not call.args or not isinstance(call.args[0], ast.Constant) or call.args[0].value != "prefix":
                    raise ValueError("Unexpected attention event")
                key = "prefix_event"
            if kind == "worker" and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_install_graph_receipts":
                if dump(call) != dump(ast.parse("self._install_graph_receipts()").body[0].value):
                    raise ValueError("Unexpected graph receipt installer")
                key = "graph_install"
            if key:
                counts[key] += 1
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if kind == "kvmat":
                if dump(node) == dump(ast.parse("original_init = Request.__init__").body[0]):
                    counts["original_init"] += 1
                    return None
                if dump(node) == dump(ast.parse("Request.__init__ = initialize").body[0]):
                    counts["request_init"] += 1
                    return None
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if kind == "kvmat" and node.name == "initialize":
                if (not node.body or dump(node.body[0]) != dump(ast.parse("original_init(self, *args, **kwargs)").body[0])
                        or any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id not in {"original_init", "isinstance", "len", "observe_runtime_event"}
                               for n in ast.walk(node))):
                    raise ValueError("Request initialization contains an unrecognized functional call")
                counts["initialize"] += 1
                return None
            if kind == "worker" and node.name == "_install_graph_receipts":
                counts["graph_method"] += 1
                return None
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if kind == "attention" and node.module == "hust_prefill_microflow.school028" and [(a.name, a.asname) for a in node.names] == [("event", None)]:
                counts["event_import"] += 1
                return None
            return node

    original = ast.parse(source)
    cleaned = ast.fix_missing_locations(Remove().visit(original))
    if dict(counts) != EXPECTED[kind]:
        raise ValueError("Observer-removal AST population differs: " + repr(dict(counts)))
    text = ast.unparse(cleaned) + "\n"
    compile(text, "<observer-removal-audit>", "exec")
    return text, dict(kind=kind, removed=dict(counts), transformed_ast_sha256=digest(dump(cleaned).encode()),
                      transformed_source_sha256=digest(text.encode()), only_declared_observers_removed=True)


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, row):
        super().__init__(fullname, row["path"])
        self.row = row

    def get_code(self, fullname):
        raw = Path(self.path).read_bytes()
        if digest(raw) != self.row["sha256"]:
            raise ValueError("Frozen source changed before native observer removal")
        text, proof = transform(raw, self.row["kind"])
        LOADED[fullname] = dict(proof, source=self.path, source_sha256=digest(raw))
        return compile(text, self.path, "exec", dont_inherit=True)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, rows):
        self.rows = rows

    def find_spec(self, fullname, path=None, target=None):
        row = self.rows.get(fullname)
        if row is None:
            return None
        return importlib.util.spec_from_file_location(fullname, row["path"], loader=Loader(fullname, row))


def install(plan_path):
    plan = json.loads(Path(plan_path).read_bytes())
    rows = plan["observer_sources"]
    if {r["kind"] for r in rows.values()} != set(EXPECTED) or len(rows) != 3:
        raise ValueError("Expected exactly the three qualified observer source families")
    if any(name in sys.modules for name in rows):
        raise ValueError("Native observer removal was installed after frozen imports")
    if any(isinstance(finder, Finder) for finder in sys.meta_path):
        raise ValueError("Native observer-removal import owner already exists")
    for row in rows.values():
        raw = Path(row["path"]).read_bytes()
        if digest(raw) != row["sha256"]:
            raise ValueError("Native observer-removal plan has a foreign source")
        transform(raw, row["kind"])
    sys.meta_path.insert(0, Finder(rows))


def audit_loaded():
    if {r["kind"] for r in LOADED.values()} != set(EXPECTED) or len(LOADED) != 3:
        raise ValueError("Not all native observer removals were imported in this process")
    for name, row in LOADED.items():
        module = sys.modules[name]
        if not isinstance(module.__loader__, Loader) or digest(Path(row["source"]).read_bytes()) != row["source_sha256"]:
            raise ValueError("Native observer-removal module custody changed")
    return dict(modules=dict(LOADED), process_imports_verified=True, sdk_files_edited=False)
