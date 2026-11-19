import ast
import sys
from pathlib import Path


def test_downstream_producer_record_consumer_is_outside_reused_operator_import_closure():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from reference_qualification_reuse import OPERATOR_INDEPENDENT
    excluded = {Path(name).stem for name in OPERATOR_INDEPENDENT}
    directories = [root / "third-party-test-kit", root / "third-party-test-kit/proc_capture", root / "scripts"]
    # GDN drivers load these two reference files and the guard through the
    # hash-pinned plan's importlib paths; include those declared dynamic roots.
    pending = ["qualify_gdn_reference", "qualify_gdn_prefill_reference", "qualify_route_capture",
               "gdn_reference", "gdn_prefill_reference", "resource_guard"]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in {"draft_producer_geometry", "third_party_report"}
        assert module not in excluded
        paths = [d / (module + ".py") for d in directories]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            continue
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split(".")[0])
    assert {"gdn_reference", "gdn_prefill_reference", "m01_route_device"} <= visited


def test_current_report_is_outside_the_native_clock_and_observation_import_closure():
    root = Path(__file__).resolve().parents[1]
    directories = [root / "scripts", root / "third-party-test-kit", root / "third-party-test-kit/proc_capture"]
    pending = ["run_native_reference", "native_timing_entry", "native_timing_prepare",
        "baseline_native_prepare", "baseline_native_entry", "native_timing_session", "native_timing_worker",
        "reference_observation"]
    assert all((root / "scripts" / (module + ".py")).is_file() for module in pending)
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in {"third_party_report", "reference_observation_reuse"}
        path = next((d / (module + ".py") for d in directories if (d / (module + ".py")).exists()), None)
        if path is None:
            continue
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split(".")[0])
    assert {"native_public_client", "runtime_identity", "native_timing_session"} <= visited
