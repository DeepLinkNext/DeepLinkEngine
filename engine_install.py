from release_selection import kit_path, runtime_lock, profile as selected_profile
"""Install the four frozen distributions from verified repository release assets."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME_ROOTS = ("vllm", "vllm-ascend", "vllm-hust-engine", "torchvision")
# The frozen SDK imports these manager and FlashLB dependencies outside the wheel metadata.
ADAPTER_REQUIREMENTS = ("platformdirs>=4", "jcs>=0.2.1", "packaging>=24", "cryptography>=3.4",
                        "numba==0.67.0", "llvmlite==0.49.0")


def dependency_audit():
    from packaging.requirements import Requirement
    queue = [(name, frozenset()) for name in RUNTIME_ROOTS]
    seen, observed, issues = set(), {}, []
    for text in ADAPTER_REQUIREMENTS:
        requirement = Requirement(text)
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            issues.append(dict(package="runtime-adapter", requires=text, found=version))
        else:
            queue.append((requirement.name, frozenset(requirement.extras)))
    while queue:
        name, extras = queue.pop()
        key = (name.lower().replace("_", "-"), extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            issues.append(dict(package=name, requires=name, found=None))
            continue
        observed[name] = distribution.version
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra})
                                              for extra in {"", *extras}):
                continue
            try:
                version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                version = None
            if version is None or not requirement.specifier.contains(version, prereleases=True):
                issues.append(dict(package=name, requires=text, found=version))
            else:
                queue.append((requirement.name, frozenset(requirement.extras)))
    return dict(verified=not issues, issues=issues, observed_versions=observed)


def resolve_runtime_dependencies(cache, *, wheelhouse=None):
    """Resolve engine API dependencies while retaining all frozen vendor versions."""
    links = [str(Path(wheelhouse).resolve())] if wheelhouse else []
    if links and not Path(links[0]).is_dir():
        raise ValueError("Runtime wheelhouse is not a directory")
    from packaging.requirements import Requirement
    from environment_setup import vendor_integrity, cached_wheel, distribution_integrity
    import os
    sys.path.insert(0, str(kit_path() / "scripts"))
    from child_runner import run_stage
    from one_click import verify_kit
    verify_kit()
    lock = runtime_lock(ROOT)
    expected = dict(lock["required_python_packages"])
    for row in sources():
        stem = row["filename"].removesuffix(".tar.gz")
        name, version = stem.split("-", 2)[:2]
        expected[name.replace("_", "-")] = version
    for name, version in expected.items():
        if metadata.version(name) != version:
            raise ValueError("Frozen runtime version differs before dependency resolution: " + name)
    for name, source in lock.get("runtime_compatibility_sources", {}).items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = None
        if observed != source["version"] or not distribution_integrity(name)["verified"]:
            wheel = cached_wheel(source, links, Path(cache) / "wheels")
            completed = run_stage([sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check",
                                   "--no-deps", "--ignore-installed", "--no-compile", str(wheel)])
            if completed.returncode:
                raise RuntimeError("Runtime compatibility package installation failed: " + name)
        expected[name] = source["version"]
        if name == "opencv-python-headless":
            expected["numpy"] = "1.26.4"
    folder = Path(cache) / "engine-install"
    folder.mkdir(parents=True, exist_ok=True)
    requirements = sorted(set(ADAPTER_REQUIREMENTS) | {str(Requirement(text)) for name in RUNTIME_ROOTS for text in metadata.requires(name) or []})
    content = "\n".join(requirements) + "\n"
    key = hashlib.sha256(content.encode()).hexdigest()[:16]
    required = folder / ("runtime-" + key + ".txt")
    constraints = folder / ("runtime-" + key + "-constraints.txt")
    required.write_text(content)
    constraints.write_text("\n".join(name + "==" + version for name, version in sorted(expected.items())) + "\n")
    before = dependency_audit()
    if not before["verified"]:
        print("RUNTIME_DEPENDENCIES repairing", len(before["issues"]), "requirements", flush=True)
        command = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check",
                   "--only-binary=:all:", "--no-compile"]
        for wheel_directory in links:
            command += ["--find-links", wheel_directory]
        completed = run_stage([*command, "-c", str(constraints), "-r", str(required)])
        if completed.returncode:
            raise RuntimeError("Engine Python dependency resolution failed: " + str(completed.returncode))
    integrity = vendor_integrity(sys.executable, lock["required_python_packages"], dict(os.environ))
    if not integrity["verified"]:
        source = lock["environment_sources"]["triton-ascend"]
        wheel = cached_wheel(source, links, Path(cache) / "wheels")
        result = run_stage([sys.executable, "-I", "-m", "pip", "install", "--no-deps", "--ignore-installed", "--no-compile", str(wheel)])
        if result.returncode:
            raise RuntimeError("Could not restore Ascend Triton after engine dependency resolution")
        integrity = vendor_integrity(sys.executable, lock["required_python_packages"], dict(os.environ))
    after = dependency_audit()
    compatibility = {name: distribution_integrity(name) for name in lock.get("runtime_compatibility_sources", {})}
    if not after["verified"] or not integrity["verified"] or not all(r["verified"] for r in compatibility.values()):
        raise ValueError("Engine dependency/file verification failed: " + json.dumps(after["issues"]))
    # Metadata can match while inherited native extensions are ABI-incompatible.
    # Probe the FlashLB import and CPU execution as well as the multimodal path
    # before starting a model service; this work is outside benchmark timing.
    smoke = folder / "runtime-import-smoke.json"
    probe = ROOT / "tools/runtime_import_smoke.py"
    completed = run_stage([sys.executable, "-I", "-B", str(probe), "--out", str(smoke)],
                          env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", ASCEND_RT_VISIBLE_DEVICES="",
                                   NUMBA_CACHE_DIR=str(folder / "numba")))
    if completed.returncode:
        raise RuntimeError("Runtime import and CPU operator smoke failed; see " + str(smoke))
    manager_smoke = folder / "manager-import-smoke.json"
    completed = run_stage([sys.executable, "-I", "-B", str(ROOT / "tools/manager_import_smoke.py"),
                           "--out", str(manager_smoke)])
    if completed.returncode:
        raise RuntimeError("Runtime manager entry check failed; see " + str(manager_smoke))
    report = dict(before=before, after=after, vendor_integrity=integrity, compatibility_integrity=compatibility,
                  manager_smoke=json.loads(manager_smoke.read_text()), adapter_requirements=ADAPTER_REQUIREMENTS,
                  import_smoke=json.loads(smoke.read_text()),
                  requirements_sha256=hashlib.sha256(content.encode()).hexdigest(), constraints=expected)
    (folder / "runtime-dependencies.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RUNTIME_DEPENDENCIES_VERIFIED", len(after["observed_versions"]), "distributions", flush=True)
    return report


def sources():
    from release_selection import active
    selected = active()
    if selected and not selected["legacy"]:
        return [selected["sdist"], *selected["runtime_artifacts"]]
    from tools.verify_dependency_index import build_index
    runtime = runtime_lock(ROOT)
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
        sys.path.insert(0, str(kit_path() / "scripts"))
        from child_runner import run_stage
        command = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-deps", "--no-compile", "--force-reinstall"]
        # pip also needs this local source when resolving isolated sdist build requirements.
        for wheel_directory in links:
            command += ["--find-links", wheel_directory]
        command += ["--require-hashes", "-r", str(requirements)]
        completed = run_stage(command)
        result.update(status="installed" if completed.returncode == 0 else "failed",
                      returncode=completed.returncode)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError("Frozen engine installation failed; see " + str(receipt))
        try:
            from package_metadata import prepare
            result["package_metadata"] = prepare(sys.executable)
            result["runtime_dependencies"] = resolve_runtime_dependencies(cache, wheelhouse=wheelhouse)
        except BaseException:
            result["status"] = "failed"
            receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    print("ENGINE_ARTIFACTS", receipt, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--repair-dependencies", action="store_true", help="Verify/repair dependencies of the already installed frozen runtime")
    parser.add_argument("--engine-version", help="Explicit historical stable version; default follows latest stable")
    args = parser.parse_args()
    try:
        from release_selection import select, active
        if not active():
            version = args.engine_version
            if args.repair_dependencies and version is None:
                version = metadata.version("vllm-hust-engine")
            select(args.cache, version=version)
        selected = active()
        try:
            installed = metadata.version("vllm-hust-engine")
        except metadata.PackageNotFoundError:
            installed = None
        target_changed = selected is not None and selected["version"] != installed
        if args.repair_dependencies and not target_changed:
            resolve_runtime_dependencies(args.cache, wheelhouse=args.wheelhouse)
        else:
            # Reusing B0 preserves its separate SDK, not the old B1 package.
            # Install all four hash-pinned target artifacts before admitting it.
            install(args.cache, wheelhouse=args.wheelhouse, download_only=args.download_only)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
