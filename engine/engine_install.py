"""Install the four frozen distributions from verified repository release assets."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))

from engine.release_selection import kit_path, runtime_lock, profile as selected_profile

RUNTIME_ROOTS = ("vllm", "vllm-ascend", "vllm-hust-engine", "torchvision")
# The frozen SDK imports these manager and FlashLB dependencies outside the wheel metadata.
ADAPTER_REQUIREMENTS = ("platformdirs>=4", "jcs>=0.2.1", "packaging>=24", "cryptography>=3.4",
                        "numba==0.67.0", "llvmlite==0.49.0")


_R10_REVISED_SOURCE = "b11bcc9c2064268112b5386de97d103a45eab5f6956ea35477f047d2e0725c0d"
_R10_REVISED_RUNTIME = "2026.11.24.10+r10.4331740a6a6c"
_R10_COMPAT_REQUIREMENT = "vllm-hust-runtime==2026.11.21.10+r10.736ef2311da0"
_R10_COMPAT_WHEELS = {
    "vllm": (
        "vllm-0.28.1.post1.dev143+gf18cf803c.empty-cp312-cp312-linux_aarch64.whl",
        "bf824c913a513b66541a85b00055374012bf8c9da2ee8145ec2fc43b641c835f",
    ),
    "vllm-ascend": (
        "vllm_ascend-0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27-cp312-cp312-linux_aarch64.whl",
        "d1cbcc7a604681b8c56120d66f09ccbf1ebd2020192b34b54d98deb0f27e5bf4",
    ),
}
_R10_RUNTIME_WHEEL = (
    f"vllm_hust_runtime-{_R10_REVISED_RUNTIME}-cp312-cp312-linux_aarch64.whl",
    "b29e7e70ca61e2c5c15fdb6e8cf3cd81feb6d229f800aebb380409df45778990",
)


def runtime_dependency_pins(selection):
    if not selection or selection.get("release") != "full-answer-b1-20261124-r10":
        return {}
    pinned_artifacts = [
        selection["sdist"],
        *selection["runtime_artifacts"],
    ]
    observed = {row["filename"]: row["sha256"] for row in pinned_artifacts}
    required = {
        "vllm_hust_engine-2026.11.24.10.tar.gz": _R10_REVISED_SOURCE,
        **dict(_R10_COMPAT_WHEELS.values()),
        _R10_RUNTIME_WHEEL[0]: _R10_RUNTIME_WHEEL[1],
    }
    if selection["legacy"] or selection["version"] != "2026.11.24.10" or observed != required:
        raise ValueError("The revised r10 compatibility dependency requires the reviewed artifact set")
    # The two unchanged compatibility wheels bind the old runtime in METADATA.
    # The reviewed revised source archive pins its replacement runtime exactly.
    return {
        owner: (_R10_COMPAT_REQUIREMENT, f"vllm-hust-runtime=={_R10_REVISED_RUNTIME}")
        for owner in _R10_COMPAT_WHEELS
    }


def runtime_requirement(owner, text, dependency_pins):
    pin = (dependency_pins or {}).get(owner.lower().replace("_", "-"))
    return pin[1] if pin and text == pin[0] else text


def runtime_install_requirements(expected, dependency_pins):
    from packaging.requirements import Requirement

    requirements = set(ADAPTER_REQUIREMENTS)
    for owner in RUNTIME_ROOTS:
        for text in metadata.requires(owner) or []:
            requirement = Requirement(runtime_requirement(owner, text, dependency_pins))
            package = requirement.name.lower().replace("_", "-")
            # Installed wrapper versions already match the reviewed artifacts.
            # Their external dependencies remain explicit roots; the audit
            # below still follows every edge through both installed wheels.
            if (owner in ("vllm-hust-engine", "vllm-ascend")
                    and package in dependency_pins and not requirement.extras
                    and requirement.marker is None and requirement.url is None
                    and str(requirement.specifier) == f"=={expected.get(package)}"):
                continue
            requirements.add(str(requirement))
    return sorted(requirements)


def dependency_audit(dependency_pins=None):
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
            requirement = Requirement(runtime_requirement(name, text, dependency_pins))
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
    from engine.environment_setup import vendor_integrity, cached_wheel, distribution_integrity
    import os
    sys.path.insert(0, str(kit_path() / "scripts"))
    from child_runner import run_stage
    from benchmark.one_click import verify_kit
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
            # Replace the old distribution and its metadata before invoking the resolver.
            completed = run_stage([sys.executable, "-I", "-B", "-m", "pip", "install", "--disable-pip-version-check",
                                   "--no-deps", "--force-reinstall", "--no-compile", str(wheel)])
            if completed.returncode:
                raise RuntimeError("Runtime compatibility package installation failed: " + name)
            if metadata.version(name) != source["version"] or not distribution_integrity(name)["verified"]:
                raise ValueError("Installed runtime compatibility package differs from the frozen wheel: " + name)
        expected[name] = source["version"]
        if name == "opencv-python-headless":
            expected["numpy"] = "1.26.4"
    folder = Path(cache) / "engine-install"
    folder.mkdir(parents=True, exist_ok=True)
    from engine.release_selection import active
    dependency_pins = runtime_dependency_pins(active())
    requirements = runtime_install_requirements(expected, dependency_pins)
    content = "\n".join(requirements) + "\n"
    key = hashlib.sha256(content.encode()).hexdigest()[:16]
    required = folder / ("runtime-" + key + ".txt")
    constraints = folder / ("runtime-" + key + "-constraints.txt")
    required.write_text(content)
    constraints.write_text("\n".join(name + "==" + version for name, version in sorted(expected.items())) + "\n")
    before = dependency_audit(dependency_pins)
    if not before["verified"]:
        print("RUNTIME_DEPENDENCIES repairing", len(before["issues"]), "requirements", flush=True)
        command = [sys.executable, "-I", "-B", "-m", "pip", "install", "--disable-pip-version-check",
                   "--only-binary=:all:", "--no-compile"]
        for wheel_directory in links:
            command += ["--find-links", wheel_directory]
        from engine.pip_transport import run_stage as audited_pip
        completed = audited_pip([*command, "-c", str(constraints), "-r", str(required)], cache=cache,
                                phase='runtime-dependencies')
        if completed.returncode:
            raise RuntimeError("Engine Python dependency resolution failed: " + str(completed.returncode))
    integrity = vendor_integrity(sys.executable, lock["required_python_packages"], dict(os.environ))
    if not integrity["verified"]:
        source = lock["environment_sources"]["triton-ascend"]
        wheel = cached_wheel(source, links, Path(cache) / "wheels")
        result = run_stage([sys.executable, "-I", "-B", "-m", "pip", "install", "--no-deps", "--ignore-installed", "--no-compile", str(wheel)])
        if result.returncode:
            raise RuntimeError("Could not restore Ascend Triton after engine dependency resolution")
        integrity = vendor_integrity(sys.executable, lock["required_python_packages"], dict(os.environ))
    after = dependency_audit(dependency_pins)
    compatibility = {name: distribution_integrity(name) for name in lock.get("runtime_compatibility_sources", {})}
    if not after["verified"] or not integrity["verified"] or not all(r["verified"] for r in compatibility.values()):
        raise ValueError("Engine dependency/file verification failed: " + json.dumps(after["issues"]))
    # Metadata can match while inherited native extensions are ABI-incompatible.
    # Probe the FlashLB import and CPU execution as well as the multimodal path
    # before starting a model service; this work is outside benchmark timing.
    smoke = folder / "runtime-import-smoke.json"
    probe = ROOT / "tools/runtime_import_smoke.py"
    smoke_command = [sys.executable, "-I", "-B", str(probe), "--out", str(smoke)]
    if 'triton-ascend' in lock['required_python_packages']:
        source = lock['environment_sources']['triton-ascend']
        wheel = cached_wheel(source, links, Path(cache) / 'wheels')
        smoke_command += ['--triton-wheel', str(wheel), '--triton-sha256', source['sha256']]
    completed = run_stage(smoke_command,
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
                  requirements_sha256=hashlib.sha256(content.encode()).hexdigest(), constraints=expected,
                  compatibility_dependency_pins=dependency_pins)
    (folder / "runtime-dependencies.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RUNTIME_DEPENDENCIES_VERIFIED", len(after["observed_versions"]), "distributions", flush=True)
    return report


def sources():
    from engine.release_selection import active
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
    from engine.environment_setup import cached_wheel
    from benchmark.one_click import verify_kit
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
        command = [sys.executable, "-I", "-B", "-m", "pip", "install", "--disable-pip-version-check",
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
            from engine.package_metadata import prepare
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
        from engine.release_selection import select, active
        if not active():
            version = args.engine_version
            if args.repair_dependencies and version is None:
                version = metadata.version("vllm-hust-engine")
            select(args.cache, version=version, wheelhouse=args.wheelhouse)
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
