"""Regression coverage for launch paths under a legitimate shared mount."""
import os

import pytest

from benchmark import launch_path_client as gateway


@pytest.fixture
def bound_paths(tmp_path):
    module = {}
    exec(compile(gateway.PATH_CHECK_SOURCE, gateway.PATH_CHECK, "exec"), module)
    base = tmp_path / "gennova-volume"
    sdk, scripts, unrelated = base / "sdk", base / "scripts", base / "foreign"
    modules = sdk / "modules"
    for folder in (scripts, unrelated, modules):
        folder.mkdir(parents=True)
    env = {"PYTHONPATH": os.pathsep.join((str(scripts), str(modules)))}
    runtime = sdk / module["DIRECTORIES"]["BETTERSCALE_QWEN35_RUNTIME"]
    runtime.mkdir(parents=True)
    env["BETTERSCALE_QWEN35_RUNTIME"] = str(runtime)
    for key, name in module["LIBRARIES"].items():
        library = sdk / name
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"test library")
        env[key] = str(library)
    return module["verify"], env, sdk, scripts, unrelated


def test_accepts_active_sdk_and_exact_client_scripts(bound_paths):
    verify, env, sdk, scripts, _ = bound_paths
    verify(env, sdk, scripts)


def test_rejects_legacy_project_path_or_unexpected_library(bound_paths):
    verify, env, sdk, scripts, unrelated = bound_paths
    pythonpath = env["PYTHONPATH"] + os.pathsep + str(unrelated)
    with pytest.raises(ValueError, match="PYTHONPATH"):
        verify(dict(env, PYTHONPATH=pythonpath), sdk, scripts)
    with pytest.raises(ValueError, match="BETTERSCALE_QWEN35_RUNTIME"):
        verify(dict(env, BETTERSCALE_QWEN35_RUNTIME=str(sdk / "modules")), sdk, scripts)
    with pytest.raises(ValueError, match="BETTERSCALE_UNKNOWN"):
        verify(dict(env, BETTERSCALE_UNKNOWN=str(unrelated)), sdk, scripts)
    with pytest.raises(ValueError, match="BETTERSCALE_GDN_HOST_LIBRARY"):
        verify(dict(env, BETTERSCALE_GDN_HOST_LIBRARY=env["BETTERSCALE_CP_LIBRARY"]), sdk, scripts)


def test_rejects_symlink_escape(bound_paths):
    verify, env, sdk, scripts, unrelated = bound_paths
    link = sdk / "linked-modules"
    try:
        link.symlink_to(unrelated, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlink creation unavailable")
    with pytest.raises(ValueError, match="PYTHONPATH"):
        verify(dict(env, PYTHONPATH=str(link)), sdk, scripts)


@pytest.fixture
def sealed_reference_client(tmp_path):
    import hashlib
    import json

    checker = {}
    exec(compile(gateway.PATH_CHECK_SOURCE, gateway.PATH_CHECK, "exec"), checker)
    bundled = {
        "third-party-test-kit/reference_operator_devices.py": b"operator source",
        "third-party-test-kit/proc_capture/m01_proc_worker.py": b"observer source",
    }
    manifest = json.dumps({"files": {name: hashlib.sha256(source).hexdigest()
                                      for name, source in bundled.items()}}, sort_keys=True).encode()
    base = tmp_path / "gennova-volume"
    client = base / hashlib.sha256(manifest).hexdigest()
    scripts = client / "scripts"
    scripts.mkdir(parents=True)
    (client / "CLIENT-MANIFEST.json").write_bytes(manifest)
    for name, source in bundled.items():
        target = client / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    sdk = base / "sdk"
    sdk.mkdir()
    return checker, client, scripts, sdk


def test_observation_accepts_only_exact_verified_kit_paths(sealed_reference_client):
    checker, client, scripts, sdk = sealed_reference_client
    verify = checker["verify"]
    paths = checker["verify_reference_kit"](scripts)
    assert paths == (client / "third-party-test-kit", client / "third-party-test-kit/proc_capture")
    env = {"PYTHONPATH": os.pathsep.join(map(str, (scripts, *paths)))}
    verify(env, sdk, scripts, reference_paths=paths)
    with pytest.raises(ValueError, match="PYTHONPATH"):
        verify(env, sdk, scripts)
    other = client / "unexpected"
    other.mkdir()
    with pytest.raises(ValueError, match="import paths"):
        verify(env, sdk, scripts, reference_paths=(paths[0], other))
    with pytest.raises(ValueError, match="PYTHONPATH"):
        verify({"PYTHONPATH": os.pathsep.join(map(str, (scripts, *paths, other)))},
               sdk, scripts, reference_paths=paths)


def test_observation_rejects_changed_or_unlisted_code(sealed_reference_client):
    checker, client, scripts, _ = sealed_reference_client
    capture = client / "third-party-test-kit/proc_capture"
    unexpected = capture / "unlisted.py"
    unexpected.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="kit inventory"):
        checker["verify_reference_kit"](scripts)
    unexpected.unlink()
    (capture / "m01_proc_worker.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="kit file differs"):
        checker["verify_reference_kit"](scripts)


def test_observation_rejects_rewritten_manifest(sealed_reference_client):
    checker, client, scripts, _ = sealed_reference_client
    manifest = client / "CLIENT-MANIFEST.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest differs"):
        checker["verify_reference_kit"](scripts)
