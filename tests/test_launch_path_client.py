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
