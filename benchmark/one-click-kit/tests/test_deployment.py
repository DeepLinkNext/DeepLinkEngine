import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import socket
import uuid
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_b1_r4 as fetch
import resource_guard
from owned_processes import Owned
from engine_entry import ipc_directory, temporary_directory
from restore_b1_r4 import relocate_signature_source
from qualification import sealed_files


class DeploymentTests(unittest.TestCase):
    def test_full_tmp_uses_private_tmpfs_and_binds_a_real_socket(self):
        import errno
        original = tempfile.mkdtemp

        def allocate(*args, **kwargs):
            if kwargs.get("dir") == "/tmp":
                raise OSError(errno.ENOSPC, "fixture root volume full")
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch("engine_entry.tempfile.mkdtemp", side_effect=allocate):
                path = ipc_directory(out)
            try:
                self.assertEqual(path.parent, Path("/dev/shm"))
                with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                    channel.bind(str(path / "fixture.sock"))
                self.assertEqual(ipc_directory(out), path)
                self.assertEqual(json.loads((out / "ipc-directory.json").read_bytes())["storage_fallback"]["errno"], errno.ENOSPC)
            finally:
                shutil.rmtree(path)

    def test_ipc_permission_errors_are_not_hidden_by_storage_fallback(self):
        import errno
        with tempfile.TemporaryDirectory() as temp:
            with patch("engine_entry.tempfile.mkdtemp", side_effect=OSError(errno.EACCES, "fixture denied")) as allocate:
                with self.assertRaises(PermissionError):
                    ipc_directory(Path(temp))
            self.assertEqual(allocate.call_count, 1)

    def test_recorded_ipc_path_cannot_redirect_to_a_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            target = Path(tempfile.mkdtemp(prefix="vhe-ipc-", dir="/dev/shm"))
            alias = target.with_name(target.name + "-alias")
            alias.symlink_to(target)
            try:
                (out / "ipc-directory.json").write_text(json.dumps({"path": str(alias)}))
                with self.assertRaisesRegex(ValueError, "Unexpected private IPC"):
                    ipc_directory(out)
            finally:
                alias.unlink()
                shutil.rmtree(target)

    def test_deep_tempdir_can_bind_datagram_after_private_temp_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / ("deep-output-" * 12)
            out.mkdir()
            code = "import socket,tempfile;d=tempfile.TemporaryDirectory(prefix='bs-cache-');s=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);s.bind(d.name+'/done.sock');s.close();d.cleanup()"
            old = subprocess.run(
                [sys.executable, "-S", "-c", code],
                env=dict(os.environ, TMPDIR=str(out)),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(old.returncode, 0)
            self.assertIn("AF_UNIX path too long", old.stderr)
            private = temporary_directory(out)
            try:
                fixed = subprocess.run(
                    [sys.executable, "-S", "-c", code],
                    env=dict(os.environ, TMPDIR=str(private)),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(fixed.returncode, 0, fixed.stderr)
            finally:
                shutil.rmtree(private.parent)

    def test_relocated_abi_uses_exact_canonical_source_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "mods/correction/pinned_metadata_signature.py"
            helper.parent.mkdir(parents=True)
            helper.write_text(
                "from pathlib import Path\nSOURCE = Path('/root/modcost-runtime-028-20261017/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')\n"
            )
            bindings = {
                "mods/correction/pinned_metadata_signature.py": hashlib.sha256(
                    helper.read_bytes()
                ).hexdigest()
            }
            change = relocate_signature_source(root, bindings)
            source = root / change["canonical_source"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                "ceabfc9db30c9d8bf8dc7c9bddf523c4a8d5ef04165994635dca3b94a8e929f4",
            )
            spec = importlib.util.spec_from_file_location(
                "relative_abi_fixture", helper
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.SOURCE, source)
            self.assertEqual(
                bindings[change["canonical_source"]], change["canonical_source_sha256"]
            )
            bad_bundle = root / "bad-bundle"
            bad_source = bad_bundle / change["canonical_source"]
            bad_source.parent.mkdir(parents=True)
            bad_source.write_bytes(b"corrupted canonical ABI")
            with patch(
                "restore_b1_r4.__file__", str(bad_bundle / "scripts/restore_b1_r4.py")
            ):
                with self.assertRaisesRegex(
                    ValueError, "Canonical Ascend ABI source differs"
                ):
                    relocate_signature_source(root, bindings)

    def test_sealed_run_rejects_missing_modified_and_extra_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fixture").write_bytes(b"fixed")
            (root / "checksums.json").write_text(
                json.dumps({"fixture": hashlib.sha256(b"fixed").hexdigest()})
            )
            self.assertEqual(len(sealed_files(root)), 1)
            (root / "fixture").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                sealed_files(root)
            (root / "fixture").write_bytes(b"fixed")
            (root / "extra").write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                sealed_files(root)
            (root / "extra").unlink()
            (root / "fixture").unlink()
            with self.assertRaises(ValueError):
                sealed_files(root)

    def test_long_output_directory_can_bind_a_real_unix_socket(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / ("long-output-" * 10) / ("nested-cache-" * 8)
            output.mkdir(parents=True)
            ipc = ipc_directory(output)
            try:
                path = ipc / str(uuid.uuid4())
                with socket.socket(socket.AF_UNIX) as listener:
                    listener.bind(str(path))
                self.assertEqual(ipc_directory(output), ipc)
            finally:
                shutil.rmtree(ipc)

    def test_cleanup_finds_orphan_in_a_new_session(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            child_code = (
                "import time;from pathlib import Path;time.sleep(2);Path("
                + repr(str(output / "late-write"))
                + ").write_text('bad')"
            )
            script = (
                "import subprocess,sys;p=subprocess.Popen([sys.executable,'-c',"
                + repr(child_code)
                + "],start_new_session=True);print(p.pid,flush=True)"
            )
            env = dict(os.environ, VLLM_HUST_TEST_OUTPUT=str(output))
            parent = subprocess.Popen(
                [sys.executable, "-c", script],
                env=env,
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            owned = Owned(output)
            owned.add(parent.pid)
            orphan = int(parent.stdout.readline())
            parent.stdout.close()
            self.assertEqual(parent.wait(timeout=5), 0)
            self.assertTrue(Path(f"/proc/{orphan}").exists())
            owned.stop()
            self.assertFalse(owned.scan())
            self.assertFalse((output / "late-write").exists())

    def test_cleanup_keeps_ownership_when_exec_clears_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import os;os.execve('/bin/sleep',['sleep','30'],{})",
                ],
                start_new_session=True,
            )
            other = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
            try:
                owned = Owned(output)
                owned.add(child.pid)
                owned.stop()
                child.wait(timeout=5)
                self.assertIsNone(other.poll())
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
                other.terminate()
                other.wait()

    def test_dotenv_is_parsed_without_execution_and_env_has_precedence(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {}, clear=True),
        ):
            root = Path(temp)
            env = root / ".env"
            value = "literal-$(touch " + str(root / "must-not-exist") + ")"
            env.write_text(
                "export GITHUB_TOKEN='"
                + value
                + "' # ignored\nUNRELATED_SECRET=ignored\n"
            )
            token, source = fetch.resolve_token(env)
            self.assertEqual(token, value)
            self.assertEqual(source, str(env) + ":GITHUB_TOKEN")
            self.assertFalse((root / "must-not-exist").exists())
            os.environ["GH_TOKEN"] = "environment-fixture"
            self.assertEqual(
                fetch.resolve_token(env),
                ("environment-fixture", "environment:GH_TOKEN"),
            )

    def test_redirect_does_not_forward_authentication_to_asset_host(self):
        request = urllib.request.Request(
            "https://api.github.com/example",
            headers={"Authorization": "Bearer fixture"},
        )
        target = fetch.SafeRedirect().redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/example",
        )
        self.assertFalse(target.has_header("Authorization"))
        with self.assertRaises(ValueError):
            fetch.SafeRedirect().redirect_request(
                request, io.BytesIO(), 302, "Found", {}, "http://example.com/fixture"
            )

    def test_verified_cache_needs_no_credential_and_corruption_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = b"fixed fixture"
            digest = hashlib.sha256(value).hexdigest()
            (root / "fixture").write_bytes(value)
            with (
                patch.object(fetch, "ASSETS", {"fixture": (1, len(value), digest)}),
                patch.object(
                    fetch,
                    "resolve_token",
                    side_effect=RuntimeError("No credential permitted"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                receipt = fetch.fetch(root)
                self.assertEqual(receipt["source"], "verified-local-cache")
                self.assertIsNone(receipt["credential_source"])
                (root / "fixture").write_bytes(b"X" + value[1:])
                with self.assertRaisesRegex(RuntimeError, "No credential permitted"):
                    fetch.fetch(root)

    def test_busy_lease_stops_before_any_device_probe_and_releases_partial_locks(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {}, clear=True),
        ):
            root = Path(temp)
            names = [str(root / "first.lock"), str(root / "busy.lock")]
            with open(names[1], "a+b") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    patch.object(resource_guard, "paths", return_value=names),
                    patch.object(
                        resource_guard.subprocess,
                        "run",
                        side_effect=AssertionError("Device probe forbidden"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaises(TimeoutError):
                        resource_guard.acquire(root, 0)
                with open(names[0], "a+b") as free:
                    fcntl.flock(free, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(
                    json.loads((root / "resource-wait.json").read_text())[
                        "model_started"
                    ]
                )


class PinnedHostPreparationTests(unittest.TestCase):
    MOUNTS = ("1 0 0:1 / / rw - overlay overlay rw\n"
              "2 1 0:2 / /network rw - nfs4 server:/volume rw\n"
              "3 2 0:3 / /network/local rw - ext4 /dev/local rw\n"
              "4 1 0:4 / /windows-share rw - cifs //server/share rw\n")

    def test_network_imports_have_a_bounded_preparation_allowance(self):
        from engine_entry import pinned_memory_preflight_timeout
        for path in ("/network/python/bin/python3", "/windows-share/env/lib"):
            self.assertEqual(pinned_memory_preflight_timeout([path], self.MOUNTS), 900)
        self.assertEqual(pinned_memory_preflight_timeout(["/usr/bin/python3"], self.MOUNTS), 120)
        self.assertEqual(pinned_memory_preflight_timeout(["/network/local/bin/python3"], self.MOUNTS), 120)
        self.assertEqual(pinned_memory_preflight_timeout(["/network-archive/bin/python3"], self.MOUNTS), 120)

    def test_mount_escaping_and_unknown_filesystem_remain_bounded(self):
        from engine_entry import pinned_memory_preflight_timeout
        mounts = self.MOUNTS + "5 1 0:5 / /slow" + chr(92) + "040store rw - nfs server:/slow rw\n"
        self.assertEqual(pinned_memory_preflight_timeout(["/slow store/bin/python"], mounts), 900)
        self.assertEqual(pinned_memory_preflight_timeout(["/anywhere"], "not a mount record"), 900)

    def test_actual_gate_payload_and_wrapper_validate_two_admissions(self):
        import types
        import engine_entry
        calls = []
        current = [None]
        dtype = object()

        class Tensor:
            def __init__(self, count, value=None):
                self.count, self.value = count, value
            def fill_(self, value):
                self.value = value
            def is_pinned(self):
                return True
            def numel(self):
                return self.count
            def to(self, device, non_blocking):
                calls.append(("h2d", current[0], device, non_blocking, self.count))
                return Tensor(self.count, self.value)
            def cpu(self):
                calls.append(("d2h", current[0], self.count))
                return Tensor(self.count, self.value)

        def empty(count, **options):
            self.assertEqual(options, dict(dtype=dtype, device="cpu", pin_memory=True))
            return Tensor(count)
        def select(device):
            current[0] = device
            calls.append(("select", device))
        fake_torch = types.SimpleNamespace(uint8=dtype, empty=empty,
            equal=lambda a, b: a.count == b.count and a.value == b.value)
        fake_npu = types.SimpleNamespace(npu=types.SimpleNamespace(set_device=select,
            synchronize=lambda: calls.append(("synchronize", current[0]))))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def run(command, **options):
                code = command[command.index("-c") + 1]
                self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), "8bdaea3ee965efa4795e64ad1fe09c91990594bd88d09b393fecd4dc58f0b9ed")
                self.assertIn("-S", command)
                self.assertEqual(options["timeout"], 900)
                with patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": fake_npu}), patch.object(sys, "argv", ["cpu-pin-fixture", command[-1]]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(code, "<original-pin-code-cpu-fixture>", "exec"), {})
                return types.SimpleNamespace(returncode=0, stdout="fixture complete\n", stderr="")
            with patch.object(engine_entry, "pinned_memory_preflight_timeout", return_value=900), patch.object(engine_entry.subprocess, "run", side_effect=run):
                engine_entry.check_pinned_host_memory(out, {"VLLMHUST_NO_SITE": "1"})
            result = json.loads((out / "pinned-memory-preflight.json").read_bytes())
            self.assertEqual([v["device"] for v in result["devices"]], [0, 1])
            self.assertEqual([v for v in calls if v[0] == "h2d"], [("h2d", 0, "npu:0", True, 32 * 2**20), ("h2d", 1, "npu:1", True, 32 * 2**20)])
            self.assertEqual([v for v in calls if v[0] == "d2h"], [("d2h", 0, 32 * 2**20), ("d2h", 1, 32 * 2**20)])
            self.assertFalse(result["model_loaded"])

    def test_timeout_preserves_partial_raw_log_and_previous_attempt(self):
        import engine_entry
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            failure = subprocess.TimeoutExpired(["fixture"], 900, output=b"partial stdout\n", stderr=b"partial stderr\xff")
            with patch.object(engine_entry, "pinned_memory_preflight_timeout", return_value=900), patch.object(engine_entry.subprocess, "run", side_effect=failure) as run:
                with self.assertRaisesRegex(TimeoutError, "before model loading"):
                    engine_entry.check_pinned_host_memory(out, {})
                self.assertEqual((out / "pinned-memory-preflight.log").read_bytes(), b"partial stdout\npartial stderr\xff")
                with self.assertRaises(FileExistsError):
                    engine_entry.check_pinned_host_memory(out, {})
                self.assertEqual(run.call_count, 1)

    def test_a_failed_transfer_still_fails_admission(self):
        import engine_entry
        import types
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def run(command, **options):
                rows = [dict(device=i, bytes=32 * 2**20, pinned=True, h2d_d2h_equal=i == 0) for i in (0, 1)]
                Path(command[-1]).write_text(json.dumps(dict(model_loaded=False, devices=rows)))
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch.object(engine_entry, "pinned_memory_preflight_timeout", return_value=120), patch.object(engine_entry.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "does not cover both devices"):
                    engine_entry.check_pinned_host_memory(out, {})


if __name__ == "__main__":
    unittest.main()
