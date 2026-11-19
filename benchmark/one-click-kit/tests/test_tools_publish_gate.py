"""A failed artifact acceptance must never publish a draft Release."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


publisher = load(ROOT / "ci/publish_tools.py", "test_tools_publisher")


class PublishGateTests(unittest.TestCase):
    def attempt(self, reject):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            (dist / "engine-tools.tar.gz").write_bytes(b"synthetic uploaded bytes")
            (dist / "build.json").write_text(
                json.dumps(
                    {
                        "source_commit": "1" * 40,
                        "archive_sha256": publisher.hashlib.sha256(
                            b"synthetic uploaded bytes"
                        ).hexdigest(),
                    }
                )
            )
            for name in ("download-and-run.py", "download-and-run-r8.py"):
                (dist / name).write_text("ASSET_ID = 0\n")
            calls = []
            checks = []

            def request(url, body=None, method=None, binary=False):
                calls.append((url, body, method))
                if url.endswith("/releases"):
                    return {
                        "id": 7,
                        "upload_url": "https://uploads.github.com/test{?name}",
                        "html_url": "https://github.com/test/release",
                    }
                return {"id": 11}

            class Opener:
                def open(self, *_args, **_kwargs):
                    return io.BytesIO(
                        b"changed bytes"
                        if reject == "download"
                        else b"synthetic uploaded bytes"
                    )

            class Downloader:
                SafeRedirect = publisher.urllib.request.HTTPRedirectHandler

                @staticmethod
                def extract(_archive, _tools):
                    return (
                        None  # extraction/identity covered by artifact-contract suite
                    )

            def run(args, **_kwargs):
                if args[0] == sys.executable:
                    checks.append(args)
                    if reject == "acceptance":
                        raise subprocess.CalledProcessError(1, args)
                    Path(args[args.index("--report") + 1]).write_text('{"passed":true}')

            with (
                patch.object(publisher, "ROOT", root),
                patch.dict(
                    publisher.os.environ,
                    {
                        "GITHUB_REF_NAME": "engine-tools-vtest",
                        "GITHUB_SHA": "1" * 40,
                        "GITHUB_REPOSITORY": publisher.REPO,
                        "GH_TOKEN": "synthetic",
                    },
                ),
                patch.object(publisher, "request", side_effect=request),
                patch.object(publisher, "subprocess") as process,
                patch.object(
                    publisher.importlib.util,
                    "module_from_spec",
                    return_value=Downloader,
                ),
                patch.object(
                    publisher.importlib.util, "spec_from_file_location"
                ) as spec,
                patch.object(
                    publisher.urllib.request, "build_opener", return_value=Opener()
                ),
            ):
                process.run.side_effect = run
                spec.return_value.loader.exec_module.return_value = None
                if reject:
                    with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                        publisher.main()
                else:
                    publisher.main()
            creates = [body for url, body, method in calls if url.endswith("/releases")]
            publishes = [body for _, body, method in calls if method == "PATCH"]
            self.assertEqual(len(creates), 1)
            self.assertTrue(creates[0]["draft"])
            self.assertEqual(creates[0]["make_latest"], "false")
            if reject:
                self.assertEqual(publishes, [])
            else:
                self.assertEqual(len(checks), 2)
                self.assertEqual(publishes, [{"draft": False, "make_latest": "false"}])

    def test_changed_uploaded_archive_keeps_release_draft(self):
        self.attempt("download")

    def test_failed_entry_acceptance_keeps_release_draft(self):
        self.attempt("acceptance")

    def test_publish_requires_both_runner_acceptances(self):
        self.attempt(None)


if __name__ == "__main__":
    unittest.main()
