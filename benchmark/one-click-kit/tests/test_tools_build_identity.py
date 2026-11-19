"""Tools archives must not claim a source commit after tracked files change."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "test_tools_builder", ROOT / "ci/build_tools.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildIdentityTests(unittest.TestCase):
    def fixture(self, directory):
        root = directory / "checkout"
        root.mkdir()
        paths = [
            "download_and_run.py",
            "runners/r8/download_and_run.py",
            "publication-files.json",
        ]
        for name in paths[:-1]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (root / "publication-files.json").write_text(json.dumps(paths))

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()

        git("init", "-q")
        git("add", ".")
        git(
            "-c",
            "user.name=CI fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        return root, git("rev-parse", "HEAD")

    def test_clean_checkout_build_records_exact_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, commit = self.fixture(Path(tmp))
            result = builder.build(root, Path(tmp) / "dist", commit)
            self.assertEqual(result["source_commit"], commit)
            self.assertEqual(result["build_kind"], "commit")
            self.assertFalse(result["downloadable"])

    def test_dirty_checkout_cannot_claim_the_original_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, commit = self.fixture(Path(tmp))
            with (root / "download_and_run.py").open("a") as stream:
                stream.write("\n# changed after freeze\n")
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build(root, Path(tmp) / "dist", commit)
            self.assertFalse((Path(tmp) / "dist/engine-tools.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
