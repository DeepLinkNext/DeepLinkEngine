"""Check ownership metadata repair without modifying installed runtime payloads."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import package_metadata

class PackageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prefix = Path(self.tmp.name) / "venv"
        self.site = self.prefix / "lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        self.python = self.prefix / "bin/python"
        (self.prefix / "pyvenv.cfg").touch()
        self.dist = self.site / "fixture-1.dist-info"
        self.dist.mkdir()
        (self.site / "namespace").mkdir()
        (self.site / "namespace/worker.py").write_bytes(b"# original source\n")
        (self.site / "standalone.py").write_bytes(b"# original module\n")
        self.rows = []
        for name in ["namespace/worker.py", "standalone.py"]:
            data = (self.site / name).read_bytes()
            self.rows.append([name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), str(len(data))])
        self.write_record(self.rows + [["../../../outside.py", "", ""], ["", "", ""]])

    def write_record(self, rows):
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(rows)
        (self.dist / "RECORD").write_text(stream.getvalue())

    def test_adds_namespace_and_module_index_with_valid_record_hash(self):
        result = package_metadata.prepare(self.python)
        self.assertEqual(result["generated"][0]["modules"], ["namespace", "standalone"])
        rows = list(csv.reader(io.StringIO((self.dist / "RECORD").read_text())))
        self.assertEqual(rows[:2], self.rows)
        for name, checksum, size in self.rows + [rows[-1]]:
            data = (self.site / name).read_bytes()
            self.assertEqual(len(data), int(size))
            self.assertEqual(checksum, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode())
        before = (self.dist / "RECORD").read_bytes()
        self.assertEqual(package_metadata.prepare(self.python)["generated"], [])
        self.assertEqual((self.dist / "RECORD").read_bytes(), before)

    def test_preserves_existing_package_metadata(self):
        (self.dist / "top_level.txt").write_text("explicit_name\n")
        self.assertEqual(package_metadata.prepare(self.python)["generated"], [])
        self.assertEqual((self.dist / "top_level.txt").read_text(), "explicit_name\n")

    def test_does_not_modify_a_shared_python(self):
        (self.prefix / "pyvenv.cfg").unlink()
        self.assertEqual(package_metadata.prepare(self.python)["status"], "not_private")
        self.assertFalse((self.dist / "top_level.txt").exists())

    def test_rejects_metadata_symlinks(self):
        outside = self.prefix.parent / "outside.txt"
        outside.write_text("unchanged")
        (self.dist / "top_level.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            package_metadata.prepare(self.python)
        self.assertEqual(outside.read_text(), "unchanged")

    def test_rejects_site_outside_private_prefix(self):
        alternate = self.prefix.parent / "shared-site"
        self.site.rename(alternate)
        self.site.symlink_to(alternate, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside"):
            package_metadata.prepare(self.python)

if __name__ == "__main__":
    unittest.main()
