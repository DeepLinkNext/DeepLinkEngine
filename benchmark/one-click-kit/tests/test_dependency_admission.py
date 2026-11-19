"""SDK metadata is checked without executing package code or native imports."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dependency_admission import inspect_sdk


class DependencyAdmissionTests(unittest.TestCase):
    def test_native_package_code_is_never_imported_and_version_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "bin/python"
            python.parent.mkdir()
            python.symlink_to(sys.executable)
            site = (
                root
                / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            source = site / "torch/__init__.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "raise RuntimeError('Native package imported during admission')"
            )
            metadata = site / "torch-2.13.0.dist-info/METADATA"
            metadata.parent.mkdir()
            metadata.write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.13.0\n")
            actual = inspect_sdk(python, {"torch": "2.13.0"})
            self.assertEqual(actual["metadata"], {"torch": "2.13.0"})
            self.assertFalse(actual["native_packages_imported"])
            self.assertEqual(actual["site_packages"], str(site))
            with self.assertRaisesRegex(ValueError, "dependency metadata differs"):
                inspect_sdk(python, {"torch": "2.99.0"})


if __name__ == "__main__":
    unittest.main()
