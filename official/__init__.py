"""Locate the official evaluation modules."""

import os
from pathlib import Path
import sys


MODULES = Path(__file__).resolve().parent


def enable_module_imports():
    # Preserve worker-extension names and the imports of source-pinned modules.
    directory = str(MODULES)
    if directory not in sys.path:
        sys.path.insert(0, directory)


def source_path(root, name):
    """Resolve a source label without changing its name in provenance records."""
    relative = Path(name)
    if relative.parent == Path('.') and relative.match('official_*.py'):
        return root / 'official' / relative
    return root / relative


def worker_pythonpath(root, scripts, inherited):
    """Expose the original top-level module names in SDK child interpreters."""
    return os.pathsep.join((str(root), str(root / 'official'), str(scripts), inherited))
