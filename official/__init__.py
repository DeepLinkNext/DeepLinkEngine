"""Locate the official evaluation modules."""

import os
from pathlib import Path
import sys


MODULES = Path(__file__).resolve().parent
# The official worker adds this distribution's parent directory to PYTHONPATH.
FT_PLUGIN_METADATA = Path('official/deeplinkengine_official_ft-1.0.dist-info')


def enable_module_imports():
    # Preserve worker-extension names and the imports of source-pinned modules.
    directory = str(MODULES)
    if directory not in sys.path:
        sys.path.insert(0, directory)


def source_path(root, name):
    """Resolve a source label without changing its name in provenance records."""
    relative = Path(name)
    if relative.parts[:1] == (FT_PLUGIN_METADATA.name,):
        return root / FT_PLUGIN_METADATA.parent / relative
    if relative.parent == Path('.'):
        if relative.match('official_*.py'):
            return root / 'official' / relative
        if relative.name in ('logging_entry.py', 'native_sdk_controller.py', 'sdk_dependency_probe.py',
                             'service_logging.py', 'transport_trust.py'):
            return root / 'engine' / relative
    return root / relative


def worker_pythonpath(root, scripts, inherited):
    """Expose the original top-level module names in SDK child interpreters."""
    return os.pathsep.join((str(root), str(root / 'official'), str(scripts), inherited))
