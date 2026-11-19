"""Check exact SDK package metadata without importing native model packages."""

import json
from pathlib import Path
import subprocess

PROBE = """
import importlib.metadata,json,sys
from pathlib import Path
python=Path(sys.argv[1]);site=python.parent.parent/('lib/python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'/site-packages')
if not site.is_dir():raise ValueError('Expected an isolated SDK site-packages directory')
sys.path.insert(0,str(site))
expected=json.loads(sys.argv[2])
metadata={name:importlib.metadata.version(name) for name in expected}
assert not {'torch','torch_npu','vllm','vllm_ascend'} & set(sys.modules)
print(json.dumps({'python':str(python),'python_version':sys.version.split()[0], 'site_packages':str(site),'metadata':metadata,'native_packages_imported':False}))
"""


def inspect_sdk(python, expected):
    python = Path(python).expanduser().absolute()
    result = subprocess.run(
        [str(python), "-I", "-S", "-c", PROBE, str(python), json.dumps(expected)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    observed = json.loads(result.stdout)
    if observed["metadata"] != expected:
        raise ValueError(
            "SDK dependency metadata differs from the source-verified host contract: "
            + json.dumps({"expected": expected, "observed": observed["metadata"]})
        )
    return observed


def admit_dependencies(deployment):
    # Caller must first verify the contract against the pinned source manifests.
    deployment = Path(deployment)
    profile = json.loads((deployment / "deployment.json").read_bytes())
    contract = json.loads(
        (
            deployment
            / "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
        ).read_bytes()
    )
    return inspect_sdk(profile["python"], contract["versions"])
