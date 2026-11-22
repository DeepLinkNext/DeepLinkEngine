"""Declare CPU-only admission adapters for source-pinned producer installers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = 'cbb124ca19737524e73181930860a31d7da76ed55e1bd8b283ae71f103751edd'
PROFILES = {
    ORIGINAL_SHA256: {'profile': 'r8', 'release': 'full-answer-b1-20261117-r8'},
    '01d0a5f98d231a27b499adea9a2e90edeb62419f8c73ee2c26828b1efc77a2bf':
        {'profile': 'r9', 'release': 'full-answer-b1-20261119-r9'},
}
SCHEMA = 'deeplinkengine-producer-cpu-admission/v1'
ENTRY = """import importlib.util,sys
path=sys.argv.pop(1)
spec=importlib.util.spec_from_file_location('vllm_hust_engine._cpu_admission',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raise SystemExit(module.main())
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def derive(raw, *, payload, receipts):
    """Retain all producer AST except its CPU admission code and payload locator."""
    profile = PROFILES.get(sha(raw))
    if profile is None:
        raise ValueError('Producer CLI source requires review before CPU admission adaptation')
    tree = ast.parse(raw)
    releases = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'RELEASE' for t in n.targets)
                and isinstance(n.value, ast.Constant)]
    if releases != [profile['release']]:
        raise ValueError('Producer CLI release differs from its exact source profile')
    audits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_install']
    payloads = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)]
    if len(audits) != 1 or len(payloads) != 1:
        raise ValueError('Expected one producer CPU admission and one payload locator')
    codes = [n for n in audits[0].body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets)]
    if len(codes) != 1 or not isinstance(codes[0].value, ast.Constant):
        raise ValueError('Expected the original producer CPU admission code')
    original_code = codes[0].value.value
    old_import = 'from dependency_admission import admit_dependencies'
    old_call = 'dependencies=admit_dependencies(root)'
    if original_code.count(old_import) != 1 or original_code.count(old_call) != 1:
        raise ValueError('Producer CPU admission source changed')
    adapted = original_code.replace(old_import,
        'sys.path.insert(0,' + repr(str(ROOT)) + ')\n'
        'from engine.sdk_dependency_probe import admit_dependencies, receipt_path')
    adapted = adapted.replace(old_call,
        'dependencies=admit_dependencies(root, tools=Path(sys.argv[2]), '
        'receipt=receipt_path(Path(' + repr(str(receipts)) + ')))')
    # The installed payload remains read-only and is still verified by the
    # original producer. The derived module has its own honest __file__.
    old_payload = ast.unparse(payloads[0].value)
    payloads[0].value = ast.parse('Path(' + repr(str(payload)) + ')', mode='eval').body
    codes[0].value = ast.Constant(adapted)
    derived = (ast.unparse(ast.fix_missing_locations(tree)) + '\n').encode()
    restored = ast.parse(derived)
    audit = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_install')
    next(n for n in audit.body if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets)).value = ast.Constant(original_code)
    next(n for n in restored.body if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)).value = ast.parse(old_payload, mode='eval').body
    if ast.dump(ast.parse(raw), include_attributes=False) != ast.dump(restored, include_attributes=False):
        raise ValueError('Producer adaptation changed other AST')
    from engine.sdk_dependency_probe import original, probe
    return derived, dict(schema=SCHEMA, source_profile=profile['profile'], source_release=profile['release'],
        original_source_sha256=sha(raw),
        derived_source_sha256=sha(derived), unchanged_other_ast=True,
        original_probe_sha256=sha(probe(original()).encode()),
        cpu_metadata_adapter_sha256=sha((ROOT/'engine/sdk_dependency_probe.py').read_bytes()),
        payload_locator=str(payload), metadata_receipts=str(receipts),
        changes=['CPU metadata admission uses the declared -I -B -S adapter',
                 'payload locator explicitly retains the original installed payload'],
        model_started=False, formal_measurement=False)


def command(target, evidence):
    """Materialize the declared producer derivation before the installer imports it."""
    from vllm_hust_engine import cli
    source = Path(cli.__file__).absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError('Producer CLI must be an ordinary installed source file')
    payload = source.parent/'payload'
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError('Producer payload must remain its original ordinary directory')
    evidence = Path(evidence).absolute()
    evidence.mkdir(parents=True, exist_ok=True)
    if evidence.is_symlink():
        raise ValueError('Producer admission evidence cannot be a symlink')
    raw = source.read_bytes()
    derived, record = derive(raw, payload=payload, receipts=evidence/'metadata')
    destination = evidence/(sha(derived)+'.py')
    if os.path.lexists(destination):
        if destination.is_symlink() or destination.read_bytes() != derived:
            raise ValueError('Existing producer admission adapter differs')
    else:
        with destination.open('xb') as stream:
            stream.write(derived)
    record.update(original_source=str(source), executed_source=str(destination),
        owned_adapter_source_sha256=sha(Path(__file__).read_bytes()),
        original_package_modified=False, original_tools_modified=False)
    receipt = destination.with_suffix('.json')
    if receipt.exists():
        if receipt.is_symlink() or json.loads(receipt.read_bytes()) != record:
            raise ValueError('Existing producer admission provenance differs')
    else:
        with receipt.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
    return [sys.executable, '-I', '-B', '-c', ENTRY, str(destination),
            'install', '--out', str(target)]
