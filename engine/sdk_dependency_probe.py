"""Explicit CPU metadata admission: retain the frozen probe and disable pyc writes."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/dependency_admission.py'
ORIGINAL_SHA256 = 'ef04dccfee4dd68ff5fd1be480e86f63efae0d91b1ddac31a5b74cd733514d54'
VERSION = 'deeplinkengine-sdk-metadata-bytecode-guard/v1'
OLD_COMMAND = b'[str(python), "-I", "-S", "-c", PROBE, str(python), json.dumps(expected)]'
NEW_COMMAND = b'[str(python), "-I", "-B", "-S", "-c", PROBE, str(python), json.dumps(expected)]'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original():
    raw = (ROOT / 'benchmark/one-click-kit' / SOURCE).read_bytes()
    if sha(raw) != ORIGINAL_SHA256:
        raise ValueError('Original frozen dependency-admission source differs')
    return raw


def probe(raw):
    values = [n.value.value for n in ast.parse(raw).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROBE' for t in n.targets)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    if len(values) != 1:
        raise ValueError('Expected exactly one original CPU metadata PROBE')
    return values[0]


def derive(raw):
    """Change exactly one argv flag, leaving the complete remaining AST intact."""
    base = original()
    derived = base.replace(OLD_COMMAND, NEW_COMMAND)
    if base.count(OLD_COMMAND) != 1 or derived == base:
        raise ValueError('Frozen metadata command requires source review')
    if raw not in (base, derived):
        raise ValueError('Unreviewed metadata source or additional AST change')
    before, after = ast.parse(base), ast.parse(derived)
    changed = 0
    for node in ast.walk(after):
        if isinstance(node, ast.List):
            strings = [n.value if isinstance(n, ast.Constant) else None for n in node.elts]
            if strings[1:5] == ['-I', '-B', '-S', '-c']:
                del node.elts[2]
                changed += 1
    if changed != 1 or ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError('CPU metadata adaptation changed more than one -B flag')
    if probe(base) != probe(derived):
        raise ValueError('Original frozen metadata PROBE changed')
    return derived, dict(schema=VERSION, original_source_sha256=sha(base),
        derived_source_sha256=sha(derived), original_probe_sha256=sha(probe(base).encode()),
        derived_probe_sha256=sha(probe(derived).encode()), added_flags=['-B'],
        unchanged_other_ast=True, scope='CPU metadata only; model, arithmetic and formal clocks unchanged')


def binding(tools):
    tools = Path(tools).resolve()
    path = tools / SOURCE
    if path.is_symlink() or not path.is_file():
        raise ValueError('Frozen metadata source must be an ordinary file')
    raw = path.read_bytes()
    derived, record = derive(raw)
    manifest = tools / ('CLIENT-MANIFEST.json' if (tools / 'CLIENT-MANIFEST.json').is_file()
                        else 'DEPLOYMENT-TOOLS.json')
    value = json.loads(manifest.read_bytes())
    if value.get('files', {}).get(SOURCE) != sha(raw):
        raise ValueError('Metadata source differs from its actual tools inventory')
    if raw == derived and value.get('cpu_metadata_admission', {}).get('derived_source_sha256') != sha(raw):
        raise ValueError('Derived metadata admission requires its declared adapter binding')
    return probe(raw), dict(record, executed_source=str(path), executed_source_sha256=sha(raw),
        tools_manifest=str(manifest), tools_manifest_sha256=sha(manifest.read_bytes()),
        owned_adapter_source_sha256=sha(Path(__file__).read_bytes()))


def inspect_sdk(python, expected, *, tools=None, receipt=None):
    """Keep the original metadata comparison and JSON result; only add -B."""
    python = Path(python).expanduser().absolute()
    tools = Path(tools) if tools is not None else ROOT / 'benchmark/one-click-kit'
    record = dict(schema=VERSION, status='started', started=time.time(),
                  model_started=False, formal_measurement=False, native_packages_imported=False)
    try:
        code, origin = binding(tools)
        record.update(origin, command_flags=['-I', '-B', '-S'],
                      expected_metadata_sha256=sha(json.dumps(expected, sort_keys=True).encode()))
        result = subprocess.run([str(python), '-I', '-B', '-S', '-c', code,
            str(python), json.dumps(expected)], capture_output=True, text=True,
            timeout=30, check=True)
        observed = json.loads(result.stdout)
        if observed.get('metadata') != expected or observed.get('native_packages_imported') is not False:
            raise ValueError('SDK dependency metadata differs from the original host contract')
        if observed.get('python') != str(python):
            raise ValueError('SDK metadata interpreter path differs')
        record.update(status='completed', observed=observed)
        return observed
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        record['ended'] = time.time()
        if receipt is not None:
            path = Path(receipt)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')


def admit_dependencies(deployment, *, tools=None, receipt=None):
    deployment = Path(deployment)
    if receipt is not None and Path(receipt).resolve().is_relative_to(deployment.resolve()):
        raise ValueError('Metadata receipts must stay outside the immutable SDK')
    profile = json.loads((deployment / 'deployment.json').read_bytes())
    contract = json.loads((deployment / 'mods/microflow/hust_prefill_microflow/school028/host-contract.json').read_bytes())
    return inspect_sdk(profile['python'], contract['versions'], tools=tools, receipt=receipt)


def client_bridge(files):
    """Declare the same one-flag adaptation for every frozen importer."""
    key = SOURCE
    before = dict(files)
    body, record = derive(files[key])
    files[key] = body
    files['adapters/sdk_dependency_probe.py'] = Path(__file__).read_bytes()
    record['owned_adapter_source_sha256'] = sha(files['adapters/sdk_dependency_probe.py'])
    compatibility = 'protocols/task-warmup-v3-c8-compatibility.json'
    if compatibility in files:
        value = json.loads(files[compatibility])
        value.setdefault('unchanged_continuous_source_adaptations', {})[key] = dict(
            previous_sha256=[record['original_source_sha256']], current_sha256=record['derived_source_sha256'],
            scope=record['scope'], original_probe_sha256=record['original_probe_sha256'])
        files[compatibility] = (json.dumps(value, indent=2) + '\n').encode()
    verifier = 'scripts/context_baseline_reuse.py'
    if verifier in files and before[key] != body:
        def scripts(values):
            return {n: sha(v) for n, v in values.items() if n.startswith('scripts/') and '/' not in n[8:]}
        digest = lambda values: sha(json.dumps(values, sort_keys=True, separators=(',', ':')).encode())
        old = scripts(before)
        expected = digest({n: h for n, h in old.items() if n != verifier})
        current = digest({n: h for n, h in scripts(files).items() if n != verifier})
        text = files[verifier].decode()
        needle = "CURRENT_SOURCES = " + repr(expected)
        if text.count(needle) != 1 or text.count('digest(previous) not in (OLD_SOURCES, ') != 1:
            raise ValueError('Metadata-only context inventory binding requires review')
        text = text.replace(needle, "CURRENT_SOURCES = " + repr(current))
        text = text.replace('digest(previous) not in (OLD_SOURCES, ',
                            'digest(previous) not in (OLD_SOURCES, ' + repr(digest(old)) + ', ')
        files[verifier] = text.encode()
        record['context_inventory_binding'] = dict(previous_exact_inventory=digest(old),
            current_without_verifier=current, source_verifier_only=True)
    files['adapters/sdk-metadata-admission.json'] = (json.dumps(record, indent=2) + '\n').encode()
    return record


def receipt_path(cache):
    import uuid
    return Path(cache) / 'sdk-metadata-admissions' / (uuid.uuid4().hex + '.json')
