"""Bind the sealed assessment client to the selected producer's runtime adapter.

The existing reference/cgroup/recovery client stays byte-identical except for
declared admission and launch bridges. Originals are never edited in place.
"""
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
ADMISSION = ('model_adapter.py', 'runtime_geometry.py', 'runtime_overlay.py',
             'restore_b1_r4.py', 'capacity_snapshot.py', 'project_evidence.py',
             'serving_warmup.py', 'completion_warmup.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p.read_bytes()) for p in folder.rglob('*')
            if p.is_file() and p.name != 'CLIENT-MANIFEST.json' and '__pycache__' not in p.parts}


def verify(record):
    root = Path(record['path'])
    manifest = root / 'CLIENT-MANIFEST.json'
    if manifest.is_symlink() or sha(manifest.read_bytes()) != record['manifest_sha256']:
        raise ValueError('Selected evaluation client manifest changed')
    data = json.loads(manifest.read_bytes())
    if any(p.is_symlink() for p in root.rglob('*')) or inventory(root) != data['files']:
        raise ValueError('Selected evaluation client source inventory changed')
    return data


def files_from_tar(raw, prefix):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as source:
        for member in source:
            p = PurePosixPath(member.name)
            if (not member.isfile() or p.is_absolute() or '..' in p.parts
                    or len(p.parts) < 2 or p.parts[0] != prefix):
                raise ValueError('Unsafe tools archive member')
            name = PurePosixPath(*p.parts[1:]).as_posix()
            if name in result or member.size > 16 * 1024**2:
                raise ValueError('Duplicate or oversized tools archive member')
            result[name] = source.extractfile(member).read()
    manifest = json.loads(result['DEPLOYMENT-TOOLS.json'])
    if manifest['files'] != {n: sha(b) for n, b in result.items() if n != 'DEPLOYMENT-TOOLS.json'}:
        raise ValueError('Producer tools inventory differs from its source manifest')
    return result


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Selected producer/client bridge requires review: ' + old[:70])
    return text.replace(old, new)


def bridge(client, provider):
    """Keep assessment and reference protocols; import producer model admission."""
    result = dict(client)
    for name in ADMISSION:
        key = 'scripts/' + name
        result[key] = provider[key]
    for name, body in provider.items():
        if name.startswith(('engines/', 'mods/', 'overlays/')):
            result[name] = body
    text = result['one_click.py'].decode()
    text = once(text, 'PROFILES = ("r4", "r5", "r6", "r7", "r8")',
                'PROFILES = tuple(__import__("runtime_overlay").RELEASES)')
    result['one_click.py'] = text.encode()
    text = result['scripts/third_party.py'].decode()
    text = once(text, 'choices=["r4", "r5", "r6", "r7", "r8"]',
                'choices=tuple(__import__("runtime_overlay").RELEASES)')
    text = once(text, 'in ("r5", "r6", "r7", "r8"):',
                'in {k for k in __import__("runtime_overlay").RELEASES if k != "r4"}:')
    result['scripts/third_party.py'] = text.encode()

    # The producer owns model selection, temporary storage and runtime env.
    # Retain the client's newer raw reference observation and startup fixes.
    key = 'scripts/engine_entry.py'
    text = result[key].decode()
    upstream = provider[key].decode()
    tree = ast.parse(upstream)
    functions = {n.name: ast.get_source_segment(upstream, n) for n in tree.body
                 if isinstance(n, ast.FunctionDef)}
    target = ast.parse(text)
    for node in reversed(target.body):
        if isinstance(node, ast.FunctionDef) and node.name in ('ipc_directory', 'temporary_directory', 'environment'):
            start = sum(len(x) for x in text.splitlines(keepends=True)[:node.lineno-1])
            end = sum(len(x) for x in text.splitlines(keepends=True)[:node.end_lineno])
            text = text[:start] + functions[node.name] + '\n' + text[end:]
    text = functions['local_runtime_root'] + '\n\n' + text
    text = once(text, '    geometry = apply_geometry(profile, command, env)',
                '    geometry = apply_geometry(profile, command, env)\n'
                '    from model_adapter import configure as select_model\n'
                '    adapter = select_model(profile, args.model, command, env, event_dir=out / "mod-events")')
    text = once(text, '    if geometry is not None:',
                '    if adapter is not None:\n        plan["model_adapter"] = adapter\n    if geometry is not None:')
    result[key] = text.encode()
    for key in ('scripts/engine_entry.py', 'scripts/baseline_entry.py', 'scripts/run_paired.py', 'scripts/third_party.py'):
        text = result[key].decode()
        for expression in ('profile["release"]', 'actual_profile["release"]'):
            # Negative lookbehind is necessary: actual_profile contains profile.
            import re
            text = re.sub(r'(?<![A-Za-z_])' + re.escape(expression) + r' == "full-answer-b1-20261117-r8"',
                          '__import__("runtime_geometry").capacity36(' + expression + ')', text)
        ast.parse(text)
        result[key] = text.encode()
    return result


def prepare(cache, archive, frozen):
    from one_click import verify_kit
    lock = verify_kit(legacy=True)
    legacy = ROOT / 'benchmark/one-click-kit'
    old = json.loads((legacy / 'DEPLOYMENT-TOOLS.json').read_bytes())
    client = {name: (legacy / name).read_bytes() for name in old['files']}
    client['DEPLOYMENT-TOOLS.json'] = (legacy / 'DEPLOYMENT-TOOLS.json').read_bytes()
    tools = [n for n in frozen['archives'] if n.startswith('tools-')]
    if len(tools) != 1:
        raise ValueError('Selected engine has no unique packaged evaluation tools')
    name = tools[0]
    with tarfile.open(archive) as source:
        raw = source.extractfile('vllm_hust_engine-' + frozen['version'] + '/src/vllm_hust_engine/payload/' + name).read()
    if len(raw) != frozen['archives'][name]['bytes'] or sha(raw) != frozen['archives'][name]['sha256']:
        raise ValueError('Packaged producer tools archive changed')
    provider = files_from_tar(raw, 'vllm-hust-engine')
    if sha(provider['DEPLOYMENT-TOOLS.json']) != frozen['tools_manifest_sha256']:
        raise ValueError('Selected producer tools manifest differs')
    if frozen.get('public_api', {}).get('schema') != 'vllm-hust-engine-public-api/v1':
        raise ValueError('Selected release requires a new inference adapter protocol')
    files = bridge(client, provider)
    manifest = dict(schema='deeplinkengine-selected-assessment-client/v1',
                    release=frozen['release'], producer_manifest_sha256=frozen['tools_manifest_sha256'],
                    assessment_manifest_sha256=lock['tools']['manifest_sha256'],
                    files={n: sha(b) for n, b in files.items()},
                    bridges={n: dict(before_sha256=sha(client[n]) if n in client else None, after_sha256=sha(b))
                             for n, b in files.items() if client.get(n) != b})
    raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    key = sha(raw_manifest)
    parent = Path(cache) / 'assessment-clients'
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / key
    record = dict(path=str(root), manifest_sha256=key, release=frozen['release'])
    if root.exists():
        verify(record)
        return record
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=parent) as folder:
        attempt = Path(folder) / 'client'
        attempt.mkdir()
        for name, body in files.items():
            dest = attempt / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
        (attempt / 'CLIENT-MANIFEST.json').write_bytes(raw_manifest)
        try:
            attempt.rename(root)
        except FileExistsError:
            pass
    verify(record)
    return record
