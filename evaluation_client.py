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


# Reviewed source-progress additions leave argv/warmup/clocks unchanged. R9
# then broadens the release predicate and routes the unchanged capacity-36 policy through runtime_geometry.capacity36. Pin both sides;
# a future warmup/argv change requires review instead of inheriting permission.
BASELINE_COMPATIBILITY = {
    "scripts/serving_warmup.py": (
        "fc2bcb64dbdb8780d5cf0add7d26adecd64043326263d7da7be241954db173ba",
        "37cc6f16fc16b33d04eaa9c45a68576d250c534eff5b1d8f372138ae992c5c3f"),
    "scripts/baseline_entry.py": (
        "184fc9efff9461dc801c1aaf783eaeb3bf6efc872ef5deb1872c65cdd24a7477",
        "8d81ec530a05ac4d3ed28bd51423f514c2f3f99fb591ebabd6c6f661f6f4f739"),
}


def baseline_reuse_compatibility(client, combined):
    key = "protocols/task-warmup-v3-c8-compatibility.json"
    compat = json.loads(client[key])
    for name, pair in BASELINE_COMPATIBILITY.items():
        before, after = sha(client[name]), sha(combined[name])
        if before == after:
            continue
        if (before, after) != pair:
            raise ValueError("Baseline reuse bridge requires source review: " + name)
        target = (compat if name.endswith("/serving_warmup.py") else
                  compat["unchanged_continuous_source_adaptations"][name])
        target["previous_sha256"] = sorted(set(target["previous_sha256"] + [before]))
        target["current_sha256"] = after
    baseline = compat["unchanged_continuous_source_adaptations"]["scripts/baseline_entry.py"]
    if baseline["current_sha256"] != "8d81ec530a05ac4d3ed28bd51423f514c2f3f99fb591ebabd6c6f661f6f4f739":
        raise ValueError("Baseline source-progress compatibility requires source review")
    baseline["previous_sha256"] = sorted(set(baseline["previous_sha256"] + [
        "efa3fbb979eed315238c859a6d44459fb02838d3f35e6d33d107b8a3524348ee",
        "860ebda40197fa05f43d28c6cb5748d677a42a01d379564f8beaebebab7ee3a1"]))
    compat["r9_change"] = ("Reviewed source-preparation progress and release-predicate/capacity36 routing; "
        "Qwen R8/R9 share the same cohorts, budgets and continuous protocol. "
        "B0 reuse still audits effective serve argv, recipe, source and weights.")
    combined[key] = (json.dumps(compat, indent=2) + "\n").encode()


BASELINE_LOCATION_HELPER = '''"""Allow a fixed B0 installation to move without changing its observed identity."""
import hashlib
import json
from pathlib import Path


def verify(recorded, deployment):
    from baseline_admission import PINNED_SOURCE_MAP_SHA256
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    # Controllers run baseline_admission.admit (full actual-file and package
    # verification) before calling reuse. This additionally binds the old full
    # inventory to the currently admitted fixed image, never to a B1 companion.
    current = json.loads((Path(deployment) / 'sdk-files.json').read_bytes())
    if (recorded.get('source_map_sha256') != PINNED_SOURCE_MAP_SHA256
            or digest(recorded['files']) != PINNED_SOURCE_MAP_SHA256
            or digest(current) != PINNED_SOURCE_MAP_SHA256
            or current != recorded['files']):
        raise ValueError('Reused B0 relocated fixed image/source inventory differs')
    return dict(observed_deployment=recorded['deployment'],
        requested_deployment=str(Path(deployment).resolve()),
        source_map_sha256=PINNED_SOURCE_MAP_SHA256,
        scope='Same pinned fixed B0 bytes; original execution path and dates retained.')
'''


def baseline_location_bridge(result):
    """Use full pinned-image identity rather than an SDK's installation path."""
    result['scripts/baseline_relocation.py'] = BASELINE_LOCATION_HELPER.encode()
    key = 'scripts/reuse_campaign.py'
    text = result[key].decode()
    text = once(text, '        keys = ("model", "physical_cards", "logical_devices", "baseline_deployment",',
        '        if previous.get("baseline_deployment") != current.get("baseline_deployment"):\n'
        '            from baseline_relocation import verify as verify_relocation\n'
        '            from baseline_admission import PINNED_SOURCE_MAP_SHA256\n'
        '            if previous.get("baseline_source_map_sha256") != PINNED_SOURCE_MAP_SHA256 or current.get("baseline_source_map_sha256") != PINNED_SOURCE_MAP_SHA256:\n'
        '                raise ValueError("Reuse baseline fixed source pin differs")\n'
        '            verify_relocation(read(root / stage / "baseline-source-admission.json"), current["baseline_deployment"])\n'
        '        keys = ("model", "physical_cards", "logical_devices",')
    text = once(text, '            "parent_status": receipt["status"],',
        '            "parent_status": receipt["status"],\n'
        '            "original_baseline_deployment": previous["baseline_deployment"],\n'
        '            "requested_baseline_deployment": plan["baseline_deployment"],\n'
        '            "baseline_source_map_sha256": previous["baseline_source_map_sha256"],')
    result[key] = text.encode()
    key = 'scripts/arm_reuse.py'
    text = result[key].decode()
    text = once(text, "                or str(Path(baseline['deployment']).resolve()) != conditions['baseline_deployment']):",
                '                ):')
    text = once(text, "            raise ValueError('Reused B0 actual pinned image/source identity differs')",
        "            raise ValueError('Reused B0 actual pinned image/source identity differs')\n"
        "        if str(Path(baseline['deployment']).resolve()) != conditions['baseline_deployment']:\n"
        "            from baseline_relocation import verify as verify_relocation\n"
        "            verify_relocation(baseline, conditions['baseline_deployment'])")
    text = once(text, '    return binding\n',
        "    if role == 'B0' and str(Path(read(root / 'baseline-source-admission.json')['deployment']).resolve()) != conditions['baseline_deployment']:\n"
        "        from baseline_relocation import verify as verify_relocation\n"
        "        binding['baseline_location'] = verify_relocation(read(root / 'baseline-source-admission.json'), conditions['baseline_deployment'])\n"
        '    return binding\n')
    text = once(text, "for key in ('original_started', 'original_release', 'weights', 'original_quality'):",
        "for key in ('original_started', 'original_release', 'weights', 'original_quality', 'baseline_location'):")
    text = once(text, "        if current[key] != binding[key]:", "        if current.get(key) != binding.get(key):")
    result[key] = text.encode()


CONTEXT_BASELINE_HELPER = '''"""Admit the reviewed fixed-B0 context measurement without relabelling it."""
import hashlib
import json
from pathlib import Path

OLD_SOURCES = '026d8526833090abce577801e4f6a767f6aea086e34c75fbb8716a59a2afe19d'
CURRENT_SOURCES = '__CURRENT_SOURCES__'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(prior, args, current):
    read = lambda path: json.loads(Path(path).read_bytes())
    previous = read(prior / 'controller-source.json')
    # Old66-file source inventory was checked against its complete original
    # candidate-v12 source. Only the reviewed R9 controller and its declared
    # fixed-B0 reference reuse bridge may adapt it.
    # Current inventory excludes this generated verifier to avoid a self hash.
    adapted = previous != current
    observed = {n: h for n, h in current.items() if n != 'scripts/context_baseline_reuse.py'}
    if adapted and (digest(previous) != OLD_SOURCES or digest(observed) != CURRENT_SOURCES):
        raise ValueError('B0 context controller sources require compatibility review')
    from runtime_geometry import capacity36
    from completion_warmup import protocol
    release = read(Path(args.deployment) / 'deployment.json')['release']
    plan = read(prior / 'service/plan.json')
    if plan.get('serving_warmup') != protocol('continuous', acceptance=True, capacity36=capacity36(release)):
        raise ValueError('B0 context shared warmup protocol differs')
    mapping = read(prior / 'device-mapping.json')
    if mapping['logical_cards'] != [int(n) for n in args.devices.split(',')]:
        raise ValueError('B0 context logical card mapping differs')
    from reuse_campaign import verify_baseline_serve_args
    verify_baseline_serve_args('continuous-B0', prior / 'service/plan.json')
    old_companion = read(prior / 'source-admission.json')
    return dict(schema='deeplinkengine-fixed-b0-context-reuse/v1',
        original_run=str(prior), original_companion=old_companion,
        requested_companion=str(Path(args.deployment).resolve()),
        original_sources_sha256=digest(previous), current_sources_sha256=digest(current),
        source_adaptation=adapted, shared_warmup=plan['serving_warmup'],
        scope='Original fixed B0 dates, actual runtime, raw phases and responses retained; companion B1 identity is not B0 identity.')
'''


def native_baseline_bridge(result):
    """Keep the original B0 native clocks when only the optimized arm changes."""
    result['scripts/reference_baseline_reuse.py'] = (ROOT / 'reference_baseline_reuse.py').read_bytes()
    key = 'scripts/third_party.py'
    text = result[key].decode()
    text = once(text, '    parser.add_argument("--reference-qualification-source", type=Path,',
        '    parser.add_argument("--reference-baseline-source", type=Path, help="Reuse a complete sealed fixed-B0 native arm; invalid evidence fails without launching B0")\n'
        '    parser.add_argument("--reference-qualification-source", type=Path,')
    result[key] = text.encode()
    key = 'scripts/automatic_reference.py'
    text = result[key].decode()
    text = once(text, '        resume_source = getattr(args, "reference_resume_source", None)',
        '        baseline_source = getattr(args, "reference_baseline_source", None)\n'
        '        if baseline_source:\n'
        '            from reference_baseline_reuse import inspect as inspect_baseline\n'
        '            baseline_source = Path(baseline_source).resolve()\n'
        '            identity = dict(source=str(baseline_source), checksums_sha256=sha(baseline_source / "checksums.json"))\n'
        '            inspect_baseline(identity["source"], identity["checksums_sha256"], plan)\n'
        '            plan["baseline_reuse"] = identity\n'
        '        resume_source = getattr(args, "reference_resume_source", None)\n'
        '        if baseline_source and resume_source:\n'
        '            raise ValueError("Choose fixed-B0 reuse or complete model-stage continuation")')
    text = once(text, '            if resumed_models and stage["id"] in resumed_models["stages"]:',
        '            if stage["id"] == "baseline-native-synchronized-timing" and plan.get("baseline_reuse"):\n'
        '                row["status"] = "running"\n'
        '                save(folder / "status.json", result)\n'
        '                from reference_baseline_reuse import inspect as inspect_baseline\n'
        '                identity = plan["baseline_reuse"]\n'
        '                proof = inspect_baseline(identity["source"], identity["checksums_sha256"], plan)\n'
        '                save(folder / "baseline-native-reuse.json", proof)\n'
        '                row.update(status="completed", returncode=0, reused_model_stage=True,\n'
        '                    original_stage_source=proof["source"], receipt_sha256=proof["receipt_sha256"],\n'
        '                    full_model_certificate=False, model_inference_rerun=False)\n'
        '                save(folder / "status.json", result)\n'
        '                print("THIRD_PARTY_REFERENCE_FIXED_B0_REUSED", proof["source"], flush=True)\n'
        '                continue\n'
        '            if resumed_models and stage["id"] in resumed_models["stages"]:')
    result[key] = text.encode()
    key = 'scripts/reference_final_stages.py'
    text = result[key].decode()
    text = once(text, '    return dict(B0=Path(folder) / "native-baseline-timing", B1=Path(folder) / "native-current-timing")',
        '    from reference_baseline_reuse import resolve\n'
        '    proof = resolve(folder)\n'
        '    return dict(B0=Path(proof["source"]) if proof else Path(folder) / "native-baseline-timing",\n'
        '                B1=Path(folder) / "native-current-timing")')
    text = once(text, '        keywords = dict(baseline_deployment=plan["baseline_deployment"]) if role == "B0" else {}',
        '        baseline = plan["baseline_deployment"]\n'
        '        if role == "B0" and (origin / "baseline-native-reuse.json").exists():\n'
        '            from reference_baseline_reuse import resolve\n'
        '            baseline = resolve(origin)["original_baseline_deployment"]\n'
        '        keywords = dict(baseline_deployment=baseline) if role == "B0" else {}')
    result[key] = text.encode()


def warmup_memory_bridge(result):
    """Release bounded cache pressure before any warmup or formal request."""
    result['scripts/warmup_memory_recovery.py'] = (ROOT / 'warmup_memory_recovery.py').read_bytes()
    key = 'scripts/checkpoint_cache.py'
    text = result[key].decode()
    text = once(text,
        'at the healthy, idle service boundary. No pressure allocation runs against a\n    live model and no buffer survives into warmup or a measured request.',
        'at the healthy, idle service boundary. Bounded cache recovery may run only\n    at that idle boundary; no buffer survives into warmup or a measured request.')
    old = ('        value=self.sample()\n'
           '        self.record[\'warmup_admission\']=value\n'
           '        self.record[\'required_warmup_raw_bytes\']=12*GIB\n'
           '        self.record[\'warmup_raw_basis\']=\'two frozen4GiB/rank State budgets plus original4GiB raw reserve\'\n'
           '        self.save()\n'
           '        if value[\'limit\']-value[\'current\'] < 12*GIB:\n'
           '            raise RuntimeError(\'Less than12GiB raw memory before warmup after startup reserve release; no formal request submitted\')')
    new = ('        value=self.sample()\n'
           '        required=24*GIB\n'
           '        self.record[\'required_warmup_raw_bytes\']=required\n'
           '        self.record[\'warmup_raw_basis\']=\'two frozen4GiB/rank State budgets plus16GiB transient host allocator/cache headroom; original working-set and OOM gates unchanged\'\n'
           '        if value[\'limit\']-value[\'current\'] < required:\n'
           '            from warmup_memory_recovery import reclaim_idle_headroom\n'
           '            recovery=self.path.with_name(\'memory-recovery-before-warmup.json\')\n'
           '            self.record[\'healthy_idle_recovery\']=reclaim_idle_headroom(self.sample,self.check,required,recovery)\n'
           '            value=self.sample()\n'
           '        self.record[\'warmup_admission\']=value\n'
           '        self.save()\n'
           '        if value[\'limit\']-value[\'current\'] < required:\n'
           '            raise RuntimeError(\'Less than24GiB raw memory before warmup after idle cache recovery; no formal request submitted\')')
    text = once(text, old, new)
    result[key] = text.encode()


def context_baseline_bridge(result):
    """Preserve completed fixed-B0 context phases across a B1-only change."""
    def scripts(files):
        return {name: sha(body) for name, body in files.items()
                if name.startswith('scripts/') and '/' not in name[len('scripts/'): ]}
    def digest(value):
        return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    # The second inventory changes only run_b1_r4.service_admission_phase:
    # read the child's source-preparation marker before sampling its clock.
    # Formal requests, warmup and all measurement clocks are unchanged.
    reviewed_sources = {
        '378c81fc3b0d27ccbf0f9a78bbae401053a0a4d6b318d0bfa921669dd82fce0f',
        '67bf9560a03a0b24ce7d0678fb742aa3fe57c5276fe376c74918d1bcbdac96fb',
        # Same pinned-copy payload and validators; only network import allowance/logs.
        'ee46611022f688bb96adb5a7604bcfedc02888b416dfcc366e546c15182d082b',
    }
    if digest(scripts(result)) not in reviewed_sources:
        raise ValueError('B0 context assessment-source compatibility requires review')
    native_baseline_bridge(result)
    warmup_memory_bridge(result)
    key = 'scripts/acceptance_suite.py'
    text = result[key].decode()
    old = ('    if admission["deployment"] != str(args.deployment.resolve()):\n'
           '        raise ValueError("Reuse uses a different deployment")\n'
           '    if json.loads((prior / "controller-source.json").read_bytes()) != provenance():\n'
           '        raise ValueError("Reuse controller code differs; no silent adaptation")')
    new = ('    if role == "B0":\n'
           '        from context_baseline_reuse import validate as validate_baseline_sources\n'
           '        source_binding = validate_baseline_sources(prior, args, provenance())\n'
           '    else:\n'
           '        if admission["deployment"] != str(args.deployment.resolve()):\n'
           '            raise ValueError("Reuse uses a different deployment")\n'
           '        if json.loads((prior / "controller-source.json").read_bytes()) != provenance():\n'
           '            raise ValueError("Reuse controller code differs; no silent adaptation")')
    text = once(text, old, new)
    text = once(text, '        if target != captured:\n            raise ValueError("Reuse baseline identity differs")',
        '        from baseline_relocation import verify as verify_baseline_location\n'
        '        location = verify_baseline_location(captured, args.baseline_deployment)\n'
        '        if target["versions"] != captured["versions"] or target["variant"] != captured["variant"]:\n'
        '            raise ValueError("Reuse baseline identity differs")\n'
        '        source_binding.update(baseline_location=location, seal_sha256=sha(prior / "checksums.json"),\n'
        '            original_started=receipt["started"], original_ended=receipt["ended"],\n'
        '            complete_original_phases_verified=True)\n'
        '        save(args.out / "reused-B0-context-provenance.json", source_binding)')
    result[key] = text.encode()
    current = digest(scripts(result))
    result['scripts/context_baseline_reuse.py'] = CONTEXT_BASELINE_HELPER.replace('__CURRENT_SOURCES__', current).encode()


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
        if isinstance(node, ast.FunctionDef) and node.name == 'environment' :
            start = sum(len(x) for x in text.splitlines(keepends=True)[:node.lineno-1])
            end = sum(len(x) for x in text.splitlines(keepends=True)[:node.end_lineno])
            text = text[:start] + functions[node.name] + '\n' + text[end:]
    # Preserve the assessment client's reviewed local storage repair, including
    # its explicit producer-compatible /tmp or /dev/shm selection.
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
    # These clients keep the same Qwen MMLU200 native work/clock protocol.
    # R9 inherits its Qwen worker and capacity-36 geometry; do not treat the
    # producer's additional Intern adapter as a new reference workload.
    for key in ('scripts/native_timing_prepare.py', 'scripts/baseline_native_prepare.py'):
        text = result[key].decode()
        text = once(text, '!= "full-answer-b1-20261117-r8":',
                    'not in ("full-answer-b1-20261117-r8", "full-answer-b1-20261119-r9"):')
        text = text.replace('requires the frozen r8 configuration', 'requires an admitted frozen Qwen capacity-36 configuration')
        text = text.replace('requires the frozen r8 campaign', 'requires an admitted frozen Qwen capacity-36 campaign')
        result[key] = text.encode()
    baseline_reuse_compatibility(client, result)
    key = 'protocols/task-warmup-v3-c8-compatibility.json'
    compat = json.loads(result[key])
    current_inputs = sha(result['scripts/continuous_inputs.py'])
    if current_inputs != '8bef6275f3bc75cfbb8f49333b7c64060da683ca961d24435d22783e09160562':
        raise ValueError('Baseline input CLI compatibility requires source review')
    compat['unchanged_continuous_source_adaptations']['scripts/continuous_inputs.py'] = dict(
        previous_sha256=['f9787d060f4740b21d570a077652c0e7eaac7bd08b3d8882f5214e118fc6f186'],
        current_sha256=current_inputs,
        scope='Only imports and a CLI result-file channel were added; prepare() and original input recipe are byte-identical. All frozen dataset/task/budget/source checks remain required.')
    result[key] = (json.dumps(compat, indent=2) + '\n').encode()
    baseline_location_bridge(result)
    context_baseline_bridge(result)
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
