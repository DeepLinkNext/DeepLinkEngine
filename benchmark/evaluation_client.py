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

ROOT = Path(__file__).resolve().parents[1]
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
    if data.get('cpu_metadata_admission'):
        from engine.sdk_dependency_probe import derive
        _, expected = derive((root / 'scripts/dependency_admission.py').read_bytes())
        declared = data['cpu_metadata_admission']
        if (declared != json.loads((root / 'adapters/sdk-metadata-admission.json').read_bytes())
                or any(declared.get(k) != v for k, v in expected.items())
                or sha((root / 'adapters/sdk_dependency_probe.py').read_bytes()) != declared['owned_adapter_source_sha256']):
            raise ValueError('Declared CPU metadata source/PROBE adapter differs')
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
        # The R10 revision only imports the release catalog and changes
        # selected() to admit its named profiles. Timed warmup is unchanged.
        r10_warmup = (name == "scripts/serving_warmup.py" and before == pair[0]
                      and after == "7c39f8ec0cd24efeb3da03c619216d0c453f7eafd13a86e92a404127a7e831c7")
        if (before, after) != pair and not r10_warmup:
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


HISTORICAL_BASELINE_LOCATION_HELPER = '''"""Allow a fixed B0 installation to move without changing its observed identity."""
import hashlib
import json
from pathlib import Path


def verify(recorded, deployment, *, prior_binding=None):
    from baseline_admission import PINNED_SOURCE_MAP_SHA256
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if (recorded.get('source_map_sha256') != PINNED_SOURCE_MAP_SHA256
            or digest(recorded['files']) != PINNED_SOURCE_MAP_SHA256):
        raise ValueError('Reused B0 relocated fixed image/source inventory differs')
    proof = dict(observed_deployment=recorded['deployment'],
        requested_deployment=str(Path(deployment).resolve()),
        source_map_sha256=PINNED_SOURCE_MAP_SHA256,
        scope='Same pinned fixed B0 bytes; original execution path and dates retained.')
    # A sealed historical pair may outlive its temporary SDK. Recheck its
    # exact admission binding; live admission still requires the actual map.
    inventory = Path(deployment) / 'sdk-files.json'
    if prior_binding == proof and not Path(deployment).exists() and not Path(deployment).is_symlink():
        return proof
    current = json.loads(inventory.read_bytes())
    if current != recorded['files']:
        raise ValueError('Reused B0 relocated fixed image/source inventory differs')
    return proof
'''


def historical_mmlu_location_bridge(result):
    """Replay historical MMLU bindings after a temporary SDK expires."""
    result['scripts/baseline_relocation.py'] = HISTORICAL_BASELINE_LOCATION_HELPER.encode()
    key = 'scripts/arm_reuse.py'
    text = result[key].decode()
    text = once(text, 'def validate_conditions(root, role, conditions, inspected):',
        'def validate_conditions(root, role, conditions, inspected, *, prior_location=None):')
    text = once(text, 'def admit(root, role, conditions):',
        'def admit(root, role, conditions, *, prior_location=None):')
    text = once(text, '    validate_conditions(root, role, conditions, data)',
        '    location = validate_conditions(root, role, conditions, data, prior_location=prior_location)')
    text = once(text, "    current = admit(root, binding['role'], binding['conditions'])",
        "    location = binding.get('baseline_location')\n"
        "    options = {'prior_location': location} if location else {}\n"
        "    current = admit(root, binding['role'], binding['conditions'], **options)")
    text = once(text, "            verify_relocation(baseline, conditions['baseline_deployment'])",
        "            return verify_relocation(baseline, conditions['baseline_deployment'], prior_binding=prior_location)")
    text = once(text,
        "    if role == 'B0' and str(Path(read(root / 'baseline-source-admission.json')['deployment']).resolve()) != conditions['baseline_deployment']:\n"
        "        from baseline_relocation import verify as verify_relocation\n"
        "        binding['baseline_location'] = verify_relocation(read(root / 'baseline-source-admission.json'), conditions['baseline_deployment'])",
        "    if location is not None:\n"
        "        binding['baseline_location'] = location")
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
    result['scripts/reference_baseline_reuse.py'] = Path(__file__).with_name('reference_baseline_reuse.py').read_bytes()
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
        '        resume_source = getattr(args, "reference_resume_source", None)')
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
    result[key] = text.encode()


    # Check the observed paths against the original layout and re-hash the
    # reviewed implementation from its fully admitted, byte-identical copy.
    key = 'third-party-test-kit/qualify_baseline_capacity.py'
    text = result[key].decode()
    # The sealed B0 native plan may be paired with any reviewed B1 reference
    # release; the permission set tracks the pinned reference releases.
    text = once(text, '            or plan["paired_engine_release"] != "full-answer-b1-20261117-r8"\n',
        '            or plan["paired_engine_release"] not in ' + native_reference_releases(result) + '\n')
    text = once(text, '    arguments = read("service/client/effective-engine-args.json")',
        '    observed_root = Path(plan["baseline_deployment"]).resolve()\n'
        '    arguments = read("service/client/effective-engine-args.json")')
    text = once(text, '            or Path(plan["baseline_deployment"]).resolve() != root\n',
        '')
    text = once(text, '            or Path(admitted["deployment"]).resolve() != root',
        '            or Path(admitted["deployment"]).resolve() != observed_root')
    text = once(text, '    for package, relative in (("vllm",',
        '    if observed_root != root:\n'
        '        from baseline_relocation import verify as verify_relocation\n'
        '        verify_relocation(admitted, root)\n'
        '    for package, relative in (("vllm",')
    text = once(text, 'Path(admitted["origins"][package]).resolve() != root / relative',
        'Path(admitted["origins"][package]).resolve() != observed_root / relative')
    text = once(text, 'Path(composition["identity"][observed]).resolve() != root / relative',
        'Path(composition["identity"][observed]).resolve() != observed_root / relative')
    result[key] = text.encode()


def warmup_memory_bridge(result):
    """Release bounded cache pressure before any warmup or formal request."""
    result['scripts/warmup_memory_recovery.py'] = Path(__file__).with_name('warmup_memory_recovery.py').read_bytes()
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
    # Raw pressure can exhaust the driver's pinned allocator while the original
    # cache-adjusted working-set gate still passes. Return only our optional
    # startup pages early; retain every original memory and OOM check.
    result['scripts/startup_memory_recovery.py'] = Path(__file__).with_name('startup_memory_recovery.py').read_bytes()
    key = 'scripts/run_b1_r4.py'
    text = result[key].decode()
    anchor = '    # A later warmup still requires the original raw-headroom admission, even\n'
    text = once(text, anchor,
        '    if startup_reserve is not None and not startup_reserve.record["released"]:\n'
        '        from startup_memory_recovery import release_startup_headroom\n'
        '        release_startup_headroom(startup_reserve, lambda: check_memory(guard, initial, root))\n'
        + anchor)
    result[key] = text.encode()


def campaign_pair_weights_bridge(result):
    """Resolve prospectively bound pair arms before reading their weight records."""
    key = 'scripts/reuse_campaign.py'
    text = result[key].decode()
    old = ('        for child in (\n'
           '            [stage / "B0", stage / "B1"] if entry["stage"] == "mmlu-pair" else [stage]\n'
           '        ):')
    new = ('        if entry["stage"] == "mmlu-pair":\n'
           '            from arm_reuse import resolve\n'
           '            pair_plan = read(stage / "frozen-pair-plan.json")\n'
           '            children = [resolve(stage, role, pair_plan) for role in ("B0", "B1")]\n'
           '        elif entry["stage"] in ("mmlu-pair/B0", "mmlu-pair/B1"):\n'
           '            from arm_reuse import resolve\n'
           '            pair = stage.parent\n'
           '            children = [resolve(pair, stage.name, read(pair / "frozen-pair-plan.json"))]\n'
           '        else:\n'
           '            children = [stage]\n'
           '        for child in children:')
    text = once(text, old, new)
    text = once(text, '                    arm = root / "mmlu-pair/B0"',
        '                    from arm_reuse import resolve as resolve_pair_arm\n'
        '                    arm = resolve_pair_arm(root / "mmlu-pair", "B0",\n'
        '                        read(root / "mmlu-pair/frozen-pair-plan.json"))')
    result[key] = text.encode()


def b0_run_policy_bridge(result):
    """Default to sealed fixed-B0 reuse at every model-launch boundary."""
    result['scripts/b0_run_policy.py'] = Path(__file__).with_name('b0_run_policy.py').read_bytes()
    key = 'scripts/third_party.py'
    text = result[key].decode()
    text = once(text, '        composed_pairs = [stage for stage in stages if stage["name"] == "mmlu-pair"',
        '        if not args.prepare_only:\n'
        '            from b0_run_policy import require_campaign_reuse\n'
        '            require_campaign_reuse(stages, locals().get("reused", {}), out, frozen)\n'
        '        composed_pairs = [stage for stage in stages if stage["name"] == "mmlu-pair"')
    result[key] = text.encode()
    key = 'scripts/automatic_reference.py'
    text = result[key].decode()
    text = once(text, '        plan = json.loads(path.read_bytes())',
        '        plan = json.loads(path.read_bytes())\n'
        '        from b0_run_policy import require_reference_reuse\n'
        '        require_reference_reuse(plan)')
    result[key] = text.encode()
    key = 'scripts/acceptance_suite.py'
    text = result[key].decode()
    text = once(text, 'def run(args, checkpoint=None):\n',
        'def run(args, checkpoint=None):\n'
        '    from b0_run_policy import require_context_reuse\n'
        '    require_context_reuse(args)\n')
    result[key] = text.encode()


def device_change_bridge(result):
    """Bind changed physical pairs while retaining each original measurement."""
    result['scripts/device_reuse.py'] = Path(__file__).with_name('device_reuse.py').read_bytes()

    def edit(name, replacements):
        key = 'scripts/' + name + '.py'
        text = result[key].decode()
        for old, new in replacements:
            text = once(text, old, new)
        ast.parse(text)
        result[key] = text.encode()

    option = '    parser.add_argument("--allow-device-change", action="store_true")\n'
    physical = '    parser.add_argument("--physical-devices",'
    edit('third_party', [
        (physical, option + physical),
        ('    from compute_reference import prepare as prepare_reference',
         '    if getattr(args, "allow_device_change", False):\n'
         '        frozen["device_reuse_mapping"] = json.loads((out / "device-mapping.json").read_bytes())\n'
         '    from compute_reference import prepare as prepare_reference'),
        ('            wire_dependencies(out, stages, reused)',
         '            wire_dependencies(out, stages, reused, allow_device_change=getattr(args, "allow_device_change", False))'),
        ('                new_inference_stages=len(stages) - len(reused),',
         '                new_inference_stages=len(stages) - len(reused),\n'
         '                **({"device_reuse": "explicit_ascend_910b2_continuation"} if getattr(args, "allow_device_change", False) else {}),'),
    ])
    edit('reuse_campaign', [
        ('def identity(plan):', 'def identity(plan, *, compare_devices=True):'),
        ('    result["engine_release"] =',
         '    if not compare_devices:\n        result.pop("physical_cards")\n    result["engine_release"] ='),
        ('    if identity(previous) != identity(current):',
         '    from device_reuse import for_campaign\n\n'
         '    device_change = for_campaign(root, previous, current)\n'
         '    if identity(previous, compare_devices=not device_change) != identity(current, compare_devices=not device_change):'),
        ('        keys = ("model", "physical_cards", "logical_devices",',
         '        keys = ("model", "logical_devices",'),
        ('        for row in receipt["stages"]:',
         '        compare_devices = plan.get("device_reuse_mapping") is None\n'
         '        different_runtime = identity(previous, compare_devices=compare_devices) != identity(plan, compare_devices=compare_devices)\n'
         '        for row in receipt["stages"]:'),
        ('            if identity(previous) != identity(plan) and name == "mmlu-pair":',
         '            if different_runtime and name == "mmlu-pair":'),
        ('            if identity(previous) != identity(plan) and not name.endswith("-B0"):',
         '            if different_runtime and not name.endswith("-B0"):'),
        ('            if identity(previous) != identity(plan) and "-judge-" in name:',
         '            if different_runtime and "-judge-" in name:'),
        ('            origin["stages"].append(name)',
         '            origin["stages"].append(name)\n'
         '        if origin["stages"] or origin.get("paired_arms"):\n'
         '            from device_reuse import for_campaign\n'
         '            device_change = for_campaign(root, previous, plan)\n'
         '            if device_change:\n                origin["device_change"] = device_change'),
        ('"scope": "Same-host composition of earlier sealed native stages; not one new uninterrupted inference run.",',
         '"scope": "Composition of earlier sealed native stages with each source retaining its original device mapping.",'),
        ('        for role in origin.get("paired_arms", []):',
         '        if origin["stages"] or origin.get("paired_arms"):\n'
         '            from device_reuse import for_campaign\n'
         '            if origin.get("device_change") != for_campaign(source_root, previous, plan):\n'
         '                raise ValueError("Reused stage device provenance differs")\n'
         '        for role in origin.get("paired_arms", []):'),
        ('def wire_dependencies(out, stages, reused):',
         'def wire_dependencies(out, stages, reused, *, allow_device_change=False):'),
        ('            args[index] = str(Path(entry["source_root"]) / "continuous-B1")',
         '            args[index] = str(Path(entry["source_root"]) / "continuous-B1")\n'
         '            if allow_device_change and stage["name"] not in reused and stage["script"] == "run_b1_r4.py":\n'
         '                args.append("--allow-device-change")'),
    ])
    for name, indent, error in (
        ('run_b1_r4', '        ', 'Continuous qualification belongs to another physical NPU pair'),
        ('run_native_reference', '        ', 'Current native qualification is bound to another physical pair'),
    ):
        failure = ('            raise ValueError(\n'
                   '                "' + error + '"\n'
                   '            )') if name == 'run_b1_r4' else '            raise ValueError("' + error + '")'
        edit(name, [
            ('    parser.add_argument("--qualified-continuous",', option + '    parser.add_argument("--qualified-continuous",'),
            ('        if qualification["physical_cards"] != cards():\n' + failure,
             indent + 'from device_reuse import for_continuous\n'
             + indent + 'device_change = for_continuous(args.qualified_continuous, qualification, args.devices,\n'
             + indent + '                               allowed=getattr(args, "allow_device_change", False))\n'
             + indent + 'if device_change:\n'
             + indent + '    save(root / "continuous-device-reuse.json", device_change)'),
        ])
    edit('automatic_reference', [
        ('        baseline_source = getattr(args, "reference_baseline_source", None)',
         '        if frozen.get("device_reuse_mapping") is not None:\n'
         '            plan["device_reuse_mapping"] = frozen["device_reuse_mapping"]\n'
         '        baseline_source = getattr(args, "reference_baseline_source", None)'),
        ('                with (folder / "http-current-work.log").open("x") as log:',
         '                if plan.get("device_reuse_mapping"):\n'
         '                    command.append("--allow-device-change")\n'
         '                with (folder / "http-current-work.log").open("x") as log:'),
        ('                if baseline_native:\n                    command +=',
         '                if not baseline_native and plan.get("device_reuse_mapping"):\n'
         '                    command.append("--allow-device-change")\n'
         '                if baseline_native:\n                    command +='),
    ])
    # Context verification and plotting do not contribute to reference measurements.
    for name in ('reference_model_reuse', 'reference_qualification_reuse'):
        edit(name, [
            ('"scripts/live_console.py",',
             '"scripts/live_console.py", "scripts/context_baseline_reuse.py", "scripts/acceptance_report.py",'),
        ])
    edit('reference_model_reuse', [
        ('        path = source / directory / "receipt.json"',
         '        if stage == "baseline-native-synchronized-timing" and old.get("baseline_reuse"):\n'
         '            from reference_baseline_reuse import resolve\n\n'
         '            if "baseline-native-reuse.json" not in seal or row.get("status") != "completed":\n'
         '                raise ValueError("Prior fixed B0 native reuse is unsealed or incomplete")\n'
         '            proof = resolve(source)\n'
         '            if (not proof or row.get("receipt_sha256") != proof["receipt_sha256"]\n'
         '                    or row.get("original_stage_source") != proof["source"]):\n'
         '                raise ValueError("Prior fixed B0 native reuse receipt differs")\n'
         '            continue\n'
         '        path = source / directory / "receipt.json"'),
    ])
    edit('full_reference_reuse', [
        ('    save(folder / "matched-reference-pair.json", result)',
         '    if plan.get("device_reuse_mapping"):\n'
         '        from device_reuse import compare, read_mapping\n'
         '        b0 = read_mapping(native_sources["B0"])\n'
         '        b1 = read_mapping(native_sources["B1"])\n'
         '        if any(b1[key] != plan["device_reuse_mapping"][key] for key in ("physical_cards", "logical_cards")):\n'
         '            raise ValueError("Current native timing used a different frozen device pair")\n'
         '        device_change = compare(b0, b1, allowed=True)\n'
         '        if device_change:\n'
         '            result["native_device_change"] = device_change\n'
         '            result["reference_reuse_scope"] = "Original complete reference denominator; original B0 and new B1 work and timing independently audited on the recorded Ascend 910B2 pairs."\n'
         '    save(folder / "matched-reference-pair.json", result)'),
    ])
    edit('acceptance_suite', [
        ('    p.add_argument("--reuse-b0", type=Path)', '    p.add_argument("--allow-device-change", action="store_true")\n    p.add_argument("--reuse-b0", type=Path)'),
        ('    if list(map(int, args.physical_devices.split(","))) != cards:\n'
         '        raise ValueError("Reuse uses another physical card pair")',
         '    if list(map(int, args.physical_devices.split(","))) != cards:\n'
         '        if role != "B0" or not getattr(args, "allow_device_change", False):\n'
         '            raise ValueError("Reuse uses another physical card pair")\n'
         '        from device_reuse import compare, read_mapping\n'
         '        from resource_guard import verify_mapping\n\n'
         '        original = read_mapping(prior)\n'
         '        if original["physical_cards"] != cards:\n'
         '            raise ValueError("Original context mapping differs from its resource admission")\n'
         '        source_binding["device_change"] = compare(original, verify_mapping(args.devices), allowed=True)'),
        ('def paired(arms):', 'def paired(arms, *, device_change=None):\n    proof = None'),
        ('        if name == "resource-admission.json":\n'
         '            left, right = left["physical_cards"], right["physical_cards"]\n'
         '        if left != right:\n'
         '            raise ValueError("Paired cards or model weights differ")',
         '        if name == "resource-admission.json":\n'
         '            left, right = left["physical_cards"], right["physical_cards"]\n'
         '            if left != right and device_change:\n'
         '                from device_reuse import compare, read_mapping\n\n'
         '                original, current = (read_mapping(arms[role]) for role in ("B0", "B1"))\n'
         '                proof = compare(original, current, allowed=True)\n'
         '                if (not proof or original["physical_cards"] != left or current["physical_cards"] != right\n'
         '                        or any(proof[side][key] != device_change[side][key]\n'
         '                               for side in ("original", "current") for key in ("physical_cards", "logical_cards"))):\n'
         '                    raise ValueError("Paired context device provenance differs")\n'
         '                continue\n'
         '        if left != right:\n'
         '            raise ValueError("Paired cards or model weights differ")'),
        ('        acceptance="Measurements collected;', '        **({"device_change": proof} if proof else {}),\n'
         '        acceptance="Measurements collected;'),
        ('    from b0_run_policy import require_context_reuse\n    require_context_reuse(args)',
         '    from b0_run_policy import require_context_reuse\n    require_context_reuse(args)\n'
         '    if getattr(args, "allow_device_change", False):\n'
         '        os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices'),
        ('        save(output / "paired-summary.json", paired(arms))',
         '        binding = output / "reused-B0-context-provenance.json"\n'
         '        device_change = json.loads(binding.read_bytes()).get("device_change") if binding.is_file() else None\n'
         '        if device_change:\n'
         '            state["device_change"] = device_change\n'
         '        save(output / "paired-summary.json", paired(arms, device_change=device_change) if device_change else paired(arms))'),
    ])
    edit('acceptance_report', [
        ('    checked = paired({role: Path(path) for role, path in data["arms"].items()})',
         '    arms = {role: Path(path) for role, path in data["arms"].items()}\n'
         '    checked = paired(arms, device_change=data.get("device_change"))'),
    ])
    edit('third_party_report', [
        ('        "current_mmlu_pair": pair,',
         '        **({"execution_devices": {\n'
         '            "current": plan["physical_cards"],\n'
         '            "stages": {row["name"]: (\n'
         '                row.get("source", {}).get("physical_cards", plan["physical_cards"])) for row in observed},\n'
         '        }} if plan and plan.get("device_reuse_mapping") else {}),\n'
         '        "current_mmlu_pair": pair,'),
        ('    scope_complete = (set(',
         '    device_note = ""\n'
         '    if data.get("execution_devices"):\n'
         '        device_note = "<h2>物理设备</h2><p>跨卡阶段保留各自原始设备。跨卡差值包含设备更换的影响。</p><pre>" + esc(json.dumps(data["execution_devices"], ensure_ascii=False, indent=2)) + "</pre>"\n'
         '    scope_complete = (set('),
        ('<h2>本次执行清单</h2>', '{device_note}<h2>本次执行清单</h2>'),
    ])
    edit('reuse_campaign', [
        ('                "source_stage_receipt": row,\n                "verified_at": time.time(),',
         '                "source_stage_receipt": row,\n'
         '                **({"physical_cards": previous["physical_cards"]} if not compare_devices else {}),\n'
         '                "verified_at": time.time(),'),
        ('                or rows[name]["status"] != "completed"',
         '                or rows[name]["status"] != "completed"\n'
         '                or entry.get("physical_cards", previous["physical_cards"]) != previous["physical_cards"]'),
    ])


def http_state_drain_bridge(result):
    """Recheck the existing cold-state boundary when a new cache operation wins the RPC race."""
    key = 'third-party-test-kit/proc_capture/m01_http_capture.py'
    old = '                operation = await self.core("state_cache", {"kind": "drop", "key": key})'
    new = (
        '                try:\n'
        '                    operation = await self.core("state_cache", {"kind": "drop", "key": key})\n'
        '                except Exception as error:\n'
        '                    if str(error) != "Call to state_cache method failed: incremental cache transaction is still in flight":\n'
        '                        raise\n'
        '                    # The scheduler rejected this command before queuing a drop.\n'
        '                    # A new snapshot must retire any reappearing State operations.\n'
        '                    record.setdefault("drop_dispatch_conflicts", []).append(key)\n'
        '                    await asyncio.sleep(.05)\n'
        '                    continue'
    )
    result[key] = once(result[key].decode(), old, new).encode()
    evidence_key = 'scripts/reference_evidence.py'
    result[evidence_key] += (
        '\n\ndef http_state_drain_fix_only(original, current, pin):\n'
        '    """Qualify only the reviewed pre-capture change against the original source pin."""\n'
        '    if not original.is_file() or sha(original) != pin:\n'
        '        return False\n'
        '    source = original.read_text()\n'
        f'    old, new = {old!r}, {new!r}\n'
        '    return source.count(old) == 1 and current.read_text() == source.replace(old, new)\n'
    ).encode()
    for name in ('reference_model_reuse', 'reference_qualification_reuse'):
        script = 'scripts/' + name + '.py'
        text = result[script].decode()
        text = once(text, 'from reference_evidence import transient_cache,',
                    'from reference_evidence import http_state_drain_fix_only, transient_cache,')
        anchor = '            if (relative == "scripts/run_b1_r4.py" and current_pin == sha(current)'
        source_pin = 'pin' if name == 'reference_model_reuse' else 'expected'
        text = once(text, anchor,
            f'            if (relative == "{key}" and current_pin == sha(current)\n'
            f'                    and http_state_drain_fix_only(path, current, {source_pin})):\n'
            '                continue\n' + anchor)
        result[script] = text.encode()


def stable_r10_provider(files):
    return b'R10_STABLE_RELEASE = "full-answer-b1-20261121-r10"' in files['scripts/runtime_overlay.py']


def native_reference_releases(files):
    names = ('full-answer-b1-20261117-r8', 'full-answer-b1-20261119-r9')
    if stable_r10_provider(files):
        names += ('full-answer-b1-20261121-r10',)
    if reviewed_r10_3072_provider(files):
        names += (R10_3072_REFERENCE_RELEASE,)
    if reviewed_r11_provider(files):
        names += (R11_REFERENCE_RELEASE,)
    return json.dumps(names).replace('[', '(').replace(']', ')')


R10_REFERENCE_RELEASE = 'full-answer-b1-20261121-r10'
R10_REFERENCE_OVERLAY_SHA256 = '96ecd0418ada8f8f7ea6a91af875676f2249c34a5559efcaf707496cf5ecb749'
R10_REFERENCE_SOURCES = {
    'runtime/ascend/vllm_ascend/ops/gdn.py': (
        'b1-r10', 'eaba7292690dd75dfbdb58616aedbe73a004dbad8a98021ef77f23e41bc04a73'),
    'mods/betterscale/betterscale/models/qwen35/service_metadata.py': (
        'b1-r10', '42de8c1014fcdefa0ce61c1965db1c1956f552766c4c2be23949b9af4ff3ec92'),
    'mods/betterscale/betterscale/models/qwen35/mixed_core.py': (
        'b1-r10-contiguous-restore', 'a9981e438e52702be1f3ad6b03a8e0f4c1d2e8b38161140c11a5b7896c10829b'),
}


R10_3072_REFERENCE_RELEASE = 'full-answer-b1-20261124-r10'
R10_3072_OVERLAY = 'engines/b1-r10-3072-overlay.json'
R10_3072_OVERLAY_SHA256 = 'eebe8d188e7a82527335a2bd40a7b00e7ab915be4b43e47251c2565bf1040c3c'
R10_3072_RUNTIME_SHA256 = '7d0bc533e0106b9c5d7ea19f9c2cb9a3a6242dfbf6cc73ead63513f6bb53a5fa'
R10_3072_MICROFLOW = 'mods/microflow/hust_prefill_microflow/runtime/microflow.py'
R10_3072_MICROFLOW_SHA256 = 'ea2fd1c91daf88df47bab95d2b21050311bbd5a13058f56688f9a01f09b04d23'


def reviewed_r10_3072_provider(files):
    if R10_3072_OVERLAY not in files:
        return False
    raw = files[R10_3072_OVERLAY]
    microflow = 'overlays/b1-r10-3072/' + R10_3072_MICROFLOW
    if (sha(raw) != R10_3072_OVERLAY_SHA256
            or sha(files.get(microflow, b'')) != R10_3072_MICROFLOW_SHA256):
        raise ValueError('Revised R10 reference implementation requires source review')
    overlay = json.loads(raw)
    if (overlay['release'] != R10_3072_REFERENCE_RELEASE
            or overlay['base_manifest_sha256'] != R10_REFERENCE_OVERLAY_SHA256
            or overlay['measurement']['installed_files_canonical_sha256'] != R10_3072_RUNTIME_SHA256):
        raise ValueError('Revised R10 reference runtime profile requires source review')
    return True


R10_3072_STACK = 'mods/active-stack-r10-3072.json'


def r10_3072_report_catalog(files, frozen):
    """Build the full-report MOD catalog from the reviewed producer metadata."""
    geometry = frozen['runtime_geometry']
    overlay = json.loads(files[R10_3072_OVERLAY])
    if (not reviewed_r10_3072_provider(files)
            or frozen['release'] != R10_3072_REFERENCE_RELEASE
            or geometry['mixed_capture_tokens'] != 3072
            or frozen['measured_runtime'] != overlay['measurement']
            or frozen['publication_scope'] != overlay['measurement']['scope']):
        raise ValueError('Revised R10 assessment profile requires source review')

    parent_path = 'mods/active-stack-r10-stable.json'
    parent = json.loads(files[parent_path])
    shipped = frozen['included_mods']
    parent_mods = {row['id']: row for row in parent['enabled_mods']}
    if (parent['release'] != R10_REFERENCE_RELEASE or len(shipped) != len(parent_mods)
            or set(shipped) != set(parent_mods)
            or any(row['manifest'] not in files for row in parent_mods.values())):
        raise ValueError('Revised R10 MOD catalog differs from the reviewed base and package')
    enabled = [dict(parent_mods[name], enabled_in='r10-3072', badge='r10-3072 included')
               for name in shipped if name != 'intern-s2']
    for row in enabled:
        if row['id'] == 'microflow':
            row['reviewed_runtime_sha256'] = R10_3072_MICROFLOW_SHA256
    stack = dict(
        schema=parent['schema'], release=frozen['release'], version=frozen['version'],
        status='derived-from-reviewed-producer', default_engine='r10-3072',
        model_adapters=['qwen35'], runtime_geometry=geometry, runtime_overlay=R10_3072_OVERLAY,
        runtime_overlay_sha256=R10_3072_OVERLAY_SHA256, measured_runtime=frozen['measured_runtime'],
        parent_catalog=parent_path, parent_catalog_sha256=sha(files[parent_path]),
        package_included_mods=shipped, enabled_mods=enabled,
        excluded=[*parent['excluded'], 'intern-s2'], publication_scope=frozen['publication_scope'],
    )
    files[R10_3072_STACK] = (json.dumps(stack, sort_keys=True, indent=2) + '\n').encode()



R11_REFERENCE_RELEASE = 'full-answer-b1-20261126-r11'
R11_OVERLAY = 'engines/b1-r11-overlay.json'
R11_OVERLAY_SHA256 = 'bd31dbdbe3a740c8d9da49f80ea89403225a9c6157a001f78352be3e65401d2d'
R11_RUNTIME_SHA256 = '882d24fd1232b3c39d9360b6212e720450743ac7b522b24fa14b8b29bb7d5682'
R11_RUNTIME_DELTA = {'mods/betterscale/betterscale/models/qwen35/prefill_round_robin.py': '71d8b19cd1fc7b020065a2065a327a156735b5c312d9866edcfc12e45ea6345b', 'mods/microflow/hust_prefill_microflow/school028/host-contract.json': '668916cc8067f57f16a51be1941cc084079c649d678f67d62c052a1916fe7449'}
R11_STACK = 'mods/active-stack-r11.json'


def reviewed_r11_provider(files):
    if R11_OVERLAY not in files:
        return False
    raw = files[R11_OVERLAY]
    if not reviewed_r10_3072_provider(files) or sha(raw) != R11_OVERLAY_SHA256:
        raise ValueError('R11 scheduler requires the reviewed overlay')
    overlay = json.loads(raw)
    if (overlay['release'] != R11_REFERENCE_RELEASE
            or overlay['base_release'] != R10_3072_REFERENCE_RELEASE
            or overlay['base_manifest_sha256'] != R10_3072_OVERLAY_SHA256
            or overlay['measurement']['installed_files_canonical_sha256'] != R11_RUNTIME_SHA256
            or {name: row['after_sha256'] for name, row in overlay['files'].items()} != R11_RUNTIME_DELTA):
        raise ValueError('R11 overlay is outside the measured source domain')
    for name, digest in R11_RUNTIME_DELTA.items():
        if sha(files.get('overlays/b1-r11/' + name, b'')) != digest:
            raise ValueError('R11 deployed scheduler source differs: ' + name)
    return True


def r11_report_catalog(files, frozen):
    overlay = json.loads(files[R11_OVERLAY])
    stack = json.loads(files[R11_STACK])
    rows = {row['id']: row for row in stack['enabled_mods']}
    if (not reviewed_r11_provider(files)
            or frozen['release'] != stack['release'] or stack['release'] != R11_REFERENCE_RELEASE
            or frozen['measured_runtime'] != overlay['measurement']
            or frozen['runtime_geometry']['mixed_capture_tokens'] != 3072
            or set(frozen['included_mods']) != set(rows)
            or any(row['manifest'] not in files for row in rows.values())):
        raise ValueError('R11 report catalog differs from the reviewed producer')
    stack.update(version=frozen['version'], model_adapters=['qwen35'],
                 runtime_geometry=frozen['runtime_geometry'], runtime_overlay=R11_OVERLAY,
                 runtime_overlay_sha256=R11_OVERLAY_SHA256,
                 measured_runtime=frozen['measured_runtime'],
                 enabled_mods=[row for name, row in rows.items() if name != 'intern-s2'],
                 compatibility_profiles=frozen['compatibility_profiles'])
    files[R11_STACK] = (json.dumps(stack, sort_keys=True, indent=2) + '\n').encode()


def r10_reference_source_bridge(result):
    """Bind the stable native-RMS variant to the reviewed arithmetic proofs."""
    if not stable_r10_provider(result):
        return
    profile = json.loads(result['engines/b1-r10-stable.json'])
    overlay = 'engines/b1-r10-contiguous-restore-overlay.json'
    if (profile['release'] != R10_REFERENCE_RELEASE or profile['runtime_overlay'] != overlay
            or profile['model_adapters']['qwen35']['restore_gated_rms'] is not False
            or sha(result[overlay]) != R10_REFERENCE_OVERLAY_SHA256):
        raise ValueError('Stable R10 reference runtime profile requires source review')
    for relative, (directory, pin) in R10_REFERENCE_SOURCES.items():
        if sha(result['overlays/' + directory + '/' + relative]) != pin:
            raise ValueError('Stable R10 reference implementation requires source review: ' + relative)

    revised = reviewed_r10_3072_provider(result)
    r11 = reviewed_r11_provider(result)
    helper = 'reference_source_domains.py'
    result['third-party-test-kit/' + helper] = Path(__file__).with_name(helper).read_bytes()
    for script, proof, read_native, output_field in (
        ('qualify_dense_source', 'canonical-dense-tail-source-domain-v1', 'read', 'retained_source_sha256'),
        ('qualify_necessary_work', 'necessary-arithmetic-source-domain-v1',
         'lambda name: sealed_read(native, name)', 'source_sha256'),
    ):
        protocol_path = 'protocols/' + proof + '.json'
        original = result[protocol_path]
        protocol = json.loads(original)
        protocol['stable_r10_source_domain'] = dict(
            release=R10_REFERENCE_RELEASE, overlay_sha256=R10_REFERENCE_OVERLAY_SHA256,
            sources={name: pin for name, (_, pin) in R10_REFERENCE_SOURCES.items()
                     if name in protocol['sources']['B1']},
            scope='Native RMS uses the reviewed convolution, recurrence and projections. '
                  'Contiguous restore copies the same values without changing the reference arithmetic.')
        if revised:
            source_pins = dict(protocol['stable_r10_source_domain']['sources'])
            if R10_3072_MICROFLOW in protocol['sources']['B1']:
                source_pins[R10_3072_MICROFLOW] = R10_3072_MICROFLOW_SHA256
            protocol['r10_3072_source_domain'] = dict(
                release=R10_3072_REFERENCE_RELEASE, overlay_sha256=R10_3072_OVERLAY_SHA256,
                installed_files_canonical_sha256=R10_3072_RUNTIME_SHA256, sources=source_pins,
                scope='3072-token capture adds a bitwise-qualified three-chunk partition. '
                      'The reviewed tensor arithmetic and fresh timing requirements are unchanged.')
        if r11:
            inherited = protocol['r10_3072_source_domain']
            protocol['r11_source_domain'] = dict(
                release=R11_REFERENCE_RELEASE, overlay_sha256=R11_OVERLAY_SHA256,
                installed_files_canonical_sha256=R11_RUNTIME_SHA256,
                sources=inherited['sources'], guarded_runtime_files=R11_RUNTIME_DELTA,
                scope='The accepted scheduler and host contract preserve the reviewed tensor arithmetic. '
                      'Native synchronized clocks and same-workload HTTP observation are still required.')
        result[protocol_path] = (json.dumps(protocol, indent=2) + '\n').encode()

        key = 'third-party-test-kit/' + script + '.py'
        imports = 'admitted_sources'
        if script == 'qualify_necessary_work':
            imports += ', source_domain_for_release, verify_http_adapter'
        text = once(result[key].decode(), '\n\nPROTOCOL =',
                    f'\nfrom reference_source_domains import {imports}\n\n\nPROTOCOL =')
        text = once(text, f'PROTOCOL_SHA256 = "{sha(original)}"',
                    f'PROTOCOL_SHA256 = "{sha(result[protocol_path])}"')
        text = once(text, '    for name, pin in protocol["sources"][role].items():',
                    f'    sources = admitted_sources(protocol, role, deployment, {read_native})\n'
                    '    for name, pin in sources.items():')
        text = once(text, f'{output_field}=protocol["sources"][role]', f'{output_field}=sources')
        if script == 'qualify_necessary_work':
            text = once(text, '    if sealed_read(original, "installed-files.json") != installed:',
                        '    deployment_release = json.loads((deployment / "deployment.json").read_bytes())["release"]\n'
                        '    if source_domain_for_release(protocol, deployment_release) is not None:\n'
                        '        verify_http_adapter(sealed_read(original, "service/plan.json"), deployment_release)\n'
                        '    if sealed_read(original, "installed-files.json") != installed:')
        result[key] = text.encode()


def http_replay_bridge(result):
    """Reuse an independently sealed HTTP observation with prior native clocks."""
    key = 'third-party-test-kit/public_output_binding.py'
    text = result[key].decode()
    text = once(text, 'plan["engine_release"] != "full-answer-b1-20261117-r8"',
        'plan["engine_release"] not in ' + native_reference_releases(result))
    result[key] = text.encode()

    key = 'scripts/reference_model_reuse.py'
    text = result[key].decode()
    text = once(text, '"qualification_reuse", "stage_resume", "canonical_reuse")',
                '"qualification_reuse", "stage_resume", "canonical_reuse", "observation_reuse")')
    result[key] = text.encode()

    key = 'scripts/reference_observation_reuse.py'
    text = result[key].decode()
    text = once(text, '    "scripts/reference_qualification_reuse.py", "scripts/run_native_reference.py", "scripts/third_party.py",',
        '    "scripts/reference_qualification_reuse.py", "scripts/run_native_reference.py", "scripts/third_party.py",\n'
        '    "scripts/reference_observation_reuse.py", "scripts/context_baseline_reuse.py", "scripts/acceptance_report.py",')
    result[key] = text.encode()

    key = 'scripts/automatic_reference.py'
    text = result[key].decode()
    text = once(text, '            if resume_source:\n                raise ValueError("Choose a completed model-stage source or a separate HTTP observation source")',
        '            if resume and "current-http-work-observation" in resume["stages"]:\n'
        '                raise ValueError("Completed model-stage continuation already supplies a sealed HTTP observation")')
    text = once(text, '                    if "current-http-work-observation" not in resumed_models["stages"]:',
        '                    if ("current-http-work-observation" not in resumed_models["stages"]\n'
        '                            and not plan.get("observation_reuse")):')
    result[key] = text.encode()


def managed_shutdown_bridge(result):
    """Change only completed B1 HTTP cleanup and preserve prior source bindings."""
    from benchmark.managed_shutdown import DIRECT_STOP, MANAGED_STOP

    module = 'scripts/managed_shutdown.py'
    result[module] = Path(__file__).with_name('managed_shutdown.py').read_bytes()
    key = 'scripts/run_b1_r4.py'
    result[key] = once(result[key].decode(), DIRECT_STOP, MANAGED_STOP).encode()

    key = 'scripts/reference_evidence.py'
    result[key] = once(result[key].decode(),
        '    old, new = ast.parse(original.read_text()), ast.parse(current.read_text())\n    removed = 0',
        '    from managed_shutdown import without_managed_stop\n'
        '    original_body, current_body = original.read_bytes(), current.read_bytes()\n'
        '    if without_managed_stop(original_body) == original_body:\n'
        '        current_body = without_managed_stop(current_body)\n'
        '        if current_body == original_body:\n'
        '            return True\n'
        '    old, new = ast.parse(original_body), ast.parse(current_body)\n'
        '    removed = 0').encode()

    key = 'scripts/reference_observation_reuse.py'
    text = once(result[key].decode(), 'from dataset_contract import sha',
        'from dataset_contract import sha\nfrom managed_shutdown import matches_pre_shutdown_controller')
    result[key] = once(text,
        '        if Path(name).is_absolute() or not path.resolve().is_relative_to(current_root) or sha(path) != pin:',
        '        if Path(name).is_absolute() or not path.resolve().is_relative_to(current_root):\n'
        '            raise ValueError("HTTP observation implementation escapes client: " + name)\n'
        '        if (sha(path) != pin and (name != "scripts/run_b1_r4.py"\n'
        '                or not matches_pre_shutdown_controller(path, pin))):').encode()


NATIVE_SAMPLE_REPLACEMENTS = (
    ('from owned_processes import Owned\n', 'from owned_processes import Owned, state\n'),
    ('def run(args):\n', """def retired_sample_owner(pid, before):
    if pid not in before:
        return False
    try:
        current = state(pid)
    except (FileNotFoundError, ProcessLookupError):
        return True
    return current["start_ticks"] == before[pid] and current["state"] == "Z"


def run(args):
"""),
    ('                            sample = subprocess.run(["npu-smi", "info"],',
     '                            before = {row["pid"]: row["start_ticks"] for row in owned.scan()}\n'
     '                            sample = subprocess.run(["npu-smi", "info"],'),
    ('                                    if pid not in known:\n',
     '                                    if pid not in known and not retired_sample_owner(pid, before):\n'),
)


def native_owner_sampling_bridge(result):
    """Allow a sampled owned PID to disappear while the device query returns."""
    key = 'scripts/run_native_reference.py'
    source = result[key].decode()
    for old, new in NATIVE_SAMPLE_REPLACEMENTS:
        source = once(source, old, new)
    ast.parse(source)
    result[key] = source.encode()

    name = 'scripts/reference_model_reuse.py'
    text = result[name].decode()
    restore = ('def native_monitor_matches(path, expected):\n'
               '    body = path.read_bytes().decode()\n'
               '    for old, new in ' + repr(NATIVE_SAMPLE_REPLACEMENTS) + ':\n'
               '        if body.count(new) != 1:\n'
               '            return False\n'
               '        body = body.replace(new, old, 1)\n'
               '    from hashlib import sha256\n'
               '    return sha256(body.encode()).hexdigest() == expected\n\n\n')
    text = once(text, 'def load(source, expected_seal, current_plan, current_root, recovery_directory=None):',
                restore + 'def load(source, expected_seal, current_plan, current_root, recovery_directory=None):')
    failure = '            raise ValueError("Completed model-stage implementation/dependency differs: " + str(current))'
    text = once(text, failure,
        '            if (relative == "scripts/run_native_reference.py" and current_pin == sha(current)\n'
        '                    and native_monitor_matches(current, pin)):\n'
        '                continue\n' + failure)
    result[name] = text.encode()


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
        # R10 preserves the B0 warmup and input recipes. The producer-side
        # changes select the measured contiguous-restore profile for B1.
        'dfd6b42f2fc1bf076f89ac71a0745d94f7dd57fe0d49615e334ab9c04fce2083',
        # R10 3072 changes only B1 capture, model adapter, geometry and overlay
        # source in this producer inventory. B0 input and clocks remain pinned.
        '4f45f7cd53fc93a14eb28b8b647093a13420285d192b07ff14ee664f6d608e74',
        # R11 changes the producer B1 dispatch, capture geometry and release admission.
        'faffcf39d286301a5ed48e5914adf9711700433b66c7c59f10d971b6f197bac4',
    }
    if digest(scripts(result)) not in reviewed_sources:
        raise ValueError('B0 context assessment-source compatibility requires review')
    native_baseline_bridge(result)
    warmup_memory_bridge(result)
    campaign_pair_weights_bridge(result)
    b0_run_policy_bridge(result)
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
    historical_mmlu_location_bridge(result)
    device_change_bridge(result)
    http_state_drain_bridge(result)
    http_replay_bridge(result)
    key = 'scripts/context_baseline_reuse.py'
    def previous_policy_sources():
        pins = scripts(result)
        pins['scripts/b0_run_policy.py'] = '6d0ffa5640db2e93f99bd6d31e32f5cab257425bbc85daeb62dd8a3b4dd7acb5'
        return pins

    previous = previous_policy_sources()
    previous[key] = sha(CONTEXT_BASELINE_HELPER.replace('__CURRENT_SOURCES__', digest(previous)).encode())
    managed_shutdown_bridge(result)
    managed = previous_policy_sources()
    helper = once(CONTEXT_BASELINE_HELPER, 'digest(previous) != OLD_SOURCES',
                  'digest(previous) not in (OLD_SOURCES, ' + repr(digest(previous)) + ')')
    # Preserve the exact controller inventory from before this native-only change.
    managed[key] = sha(helper.replace('__CURRENT_SOURCES__', digest(managed)).encode())
    native_owner_sampling_bridge(result)
    current = digest(previous_policy_sources())
    helper = once(helper, 'digest(previous) not in (OLD_SOURCES, ',
                  'digest(previous) not in (OLD_SOURCES, ' + repr(digest(managed)) + ', ')
    result[key] = helper.replace('__CURRENT_SOURCES__', current).encode()
    # The previous controller was also emitted with the reviewed -B SDK probe.
    from engine.sdk_dependency_probe import SOURCE, derive
    legacy_pins = previous_policy_sources()
    prior = digest(legacy_pins)
    revised_probe, _ = derive(result[SOURCE])
    legacy_pins[SOURCE] = sha(revised_probe)
    del legacy_pins[key]
    legacy_helper = once(helper, 'digest(previous) not in (OLD_SOURCES, ',
                         'digest(previous) not in (OLD_SOURCES, ' + repr(prior) + ', ')
    legacy_pins[key] = sha(legacy_helper.replace('__CURRENT_SOURCES__', digest(legacy_pins)).encode())
    compatible = (prior, digest(legacy_pins))
    automatic_b0_bridge(result)
    helper = once(helper, 'digest(previous) not in (OLD_SOURCES, ',
                  'digest(previous) not in (OLD_SOURCES, ' + ', '.join(map(repr, compatible)) + ', ')
    pins = scripts(result)
    del pins[key]
    result[key] = helper.replace('__CURRENT_SOURCES__', digest(pins)).encode()


def automatic_b0_bridge(result):
    """Bind automatic B0 discovery at the three sealed admission boundaries."""
    def edit(name, pairs):
        key = "scripts/" + name + ".py"
        text = result[key].decode()
        for old, new in pairs:
            text = once(text, old, new)
        ast.parse(text)
        result[key] = text.encode()

    edit("third_party", [
        ('        if getattr(args, "reuse_completed_campaign", []) and not args.prepare_only:',
         '        from b0_run_policy import candidates as baseline_candidates\n'
         '        auto_roots = baseline_candidates(out, "campaign") if not args.prepare_only else []\n'
         '        if (getattr(args, "reuse_completed_campaign", []) or auto_roots) and not args.prepare_only:'),
        ('            reused = prepare_reuse(args.reuse_completed_campaign, out, frozen, stages)',
         '            reused = prepare_reuse(getattr(args, "reuse_completed_campaign", []), out, frozen, stages,\n'
         '                                   automatic_roots=auto_roots)'),
        ('            frozen["reuse_manifest_sha256"] = sha(out / "reused-stages.json")',
         '            manifest = out / "reused-stages.json"\n'
         '            if manifest.exists():\n'
         '                frozen["reuse_manifest_sha256"] = sha(manifest)'),
        ('                evidence_mode="composition_of_sealed_native_runs",',
         '                evidence_mode=("composition_of_sealed_native_runs" if manifest.exists() else "new_native_execution"),'),
    ])
    edit("reuse_campaign", [
        ('def prepare(roots, out, plan, stages):\n    if not roots:\n        return {}',
         '''def prepare(roots, out, plan, stages, *, automatic_roots=()):
    from b0_run_policy import probe, check_current_weights
    roots = [Path(path).resolve() for path in roots]
    automatic_roots = [path for path in dict.fromkeys(Path(p).resolve() for p in automatic_roots)
                       if path not in roots]
    if not roots and not automatic_roots:
        return {}
    rejected, checked_weights = [], {}'''),
        ('    for index, path in enumerate(roots):\n        root, previous, receipt, bindings = source(path)',
         '''    def sealed_candidate(path):
        inspected = source(path)
        _, previous, receipt, _ = inspected
        if (receipt["status"] not in ("completed", "failed")
                or not isinstance(receipt["stages"], list)
                or not previous["baseline_deployment"]
                or not previous["baseline_source_map_sha256"]):
            raise ValueError("Prior campaign lacks a stopped and sealed B0 identity")
        return inspected

    for path in [*roots, *automatic_roots]:
        automatic = path not in roots
        inspected = probe(path, sealed_candidate, rejected) if automatic else source(path)
        if inspected is None:
            continue
        index = len(origins)
        root, previous, receipt, bindings = inspected'''),
        ('            if name not in wanted or row["status"] != "completed":\n                continue',
         '''            if name not in wanted or row["status"] != "completed":
                continue
            if automatic and (name in entries or "-judge-" in name or
                    (name != "mmlu-pair" and not name.endswith("-B0"))):
                continue'''),
        ('            if different_runtime and name == "mmlu-pair":\n                compatible("mmlu-pair/B0", previous, plan, root)',
         '''            if (different_runtime or automatic) and name == "mmlu-pair":
                if automatic and "B0" in paired_arms:
                    continue'''),
        ('''                    arm = resolve_pair_arm(root / "mmlu-pair", "B0",
                        read(root / "mmlu-pair/frozen-pair-plan.json"))
                    conditions = expected_arm(plan["optimized_deployment"], plan["baseline_deployment"],
                        plan["model"], plan["physical_cards"], plan["logical_devices"],
                        read(arm / "model-weights.json"))
                    binding = admit_arm(arm, "B0", conditions)''',
         '''                    def verify_pair(pair):
                        compatible("mmlu-pair/B0", previous, plan, root)
                        arm = resolve_pair_arm(pair, "B0", read(pair / "frozen-pair-plan.json"))
                        weights = read(arm / "model-weights.json")
                        if automatic:
                            weights = check_current_weights(plan["model"], weights, checked_weights)
                        conditions = expected_arm(plan["optimized_deployment"], plan["baseline_deployment"],
                            plan["model"], plan["physical_cards"], plan["logical_devices"], weights)
                        return admit_arm(arm, "B0", conditions)
                    pair = root / "mmlu-pair"
                    binding = probe(pair, verify_pair, rejected) if automatic else verify_pair(pair)
                    if binding is None:
                        continue'''),
        ('            compatible(name, previous, plan, root)\n            audit_stage(root, name)',
         '''            def verify_stage(path):
                compatible(name, previous, plan, root)
                audit_stage(root, name)
                if automatic:
                    check_current_weights(plan["model"], read(path / "model-weights.json"), checked_weights)
                return True
            if automatic:
                if probe(root / name, verify_stage, rejected) is None:
                    continue
            else:
                verify_stage(root / name)'''),
        ('    if not entries and not paired_arms and not explicitly_selected_arms:',
         '''    if automatic_roots:
        plan["automatic_b0_campaign"] = {
            "candidates": [str(path) for path in automatic_roots], "rejected": rejected,
            "reused": [name for name in entries if name.endswith("-B0") or name == "mmlu-pair"]
                      + (["mmlu-pair/B0"] if "B0" in paired_arms else []),
        }
    if not entries and not paired_arms and not explicitly_selected_arms and not roots:
        return {}
    if not entries and not paired_arms and not explicitly_selected_arms:'''),
    ])
    edit("automatic_reference", [
        ('''        baseline_source = getattr(args, "reference_baseline_source", None)
        if baseline_source:''',
         '''        baseline_source = getattr(args, "reference_baseline_source", None)
        from b0_run_policy import decision, mode, select
        if baseline_source:
            plan["baseline_execution_policy"] = decision("reference", baseline_source, [])
        elif not getattr(args, "reference_resume_source", None):
            def verify_candidate(candidate):
                from reference_baseline_reuse import inspect as inspect_candidate
                identity = dict(source=str(candidate), checksums_sha256=sha(candidate / "checksums.json"))
                inspect_candidate(identity["source"], identity["checksums_sha256"], plan)
                return identity

            if mode() == "auto" and not getattr(args, "prepare_only", False):
                _, identity, proof = select(out, "reference", verify_candidate)
                plan["baseline_execution_policy"] = proof
                if identity:
                    plan["baseline_reuse"] = identity
            else:
                plan["baseline_execution_policy"] = decision("reference", None, [])
        if baseline_source:'''),
    ])
    edit("acceptance_suite", [
        ('''            if prior:
                root, _, _ = admit_reuse(prior, role, args, contract)
                stage = dict(role=role, run=str(root), reused=True, returncode=0)''',
         '''            accepted = None
            if role == "B0":
                from b0_run_policy import decision, select

                if prior:
                    proof = decision("context", prior, [])
                else:
                    prior, accepted, proof = select(
                        output, "context", lambda source: admit_reuse(source, "B0", args, contract))
                state["baseline_execution_policy"] = proof
                save(output / "execution.json", state)
            if prior:
                root, _, _ = accepted if accepted is not None else admit_reuse(prior, role, args, contract)
                stage = dict(role=role, run=str(root), reused=True, returncode=0)'''),
    ])




def revised_microflow_partition_bridge(result):
    """Use the reviewed four-size equivalence contract for revised r10."""
    context = "scripts/context_baseline_reuse.py"

    def context_inventory():
        sources = {n: sha(raw) for n, raw in result.items() if n.startswith("scripts/")
                   and "/" not in n[8:] and n != context}
        return sha(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode())

    original_inventory = context_inventory()
    key = "scripts/original_audit.py"
    text = result[key].decode()
    text = once(text, 'def audit_activation(files, prefix="retrieval-b1-r1/", expected_requests=7):',
        "def partition_tokens_for_release(deployment, installed):\n"
        "    if deployment.get('release') not in " + repr((R10_3072_REFERENCE_RELEASE, R11_REFERENCE_RELEASE)) + ":\n"
        "        return None\n"
        "    if installed.get(" + repr(R10_3072_MICROFLOW) + ") != " + repr(R10_3072_MICROFLOW_SHA256) + ":\n"
        "        raise ValueError('Revised r10 Microflow partition source differs')\n"
        "    return {1024, 2048, 3072, 4096}\n\n\n"
        'def audit_activation(files, prefix="retrieval-b1-r1/", expected_requests=7, expected_partition_tokens=None):')
    text = once(text, '''        if {r["tokens"] for r in equivalence} != {1024, 2048, 4096} or any(
            r.get("exact") != [True] * 5 or r.get("rank") != rank for r in equivalence
        ):''', '''        expected = {1024, 2048, 4096} if expected_partition_tokens is None else expected_partition_tokens
        if expected_partition_tokens is not None and (
            len(equivalence) != len(expected)
            or any(type(r.get("tokens")) is not int or type(r.get("rank")) is not int
                   or not isinstance(r.get("exact"), list)
                   or any(flag is not True for flag in r["exact"]) for r in equivalence)
        ):
            raise ValueError("Revised Microflow partition evidence must be unique and exact")
        if {r["tokens"] for r in equivalence} != expected or any(
            r.get("exact") != [True] * 5 or r.get("rank") != rank for r in equivalence
        ):''')
    result[key] = text.encode()

    key = "scripts/qualification.py"
    text = result[key].decode()
    text = once(text, "from original_audit import audit_activation, reconstruct",
                "from original_audit import audit_activation, partition_tokens_for_release, reconstruct")
    text = once(text, "        expected_requests=len(rows)\n",
                '        expected_partition_tokens=partition_tokens_for_release(read("deployment.json"), read("installed-files.json")),\n'
                "        expected_requests=len(rows)\n")
    result[key] = text.encode()

    key = "scripts/report_execution.py"
    text = result[key].decode()
    text = once(text, '        joint_activation = audit_activation(\n            activation_files,\n',
                "        from original_audit import partition_tokens_for_release\n"
                "        joint_activation = audit_activation(\n"
                "            activation_files,\n"
                "            expected_partition_tokens=partition_tokens_for_release(\n"
                '                json.loads((root / "deployment.json").read_bytes()),\n'
                '                json.loads((root / "installed-files.json").read_bytes())),\n')
    result[key] = text.encode()
    if context in result:
        # Reports and qualification do not change B0 execution. Preserve the
        # original historical allowlist and bind this exact derived inventory.
        text = result[context].decode()
        text = once(text, "CURRENT_SOURCES = " + repr(original_inventory),
                    "CURRENT_SOURCES = " + repr(context_inventory()))
        result[context] = text.encode()
    for key in ("scripts/original_audit.py", "scripts/qualification.py", "scripts/report_execution.py"):
        compile(result[key], key, "exec")


REFERENCE_OPERATOR_HELPER = "third-party-test-kit/reference_operator_devices.py"
REFERENCE_OPERATOR_SOURCES = {
    "qualify_gdn_reference.py": (
        "2e2dd34b3d8e5bfc37e832ab99bf650c0c3bde4e58815517e24858a936ad1b74", "output", '"'),
    "qualify_gdn_prefill_reference.py": (
        "330a025d34482f8fa36551d0a49aae7b7f98c47248c172ac6014e692fb9dbc09", "out", '"'),
    "qualify_route_capture.py": (
        "d5e35eba3cb27b99dc5100b54780fcd411e98cff52e8e9c2b1322c96fe01e424", "out", "'"),
}


def reference_operator_devices_bridge(result):
    """Use mask-local ordinals while retaining the verified host mapping."""
    bundle = "third-party-test-kit/"
    helper = Path(__file__).with_name(Path(REFERENCE_OPERATOR_HELPER).name)
    result[REFERENCE_OPERATOR_HELPER] = helper.read_bytes()
    for name, (expected, output, quote) in REFERENCE_OPERATOR_SOURCES.items():
        key = bundle + name
        if sha(result[key]) != expected:
            raise ValueError("Reference operator device binding requires source review: " + name)
        text = result[key].decode()
        original_mapping = f'        write({output} / "mapping.json", guard.verify_mapping(plan["logical_devices"]))'
        text = once(text, original_mapping,
                    '        mapping = guard.verify_mapping(plan["logical_devices"])\n'
                    f'        write({output} / "mapping.json", mapping)')
        original_loop = (
            f'        for device, card in zip(map(int, plan["logical_devices"].split({quote},{quote})), '
            'plan["physical_cards"], strict=True):'
        )
        text = once(text, original_loop,
                    '        from reference_operator_devices import device_bindings\n\n'
                    '        bindings = device_bindings(mapping, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),\n'
                    '                                   torch.npu.device_count())\n'
                    '        for device, logical, card in bindings:')
        text = once(text, "logical_device=device,",
                    "logical_device=logical, torch_device_ordinal=device,")
        compile(text, key, "exec")
        result[key] = text.encode()


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
                    'not in ' + native_reference_releases(result) + ':')
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
    r10_reference_source_bridge(result)
    from engine.sdk_dependency_probe import client_bridge
    client_bridge(result)
    if reviewed_r10_3072_provider(result):
        revised_microflow_partition_bridge(result)
        reference_operator_devices_bridge(result)
    return result


def prepare(cache, archive, frozen):
    from benchmark.one_click import verify_kit
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
    if frozen['release'] == R10_3072_REFERENCE_RELEASE:
        r10_3072_report_catalog(files, frozen)
    if frozen['release'] == R11_REFERENCE_RELEASE:
        r11_report_catalog(files, frozen)
    manifest = dict(schema='deeplinkengine-selected-assessment-client/v1',
                    release=frozen['release'], producer_manifest_sha256=frozen['tools_manifest_sha256'],
                    assessment_manifest_sha256=lock['tools']['manifest_sha256'],
                    files={n: sha(b) for n, b in files.items()},
                    bridges={n: dict(before_sha256=sha(client[n]) if n in client else None, after_sha256=sha(b))
                             for n, b in files.items() if client.get(n) != b})
    manifest['cpu_metadata_admission'] = json.loads(files['adapters/sdk-metadata-admission.json'])
    if R10_3072_STACK in files:
        manifest['r10_3072_report_catalog'] = R10_3072_STACK
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


def metadata_client(cache, *, release, record=None, selection=None):
    """Bind current admission to a sealed selection without changing its original files."""
    from engine.sdk_dependency_probe import client_bridge
    prior = verify(record) if record else {}
    if (release == R10_3072_REFERENCE_RELEASE
            and (prior.get('r10_3072_report_catalog') != R10_3072_STACK
                 or prior.get('files', {}).get(REFERENCE_OPERATOR_HELPER) != sha(
                     Path(__file__).with_name(Path(REFERENCE_OPERATOR_HELPER).name).read_bytes()))):
        if selection is None or selection['release'] != release:
            raise ValueError('Revised R10 client update requires the selected producer archive')
        from engine.environment_setup import cached_wheel
        archive = cached_wheel(selection['sdist'], [], Path(cache) / 'wheels')
        return prepare(cache, archive, selection['frozen'])
    if record:
        if prior.get('cpu_metadata_admission'):
            return record
        root = Path(record['path'])
        files = {n: (root / n).read_bytes() for n in prior['files']}
        manifest = {k: v for k, v in prior.items() if k not in ('files', 'bridges')}
        old_bridges = prior.get('bridges', {})
    else:
        from benchmark.one_click import verify_kit
        lock = verify_kit(legacy=True)
        root = ROOT / 'benchmark/one-click-kit'
        originals = json.loads((root / 'DEPLOYMENT-TOOLS.json').read_bytes())
        files = {n: (root / n).read_bytes() for n in originals['files']}
        files['DEPLOYMENT-TOOLS.json'] = (root / 'DEPLOYMENT-TOOLS.json').read_bytes()
        manifest = dict(schema='deeplinkengine-selected-assessment-client/v1', release=release,
                        assessment_manifest_sha256=lock['tools']['manifest_sha256'],
                        legacy_metadata_only=True)
        old_bridges = {}
    before = dict(files)
    guard = client_bridge(files)
    manifest.update(files={n: sha(b) for n, b in files.items()},
        cpu_metadata_admission=guard, parent_client=record,
        bridges={**old_bridges, **{n: dict(before_sha256=sha(before[n]) if n in before else None,
                    after_sha256=sha(b)) for n, b in files.items() if before.get(n) != b}})
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    parent = Path(cache) / 'assessment-clients'
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / sha(raw)
    result = dict(path=str(output), manifest_sha256=sha(raw), release=release)
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix='metadata-', dir=parent) as folder:
            attempt = Path(folder) / 'client'
            attempt.mkdir()
            for name, body in files.items():
                target = attempt / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            (attempt / 'CLIENT-MANIFEST.json').write_bytes(raw)
            try:
                attempt.rename(output)
            except FileExistsError:
                pass
    verify(result)
    return result
