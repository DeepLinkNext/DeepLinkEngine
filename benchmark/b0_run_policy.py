"""Select an admitted B0 source before deciding whether to measure the stage."""
import hashlib
import json
import os
from pathlib import Path

ENVIRONMENT_KEY = 'DEEPLINKENGINE_RERUN_B0'
DISCOVERY_ROOT = 'DEEPLINKENGINE_B0_DISCOVERY_ROOT'
DISCOVERY_SOURCE = 'DEEPLINKENGINE_B0_DISCOVERY_SOURCE'
MESSAGE = '当前选择为仅复用 B0。请提供匹配封存，或移除 --no-rerun-b0（配置项 rerun_b0: false）。'


def boolean(value):
    if type(value) is not bool:
        raise ValueError('rerun_b0 必须是 true 或 false，省略则自动选择')
    return value


def mode():
    return {'0': 'reuse_only', '1': 'force', 'auto': 'auto'}[os.environ.get(ENVIRONMENT_KEY, '0')]


def enabled():
    return mode() != 'reuse_only'


def candidates(out, kind):
    if mode() != 'auto':
        return []
    out = Path(out).resolve()
    parent = Path(os.environ.get(DISCOVERY_ROOT) or out.parent).resolve()
    locations = {
        'campaign': ('fixed-five', '.'),
        'context': ('context-tenant/B0', 'context-tenant/native/B0', 'B0', 'native/B0'),
        'reference': ('fixed-five/compute-reference/native-baseline-timing',
                      'compute-reference/native-baseline-timing', 'native-baseline-timing'),
    }[kind]
    marker = 'campaign-checksums.json' if kind == 'campaign' else 'checksums.json'
    if not parent.is_dir():
        return []
    result = []
    roots = []
    for path in parent.iterdir():
        if path.is_dir() and not path.is_symlink():
            try:
                roots.append((path.stat().st_mtime_ns, path.name, path))
            except FileNotFoundError:
                continue
    siblings = [entry[-1] for entry in sorted(roots, reverse=True)]
    preferred = os.environ.get(DISCOVERY_SOURCE)
    if preferred:
        siblings.insert(0, Path(preferred).resolve())
    for sibling in dict.fromkeys(siblings):
        if out == sibling or out.is_relative_to(sibling) or sibling.is_relative_to(out):
            continue
        for relative in locations:
            candidate = sibling / relative
            if candidate.resolve().is_relative_to(sibling) and (candidate / marker).is_file():
                result.append(candidate)
    return result


def decision(scope, source, rejected):
    return {'stage': scope, 'mode': mode(), 'action': 'reuse' if source else 'measure',
            'reason': ('matching_sealed_source' if source else
                       'explicit_rerun' if mode() == 'force' else 'no_matching_sealed_source'),
            'source': str(source) if source else None, 'rejected': rejected}


def probe(path, verify, rejected):
    try:
        return verify(path)
    except (ValueError, OSError, KeyError, TypeError) as error:
        rejected.append({'source': str(path), 'reason': f'{type(error).__name__}: {error}'})
        return None


def select(out, kind, verify):
    rejected = []
    for path in candidates(out, kind):
        proof = probe(path, verify, rejected)
        if proof is not None:
            return path, proof, decision(kind, path, rejected)
    return None, None, decision(kind, None, rejected)



def check_current_weights(model, weights, checked):
    if not weights or set(weights) != {path.name for path in Path(model).glob("*.safetensors")}:
        raise ValueError("Current checkpoint file membership differs")
    identity = {}
    for name, row in weights.items():
        path = Path(model) / name
        stat = path.stat()
        if name not in checked:
            with path.open("rb") as stream:
                checked[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if (stat.st_size, checked[name]) != (row["bytes"], row["sha256"]):
            raise ValueError("Current model weight differs: " + name)
        identity[name] = dict(row, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)
    return identity


def public_arguments(arguments, *, prepare_only=False):
    """Consume the public switch before passing unchanged args to sealed parsers."""
    args = list(arguments)
    flags = [x for x in args if x in ('--rerun-b0', '--no-rerun-b0')]
    if len(flags) > 1:
        raise ValueError('只指定一次 --rerun-b0 或 --no-rerun-b0')
    rerun = flags == ['--rerun-b0']
    args = [x for x in args if x not in ('--rerun-b0', '--no-rerun-b0')]
    reuse = [x for x in args if x.split('=', 1)[0] in (
        '--reuse-completed-campaign', '--reuse-mmlu-baseline-arm',
        '--reference-baseline-source', '--reuse-b0', '--reference-resume-source')]
    if rerun and reuse:
        raise ValueError('重跑 B0 与复用 B0 来源不能同时指定；请使用新的结果目录')
    if not prepare_only and flags == ['--no-rerun-b0']:
        if '--context-tenant-only' in args:
            present = any(x.split('=', 1)[0] == '--reuse-b0' for x in args)
        elif '--compute-reference-only' in args:
            present = any(x.split('=', 1)[0] in ('--reference-baseline-source', '--reference-resume-source') for x in args)
        else:
            present = any(x.split('=', 1)[0] in ('--reuse-completed-campaign', '--reuse-mmlu-baseline-arm') for x in args)
        if not present:
            raise ValueError(MESSAGE)
    # An inherited environment variable cannot silently opt a public run in.
    os.environ[ENVIRONMENT_KEY] = '1' if rerun else '0' if flags else 'auto'
    return args


def require_campaign_reuse(stages, reused, out, frozen):
    """Run after the original full raw/source admission, before any child model."""
    missing = []
    for stage in stages:
        name = stage['name']
        baseline = name == 'continuous-B0' or (name.endswith('-B0') and '-judge-' not in name)
        if baseline and name not in reused:
            missing.append(name)
        if name == 'mmlu-pair' and name not in reused and '--reuse-baseline-arm' not in stage['args']:
            missing.append('mmlu-pair/B0')
    frozen['baseline_execution_policy'] = dict(
        mode=mode(), measure=missing, automatic_reuse=bool(mode() == 'auto'),
        discovery=frozen.get('automatic_b0_campaign', {}))
    if missing and not enabled():
        raise ValueError(MESSAGE + ' 未接入：' + ', '.join(missing))
    reference = frozen.get('compute_reference', {})
    if reference.get('mode') == 'automatic_current_certification':
        plan = json.loads((Path(out) / reference['plan']).read_text())
        require_reference_reuse(plan)


def require_reference_reuse(plan):
    if not enabled() and not plan.get('baseline_reuse') and not plan.get('stage_resume'):
        raise ValueError(MESSAGE + ' 缺少 B0 原生计时复用来源。')


def require_context_reuse(args):
    if not enabled() and not getattr(args, 'reuse_b0', None):
        raise ValueError(MESSAGE + ' 缺少 B0 上下文／四租户复用来源。')


def full_entry_policy(data, *, prior=None, context=None, compute=None, recovery=False):
    """Check the full entry before environment preparation or source downloads."""
    rerun = boolean(data["rerun_b0"]) if "rerun_b0" in data else None
    if rerun:
        if prior or context or compute:
            raise ValueError('重跑 B0 与复用来源不能同时指定；请使用新的结果目录')
        return context, compute
    if rerun is None:
        return context, compute
    if not prior and not recovery:
        raise ValueError(MESSAGE + ' 完整入口使用 --reuse-completed-run 指定旧结果。')
    source = Path(prior or data['out'])
    record = source / 'results.json'
    saved = json.loads(record.read_text()).get('reuse_selection', {}) if record.is_file() else {}
    context = context or saved.get('context_baseline_arm')
    compute = compute or saved.get('compute_baseline_arm')
    if not context:
        # The public acceptance controller writes output/B0. Some older
        # independently launched supplements had an extra native/ directory.
        for name in ('context-tenant/B0', 'context-tenant/native/B0'):
            candidate = source / name
            if candidate.is_dir():
                context = candidate
                break
    if not compute:
        candidate = source / 'fixed-five/compute-reference/native-baseline-timing'
        if candidate.is_dir():
            compute = candidate
    if rerun is False and (not context or not compute):
        raise ValueError(MESSAGE + ' 请补充 --reuse-context-baseline-arm 和 --reuse-compute-baseline-arm。')
    return Path(context) if context else None, Path(compute) if compute else None
