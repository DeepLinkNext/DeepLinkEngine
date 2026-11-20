"""Public B0 execution is opt-in; missing reusable evidence never starts B0."""
import json
import os
from pathlib import Path

ENVIRONMENT_KEY = 'DEEPLINKENGINE_RERUN_B0'
MESSAGE = ('默认不重跑 B0。请提供匹配的已封存 B0 结果；'
           '首次测量或确需重新测量时显式使用 --rerun-b0（配置项 rerun_b0: true）。')


def boolean(value):
    if type(value) is not bool:
        raise ValueError('rerun_b0 必须是 true 或 false，默认 false')
    return value


def enabled():
    value = os.environ.get(ENVIRONMENT_KEY, '0')
    if value not in ('0', '1'):
        raise ValueError('Invalid explicit B0 execution policy')
    return value == '1'


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
    if not prepare_only and not rerun:
        if '--context-tenant-only' in args:
            present = any(x.split('=', 1)[0] == '--reuse-b0' for x in args)
        elif '--compute-reference-only' in args:
            present = any(x.split('=', 1)[0] in ('--reference-baseline-source', '--reference-resume-source') for x in args)
        else:
            present = any(x.split('=', 1)[0] in ('--reuse-completed-campaign', '--reuse-mmlu-baseline-arm') for x in args)
        if not present:
            raise ValueError(MESSAGE)
    # An inherited environment variable cannot silently opt a public run in.
    os.environ[ENVIRONMENT_KEY] = '1' if rerun else '0'
    return args


def require_campaign_reuse(stages, reused, out, frozen):
    """Run after the original full raw/source admission, before any child model."""
    frozen['baseline_execution_policy'] = dict(rerun_b0=enabled(), default_rerun_b0=False)
    if enabled():
        return
    missing = []
    for stage in stages:
        name = stage['name']
        baseline = name == 'continuous-B0' or (name.endswith('-B0') and '-judge-' not in name)
        if baseline and name not in reused:
            missing.append(name)
        if name == 'mmlu-pair' and name not in reused and '--reuse-baseline-arm' not in stage['args']:
            missing.append('mmlu-pair/B0')
    if missing:
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
    rerun = boolean(data.get('rerun_b0', False))
    if rerun:
        if prior or context or compute:
            raise ValueError('重跑 B0 与复用来源不能同时指定；请使用新的结果目录')
        return context, compute
    if not prior and not recovery:
        raise ValueError(MESSAGE + ' 完整入口使用 --reuse-completed-run 指定旧结果。')
    source = Path(prior or data['out'])
    record = source / 'results.json'
    saved = json.loads(record.read_text()).get('reuse_selection', {}) if record.is_file() else {}
    context = context or saved.get('context_baseline_arm')
    compute = compute or saved.get('compute_baseline_arm')
    if not context:
        candidate = source / 'context-tenant/native/B0'
        if candidate.is_dir():
            context = candidate
    if not compute:
        candidate = source / 'fixed-five/compute-reference/native-baseline-timing'
        if candidate.is_dir():
            compute = candidate
    if not context or not compute:
        raise ValueError(MESSAGE + ' 请补充 --reuse-context-baseline-arm 和 --reuse-compute-baseline-arm。')
    return Path(context), Path(compute)
