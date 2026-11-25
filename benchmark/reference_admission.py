"""Classify a failed reference independently of fixed-five inference."""

import hashlib
import json
from pathlib import Path


CERTIFICATE = 'qualified_full_reference_denominator_reuse/v1'
MEASURED = 'current_workload_measured'
ADMISSION_FAILURE = 'MMLU200有效算力参考核验未通过，未启动性能模型'


def sealed_campaign(folder):
    path = folder / 'campaign.json'
    if path.is_symlink():
        raise ValueError('Fixed-five campaign receipt is a symlink')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    seal = json.loads((folder / 'campaign-checksums.json').read_bytes())
    if seal.get(path.name) != digest:
        raise ValueError('Fixed-five campaign receipt differs from its seal')
    return json.loads(raw), digest


def rejected_before_model(campaign):
    return (campaign.get('status') == 'failed'
            and campaign.get('selected_execution_complete') is False
            and campaign.get('stages') == []
            and str(campaign.get('failure', '')).removeprefix('RuntimeError: ').startswith(ADMISSION_FAILURE)
            and campaign.get('compute_reference', {}).get('status') == 'blocked')


def assess(reference):
    """Keep the certificate gate and expose the exact reason when it fails."""
    from compute_reference import target

    reference = reference or {}
    outcome = target(reference)
    if outcome['status'] == MEASURED and reference.get('certificate_kind') == CERTIFICATE:
        return outcome
    if outcome['status'] == MEASURED:
        return dict(outcome, accepted=False, status='current_workload_reference_not_measured',
                    display_value='缺少合格参考证书', measured_value=None, measured_value_decimal=None,
                    reason='本轮有效算力缺少同工况完整参考分母的合格认证。')
    return outcome


def archived_refusal(root, descriptor):
    root = Path(root).resolve()
    path = root / descriptor['path']
    if path.name != 'campaign.json' or not path.resolve().is_relative_to(root / 'attempts'):
        raise ValueError('Archived reference refusal is outside the current run history')
    campaign, digest = sealed_campaign(path.parent)
    if digest != descriptor['sha256'] or not rejected_before_model(campaign):
        raise ValueError('Archived reference refusal does not match its sealed admission')
    return campaign['compute_reference']
