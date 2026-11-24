"""Admit the reviewed stable R10 sources using sealed native and HTTP launch evidence."""

import hashlib
import json


def source_domain_for_release(protocol, release):
    for key in ('stable_r10_source_domain', 'r10_3072_source_domain'):
        domain = protocol.get(key)
        if domain is not None and domain['release'] == release:
            return domain
    return None


def admitted_sources(protocol, role, deployment, read_native):
    sources = protocol['sources'][role]
    if role != 'B1':
        return sources

    profile_path = deployment / 'deployment.json'
    raw_profile = profile_path.read_bytes()
    profile = json.loads(raw_profile)
    compatibility = source_domain_for_release(protocol, profile['release'])
    if compatibility is None:
        return sources

    plan = read_native('service/native-plan.json')
    overlay_path = deployment / 'provenance/runtime-overlay.json'
    overlay_pin = compatibility['overlay_sha256']
    measured = profile['measured_runtime']
    runtime_pin = compatibility.get('installed_files_canonical_sha256')
    if runtime_pin is None:
        profile_overlay_pin = measured.get('runtime_overlay_sha256')
    else:
        profile_overlay_pin = profile.get('runtime_overlay_sha256')
        if (measured.get('installed_files_canonical_sha256') != runtime_pin
                or profile.get('model_adapters') != ['qwen35']):
            raise ValueError('Revised R10 reference requires the sealed Qwen runtime')
    if (plan['engine_release'] != profile['release'] or plan['deployment'] != str(deployment)
            or plan['files'].get(str(profile_path)) != hashlib.sha256(raw_profile).hexdigest()
            or profile_overlay_pin != overlay_pin
            or hashlib.sha256(overlay_path.read_bytes()).hexdigest() != overlay_pin):
        raise ValueError('Stable R10 reference runtime differs from the sealed native deployment')

    environment = read_native('service/launch-command.json').get('environment')
    if not isinstance(environment, dict) or environment.get('HUST_RESTORE_GATED_RMS') not in (None, '', '0'):
        raise ValueError('Stable R10 reference requires a sealed native RMS launch')
    return {**sources, **compatibility['sources']}


def verify_http_adapter(plan, release):
    adapter = plan.get('model_adapter') or {}
    if (plan['engine_release'] != release or adapter.get('id') != 'qwen35'
            or adapter.get('gated_rms_mode') != 'native' or adapter.get('restore_gated_rms') is not False):
        raise ValueError('Stable R10 reference requires a sealed native RMS HTTP companion')
