"""Admit the reviewed stable R10 sources using sealed native and HTTP launch evidence."""

import hashlib
import json


def admitted_sources(protocol, role, deployment, read_native):
    sources = protocol['sources'][role]
    if role != 'B1':
        return sources

    compatibility = protocol['stable_r10_source_domain']
    profile_path = deployment / 'deployment.json'
    raw_profile = profile_path.read_bytes()
    profile = json.loads(raw_profile)
    if profile['release'] != compatibility['release']:
        return sources

    plan = read_native('service/native-plan.json')
    overlay_path = deployment / 'provenance/runtime-overlay.json'
    overlay_pin = compatibility['overlay_sha256']
    if (plan['engine_release'] != profile['release'] or plan['deployment'] != str(deployment)
            or plan['files'].get(str(profile_path)) != hashlib.sha256(raw_profile).hexdigest()
            or profile['measured_runtime']['runtime_overlay_sha256'] != overlay_pin
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
