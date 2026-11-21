"""CPU projection of selected sealed report members; never replay raw SSE or models."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from official_rebuild import member, relative, write, digest

ROOT = Path(__file__).resolve().parent

COUNTS = {'mmlu': 200, 'hle': 200, 'frontier': 160, 'swe': 64, 'terminal': 48}
LABELS = {'mmlu': 'MMLU200', 'hle': 'HLE200', 'frontier': 'Frontier160', 'swe': 'SWE trace64', 'terminal': 'Terminal trace48'}


def prepare(document, report_sha256):
    if document.get('schema') != 'deeplinkengine-official-baseline-report/v1':
        raise ValueError('Unsupported original report')
    if document.get('status') not in ('completed', 'failed', 'interrupted'):
        raise ValueError('A terminal original report is required; active runs are not inputs')
    if document.get('planned_denominators') != COUNTS:
        raise ValueError('The fixed-five denominator changed')
    if document.get('new_F_T_qualification') is not False or document.get('qualified_efficiency_percent') is not None:
        raise ValueError('This projection cannot qualify new F/T')
    profiles = []
    missing = []
    for name, requests in COUNTS.items():
        phase = document['phases'].get(name, {})
        metric = document.get('metrics', {}).get(name)
        if metric is None:
            missing.append(name)
            continue
        if phase.get('status') not in ('completed', 'reused'):
            raise ValueError('Metrics from a non-completed/non-reused solver: ' + name)
        source_key = 'service/joint.json' if name == 'mmlu' else 'performance-summary.json'
        source = phase[source_key]
        if source['value'] != metric:
            raise ValueError('The report metric differs from its embedded original: ' + name)
        if len(source.get('sha256', '')) != 64:
            raise ValueError('Missing original small-file hash')
        planned = metric['total'] if name == 'mmlu' else metric['planned']
        if planned != requests:
            raise ValueError('Wrong measured denominator: ' + name)
        if name == 'hle' and (metric.get('full_dataset_tasks') != 1811 or metric.get('execution_scope', {}).get('selected_tasks') != 200):
            raise ValueError('HLE is the fixed 200 of 1811, not an upstream full set')
        if name in ('swe', 'terminal') and metric.get('quality_measured') is not False:
            raise ValueError('Trace replay does not measure task quality')
        arm = {'role': 'B0', 'requests': requests, 'successful': metric['completed'] if name == 'mmlu' else metric['successful'], 'failed': metric['failed'],
               'output_tokens': metric['successful_output_tokens'], 'duration_seconds': metric['duration_seconds'],
               'output_tokens_per_second': metric['output_tokens_per_second'], 'mean_ttft_ms': metric.get('mean_ttft_ms'), 'mean_tpot_ms': metric.get('mean_tpot_ms'),
               'truncated': metric['truncated'] if name == 'mmlu' else metric.get('length_stops'),
               'relative_cost_per_token': None}
        if arm['successful'] + arm['failed'] != requests:
            raise ValueError('Incomplete/failed response population needs an explicit unreturned policy')
        for key in ('output_tokens_per_second', 'mean_ttft_ms', 'mean_tpot_ms'):
            if arm[key] is not None and (not isinstance(arm[key], (int, float)) or not math.isfinite(arm[key]) or arm[key] < 0):
                raise ValueError('Nonfinite original metric')
        if arm['duration_seconds'] <= 0 or not math.isclose(arm['output_tokens'] / arm['duration_seconds'], arm['output_tokens_per_second'], rel_tol=1e-8):
            raise ValueError('Original throughput/timing mismatch')
        quality = None
        if name == 'mmlu':
            arm.update(correct=metric['correct'], unparsed=metric['unparsed'])
            quality = {'quality_measured': True, 'protocol': 'Original joint MMLU200 extraction', 'correct': metric['correct'], 'total': requests, 'unparsed': metric['unparsed']}
        elif name in ('frontier', 'hle'):
            judge_name = name + '-judge'
            quality = document.get('quality', {}).get(judge_name)
            if quality is not None:
                if document['phases'][judge_name]['status'] != 'completed':
                    raise ValueError('A judge summary without its completed phase is not final')
                original = document['phases'][judge_name]['service/quality-summary/summary.json']
                if original['value'] != quality:
                    raise ValueError('Judge summary differs from its embedded original')
                if name == 'frontier':
                    tracks = quality['tracks']
                    if quality.get('denominator_policy') != 'all_cases_invalid_as_nonsuccess' or {k: v['total'] for k, v in tracks.items()} != {'olympiad': 100, 'research': 60}:
                        raise ValueError('Frontier invalid grades must retain the full 160 denominator')
                # Keep the exact track/subject grading; do not silently collapse it into accuracy.
        else:
            quality = {'quality_measured': False, 'measurement': 'Published trace inference; task success unmeasured'}
        profiles.append({'id': name, 'label': LABELS[name], 'source_phase_status': phase['status'], 'workload_kind': 'published-agent-trace' if name in ('swe', 'terminal') else 'fixed-dataset',
                         'arm': arm, 'quality': quality, 'execution_scope': metric.get('execution_scope'),
                         'source_metric': {'path': source['path'], 'sha256': source['sha256']}})
    return {'schema': 'deeplinkengine-official-publication-preview/v1', 'publication_ready': False,
            'original_report_sha256': report_sha256, 'baseline': document['baseline'], 'source_execution_status': document['status'],
            'profiles': profiles, 'missing_metric_phases': missing, 'original_quality': document.get('quality', {}),
            'pairs': None, 'F': None, 'efficiency_percent': None, 'independent_third_party': False,
            'requirements_before_publication': ['Terminal campaign and original external seal; no live-run reads',
                'Full selected archive closure and actual phase seals, including reused original source chain',
                'Runtime/model/hardware identity from sealed original records, not an earlier C8 assumption',
                'Raw execution commits per phase and distinct actual CPU projection commit',
                'Carry forward the unchanged original C8 profiles; actual context/native sections only when present',
                'Real benchmark visibility projection and website descriptor/hash/browser checks']}


def run(configuration):
    config_path = Path(configuration).absolute()
    config = json.loads(config_path.read_bytes())
    allowed = {'kind', 'source', 'out', 'report_sha256', 'original_version_seal_sha256', 'plots'}
    if (not isinstance(config, dict) or set(config) - allowed or config.get('kind') != 'sealed-report-projection'
            or any(not isinstance(config.get(key), str) or not config[key] for key in ('source', 'out'))
            or any(not re.fullmatch('[a-f0-9]{64}', config.get(key, '')) for key in ('report_sha256', 'original_version_seal_sha256'))
            or type(config.get('plots', True)) is not bool):
        raise ValueError('Sealed report projection requires explicit source, new out, report SHA256 and original version seal SHA256')
    source = Path(os.path.abspath(config_path.parent / config['source']))
    out = Path(os.path.abspath(config_path.parent / config['out']))
    if not source.is_dir() or source.is_symlink() or os.path.lexists(out) or out.resolve().is_relative_to(source.resolve()):
        raise ValueError('Select a closed source directory and a new output outside it; no symlinks or overwrites')
    seal_raw = member(source, 'checksums.json').read_bytes()
    if hashlib.sha256(seal_raw).hexdigest() != config['original_version_seal_sha256']:
        raise ValueError('External original version seal SHA256 mismatch')
    seal = json.loads(seal_raw)
    if not isinstance(seal, dict) or not seal:
        raise ValueError('A nonempty original version seal is required')
    for name, pin in seal.items():
        relative(name)
        if not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin):
            raise ValueError('Invalid original version seal member')
    selected = {}

    def read(name, expected=None):
        raw = member(source, name).read_bytes()
        pin = hashlib.sha256(raw).hexdigest()
        if seal.get(name) != pin or expected is not None and expected != pin:
            raise ValueError('Selected original report member changed: ' + name)
        selected[name] = raw
        return json.loads(raw)

    document = read('report.json', config['report_sha256'])
    campaign = read('campaign.json')
    if (campaign.get('schema') != 'deeplinkengine-official-full-suite/v1'
            or campaign.get('status') not in ('completed', 'failed')
            or campaign['status'] != document.get('status')
            or campaign.get('baseline_image', {}).get('id') != document.get('baseline')):
        raise ValueError('Original campaign/report must agree and both be terminal')
    phases = {phase['name']: phase for phase in campaign['stages']}
    if len(phases) != len(campaign['stages']) or set(phases) != set(document['phases']):
        raise ValueError('Original campaign/report phase population changed')
    reuse = {}
    for name, phase in document['phases'].items():
        original = phases[name]
        if phase['status'] != original['status']:
            raise ValueError('Original phase status differs from campaign: ' + name)
        for descriptor in phase.values():
            if not isinstance(descriptor, dict) or 'path' not in descriptor:
                continue
            if read(descriptor['path'], descriptor['sha256']) != descriptor['value']:
                raise ValueError('Embedded report value differs from its sealed original')
        if phase['status'] == 'reused':
            evidence = original.get('reuse', {})
            if any(not re.fullmatch('[a-f0-9]{64}', evidence.get(key, '')) for key in ('original_checksums_sha256', 'original_receipt_sha256')):
                raise ValueError('Reused phase needs its admitted original seal and receipt identity')
            read(name + '/checksums.json', evidence['original_checksums_sha256'])
            receipt = phase.get('receipt.json', {})
            if (receipt.get('sha256') != evidence['original_receipt_sha256']
                    or receipt.get('value', {}).get('status') != 'completed'
                    or receipt['value'].get('completed') is not True
                    or receipt['value'].get('quiescent') is not True):
                raise ValueError('Reused phase lacks a matching completed, quiescent original receipt')
            reuse[name] = evidence
    preview = prepare(document, config['report_sha256'])
    provenance = dict(schema='deeplinkengine-sealed-report-projection/v1', kind='sealed-report-projection',
        status='preparing', source_execution_status=document['status'], input_report_sha256=config['report_sha256'],
        original_version_seal_sha256=config['original_version_seal_sha256'], original_results_modified=False,
        model_requests_executed=0, raw_sse_recomputed=False, all_original_payloads_verified=False,
        selected_original_members_verified={name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(selected.items())},
        unverified_omitted_original_members=len(set(seal) - set(selected)), original_sources=campaign.get('sources', {}),
        original_reuse_sources=reuse, projection_sources={name: digest(ROOT / name) for name in
            ('run_all.py', 'official_baselines.py', 'official_rebuild.py', 'official_projection.py', 'official_plots.py')},
        publication_ready=False, plots_requested=config.get('plots', True), F=None, efficiency_percent=None,
        independent_third_party=False)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in selected.items():
        target = out / 'originals' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (out / 'originals' / 'checksums.json').write_bytes(seal_raw)
    projected = {**document, 'report_projection': provenance}
    write(out / 'report.json', projected)
    write(out / 'publication-preview.json', preview)
    if config.get('plots', True):
        try:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'official_plots.py'), str(out / 'report.json')],
                                    capture_output=True, text=True, timeout=300)
            log = result.stdout + result.stderr
            plot_error = result.returncode != 0
        except (OSError, subprocess.TimeoutExpired) as error:
            log = str(error)
            plot_error = True
        (out / 'plotting.log').write_text(log, encoding='utf-8')
        if plot_error:
            provenance['status'] = 'failed'
            write(out / 'report.json', projected)
            write(out / 'derivation.json', provenance)
            write(out / 'projection-execution.json', dict(status='failed', phase='CPU plotting', model_requests_executed=0,
                                                         source_execution_status=document['status']))
            raise ValueError('Pinned CPU plotting failed; preserve new output and plotting.log')
    provenance['status'] = 'completed'
    write(out / 'report.json', projected)
    rows = ''.join('<tr><td>' + html.escape(name) + '</td><td>' + html.escape(phase['status']) + '</td></tr>'
                   for name, phase in document['phases'].items())
    figures = ''.join('<figure><img style="max-width:100%" src="' + path.name + '" alt="Original report measurements"></figure>'
                      for path in sorted(out.glob('*.svg')))
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Official report projection</title>'
        '<h1>' + html.escape(document['baseline']) + '</h1><p>CPU projection of an existing sealed report. Original status: '
        + html.escape(document['status']) + '. No model requests or raw SSE recomputation.</p><table>' + rows + '</table>' + figures
        + '<p><a href="originals/report.json">Original report</a> · <a href="report.json">Projected report</a> · '
        '<a href="derivation.json">Sources and selected custody</a></p>', encoding='utf-8')
    write(out / 'derivation.json', provenance)
    write(out / 'checksums.json', {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob('*')) if path.is_file()})
    print('OFFICIAL_SEALED_REPORT_PROJECTION_COMPLETED', str(out), 'model_requests=0', flush=True)
    return 0

