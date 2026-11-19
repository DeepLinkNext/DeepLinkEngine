#!/usr/bin/env python3
"""Build a portable five-dataset dashboard from sealed experiment reports."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

DATASETS = ('mmlu', 'hle', 'frontier', 'swe', 'terminal')
SCHEMAS = ('vllm-hust-current-paired-report/v1', 'vllm-hust-third-party-report/v1')


def load_verified(path):
    path = Path(path)
    seal = path.parent / 'checksums.json'
    if not seal.is_file():
        raise ValueError('缺少报告校验文件')
    checksums = json.loads(seal.read_bytes())
    expected = checksums.get('summary.json')
    content = path.read_bytes()
    if not expected or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('summary.json 校验失败')
    data = json.loads(content)
    if not isinstance(data, dict) or data.get('schema') not in SCHEMAS:
        raise ValueError('非受支持的正式报告类型')
    return data

def selected_config(plan, dataset):
    prepared = (plan.get('prepared_datasets') or {}).get(dataset) or {}
    keys = ('engine_release', 'model', 'physical_cards', 'logical_devices', 'baseline_deployment', 'optimized_deployment', 'serving_warmup', 'task_serving_warmup', 'hle_profile', 'mmlu_scope', 'judge_scope', 'public_trace_scope', 'failure_policy')
    config = {key: plan[key] for key in keys if key in plan}
    config['dataset'] = {k: prepared[k] for k in ('name', 'tasks', 'modality', 'image_tasks', 'execution_scope', 'contract_sha256') if k in prepared}
    return config

def timestamp_seconds(value):
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if date.tzinfo is not None:
                return date.timestamp()
    except (ValueError, OverflowError, OSError):
        pass
    return None


def dataset_times(data, dataset, arms):
    receipt = data.get('receipt') or {}
    stages = {
        stage['name']: stage
        for stage in data.get('current_stages') or receipt.get('stages') or []
        if isinstance(stage, dict) and isinstance(stage.get('name'), str)
    }
    stage_kind = 'trace' if dataset in ('swe', 'terminal') else 'solver'
    roles = ('B0', 'B1')
    inputs = []
    for role in roles:
        arm = arms.get(role) or {}
        stage = stages.get(f'{dataset}-{stage_kind}-{role}', {})
        inputs.append((arm.get('receipt') or {}, stage))

    def inference_boundary(field, combine):
        times = []
        for arm_receipt, stage in inputs:
            value = timestamp_seconds(arm_receipt.get(field))
            if value is None:
                value = timestamp_seconds(stage.get(field))
            if value is not None:
                times.append(value)
        if times:
            return combine(times)
        if dataset == 'mmlu':
            return timestamp_seconds(stages.get('mmlu-pair', {}).get(field))
        return None

    inference_started = inference_boundary('started', min)
    inference_ended = inference_boundary('ended', max)
    started = timestamp_seconds(receipt.get('started'))
    ended = timestamp_seconds(receipt.get('ended'))
    if started is None and ended is None:
        started, ended = inference_started, inference_ended

    regraded = (
        dataset in ('hle', 'frontier')
        and all(stage.get('status') == 'reused_verified' for _, stage in inputs)
        and any(
            stages.get(f'{dataset}-judge-{role}', {}).get('status') == 'completed'
            for role in roles
        )
    )
    return {
        'started': started,
        'ended': ended,
        'inference_started': inference_started,
        'inference_ended': inference_ended,
        'regraded': regraded,
    }


def normalize(data, path, dataset, pair=None):
    receipt = data.get('receipt') or {}
    plan = data.get('plan') or {}
    perf, quality = {}, {}
    if pair:
        for role, arm in (pair.get('arms') or {}).items():
            if role in ('B0', 'B1'):
                perf[role] = (arm.get('groups') or {}).get('joint200') or {}
                quality[role] = arm.get('quality') or {}
    elif dataset in ('swe', 'terminal'):
        trace = (data.get('current_trace_performance') or {}).get(dataset) or {}
        perf = trace.get('arms') or {}
    else:
        perf = {role: (data.get('current_dataset_performance') or {}).get(f'{dataset}-solver-{role}') for role in ('B0', 'B1')}
        quality = {role: (data.get('current_dataset_quality') or {}).get(f'{dataset}-{role}') for role in ('B0', 'B1')}
    perf = {k: v for k, v in perf.items() if isinstance(v, dict)}
    quality = {k: v for k, v in quality.items() if isinstance(v, dict)}
    if not perf and not quality:
        return None
    timeline = dataset_times(data, dataset, (pair.get('arms') or {}) if pair else perf)
    fields = ('output_tokens_per_second', 'mean_ttft_ms', 'mean_tpot_ms', 'planned', 'recorded', 'successful', 'failed', 'unreturned', 'cohort_wall_seconds', 'successful_output_tokens')
    perf = {r: {k: v[k] for k in fields if k in v} for r, v in perf.items()}
    qfields = ('planned_denominator', 'recorded', 'correct', 'accuracy_pct_of_planned200', 'truncated_recorded', 'unreturned', 'tracks', 'judge_model', 'solver_model', 'protocol', 'scope', 'scoring_policy')
    quality = {r: {k: v[k] for k in qfields if k in v} for r, v in quality.items()}
    config = selected_config(plan, dataset)
    if pair and not plan:
        config['mmlu_scope'] = pair.get('mmlu_task_scope')
    label = path.parent.name
    if label == 'mmlu-pair-report':
        label = path.parent.parent.name + '/' + label
    return {
        'dataset': dataset,
        'id': str(path.parent.resolve()),
        'label': label,
        'source': str(path),
        'evidence_run': (pair or {}).get('root') or data.get('run') or str(path),
        **timeline,
        'status': receipt.get('status', '已封存配对'),
        'evidence_basis': data.get('existing_evidence_basis'),
        'performance': perf,
        'quality': quality,
        'targets': data.get('formal_targets') or {},
        'scope': data.get('scope'),
        'config': config,
        'gains': pair.get('gains') if pair else None,
    }


def output_path(path):
    output = path.expanduser().resolve()
    if output.suffix.lower() != '.html':
        raise ValueError('网页输出路径需要使用 .html 后缀')
    if any(
        (parent / seal).exists()
        for parent in output.parents
        for seal in ('checksums.json', 'campaign-checksums.json', 'pair-checksums.json')
    ):
        raise ValueError('请将网页输出到封存报告目录之外')
    return output

def build(results, extra, output):
    output = output_path(output)
    paths = set(results.glob('*-report/summary.json'))
    paths.update(results.glob('*/mmlu-pair-report/summary.json'))
    paths.update(Path(p) for p in extra)
    extra_paths = {str(Path(p).resolve()) for p in extra}
    records, warnings = [], []
    for path in sorted(paths):
        try:
            data = load_verified(path)
            if data['schema'] == SCHEMAS[0]:
                item = normalize(data, path, 'mmlu', data)
                if item:
                    records.append(item)
            else:
                pair = data.get('current_mmlu_pair')
                if pair:
                    item = normalize(data, path, 'mmlu', pair)
                    if item:
                        records.append(item)
                for name in DATASETS[1:]:
                    item = normalize(data, path, name)
                    if item:
                        records.append(item)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
            warnings.append({'report': str(path), 'reason': str(error)})
    unique = {}
    for row in records:
        key = (row['dataset'], row['evidence_run'])
        priority = (str(Path(row['source']).resolve()) in extra_paths, bool(row.get('config', {}).get('engine_release')))
        if key not in unique or priority > unique[key][0]:
            unique[key] = (priority, row)
    records = [row for _, row in unique.values()]

    def time_key(row):
        time = row['started'] if row['started'] is not None else row['ended']
        return (time if time is not None else float('-inf'), row['id'])
    records.sort(key=time_key, reverse=True)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value
    payload = finite({'generated_at': datetime.now(timezone.utc).isoformat(), 'records': records, 'warnings': warnings})
    template = Path(__file__).with_name('acceptance_dashboard.html').read_text(encoding='utf-8')
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent, delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            stream.write(template.replace('/*__REPORT_DATA__*/', payload_json))
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'output': str(output), 'records': len(records), 'warnings': warnings}, ensure_ascii=False))
    return payload

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='one_click.py dashboard' if argv is not None else None,
        description=__doc__,
    )
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--extra', action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--serve', type=int, metavar='PORT', help='Loopback-only HTTP server; regenerates on each page request')
    args = parser.parse_args(argv)
    try:
        args.out = output_path(args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.serve is None:
        build(args.results, args.extra, args.out)
        return 0
    if not 0 <= args.serve <= 65535:
        parser.error('PORT must be between 0 and 65535')
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Lock
    lock = Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/index.html'):
                self.send_error(404)
                return
            try:
                with lock:
                    build(args.results, args.extra, args.out)
                    body = args.out.read_bytes()
            except (OSError, ValueError, TypeError) as error:
                self.send_error(500, 'Report generation failed')
                print('Dashboard generation error:', str(error), flush=True)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', args.serve), Handler)
    print('Dashboard http://127.0.0.1:%d/' % server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
