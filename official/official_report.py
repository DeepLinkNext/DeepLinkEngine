"""Publish actual phase state and original outputs without invented metrics."""
import hashlib
import html
import json
import os
import subprocess
from pathlib import Path

from official_baselines import save


def seal(root):
    """Seal original outputs after owned processes exit; compiler caches excluded."""
    root = Path(root)
    files = {}
    for parent, directories, names in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name != 'cache' and not (Path(parent) / name).is_symlink()]
        for name in names:
            path = Path(parent) / name
            if path == root / 'checksums.json' or path.is_symlink() or not path.is_file():
                continue
            with path.open('rb') as stream:
                files[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    save(root / 'checksums.json', files)
    return files


def report(root, config=None):
    root = Path(root)
    campaign = json.loads((root / 'campaign.json').read_bytes())
    summaries = {}
    for phase in campaign['stages']:
        directory = root / phase['name']
        summaries[phase['name']] = dict(status=phase['status'])
        for name in ('service/joint.json', 'service/continuous.json', 'service/dataset-phase.json',
                     'service/acceptance/execution.json', 'client/timings.json', 'receipt.json',
                     'performance-summary.json', 'service/quality-summary/summary.json'):
            path = directory / name
            if path.is_file():
                summaries[phase['name']][name] = dict(path=str(path.relative_to(root)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), value=json.loads(path.read_bytes()))
    qualification = None
    directory = root / 'necessary-work'
    if (directory / 'qualified-m01.json').is_file():
        from official_ft_audit import Evidence
        evidence = Evidence(directory)
        receipt = evidence.read('receipt.json')
        value = evidence.read('qualified-m01.json')
        if (receipt['completed'] is True and receipt['qualification_sha256'] == evidence.selected['qualified-m01.json']
                and value['new_F_T_qualification'] is True
                and value['source_native_seal_sha256'] == hashlib.sha256((root / 'native-timing/checksums.json').read_bytes()).hexdigest()):
            qualification = value
        else:
            raise ValueError('Official F/T qualification does not bind to this run')
        summaries['necessary-work']['qualified-m01.json'] = dict(path='necessary-work/qualified-m01.json',
            sha256=evidence.selected['qualified-m01.json'], value=value)
    metrics, quality = {}, {}
    for name, summary in summaries.items():
        if 'performance-summary.json' in summary:
            metrics[name] = summary['performance-summary.json']['value']
        if 'service/joint.json' in summary:
            metrics[name] = summary['service/joint.json']['value']
        if 'service/quality-summary/summary.json' in summary:
            quality[name] = summary['service/quality-summary/summary.json']['value']
    document = dict(schema='deeplinkengine-official-baseline-report/v1',
                    baseline=campaign['baseline_image']['id'], status=campaign['status'],
                    planned_denominators=campaign['fixed_five_denominators'], phases=summaries,
                    metrics=metrics, quality=quality, runtime_identity=campaign.get('runtime_identity'),
                    new_F_T_qualification=qualification is not None,
                    qualified_efficiency_percent=qualification['efficiency_lower_percent'] if qualification else None,
                    F_T_qualification=qualification,
                    scope='Original official-runtime measurements. Incomplete and failed phases remain explicit; archived 0.18 data and conclusions are unchanged.')
    save(root / 'report.json', document)
    figures = []
    if config is not None and metrics:
        from official_suite import ROOT, TOOLS, clean_client_environment
        sdk = Path(config['protocol_deployment'])
        cache = Path(config['cache']) / 'official-report'
        temporary = cache / 'tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env = clean_client_environment()
        env.update(XDG_CACHE_HOME=str(cache), TMPDIR=str(temporary), MPLCONFIGDIR=str(cache / 'matplotlib'))
        plan = dict(client_tools=str(TOOLS), protocol_deployment=str(sdk),
                    output=str(root), client_sources=campaign['sources'])
        existing = Path(config['protocol_run']).parents[1] / 'cache/xdg/vllm-hust-engine/report-mpl3108/bin/python'
        if existing.is_file():
            plan['report_python'] = str(existing)
        save(root / 'plotting-plan.json', plan)
        result = subprocess.run([config['protocol_python'], '-B', str(ROOT / 'official/official_http.py'),
                                 '--config', str(root / 'plotting-plan.json'), '--phase', 'plots'],
                                env=env, capture_output=True, text=True, timeout=900)
        (root / 'plotting.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError('Pinned CPU plot generation failed; preserve plotting.log')
        figures = sorted(path.name for path in root.glob('*.svg'))
    rows = ''.join('<tr><td>' + html.escape(name) + '</td><td>' + html.escape(row['status'])
                   + '</td><td>' + ' · '.join('<a href="' + html.escape(item['path'], quote=True)
                   + '">' + html.escape(label) + '</a>' for label, item in row.items() if isinstance(item, dict) and 'path' in item)
                   + '</td></tr>' for name, row in summaries.items())
    text = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(document['baseline']) + '</title><h1>' + html.escape(document['baseline']) + '</h1><p>' + html.escape(document['status']) + '</p><p>完整分母：MMLU 200 / HLE 200 / Frontier 160 / SWE 64 / Terminal 48。未完成项保留；不签发效率或第三方认证结论。</p><table><thead><tr><th>阶段</th><th>状态</th><th>原始结果</th></tr></thead><tbody>' + rows + '</tbody></table><p><a href="report.json">JSON / hashes</a></p></html>'
    if figures:
        text = text.replace('</html>', ''.join('<figure><img style="max-width:100%" src="' + html.escape(name, quote=True) + '" alt="Original official baseline measurements"></figure>' for name in figures) + '</html>')
    if qualification:
        text = text.replace('不签发效率或第三方认证结论。',
            '独立 F/T 达成率下界 ≥' + html.escape(qualification['efficiency_lower_percent'])
            + '%；第三方独立见证尚未完成。')
    (root / 'report.html').write_text(text, encoding='utf-8')
    seal(root)
    return document
