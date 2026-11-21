"""Publish actual phase state and original outputs without invented metrics."""
import hashlib
import html
import json
from pathlib import Path

from official_baselines import save


def report(root):
    root = Path(root)
    campaign = json.loads((root / 'campaign.json').read_bytes())
    summaries = {}
    for phase in campaign['stages']:
        directory = root / phase['name']
        summaries[phase['name']] = dict(status=phase['status'])
        for name in ('service/joint.json', 'service/continuous.json', 'service/dataset-phase.json',
                     'service/acceptance/execution.json', 'client/timings.json', 'receipt.json'):
            path = directory / name
            if path.is_file():
                summaries[phase['name']][name] = dict(path=str(path.relative_to(root)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), value=json.loads(path.read_bytes()))
    document = dict(schema='deeplinkengine-official-baseline-report/v1',
                    baseline=campaign['baseline_image']['id'], status=campaign['status'],
                    planned_denominators=campaign['fixed_five_denominators'], phases=summaries,
                    new_F_T_qualification=False, qualified_efficiency_percent=None,
                    scope='Original official-runtime measurements. Incomplete and failed phases remain explicit; archived 0.18 data and conclusions are unchanged.')
    save(root / 'report.json', document)
    rows = ''.join('<tr><td>' + html.escape(name) + '</td><td>' + html.escape(row['status'])
                   + '</td><td>' + ' · '.join('<a href="' + html.escape(item['path'], quote=True)
                   + '">' + html.escape(label) + '</a>' for label, item in row.items() if isinstance(item, dict) and 'path' in item)
                   + '</td></tr>' for name, row in summaries.items())
    text = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(document['baseline']) + '</title><h1>' + html.escape(document['baseline']) + '</h1><p>' + html.escape(document['status']) + '</p><p>完整分母：MMLU 200 / HLE 200 / Frontier 160 / SWE 64 / Terminal 48。未完成项保留；不签发效率或第三方认证结论。</p><table><thead><tr><th>阶段</th><th>状态</th><th>原始结果</th></tr></thead><tbody>' + rows + '</tbody></table><p><a href="report.json">JSON / hashes</a></p></html>'
    (root / 'report.html').write_text(text, encoding='utf-8')
    return document
