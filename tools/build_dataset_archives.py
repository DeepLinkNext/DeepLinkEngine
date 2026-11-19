"""Seal unchanged dataset originals; never sample, rewrite rows, or download inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def build(sdk, plans, traces, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / 'datasets/public-agent-trace-fixed32-v1.json').read_bytes())
    mmlu = json.loads((ROOT / 'datasets/mmlu-fixed200.json').read_bytes())
    parts = {name: {} for name in ('mmlu', 'hle', 'frontier', 'swe', 'terminal')}
    for name, expected in mmlu['frozen_files_sha256'].items():
        if name.startswith('inputs/'):
            p = sdk / name
            if sha(p) != expected:
                raise ValueError('Frozen MMLU input changed: ' + name)
            parts['mmlu']['mmlu/' + name] = p
    for p in (sdk / 'inputs/model-metadata').iterdir():
        if p.name in protocol['tokenizer_files']:
            if sha(p) != protocol['tokenizer_files'][p.name]:
                raise ValueError('Frozen tokenizer metadata changed')
            parts['mmlu']['mmlu/inputs/model-metadata/' + p.name] = p
    for name in ('hle', 'frontier'):
        contract = json.loads((plans / name / 'contract.json').read_bytes())
        for track, record in contract['sources'].items():
            p = plans / name / 'assets' / track / 'test.jsonl'
            if sha(p) != record['sha256']:
                raise ValueError('Frozen source changed: ' + name + '/' + track)
            parts[name][name + '/' + track + '/test.jsonl'] = p
    for name in ('swe', 'terminal'):
        for source in protocol['datasets'][name]['sources']:
            p = traces / source['cache_key']
            if p.stat().st_size != source['size'] or sha(p) != source['sha256']:
                raise ValueError('Frozen trace source changed: ' + source['cache_key'])
            parts[name]['public-agent-traces/' + source['cache_key']] = p
    manifest = dict(schema='deeplinkengine-fixed-dataset-bundles/v1',
        standard_id='five-datasets-20261113-v1',
        release='fixed-five-data-20261120-v1', repository='DeepLinkNext/DeepLinkEngine',
        components={})
    for name, files in parts.items():
        archive = out / ('fixed-five-data-' + name + '-20261113-v1.tar.gz')
        records = {n: dict(bytes=p.stat().st_size, sha256=sha(p)) for n, p in sorted(files.items())}
        with archive.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as target:
                for relative, path in sorted(files.items()):
                    item = tarfile.TarInfo(relative)
                    item.size, item.mode, item.mtime = path.stat().st_size, 0o644, 0
                    with path.open('rb') as stream:
                        target.addfile(item, stream)
        manifest['components'][name] = dict(name=archive.name, bytes=archive.stat().st_size,
            sha256=sha(archive), files=records,
            url='https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/' + manifest['release'] + '/' + archive.name)
        print(name, archive.stat().st_size, len(files), flush=True)
    paths = ['datasets/standard.json'] + ['datasets/' + n for n in json.loads((ROOT / 'datasets/standard.json').read_bytes())['artifacts']]
    manifest['frozen_protocols'] = {n: sha(ROOT / n) for n in paths}
    manifest['execution_scope'] = {n: dict(tasks=t, requests=r) for n,t,r in [('mmlu',200,200),('hle',200,200),('frontier',160,160),('swe',32,64),('terminal',32,48)]}
    notice = ROOT / 'data/DATA-NOTICES.txt'
    manifest['notice'] = dict(name=notice.name, bytes=notice.stat().st_size, sha256=sha(notice))
    (out / 'bundles.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('sdk', 'plans', 'traces', 'out'):
        parser.add_argument('--' + n, type=Path, required=True)
    a = parser.parse_args()
    build(a.sdk, a.plans, a.traces, a.out)
