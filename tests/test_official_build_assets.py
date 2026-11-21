import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_build_assets as assets
import official_source_candidate as candidate
import transport_trust


class BuildAssetsTests(unittest.TestCase):
    def wheel(self,root,name,version,requirements=()):
        path=root/(name+'-'+version+'-py3-none-any.whl')
        metadata='Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n'+''.join('Requires-Dist: '+r+'\n' for r in requirements)+'\n'
        with zipfile.ZipFile(path,'w') as archive:archive.writestr(name+'-'+version+'.dist-info/METADATA',metadata)
        return path

    def test_mirror_requires_exact_filename_and_publisher_sha(self):
        digest='a'*64;row=dict(filename='target.whl',sha256=digest)
        page=('<a href="../packages/target.whl#sha256='+digest+'">good</a><a href="../packages/wrong.whl#sha256='+digest+'">wrong</a><a href="../packages/target.whl#sha256='+'b'*64+'">bad hash</a><a href="http://invalid/target.whl#sha256='+digest+'">http</a>').encode()
        with patch.object(assets,'_read',return_value=page):links,proof=assets.matching_links('https://mirror.invalid/simple/pkg/',row)
        self.assertEqual(links,['https://mirror.invalid/simple/packages/target.whl'])
        self.assertEqual(proof['index_sha256'],hashlib.sha256(page).hexdigest())

    def test_publisher_metadata_must_match_report_whole_hash(self):
        item=dict(metadata=dict(name='sample',version='1.0'),download_info=dict(url='https://mirror.invalid/sample.whl',archive_info=dict(hashes=dict(sha256='a'*64))))
        raw=json.dumps(dict(urls=[dict(filename='sample.whl',size=12,url='https://files.pythonhosted.org/sample.whl',digests=dict(sha256='b'*64))])).encode()
        with patch.object(assets,'_read',return_value=raw):
            with self.assertRaisesRegex(ValueError,'absent from the official'):assets.authority(item,{})

    def test_wheel_metadata_is_checked_after_whole_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.wheel(Path(directory),'sample','1.0',['leaf>=2'])
            correct=dict(name='sample',version='1.0',requires_dist=['leaf >= 2'])
            self.assertEqual(assets.verify_wheel_metadata(path,correct)['kind'],'verified-wheel-metadata')
            with self.assertRaisesRegex(ValueError,'dependency metadata'):assets.verify_wheel_metadata(path,dict(correct,requires_dist=[]))

    def test_closure_checks_marker_extras_and_missing_edges(self):
        rows=[dict(metadata=dict(name='root',version='1.0',requires_dist=['leaf>=2; extra == "fast"'])),dict(metadata=dict(name='leaf',version='2.1'))]
        self.assertEqual(assets.verify_closure(['root[fast]>=1'],rows,{})['reachable_projects'],2)
        self.assertEqual(assets.verify_closure(['root>=1'],rows[:1],{})['reachable_projects'],1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):assets.verify_closure(['root[fast]>=1'],rows[:1],{})

    def test_exact_pip_cache_body_is_admitted_only_after_whole_sha(self):
        from pip._internal.network.cache import SafeFileCache
        from pip._vendor.cachecontrol.controller import CacheController
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);body=b'verified resolver HTTP response';url='https://mirror.invalid/exact.whl';pip_cache=root/'pip'
            http=SafeFileCache(str(pip_cache/'http-v2'));source=Path(http._get_cache_path(CacheController.cache_url(url))+'.body');source.parent.mkdir(parents=True);source.write_bytes(body)
            row=dict(filename='exact.whl',bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            with patch.object(assets,'transports',side_effect=AssertionError('must not re-download')):
                path,proof=assets.admit_pip_cache(row,root/'wheels',pip_cache=pip_cache,report_url=url)
            self.assertEqual(path.read_bytes(),body);self.assertEqual(proof['kind'],'verified-metadata-http-cache')
            self.assertNotEqual(path.stat().st_ino,source.stat().st_ino)
            self.assertIsNone(assets.admit_pip_cache(dict(row,sha256='a'*64),root/'other',pip_cache=pip_cache,report_url=url))
            self.assertIsNone(assets.admit_pip_cache(row,root/'other',pip_cache=pip_cache,report_url=url+'other'))

    def test_existing_range_downloader_resumes_real_http_and_checks_whole_sha(self):
        body=b'build-assets-range-fixture'*2048
        offsets=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);offsets.append(start);data=body[start:]
                self.send_response(206 if start else 200)
                if start:self.send_header('Content-Range',f'bytes {start}-{len(body)-1}/{len(body)}')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);row=dict(filename='fixture.whl',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),url=f'http://127.0.0.1:{server.server_port}/wheel',package='fixture')
                partial=root/'fixture.whl.download.partial';partial.write_bytes(body[:271])
                (root/'fixture.whl.download.json').write_text(json.dumps(dict(schema='deeplinkengine-dependency-download/v1',artifact={k:row[k] for k in ('filename','sha256','bytes')},partial_bytes=271,partial_sha256=hashlib.sha256(body[:271]).hexdigest(),attempts=[])))
                with patch.object(assets,'transports',return_value=([row['url']],[])):path,proof=assets.fetch(row,root)
                self.assertEqual(path.read_bytes(),body);self.assertEqual(offsets,[271]);self.assertFalse(proof['cached']);self.assertNotIn('torch',sys.modules)
        finally:server.shutdown();server.server_close();thread.join(timeout=5)

    def test_full_plan_preserves_original_requirements_and_only_creates_build_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);core=root/'core';core.mkdir();build=root/'build';build.mkdir()
            original=b'[build-system]\nrequires=["torch == 2.13.0"]\n';(core/'pyproject.toml').write_bytes(original)
            torch=self.wheel(root,'torch','2.13.0+cpu',['sympy>=1.13.3']);sympy=self.wheel(root,'sympy','1.14.0')
            rows=[]
            for path,name,version,requirements in [(torch,'torch','2.13.0+cpu',['sympy>=1.13.3']),(sympy,'sympy','1.14.0',[])]:
                digest=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(dict(metadata=dict(name=name,version=version,requires_dist=requirements),download_info=dict(url='https://mirror.invalid/'+path.name,archive_info=dict(hashes=dict(sha256=digest)))))
            primary=dict(filename=torch.name,bytes=torch.stat().st_size,sha256=rows[0]['download_info']['archive_info']['hashes']['sha256'],url='https://download.pytorch.org/torch.whl',upstream_url='https://download.pytorch.org/torch.whl',package='torch',stage='01-core',requirement='torch==2.13.0')
            primary_constraint=root/'primary.txt';primary_constraint.write_text('fixture')
            official=json.dumps(dict(urls=[dict(filename=sympy.name,size=sympy.stat().st_size,url='https://files.pythonhosted.org/'+sympy.name,digests=dict(sha256=rows[1]['download_info']['archive_info']['hashes']['sha256']))])).encode()
            def resolve(argv,**kwargs):
                self.assertIn('--dry-run',argv);self.assertIn('torch == 2.13.0',argv);self.assertNotIn('--only-binary',argv);self.assertNotIn('PIP_CONSTRAINT',kwargs['env'])
                report=Path(argv[argv.index('--report')+1]);report.write_text(json.dumps(dict(install=rows,environment={})))
            def fetch(row,cache):return (torch if row['package']=='torch' else sympy),dict(cpu_fixture=True)
            with patch.object(candidate,'build_dependency_artifacts',return_value={'sha256:'+primary['sha256']:primary}),patch.object(candidate,'primary_build_constraint',return_value=(primary_constraint,dict(cpu_fixture=True))), \
                 patch.object(transport_trust,'configure',return_value=dict(cpu_fixture=True)),patch.object(assets,'_run',side_effect=resolve),patch.object(assets,'fetch',side_effect=fetch),patch.object(assets,'admit_pip_cache',return_value=None),patch.object(assets,'_read',return_value=official):
                path,proof=assets.prepare_stage(Path(sys.executable),core,root/'cache',build,{},sdk=root/'sdk',stage='01-core')
            text=path.read_text();self.assertIn('torch @ file:',text);self.assertIn('sympy @ file:',text);self.assertFalse(proof['runtime_constraint_added']);self.assertEqual((core/'pyproject.toml').read_bytes(),original)


if __name__=='__main__':unittest.main()
