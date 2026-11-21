"""Build-input catalog CPU contracts; no framework or real SDK is imported."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_source_prefetch as prefetch
import official_build_assets as assets


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.environment=dict(tags=['py3-none-any','cp312-cp312-manylinux_2_28_aarch64'],markers=dict(python_full_version='3.12.13',python_version='3.12',platform_machine='aarch64',sys_platform='linux',implementation_name='cpython'))

    def wheel(self,name='demo',version='1.0',**changes):
        return dict(filename=f'{name}-{version}-py3-none-any.whl',packagetype='bdist_wheel',size=42,
            digests={'sha256':'a'*64},url='https://files.pythonhosted.org/pinned.whl',requires_python='>=3.10',yanked=False,**changes)

    def test_only_exact_active_pins_are_preloaded_without_changing_original_requirements(self):
        original=['torch==2.10.0','demo==1.0; python_version >= "3.12"','off==1.0; sys_platform == "win32"','wild==1.*','range>=1.0','almost===1.0']
        projects=[dict(project='original',requires=list(original))]
        selected=prefetch.requirements(projects,self.environment['markers'])
        self.assertEqual({r['name']:r['exact'] for r in selected},{'torch':'2.10.0','demo':'1.0','wild':None,'range':None,'almost':None})
        self.assertEqual(projects[0]['requires'],original)

    def test_supported_tags_python_and_actual_release_filename_filter_preload_only(self):
        good=self.wheel();wrong_python=dict(good,filename='demo-1.0-py2.py3-none-any.whl',requires_python='>=99')
        wrong_tag=dict(good,filename='demo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl')
        wrong_name=dict(good,filename='other-1.0-py3-none-any.whl');wrong_version=dict(good,filename='demo-2.0-py3-none-any.whl')
        release=dict(urls=[good,wrong_python,wrong_tag,wrong_name,wrong_version])
        with patch.object(assets,'_read',return_value=json.dumps(release).encode()):
            rows,proof=prefetch.published_wheels('demo','1.0',self.environment,self.root)
        self.assertEqual([r['filename'] for r in rows],[good['filename']]);self.assertEqual(proof['status'],'published')

    def test_source_only_requirement_is_delegated_instead_of_inventing_a_wheel(self):
        release=dict(urls=[dict(filename='demo-1.0.tar.gz',packagetype='sdist')])
        with patch.object(assets,'_read',return_value=json.dumps(release).encode()):
            rows,proof=prefetch.published_wheels('demo','1.0',self.environment,self.root)
        self.assertEqual(rows,[]);self.assertEqual(proof['status'],'delegated')

    def test_missing_pypi_uses_real_release_digest_for_triton_only(self):
        filename='triton_ascend-3.2.2-cp312-cp312-manylinux_2_28_aarch64.whl'
        release=dict(assets=[dict(name=filename,digest='sha256:'+'b'*64,size=269942385,id=123,browser_download_url='https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/'+filename)])
        def read(url):
            if url.startswith('https://pypi.org/'):raise urllib.error.HTTPError(url,404,'missing',{},None)
            self.assertEqual(url,'https://api.github.com/repos/triton-lang/triton-ascend/releases/tags/v3.2.2');return json.dumps(release).encode()
        with patch.object(assets,'_read',side_effect=read):rows,_=prefetch.published_wheels('triton-ascend','3.2.2',self.environment,self.root)
        self.assertEqual(rows[0]['sha256'],'b'*64);self.assertEqual(rows[0]['source_authority'],'official-github-release-asset')
        release['assets'][0].pop('digest')
        with patch.object(assets,'_read',side_effect=read):rows,_=prefetch.published_wheels('triton-ascend','3.2.2',self.environment,self.root)
        self.assertEqual(rows,[])

    def test_vendor_link_keeps_original_no_hash_identity_and_rejects_wrong_advertisement(self):
        name='triton_ascend-3.2.2-py3-none-any.whl';row=dict(filename=name,package='triton-ascend',sha256='a'*64,publisher={'fixture':True})
        for fragment in ['', '#sha256='+'0'*64]:
            raw=('<a href="'+name+fragment+'">wheel</a>').encode()
            with patch.object(assets,'_read',return_value=raw):
                if fragment:
                    with self.assertRaisesRegex(ValueError,'different publisher SHA'):prefetch.recipe_vendor(row,self.root)
                else:
                    url,proof=prefetch.recipe_vendor(row,self.root)
                    self.assertFalse(proof['link_has_digest']);self.assertEqual(proof['link'],url)

    def test_cpu_primary_variants_are_not_replaced_by_pypi_pins(self):
        import official_source_candidate as candidate
        original=[dict(project='core',requires=['torch==2.13.0','torch-npu==2.10.0.post4'])]
        with patch.object(prefetch,'runtime',return_value=self.environment),patch.object(prefetch,'original_projects',return_value=original), \
             patch.object(prefetch,'cached_build_inputs',return_value=[]),patch.object(prefetch,'published_wheels') as publisher:
            rows=prefetch.collect(self.root/'cache',self.root/'out')
        publisher.assert_not_called()
        self.assertIn(candidate.TORCH_BUILD_WHEEL['filename'],[r['filename'] for r in rows])
        self.assertIn(candidate.NESTED_BUILD_ARTIFACTS['arctic-build-torch']['filename'],[r['filename'] for r in rows])


if __name__=='__main__':unittest.main()
