import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports

enable_module_imports()
import official_source_build_graph as graph
import official_build_assets as assets
import official_source_candidate as candidate


class SourceGraph(unittest.TestCase):
    def wheel(self,root,name,version):
        file=root/(name+'-'+version+'-py3-none-any.whl')
        with zipfile.ZipFile(file,'w') as stream:
            stream.writestr(name+'-'+version+'.dist-info/METADATA','Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        return file

    def row(self,file,package,version,stage):
        return dict(package=package,version=version,stage=stage,requirement=package+'=='+version,
            filename=file.name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            url='https://publisher.invalid/'+file.name,upstream_url='https://publisher.invalid/'+file.name)

    def test_original_nested_build_nodes_keep_distinct_dependency_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build=root/'build';build.mkdir();project=root/'root';project.mkdir()
            original=b'[build-system]\nrequires=["scope_dep==2.0","scope_nested==0.1"]\n'
            (project/'pyproject.toml').write_bytes(original)
            one=self.wheel(root,'scope_dep','1.0');two=self.wheel(root,'scope_dep','2.0')
            nested=root/'scope_nested-0.1.tar.gz';nested_pyproject=b'[build-system]\nrequires=["scope_dep==1.0"]\n'
            with tarfile.open(nested,'w:gz') as stream:
                member=tarfile.TarInfo('scope_nested-0.1/pyproject.toml');member.size=len(nested_pyproject);stream.addfile(member,io.BytesIO(nested_pyproject))
            rows=[self.row(two,'scope_dep','2.0','01-core'),self.row(one,'scope_dep','1.0','nested'),self.row(nested,'scope_nested','0.1','nested-source')]
            registry={'sha256:'+row['sha256']:row for row in rows};by_name={row['filename']:row for row in rows};by_file={file.name:file for file in (one,two,nested)}
            commands=[]
            def run(argv,**kwargs):
                commands.append((argv,dict(kwargs['env'])))
                self.assertNotIn('PIP_BUILD_CONSTRAINT',kwargs['env']);self.assertNotIn('PIP_CONSTRAINT',kwargs['env'])
                self.assertIn('--constraint',argv);self.assertIn('--find-links',argv)
                current=graph.requirements(Path(kwargs['cwd'])/'pyproject.toml')
                selected=[rows[1]] if current==['scope_dep==1.0'] else [rows[0],rows[2]]
                install=[dict(metadata=dict(name=row['package'],version=row['version'],requires_dist=[]),
                    download_info=dict(url=row['url'],archive_info=dict(hashes=dict(sha256=row['sha256'])))) for row in selected]
                Path(argv[argv.index('--report')+1]).write_text(json.dumps(dict(install=install,environment={})))
            def fetch(row,cache):return by_file[row['filename']],dict(cpu_fixture=True)
            def authority(item,primary):return by_name[item['download_info']['url'].rsplit('/',1)[-1]]
            with patch.object(candidate,'build_dependency_artifacts',return_value=registry),patch('transport_trust.configure',return_value=dict(cpu_fixture=True)),patch.object(assets,'fetch',side_effect=fetch),patch.object(assets,'authority',side_effect=authority),patch.object(assets,'admit_pip_cache',return_value=None),patch.object(assets,'_run',side_effect=run):
                links,proof=graph.prepare(Path(sys.executable),project,root/'cache',build,dict(PIP_BUILD_CONSTRAINT='wrong-flat-file',PIP_CONSTRAINT='wrong-global-file'),sdk=root/'sdk',stage='01-core')
            self.assertEqual((project/'pyproject.toml').read_bytes(),original)
            self.assertEqual({item.name for item in links.iterdir()},{one.name,two.name,nested.name})
            self.assertEqual(len(commands),2);lock=json.loads(Path(proof['lock']).read_text())
            self.assertFalse(lock['global_build_constraints_added']);self.assertEqual(len(lock['nested_builds']),1)
            child=json.loads(Path(lock['nested_builds'][0]['build']['lock']).read_text())
            self.assertEqual(child['original_requirements'],['scope_dep==1.0'])
            self.assertEqual(child['artifacts'][0]['metadata']['version'],'1.0')
            self.assertEqual(lock['artifacts'][0]['metadata']['version'],'2.0')

    def test_approved_links_reject_changed_bytes_and_conflicting_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'asset.whl';file.write_bytes(b'approved')
            row=self.row(file,'example','1.0','fixture');row['path']=str(file);file.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError,'input changed'):graph.approved_links([row],root/'bad')
            file.write_bytes(b'approved')
            with self.assertRaisesRegex(ValueError,'Conflicting approved'):
                graph.approved_links([row,dict(row,sha256='a'*64)],root/'conflict')


if __name__=='__main__':unittest.main()
