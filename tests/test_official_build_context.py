import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_build_context as build
import official_execution as execution


class BuildContextTests(unittest.TestCase):
    def archive(self,directory,items):
        path=Path(directory)/'layer.tar'
        with tarfile.open(path,'w') as archive:
            for name,kind,value in items:
                member=tarfile.TarInfo(name)
                if kind=='file':
                    member.size=len(value);archive.addfile(member,io.BytesIO(value))
                else:
                    member.type=tarfile.SYMTYPE;member.linkname=value;archive.addfile(member)
        return path

    def test_restore_system_tools_and_headers_without_python_cann_or_live_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'root';root.mkdir()
            items=[('usr/bin/g++','file',b'compiler'),('usr/include/google/protobuf/struct.proto','file',b'proto'),
                   ('usr/local/Ascend/private','file',b'cann'),('proc/live','file',b'host'),
                   ('root/secret','file',b'host'),('vllm-workspace/model','file',b'source')]
            build.apply_system_layer(self.archive(directory,items),root)
            self.assertEqual((root/'usr/bin/g++').read_bytes(),b'compiler')
            self.assertTrue((root/'usr/include/google/protobuf/struct.proto').is_file())
            self.assertFalse((root/'usr/local').exists());self.assertFalse((root/'root').exists())

    def test_whiteouts_follow_official_layer_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'root';(root/'usr/bin').mkdir(parents=True)
            (root/'usr/bin/old').write_bytes(b'old')
            build.apply_system_layer(self.archive(directory,[('usr/bin/.wh.old','file',b''),('usr/bin/new','file',b'new')]),root)
            self.assertFalse((root/'usr/bin/old').exists());self.assertTrue((root/'usr/bin/new').exists())

    def test_archive_escape_and_external_symlink_rejected(self):
        for name in ['../outside','/usr/bin/absolute','usr/bin/../../../../outside']:
            with self.subTest(name=name),self.assertRaises(ValueError):build.selected(name)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'root';root.mkdir()
            with self.assertRaisesRegex(ValueError,'symlink escaped'):
                build.apply_system_layer(self.archive(directory,[('usr/bin/compiler','link','../../../../outside')]),root)

    def base(self,root):
        sdk=root/'official-sdks/base';sdk.mkdir(parents=True)
        (sdk/'installed-files.json').write_text('{}')
        (sdk/'installation.json').write_text(json.dumps(dict(image_manifest=build.BASE_MANIFEST,
            installed_files_sha256=execution.sha(sdk/'installed-files.json'))))
        (sdk/'image-lock.json').write_text(json.dumps(dict(manifest_digest=build.BASE_MANIFEST,
            registry='https://fixture.invalid/',layers=[dict(digest='sha256:'+execution.BASE_DIGEST,size=1)])))
        return sdk

    def test_failed_package_preparation_preserves_staging_and_publishes_no_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk=self.base(root)
            def initial(blob,output):
                output.mkdir();(output/'rootfs').mkdir();(output/'receipt.json').write_text('{}')
                return output/'rootfs'
            with patch('baseline_setup.fetch',return_value=root/'layer'),patch.object(execution,'prepare_root',side_effect=initial), \
                 patch.object(build,'provision_protobuf',side_effect=RuntimeError('controlled package failure')):
                with self.assertRaisesRegex(RuntimeError,'controlled package failure'):build.ensure(sdk)
            contexts=root/'execution-contexts'
            self.assertEqual(len(list(contexts.glob('*.staging-*'))),1)
            self.assertEqual([p for p in contexts.iterdir() if p.is_dir() and '.staging-' not in p.name],[])

    def test_native_image_tools_do_not_bind_host_compilers_or_headers(self):
        calls=[]
        with patch.object(execution.shutil,'which',side_effect=lambda name:calls.append(name)), \
             patch.object(execution.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')):
            bindings,libraries=execution.host_tools(toolchain=False)
        self.assertEqual(calls,['npu-smi'])
        self.assertNotIn('/usr/include',[row[1] for row in bindings])


if __name__=='__main__':unittest.main()
