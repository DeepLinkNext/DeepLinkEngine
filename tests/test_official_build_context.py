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
    def apt_root(self, directory):
        root=Path(directory)/'private-root';staging=Path(directory)/'staging'
        (root/'etc/apt').mkdir(parents=True);staging.mkdir()
        (root/'etc/apt/sources.list').write_text('deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://ports.ubuntu.com/ubuntu-ports/ jammy universe\n')
        return root,staging

    def test_private_mirror_preserves_signature_key_suite_and_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root,staging=self.apt_root(directory);logs=staging/'logs';logs.mkdir()
            source=root/'etc/apt/sources.list';original=source.read_bytes()
            sources=build.private_apt_sources(root,logs)
            hashes=build.set_private_mirror(root,sources,build.UBUNTU_PORTS_MIRRORS[0])
            text=source.read_text()
            self.assertIn('signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg',text)
            self.assertIn('jammy universe',text)
            self.assertIn('https://mirrors.ustc.edu.cn/ubuntu-ports/',text)
            self.assertEqual(hashes['etc/apt/sources.list']['original_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(json.loads((logs/'original-sources.json').read_text())['etc/apt/sources.list'],original.decode())
            with self.assertRaises(ValueError):build.set_private_mirror(root,sources,'http://untrusted.invalid')

    def test_private_apt_unknown_source_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root,staging=self.apt_root(directory)
            (root/'etc/apt/sources.list').write_text('deb https://unknown.invalid jammy main\n')
            with patch.object(execution,'popen_in_context') as launch:
                with self.assertRaisesRegex(ValueError,'recognized Ubuntu ports'):build.provision_protobuf(root,staging)
                launch.assert_not_called()

    def test_mirror_failure_journal_then_bounded_fallback_and_package_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root,staging=self.apt_root(directory);codes=iter([100,0,0]);seen=[]
            def launch(command,*args,**kwargs):
                journal=json.loads((staging/'provision/commands.json').read_text())
                self.assertEqual(journal[-1]['status'],'running')
                seen.append(journal[-1])
                self.assertLessEqual(journal[-1]['timeout_seconds'],180)
                self.assertIn('APT::Update::Error-Mode=any',command)
                self.assertNotIn('trusted=yes',' '.join(command))
                code=next(codes)
                class Process:
                    def wait(self,timeout):return code
                    def poll(self):return code
                return Process()
            with patch.object(execution,'popen_in_context',side_effect=launch),patch.object(execution,'run_in_context',return_value=subprocess.CompletedProcess([],0,stdout='protobuf-compiler\t3.12.4\n')):
                receipt=build.provision_protobuf(root,staging)
            self.assertEqual([r['mirror'] for r in seen],[build.UBUNTU_PORTS_MIRRORS[0]]+[build.UBUNTU_PORTS_MIRRORS[1]]*2)
            self.assertEqual([r['status'] for r in receipt['commands']],['failed','completed','completed'])
            self.assertIn('3.12.4',receipt['package_versions'])
            self.assertFalse(receipt['host_packages_modified'])

    def test_apt_timeout_is_recorded_and_owned_group_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root,staging=self.apt_root(directory)
            class Process:
                pid=123456
                done=False
                def wait(self,timeout):
                    if not self.done:
                        self.done=True
                        raise subprocess.TimeoutExpired('controlled fixture',timeout)
                    return -15
                def poll(self):return -15 if self.done else None
            # Explicitly model a still-live child after the initial wait timeout.
            class AliveProcess(Process):
                waits=0
                def wait(self,timeout):
                    self.waits+=1
                    if self.waits==1:raise subprocess.TimeoutExpired('controlled fixture',timeout)
                    self.done=True;return -15
            with patch.object(execution,'popen_in_context',side_effect=lambda *a,**k:AliveProcess()),patch.object(build.os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'bounded mirror'):build.provision_protobuf(root,staging)
            self.assertEqual(kill.call_count,2)
            rows=json.loads((staging/'provision/commands.json').read_text())
            self.assertEqual([r['status'] for r in rows],['timeout','timeout'])
            self.assertTrue(all(r['process_reaped'] for r in rows))

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
