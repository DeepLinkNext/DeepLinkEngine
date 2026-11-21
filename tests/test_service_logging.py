"""CPU checks using the byte-verified upstream logger, without tensor packages."""
import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import service_logging as adapter

URL = ('https://raw.githubusercontent.com/vllm-project/vllm-ascend/'
       'c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/logger.py')
VENDOR = None
PLATFORM = None


def vendor():
    global VENDOR
    if VENDOR is None:
        source = os.environ.get('DEEPLINK_TEST_ASCEND_LOGGER')
        VENDOR = Path(source).read_bytes() if source else urllib.request.urlopen(URL, timeout=45).read()
        if hashlib.sha256(VENDOR).hexdigest() != adapter.LOGGER_SHA256:
            raise ValueError('Test requires the exact real upstream logger')
    return VENDOR


def platform_logging_statements():
    global PLATFORM
    if PLATFORM is None:
        source = os.environ.get('DEEPLINK_TEST_ASCEND_PLATFORM')
        PLATFORM = Path(source).read_bytes() if source else urllib.request.urlopen(URL.replace('logger.py', 'platform.py'), timeout=45).read()
    method = next(node for node in ast.walk(ast.parse(PLATFORM))
                  if isinstance(node, ast.FunctionDef) and node.name == 'check_and_update_config')
    names = ('configure_ascend_file_logging', 'configure_ascend_logging')
    calls = [node for node in method.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id in names]
    if tuple(node.value.func.id for node in calls) != names:
        raise ValueError('Actual platform logging order changed; review the adapter')
    return compile(ast.Module(body=calls, type_ignores=[]), '<actual-vendor-platform-logging>', 'exec')


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sdk, self.out = self.base / 'sdk', self.base / 'out'
        self.site = self.sdk / 'site'
        self.site.mkdir(parents=True)
        self.out.mkdir()
        self.put('vllm/__init__.py', "__version__='pinned'\n")
        self.put('vllm/envs.py', "NO_COLOR=True\nVLLM_LOGGING_COLOR='0'\nVLLM_LOGGING_STREAM='ext://sys.stderr'\nVLLM_LOGGING_LEVEL='INFO'\n")
        self.put('vllm/logging_utils.py', "from logging import Formatter\nclass NewLineFormatter(Formatter):\n def format(self,record):\n  record.fileinfo=record.filename\n  return super().format(record)\nColoredFormatter=NewLineFormatter\n")
        self.put('vllm_ascend/__init__.py', "from .logger import configure_ascend_file_logging\nconfigure_ascend_file_logging()\n")
        self.put('vllm_ascend/logger.py', vendor())
        self.put('vllm_ascend/ascend_config.py', "def get_ascend_config():\n raise ValueError('not initialized yet')\n")
        self.env = dict(os.environ, PYTHONPATH=str(self.site), PYTHONDONTWRITEBYTECODE='1')
        # Simulate an unusable inherited configuration path, without changing HOME.
        invalid = self.base / 'unusable-config'; invalid.write_text('not a directory')
        self.env.update(XDG_CONFIG_HOME=str(invalid), VLLM_CONFIG_ROOT=str(invalid))

    def put(self, path, data):
        p = self.site / path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
        return p

    def child(self, code, *args):
        result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT), str(self.sdk), str(self.out), *args],
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_first_native_file_fallback_and_cleanup(self):
        result = self.child("""
import sys,os,json,logging
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import service_logging as a
home=os.environ.get('HOME');before=set(sys.modules);p=a.prepare(sys.argv[2],sys.argv[3],os.environ)
assert not {'vllm','torch','vllm_ascend'} & (set(sys.modules)-before)
finder=a.install(p);assert 'vllm' not in sys.modules
import vllm_ascend.logger as v
assert finder not in sys.meta_path
assert Path(v._LOG_DIR).resolve()!=Path(p['logs']).resolve()
assert Path(v._file_handler.baseFilename).parent==Path(p['logs'])
v.configure_ascend_file_logging();assert Path(v._file_handler.baseFilename).parent==Path(p['logs'])
import vllm_ascend.ascend_config as config,types
config.get_ascend_config=lambda:types.SimpleNamespace(ascend_log_path=p['logs'])
v.configure_ascend_file_logging();assert Path(v._file_handler.baseFilename).parent==Path(p['logs'])
logging.getLogger('vllm_ascend').warning('API first message')
v._file_handler.flush();assert 'API first message' in Path(v._file_handler.baseFilename).read_text()
assert os.environ.get('HOME')==home
for key in ('XDG_CONFIG_HOME','VLLM_CONFIG_ROOT'):
 Path(os.environ[key],'cpu-write-check').write_text('ok')
assert a.digest(v.__file__)==a.LOGGER_SHA256
assert json.loads(Path(p['receipt']).read_bytes())['adapter_applied'] is True
""")
        self.assertIn('API first message', result.stderr)

    def test_native_file_first_order_skips_console_without_adapter(self):
        # Real vendor source: a file handler makes its native console guard return.
        self.put('vllm_ascend/__init__.py', '')
        self.child("""
import sys,logging
import vllm_ascend.logger as v
v._setup_file_logging(log_dir=sys.argv[3])
v.configure_ascend_logging()
handlers=logging.getLogger('vllm_ascend').handlers
assert any(isinstance(x,logging.FileHandler) for x in handlers)
assert not any(isinstance(x,logging.StreamHandler) and not isinstance(x,logging.FileHandler) for x in handlers)
""")

    def test_actual_platform_sequence_keeps_native_console_and_file(self):
        # Serialize the two actual AST statements, not a rewritten platform fixture.
        statements = platform_logging_statements()
        import marshal
        compiled = self.out/'platform-logging.marshal';compiled.write_bytes(marshal.dumps(statements))
        result = self.child("""
import sys,os,marshal,logging
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import service_logging as a
p=a.prepare(sys.argv[2],sys.argv[3],os.environ);a.install(p)
import vllm_ascend.logger as v
exec(marshal.loads((Path(sys.argv[3])/'platform-logging.marshal').read_bytes()),vars(v))
logging.getLogger('vllm_ascend').warning('actual platform console and file')
v._file_handler.flush()
assert 'actual platform console and file' in Path(v._file_handler.baseFilename).read_text()
""")
        self.assertIn('actual platform console and file', result.stderr)

    def test_unknown_vendor_and_outside_sdk_refused_without_import(self):
        (self.site / 'vllm_ascend/logger.py').write_text('# unknown vendor\n')
        with self.assertRaisesRegex(ValueError, 'Unsupported Ascend logger'):
            adapter.prepare(self.sdk, self.out, self.env)
        with self.assertRaisesRegex(ValueError, 'SDK-bound'):
            adapter.prepare(self.base / 'other', self.out, self.env)
        self.assertNotIn('vllm', sys.modules)

    def test_module_absent_is_explicit_and_dry_plan_has_no_hook(self):
        (self.site / 'vllm_ascend/logger.py').unlink()
        before = tuple(sys.meta_path)
        p = adapter.prepare(self.sdk, self.out, self.env)
        self.assertEqual(p['state'], 'not-applicable-module-absent')
        self.assertEqual(tuple(sys.meta_path), before)
        self.assertNotIn('vllm_ascend', sys.modules)

    def test_changed_source_after_plan_and_late_import_refused(self):
        self.child("""
import sys,os
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import service_logging as a
p=a.prepare(sys.argv[2],sys.argv[3],os.environ)
source=Path(p['vendor_source']);source.write_bytes(source.read_bytes()+b'\\n')
try:a.install(p)
except ValueError:pass
else:raise AssertionError('changed source accepted')
sys.modules['vllm']=object()
try:a.install(p)
except ValueError as e:assert 'before framework' in str(e)
else:raise AssertionError('late hook accepted')
""")

    def test_import_origin_mismatch_removes_hook_and_records_failure(self):
        self.child("""
import sys,os,types,json
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import service_logging as a
p=a.prepare(sys.argv[2],sys.argv[3],os.environ);finder=a.install(p)
package=types.ModuleType('vllm_ascend');package.__path__=[str(Path(sys.argv[3])/'absent')]
sys.modules['vllm_ascend']=package
try:__import__('vllm_ascend.logger')
except ValueError as e:assert 'outside the admitted' in str(e)
else:raise AssertionError('wrong source accepted')
assert finder not in sys.meta_path
assert json.loads(Path(p['receipt']).read_bytes())['state']=='failed'
""")

    def test_spawn_does_not_inherit_hook_but_keeps_config_environment(self):
        fixture = self.out / 'spawn_fixture.py'
        fixture.write_text("""
import sys,os,multiprocessing
def child(pipe):
 pipe.send(dict(hooks=[type(x).__name__ for x in sys.meta_path],xdg=os.environ['XDG_CONFIG_HOME']))
 pipe.close()
if __name__=='__main__':
 sys.path.insert(0,sys.argv[1]);import service_logging as a
 p=a.prepare(sys.argv[2],sys.argv[3],os.environ);a.install(p)
 ctx=multiprocessing.get_context('spawn');receive,send=ctx.Pipe(False)
 process=ctx.Process(target=child,args=(send,));process.start();send.close()
 assert receive.poll(15);result=receive.recv();process.join(10)
 assert process.exitcode==0 and 'Finder' not in result['hooks']
 assert result['xdg']==p['config_environment']['XDG_CONFIG_HOME']
 assert 'early HOME fallback is not covered' in p['scope']
""")
        result=subprocess.run([sys.executable,'-B',str(fixture),str(ROOT),str(self.sdk),str(self.out)],env=self.env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_official_production_entry_installs_before_import(self):
        self.put('torch.py', '')
        self.put('torch_npu.py', '')
        self.child("""
import sys,os,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1]);import official_serve
out=Path(sys.argv[3]).resolve();plan=dict(sdk=sys.argv[2],output=str(out),runtime_versions={n:'pinned' for n in ('vllm','vllm-ascend','torch','torch-npu')},worker_sources={},baseline_id='cpu',serve_arguments=[],model='/cpu-only')
sdk=Path(plan['sdk']);(sdk/'image-lock.json').write_text(json.dumps(dict(manifest_digest='sha256:'+'a'*64)))
from official_runtime_identity import runtime_identity
plan['runtime_identity']=runtime_identity(sdk)
path=out/'official.json';path.write_text(json.dumps(plan));sys.argv=['official_serve',str(path)]
import types
sys.modules['engine_entry']=types.SimpleNamespace(check_pinned_host_memory=lambda *a:None)
with patch.object(official_serve.metadata,'version',return_value='pinned'),patch.object(sys,'prefix',plan['sdk']),patch.object(official_serve.runpy,'run_module') as cli:
 official_serve.main();cli.assert_called_once()
receipt=json.loads((out/'service-logging-receipt.json').read_bytes())
assert receipt['state']=='applied' and Path(receipt['actual_handler']).parent==out/'vllm-ascend-logs'
assert json.loads((out/'actual-package-imports.json').read_bytes())['logging_adapter_receipt']==str(out/'service-logging-receipt.json')
""")

    def test_public_wrapper_preserves_child_preimport_and_arguments(self):
        child_source = os.environ.get('DEEPLINK_TEST_R8_CHILD')
        if child_source:
            body = Path(child_source).read_bytes()
        else:
            body = b"import sys\nassert 'vllm' not in sys.modules\nimport vllm_ascend.logger as logger\nlogger.configure_ascend_file_logging()\n"
        child = self.sdk / 'runtime/entrypoints/serve_b1_child.py'
        child.parent.mkdir(parents=True);child.write_bytes(body)
        self.put('corrected_service_entry.py', 'def compose_environment(env):\n return dict(env)\n')
        self.put('corrected_serving_runtime.py', 'class CorrectedWorker:pass\n')
        self.put('betterscale/__init__.py', '')
        self.put('betterscale/qwen35_worker.py', 'from corrected_serving_runtime import CorrectedWorker as Worker\n')
        self.put('vllm/plugins.py', 'def load_general_plugins():\n import vllm_ascend.logger\n')
        self.put('vllm/entrypoints/__init__.py', '')
        self.put('vllm/entrypoints/cli/__init__.py', '')
        self.put('vllm/entrypoints/cli/main.py', 'def main():\n import logging\n logging.getLogger("vllm").warning("real child reached CLI")\n')
        self.put('vllm_ascend-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: vllm_ascend\nVersion: 1.0\n')
        self.env['VLLM_PLUGINS'] = 'cpu-only'
        plan = adapter.prepare(self.sdk, self.out, self.env)
        plan.update(child=str(child.resolve()), child_sha256=adapter.digest(child),
                    entry_sha256=adapter.digest(ROOT/'logging_entry.py'))
        path=self.out/'service-logging-plan.json';adapter.save(path, plan)
        service=self.out/'plan.json';service.write_text(json.dumps(dict(output=str(self.out),expected_versions=dict(vllm='pinned',vllm_ascend='1.0'))))
        result=subprocess.run([sys.executable,'-B',str(ROOT/'logging_entry.py'),str(path),'--',str(child),str(service),'serve','/cpu-only'],env=self.env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        receipt=json.loads(Path(plan['receipt']).read_bytes())
        self.assertEqual(receipt['state'],'applied')
        self.assertEqual(adapter.digest(child),plan['child_sha256'])
        if child_source:
            self.assertTrue((self.out/'runtime-identity.json').is_file())
            self.assertIn('real child reached CLI',Path(receipt['actual_handler']).read_text())


if __name__ == '__main__':
    unittest.main()
