"""CPU contracts for standalone install, serving and the HTTP example."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deeplink
spec = importlib.util.spec_from_file_location("service", ROOT / "engine/service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class Standalone(unittest.TestCase):
    def setUp(self):
        # Network selection is independently tested with real archive fixtures.
        selected=patch('engine.release_selection.select',return_value=None)
        selected.start()
        self.addCleanup(selected.stop)

    def test_help_without_runtime(self):
        for args in [["--help"], ["install", "--help"], ["serve", "--help"]]:
            result = subprocess.run([sys.executable, str(ROOT / "deeplink.py"), *args], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_devices(self):
        self.assertEqual(service.pair("0,7"), [0, 7])
        for value in ["0,0", "-1,7", "0", "0,1,2", "0,x"]:
            with self.assertRaises(ValueError):
                service.pair(value)

    def test_existing_target_including_dangling_link(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "install"
            out.symlink_to(Path(folder) / "absent", target_is_directory=True)
            args = argparse.Namespace(out=str(out), cache=folder)
            with self.assertRaisesRegex(ValueError, "new installation"):
                deeplink.install(args)

    def test_receipt_checks_version_and_root(self):
        lock = json.loads((ROOT / "engine/r8-lock.json").read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            value = dict(release=lock["upstream_release"], package_version=lock["public_distribution"].split("==")[1], deployment=str(out / "sdk"))
            (out / "sdk").mkdir()
            (out / "sdk/deployment.json").write_text(json.dumps(dict(release=value["release"])))
            receipt = out / "installation.json"
            receipt.write_text(json.dumps(value))
            self.assertEqual(deeplink.installation(out)[0], out.resolve())
            value["deployment"] = str(out / "outside")
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                deeplink.installation(out)
            value["package_version"] = "wrong"
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                deeplink.installation(out)

    def test_install_without_models_or_evaluation(self):
        from engine import environment_setup as setup
        calls, downloads = [], []
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            cache = Path(folder) / "cache"
            args = argparse.Namespace(out=str(Path(folder)/"install"), cache=str(cache), python=None, wheelhouse=None, cann_home=None, use_environment=None)
            def stage(command, env, log):
                calls.append(command)
                self.assertTrue(Path(env["TMPDIR"]).is_dir())
                self.assertTrue(Path(env["PIP_CACHE_DIR"]).is_dir())
                return types.SimpleNamespace(returncode=0)
            def artifact(source, links, destination):
                downloads.append(source)
                return cache / source["filename"]
            with patch.object(setup, "ensure_environment", return_value=None), patch.object(setup, "cached_wheel", side_effect=artifact), patch.object(setup, "run_repair_stage", side_effect=stage):
                self.assertEqual(deeplink.install(args), 0)
            self.assertEqual(downloads, [])
            self.assertEqual(calls[0][:2], [sys.executable, str(ROOT / "engine/engine_install.py")])
            self.assertEqual(calls[0][calls[0].index("--cache") + 1], str(cache))
            self.assertEqual(calls[1][1], "verify")
            self.assertEqual(Path(calls[2][1]).name, "runtime_storage.py")
            self.assertFalse(any("reference" in str(x) or "evaluate" in str(x) for x in calls))

    def test_explicit_environment_with_matching_versions_but_changed_vendor_files_is_rejected(self):
        from engine import environment_setup as setup
        lock = json.loads((ROOT / "engine/r8-lock.json").read_bytes())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            envroot = Path(folder) / "venv"
            envroot.mkdir()
            (envroot / "pyvenv.cfg").touch()
            args = argparse.Namespace(out=str(Path(folder)/"install"), cache=str(Path(folder)/"cache"),
                python=None, wheelhouse=None, cann_home=None, use_environment=str(envroot/"bin/python"))
            observed = dict(packages=lock["required_python_packages"])
            with patch.object(setup, "cann_environment", return_value=(dict(os.environ), {})), \
                 patch.object(setup, "probe", return_value=observed), \
                 patch.object(setup, "compatible_python", return_value=True), \
                 patch.object(setup, "vendor_integrity", return_value={"verified": False}), \
                 patch.object(os, "execve") as execute:
                with self.assertRaisesRegex(ValueError, "overwritten Ascend Triton"):
                    deeplink.install(args)
            execute.assert_not_called()

    def test_install_stops_on_failure(self):
        from engine import environment_setup as setup
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            args = argparse.Namespace(out=str(Path(folder)/"install"), cache=folder, python=None, wheelhouse=None, cann_home=None, use_environment=None)
            with patch.object(setup,"ensure_environment",return_value=None), patch.object(setup,"cached_wheel",return_value=Path(folder)/"pkg"), patch.object(setup,"run_repair_stage",return_value=types.SimpleNamespace(returncode=9)) as run:
                self.assertEqual(deeplink.install(args),9)
                self.assertEqual(run.call_count,1)

    def test_launch_rewrites_only_service_address(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"PYTHONPATH":"original"}):
            root=Path(folder)/"sdk"; root.mkdir()
            ascend=root/'runtime/ascend/vllm_ascend';ascend.mkdir(parents=True)
            (ascend/'__init__.py').touch()
            child=root/'runtime/entrypoints/serve_b1_child.py';child.parent.mkdir(parents=True);child.touch()
            os.environ['PYTHONPATH']=str(ascend.parent)
            out=Path(folder)/"out";out.mkdir()
            profile={"python":sys.executable,"host_contract_qualification":{"installed_metadata":{"vllm-ascend":"pinned"}}}
            command=[sys.executable,"-m","vllm.entrypoints.cli.main","serve","/model","--host","127.0.0.1","--port","18390","--served-model-name","old","--worker-extension-cls","synchronized_boundary.WorkerBoundary"]
            launcher=types.SimpleNamespace(prepare_command=lambda *a:(command,{"VLLM_PLUGINS":"hust_joint_packed,hust_joint_packed"},None))
            budget=types.SimpleNamespace(configure_host_budget=lambda command,env,size:(command,env))
            geometry=types.SimpleNamespace(apply=lambda *a:{"mtp":2})
            with patch.dict(sys.modules,launch_boundary_b1=launcher,serve_b1_budget=budget,runtime_geometry=geometry):
                built,env=service.build(root,profile,Path('/model'),out,out/'cache','0,1','127.0.0.1',8000,'deeplink')
            self.assertEqual(built[built.index('--port')+1],'8000')
            self.assertEqual(built[built.index('--served-model-name')+1],'deeplink')
            self.assertEqual(env['VLLM_PLUGINS'],'hust_joint_packed')
            plan=json.loads((out/'plan.json').read_text())
            self.assertEqual(plan['purpose'],'inference-service')
            self.assertFalse({'datasets','reference','baseline','workloads'} & plan.keys())
            self.assertIn('logging_entry.py',built[1])
            self.assertEqual(built[3],'--')
            self.assertEqual(Path(built[4]),child)
            self.assertEqual(env['XDG_CONFIG_HOME'],str(out.resolve()/'runtime-config'))

    def test_ambiguous_runtime_flag_rejected(self):
        for command in [[],['--port','1','--port','2']]:
            with self.assertRaises(ValueError):
                service.set_option(command,'--port',8000)

    def test_changed_runtime_stops_before_device_or_model_use(self):
        import one_click
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);model=out/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe_text','num_hidden_layers':40,'hidden_size':2048}))
            args=['service','--installation',str(out),'--model',str(model),'--physical-devices','0,7','--cache',str(out/'cache')]
            guard=types.SimpleNamespace(acquire=unittest.mock.Mock())
            verifier=types.SimpleNamespace(verify_installed_sources=unittest.mock.Mock(side_effect=ValueError('source changed')))
            with patch.object(deeplink,'installation',return_value=(out,{'release':'full-answer-b1-20261117-r8'})), patch.object(one_click,'verify_kit'), patch.dict(os.environ), patch.dict(sys.modules,engine_entry=types.SimpleNamespace(),resource_guard=guard,restore_b1_r4=verifier,**{"engine.sdk_dependency_probe": types.SimpleNamespace(admit_dependencies=lambda root,**kwargs:None,receipt_path=lambda cache:Path(cache)/"cpu-metadata.json")}), patch.object(sys,'argv',args):
                os.environ.pop('DEEPLINK_SERVICE_BOOTSTRAPPED',None)
                with self.assertRaisesRegex(ValueError,'source changed'):
                    service.main()
            guard.acquire.assert_not_called()

    def test_bootstrapped_service_still_requires_device_lease(self):
        import one_click
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);model=out/'model';model.mkdir();sdk=out/'sdk';sdk.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe_text','num_hidden_layers':40,'hidden_size':2048}))
            (sdk/'deployment.json').write_text('{}')
            args=['service','--installation',str(out),'--model',str(model),'--physical-devices','0,7','--devices','0,1','--cache',str(out/'cache'),'--work-dir',str(out)]
            guard=types.SimpleNamespace(inherited=lambda:None)
            with patch.object(deeplink,'installation',return_value=(out,{'release':'full-answer-b1-20261117-r8'})), patch.object(one_click,'verify_kit'), patch.dict(os.environ,DEEPLINK_SERVICE_BOOTSTRAPPED=str(out)), patch.dict(sys.modules,engine_entry=types.SimpleNamespace(),resource_guard=guard), patch.object(sys,'argv',args), patch.object(service,'build') as build:
                with self.assertRaisesRegex(ValueError,'valid inherited device lease'):
                    service.main()
                build.assert_not_called()

    def check_service_environment_rejection(self, *, changed_version=False):
        import one_click
        from engine import environment_setup as setup
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);model=out/'model';model.mkdir();sdk=out/'sdk';sdk.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe_text','num_hidden_layers':40,'hidden_size':2048}))
            (sdk/'deployment.json').write_text(json.dumps({'python':str(out/'venv/bin/python')}))
            args=['service','--installation',str(out),'--model',str(model),'--physical-devices','0,7','--cache',str(out/'cache')]
            required=json.loads((ROOT/'engine/r8-lock.json').read_bytes())['required_python_packages']
            observed=dict(required)
            if changed_version:observed['triton-ascend']='3.2.2'
            guard=types.SimpleNamespace(acquire=unittest.mock.Mock())
            with patch.object(deeplink,'installation',return_value=(out,{'release':'full-answer-b1-20261117-r8'})),patch.object(one_click,'verify_kit'),patch.dict(os.environ),patch.dict(sys.modules,engine_entry=types.SimpleNamespace(),resource_guard=guard,restore_b1_r4=types.SimpleNamespace(verify_installed_sources=lambda root:None),**{"engine.sdk_dependency_probe": types.SimpleNamespace(admit_dependencies=lambda root,**kwargs:None,receipt_path=lambda cache:Path(cache)/"cpu-metadata.json")}),patch.object(setup,'probe',return_value={'packages':observed}),patch.object(setup,'vendor_integrity',return_value={'verified':False}),patch.object(sys,'argv',args):
                os.environ.pop('DEEPLINK_SERVICE_BOOTSTRAPPED',None)
                with self.assertRaisesRegex(ValueError,'versions differ' if changed_version else 'files are overwritten'):service.main()
            guard.acquire.assert_not_called()

    def test_serving_rejects_changed_vendor_files_before_any_device_lease(self):
        self.check_service_environment_rejection()

    def test_serving_rejects_old_triton_despite_other_qualified_dependencies(self):
        self.check_service_environment_rejection(changed_version=True)

    def test_default_install_cache_follows_target_storage_instead_of_cwd(self):
        from engine import environment_setup as setup
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ):
            target=Path(folder)/'large-volume/install'
            args=argparse.Namespace(out=str(target),cache=None,python=None,wheelhouse=None,cann_home=None,use_environment=None)
            with patch.object(setup,'ensure_environment',return_value=None) as environment,patch.object(setup,'run_repair_stage',return_value=types.SimpleNamespace(returncode=0)) as run:
                self.assertEqual(deeplink.install(args),0)
            expected=target.parent/'.cache/deeplinkengine'
            self.assertEqual(environment.call_args.args[0]['cache'],str(expected))
            command=run.call_args_list[0].args[0]
            self.assertEqual(command[command.index('--cache')+1],str(expected))
            self.assertEqual(os.environ['TMPDIR'],str(expected/'tmp'))

    def test_default_service_cache_follows_installation_storage(self):
        from engine import environment_setup as setup
        with tempfile.TemporaryDirectory() as folder:
            installed=Path(folder)/'large-volume/install'
            args=argparse.Namespace(installation=str(installed),model='/models/example',physical_devices='0,7',host='127.0.0.1',port=8000,served_model_name='deeplink',cache=None,lease_timeout=0,work_dir=None,devices=None,dry_run=True)
            with patch.object(deeplink,'installation',return_value=(installed,{'python':'/private/bin/python'})),patch.object(setup,'cann_environment',return_value=({},{})),patch.object(deeplink.os,'execve',side_effect=SystemExit) as execute:
                with self.assertRaises(SystemExit):deeplink.serve(args)
            command=execute.call_args.args[1]
            self.assertEqual(command[command.index('--cache')+1],str(installed.parent/'.cache/deeplinkengine/serving'))

    def test_qualified_service_reaches_device_admission(self):
        import one_click
        from engine import environment_setup as setup
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);model=out/'model';model.mkdir();sdk=out/'sdk';sdk.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe_text','num_hidden_layers':40,'hidden_size':2048}))
            (sdk/'deployment.json').write_text(json.dumps({'python':str(out/'venv/bin/python')}))
            required=json.loads((ROOT/'engine/r8-lock.json').read_bytes())['required_python_packages']
            args=['service','--installation',str(out),'--model',str(model),'--physical-devices','0,7','--devices','0,1','--cache',str(out/'cache')]
            guard=types.SimpleNamespace(verify_mapping=lambda logical:None,acquire=unittest.mock.Mock(side_effect=RuntimeError('device-admission-boundary')))
            with patch.object(deeplink,'installation',return_value=(out,{'release':'full-answer-b1-20261117-r8'})),patch.object(one_click,'verify_kit'),patch.dict(os.environ),patch.dict(sys.modules,engine_entry=types.SimpleNamespace(),resource_guard=guard,restore_b1_r4=types.SimpleNamespace(verify_installed_sources=lambda root:None),**{"engine.sdk_dependency_probe": types.SimpleNamespace(admit_dependencies=lambda root,**kwargs:None,receipt_path=lambda cache:Path(cache)/"cpu-metadata.json")}),patch.object(setup,'probe',return_value={'packages':required}),patch.object(setup,'vendor_integrity',return_value={'verified':True}),patch.object(sys,'argv',args):
                os.environ.pop('DEEPLINK_SERVICE_BOOTSTRAPPED',None)
                with self.assertRaisesRegex(RuntimeError,'device-admission-boundary'):service.main()
            guard.acquire.assert_called_once()

    def test_chat_example_sends_real_http(self):
        observed=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'message':{'content':'hello'}}]}).encode())
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.handle_request);thread.start()
        try:
            run=subprocess.run([sys.executable,str(ROOT/'examples/chat.py'),'hi','--base-url',f'http://127.0.0.1:{server.server_port}','--max-tokens','16'],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(run.stdout.strip(),'hello')
            self.assertEqual(observed[0][0],'/v1/chat/completions')
            self.assertEqual(observed[0][1]['max_tokens'],16)
            self.assertFalse(observed[0][1]['chat_template_kwargs']['enable_thinking'])
        finally:
            server.server_close();thread.join(2)


if __name__ == '__main__':
    unittest.main()
