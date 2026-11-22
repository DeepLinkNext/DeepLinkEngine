"""Real Linux endpoint lifecycle through the judge's pre-launch path; no model."""
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from official import enable_module_imports

enable_module_imports()
import official_judge as judge
from official_endpoint import available


class LaunchBoundaryReached(Exception):
    pass


@unittest.skipUnless(sys.platform.startswith('linux'),'Official runtime uses Linux socket semantics')
class JudgeEndpointLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.installation=self.root/'installation'
        self.sdk=self.installation/'sdk'
        self.sdk.mkdir(parents=True)
        (self.sdk/'deployment.json').write_text(json.dumps(dict(release='fixture-fixed-judge')))
        (self.installation/'installation.json').write_text(json.dumps(dict(deployment=str(self.sdk),release='fixture-fixed-judge')))
        self.protocol=self.root/'protocol'
        (self.protocol/'plans').mkdir(parents=True)
        (self.protocol/'plans/frontier-judge-B0.json').write_text(json.dumps(dict(responses='original-fixture-response-path')))
        self.output=self.root/'official-version'/'frontier-judge'
        self.output.parent.mkdir()
        self.config=dict(protocol_run=str(self.protocol),model=str(self.root/'model'),cache=str(self.root/'cache'),protocol_python=sys.executable,physical_devices=[0,1])
        self.checked=False
        self.launched=False

    def invoke(self, endpoint):
        # External memory/ownership interfaces are CPU fixtures. The actual
        # judge controller, socket checker, output writes and final seal run.
        guard=SimpleNamespace(admit_startup_memory=lambda *a: {},check_memory=lambda *a: None,listener_owned=lambda *a: False)
        owned=SimpleNamespace(stop=lambda: [],scan=lambda: [])
        def check():
            available(*endpoint)
            self.checked=True
        def stop_before_child(*args,**kwargs):
            self.assertTrue(self.checked,'The judge did not pass its endpoint lifecycle check')
            self.launched=True
            self.assertEqual(kwargs['pass_fds'],())
            raise LaunchBoundaryReached('CPU fixture stops immediately before any child')
        with patch.dict(sys.modules,{'run_b1_r4':guard,'owned_processes':SimpleNamespace(Owned=lambda *a: owned)}), \
             patch.object(judge,'load',return_value=object()), \
             patch.object(judge,'memory_guard',side_effect=lambda guard: guard), \
             patch.object(judge,'judge_command',side_effect=lambda command, *_: command), \
             patch('official_endpoint.available',side_effect=check), \
             patch.object(judge.subprocess,'Popen',side_effect=stop_before_child):
            return judge.run(self.config,'frontier',self.output,self.root/'tools',self.sdk,{},[],{})

    def test_closed_previous_phase_reaches_launch_boundary(self):
        with socket.socket() as server,socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0))
            endpoint=server.getsockname()
            server.listen(1)
            client.connect(endpoint)
            connection,_=server.accept()
            connection.shutdown(socket.SHUT_WR)
            connection.close()
            self.assertEqual(client.recv(1),b'')
            client.close()
            server.close()
            # Reproduce the old bare-bind rejection on this same real port.
            with socket.socket() as old,self.assertRaises(OSError):
                old.bind(endpoint)
            with self.assertRaises(LaunchBoundaryReached):
                self.invoke(endpoint)
        self.assertTrue(self.checked and self.launched)
        receipt=json.loads((self.output/'receipt.json').read_bytes())
        self.assertTrue(receipt['quiescent'])
        self.assertFalse(receipt['completed'])
        self.assertIn('LaunchBoundaryReached',receipt['error'])
        self.assertTrue((self.output/'checksums.json').is_file())

    def test_live_listener_refuses_before_child(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0))
            listener.listen(1)
            with self.assertRaises(OSError):
                self.invoke(listener.getsockname())
        self.assertFalse(self.checked or self.launched)
        receipt=json.loads((self.output/'receipt.json').read_bytes())
        self.assertTrue(receipt['quiescent'])
        self.assertFalse(receipt['completed'])


if __name__=='__main__':
    unittest.main()
