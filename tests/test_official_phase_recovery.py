"""CPU evidence-tampering and real Linux socket lifecycle checks."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_endpoint import available
from official_reuse import admit, sha


def seal(root):
    files = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*')
             if p.is_file() and p != root / 'checksums.json'}
    (root / 'checksums.json').write_text(json.dumps(files))


class PhaseRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.current = self.root / 'old-source', self.root / 'current-source'
        for p in (self.old, self.current):
            p.mkdir()
            (p / 'client.py').write_text('unchanged client')
            (p / 'official_suite.py').write_text('previous controller' if p == self.old else 'reviewed controller')
        self.previous = self.root / 'previous'
        self.source = self.previous / 'continuous'
        (self.source / 'service').mkdir(parents=True)
        self.destination = self.root / 'retained'
        self.inputs = self.root / 'inputs.json'
        self.inputs.write_text('unchanged inputs')
        self.config = dict(model=str(self.root / 'model'), protocol_deployment=str(self.root / 'sdk'))
        self.lock = dict(id='official-vllm-0.23.0', manifest_digest='reviewed-image')
        self.mapping = dict(physical_cards=[0, 7], logical_cards=[0, 1], npu_smi_mapping='fixed hardware mapping')
        self.weights = {'shard': dict(bytes=3, sha256='checkpoint', inode=42, mtime_ns=123)}
        self.arguments = ['--port', '18390']
        self.old_sources = {str(p.resolve()): sha(p) for p in self.old.iterdir()}
        self.sources = {str(p.resolve()): sha(p) for p in self.current.iterdir()}
        self.dump(self.source / 'receipt.json', dict(completed=True, quiescent=True, model_ready=True,
                  client_started=True, baseline_id=self.lock['id'], phase='continuous', cleanup_actions=[]))
        self.dump(self.source / 'service/plan.json', dict(model=self.config['model'],
                  protocol_deployment=self.config['protocol_deployment'],
                  client_tools=str(self.old / 'benchmark/one-click-kit'),
                  serve_arguments=self.arguments))
        phases = []
        for i, name in enumerate(('4096-warmup', '4096-formal', '16384-warmup', '16384-formal')):
            phases.append(dict(name=name, seconds=60 if i % 2 == 0 else 300, requests=1, failed=0))
            self.dump(self.source / f'service/service-{i}-lane0.json', dict(phase=name, success=True))
        self.dump(self.source / 'service/continuous.json', dict(status='completed', phases=phases))
        seal(self.source)
        self.dump(self.previous / 'model-weights.json', self.weights)
        self.dump(self.previous / 'mmlu/receipt.json', dict(model_service_started=False, client_started=False, quiescent=True))
        self.campaign = dict(schema='deeplinkengine-official-full-suite/v1', status='failed',
                             baseline_image=self.lock, physical_mapping=self.mapping,
                             continuous_input_sha256=sha(self.inputs), sources=self.old_sources,
                             stages=[dict(name='continuous', status='completed'), dict(name='mmlu', status='failed')]
                             + [dict(name=f'phase-{i}', status='not_attempted') for i in range(8)])
        self.dump(self.previous / 'campaign.json', self.campaign)
        seal(self.previous)
        self.review = patch('official_reuse.REVIEWED_CONTROLLERS', {'official_suite.py': {
            sha(self.old / 'official_suite.py'): sha(self.current / 'official_suite.py')}})
        self.review.start()
        self.addCleanup(self.review.stop)

    def dump(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_admission(self, **changes):
        values = dict(config=self.config, image_lock=self.lock, sources=self.sources, weights=self.weights,
                      mapping=self.mapping, inputs=self.inputs, current_root=self.current, arguments=self.arguments)
        values.update(changes)
        return admit(self.source, self.destination, **values)

    def test_copy_preserves_every_original_byte_and_dates(self):
        result = self.run_admission()
        self.assertFalse(result['inference_repeated'])
        self.assertTrue(result['original_dates_and_outputs_preserved'])
        self.assertEqual((self.destination / 'checksums.json').read_bytes(), (self.source / 'checksums.json').read_bytes())
        for name, pin in json.loads((self.source / 'checksums.json').read_bytes()).items():
            self.assertEqual(sha(self.destination / name), pin)

    def test_tampered_raw_response_rejected(self):
        (self.source / 'service/service-0-lane0.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.run_admission()

    def test_resealed_missing_response_still_rejected(self):
        (self.source / 'service/service-3-lane0.json').unlink()
        seal(self.source)
        seal(self.previous)
        with self.assertRaises(ValueError):
            self.run_admission()

    def test_other_image_or_checkpoint_rejected(self):
        for changes in (dict(image_lock=dict(self.lock, id='official-vllm-0.30.0')),
                        dict(weights={'shard': dict(bytes=3, sha256='different')})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_admission(**changes)

    def test_actual_later_model_start_rejected_even_when_resealed(self):
        self.dump(self.previous / 'mmlu/receipt.json', dict(model_service_started=True, client_started=False, quiescent=True))
        seal(self.previous)
        with self.assertRaises(ValueError):
            self.run_admission()

    def test_unreviewed_client_or_mapping_change_rejected(self):
        with self.assertRaises(ValueError):
            self.run_admission(mapping=dict(self.mapping, physical_cards=[2, 3]))
        target = self.current / 'client.py'
        target.write_text('different client')
        self.sources[str(target.resolve())] = sha(target)
        with self.assertRaises(ValueError):
            self.run_admission()


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux bind semantics; official runtime is Linux only')
class EndpointLifecycle(unittest.TestCase):
    def test_old_phase_time_wait_does_not_block_new_phase(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            host, port = server.getsockname()
            server.listen(1)
            client.connect((host, port))
            connection, _ = server.accept()
            connection.shutdown(socket.SHUT_WR)
            connection.close()
            self.assertEqual(client.recv(1), b'')
            client.close()
            server.close()
            with socket.socket() as old_probe, self.assertRaises(OSError):
                old_probe.bind((host, port))
            available(host, port)

    def test_existing_listener_is_still_rejected(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                available(*listener.getsockname())


if __name__ == '__main__':
    unittest.main()
