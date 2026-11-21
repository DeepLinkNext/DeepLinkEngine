"""Real HTTP and public entry checks; no model framework, installer or device."""
import contextlib,hashlib,http.server,io,json,re,socket,sys,tempfile,threading,time,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import model_restore as restore
import model_receiver as receiver

class HTTP(http.server.ThreadingHTTPServer):
    daemon_threads=True
    def handle_error(self,*args):pass

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        server=self.server;server.calls.append(self.headers.get('Range'));index=len(server.calls)
        match=re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers.get('Range',''));start,end=map(int,match.groups())
        if server.mode=='blocked' and index==1:server.entered.set();server.release.wait(5)
        mode=server.mode;body=server.raw[start:end+1]
        self.send_response(200 if mode=='ignore' else 206)
        self.send_header('Content-Range',f'bytes {start}-{end}/{len(server.raw)+(1 if mode=="wrong-total" else 0)}')
        self.send_header('Content-Length',str(len(body)));self.end_headers()
        if mode=='stall' or (mode=='cut-then-stall' and index>1):time.sleep(.15);return
        if mode=='wrong-body':body=b'x'*len(body)
        if (mode in ('cut-once','cut-then-stall')) and index==1:
            body=body[:len(body)//2];self.wfile.write(body);self.wfile.flush();self.close_connection=True;return
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

class Receiver(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.output=self.root/'model';self.raw=bytes(range(251))*4178;self.name='model-00014.safetensors';self.pin=hashlib.sha256(self.raw).hexdigest()
        self.lock=self.root/'manifest.json';self.lock.write_text(json.dumps(dict(schema='deeplinkengine-model-download/v1',repository='Qwen/model',files={self.name:dict(bytes=len(self.raw),sha256=self.pin,revision='a'*40)})))
        self.server=HTTP(('127.0.0.1',0),Handler);self.server.raw=self.raw;self.server.calls=[];self.server.mode='normal';self.server.entered=threading.Event();self.server.release=threading.Event();self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.addCleanup(self.close)
        self.base='http://127.0.0.1:'+str(self.server.server_port)
        self.chunks=patch.object(receiver,'RANGE_BYTES',128*1024);self.chunks.start();self.addCleanup(self.chunks.stop)
    def close(self):self.server.release.set();self.server.shutdown();self.server.server_close();self.thread.join(2)
    def run_restore(self):return restore.restore(self.lock,self.output,transport_base_url=self.base)
    def receipt(self):return json.loads((self.output/'model-restoration.json').read_bytes())
    def states(self):return list(self.root.glob('.model-model-downloads/*/*/state.json'))
    def test_real_ranges_keep_origin_and_transport_separate(self):
        result=self.run_restore();self.assertEqual(result['status'],'completed');self.assertEqual((self.output/self.name).read_bytes(),self.raw)
        row=result['files'][self.name];self.assertTrue(row['url'].startswith('https://modelscope.cn/'));self.assertEqual(row['transport_url'],self.base+'/'+self.name)
        self.assertEqual(result['manifest_sha256'],hashlib.sha256(self.lock.read_bytes()).hexdigest());self.assertFalse(result['model_service_started'])
        self.assertGreater(len(self.server.calls),1)
    def test_disconnect_retains_its_exact_prefix_then_resumes(self):
        self.server.mode='cut-once';self.run_restore();self.assertEqual(self.server.calls[1].split('-')[0],'bytes=65536')
        state=json.loads(self.states()[0].read_bytes());first=state['segments'][0];self.assertTrue(first['interrupted']);self.assertEqual((self.states()[0].parent/first['file']).read_bytes(),self.raw[:65536])
    def test_ignored_range_and_wrong_total_never_admit_bytes(self):
        for mode in ('ignore','wrong-total'):
            with self.subTest(mode=mode):
                self.output=self.root/mode;self.server.mode=mode
                with self.assertRaisesRegex(OSError,'Content-Range'):self.run_restore()
                self.assertEqual(self.receipt()['status'],'failed');self.assertFalse((self.output/self.name).exists())
    def test_timeout_and_cross_call_resume_preserve_first_prefix_and_failed_receipt(self):
        self.server.mode='cut-then-stall'
        with patch.object(receiver,'NETWORK_TIMEOUT',.025),self.assertRaises(OSError):self.run_restore()
        failed=(self.output/'model-restoration.json').read_bytes();self.assertEqual(json.loads(failed)['status'],'failed')
        self.assertEqual(Path(json.loads(failed)['files'][self.name]['state_path']).resolve(),self.states()[0].resolve())
        state=json.loads(self.states()[0].read_bytes());self.assertEqual(state['segments'][0]['bytes'],65536)
        before=len(self.server.calls);self.server.mode='normal';self.run_restore();self.assertTrue(self.server.calls[before].startswith('bytes=65536-'))
        self.assertIn(failed,[p.read_bytes() for p in (self.output/'model-restoration-attempts').glob('previous-*.json')])
        self.assertEqual(json.loads(self.states()[0].read_bytes())['lifecycle_status'],'completed')
    def test_full_size_legacy_partial_is_whole_verified_without_network(self):
        self.output.mkdir();p=self.output/(self.name+'.download-part');p.write_bytes(self.raw)
        with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('No request needed')):result=self.run_restore()
        self.assertEqual(result['files'][self.name]['admission'],'whole-SHA-verified-full-size-partial');self.assertEqual(p.read_bytes(),self.raw)
    def test_legacy_prefix_is_captured_automatically_without_mutating_original(self):
        self.output.mkdir();p=self.output/(self.name+'.download-part');p.write_bytes(self.raw[:777]);self.run_restore()
        self.assertEqual(p.read_bytes(),self.raw[:777]);self.assertTrue(self.server.calls[0].startswith('bytes=777-'))
    def test_wrong_whole_sha_is_preserved_and_never_becomes_model(self):
        self.server.mode='wrong-body'
        with self.assertRaisesRegex(ValueError,'whole SHA-256'):self.run_restore()
        self.assertFalse((self.output/self.name).exists());self.assertEqual(self.receipt()['status'],'failed')
        state=json.loads(self.states()[0].read_bytes());self.assertTrue((self.states()[0].parent/state['assembly']).is_file())
    def test_wrong_full_size_partial_is_rejected_and_retained(self):
        self.output.mkdir();p=self.output/(self.name+'.download-part');p.write_bytes(b'x'*len(self.raw))
        with self.assertRaisesRegex(ValueError,'wrong whole SHA'):self.run_restore()
        self.assertEqual(p.read_bytes(),b'x'*len(self.raw));self.assertFalse(self.server.calls)
    def test_running_receipt_precedes_bytes_and_output_mutex_prevents_two_writers(self):
        self.server.mode='blocked'
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.run_restore);self.assertTrue(self.server.entered.wait(3));initial=self.receipt();self.assertEqual(initial['status'],'running')
            second=pool.submit(self.run_restore);time.sleep(.03);self.assertEqual(self.receipt()['attempt'],initial['attempt']);self.assertFalse(second.done())
            self.server.release.set();self.assertEqual(first.result(10)['status'],'completed');count=len(self.server.calls);self.assertEqual(second.result(10)['status'],'completed');self.assertEqual(len(self.server.calls),count)
    def public(self):
        import run_all
        config=self.root/'evaluation.json';config.write_text(json.dumps(dict(model=str(self.output),cache=str(self.root/'cache'),out=str(self.root/'official-out'),versions=['0.23.0'],physical_devices=[0,1],model_manifest=str(self.lock),model_transport_base_url=self.base+'/')))
        args=['run_all.py','--official-baselines','--config',str(config),'--prepare-environment']
        return run_all,args
    def test_public_entry_passes_normalized_mirror_and_completes_receiver_before_sdk(self):
        run_all,args=self.public();real=restore.restore
        class BoundaryStop(Exception):pass
        def receive(lock,output,**kwargs):
            self.assertEqual(kwargs['transport_base_url'],self.base);result=real(lock,output,**kwargs);self.assertEqual(result['status'],'completed');raise BoundaryStop()
        with patch.object(sys,'argv',args),patch('official_baselines.platform.system',return_value='Linux'),patch('official_baselines.platform.machine',return_value='aarch64'),patch.object(restore,'restore',side_effect=receive),patch('official_image.prepare',side_effect=AssertionError('No SDK in receiver fixture')),self.assertRaises(BoundaryStop):run_all.main()
        self.assertEqual((self.output/self.name).read_bytes(),self.raw);self.assertFalse((self.root/'official-out').exists())
    def test_public_entry_rejects_ignored_range_before_any_sdk_or_model(self):
        run_all,args=self.public();self.server.mode='ignore';error=io.StringIO()
        with patch.object(sys,'argv',args),patch('official_baselines.platform.system',return_value='Linux'),patch('official_baselines.platform.machine',return_value='aarch64'),patch('official_image.prepare',side_effect=AssertionError('No SDK')),contextlib.redirect_stderr(error),self.assertRaises(SystemExit) as stopped:run_all.main()
        self.assertEqual(stopped.exception.code,2);self.assertIn('Content-Range',error.getvalue());self.assertEqual(self.receipt()['status'],'failed');self.assertFalse((self.output/self.name).exists())

if __name__=='__main__':unittest.main()
