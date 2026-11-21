import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official_range_download import fetch


class Ranges(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.payload=b'abcdefghijkl'
        self.pin='sha256:'+hashlib.sha256(self.payload).hexdigest()
        self.calls=[];self.bad=set();self.wrong_total=False;self.status='206'

    def curl(self,args,**kwargs):
        a,b=map(int,args[args.index('--range')+1].split('-'));self.calls.append((a,b))
        if a in self.bad:raise OSError('fixture interrupted transfer')
        self.assertEqual(args[args.index('--max-filesize')+1],str(b-a+1))
        Path(args[args.index('--output')+1]).write_bytes(self.payload[a:b+1])
        Path(args[args.index('--dump-header')+1]).write_text('HTTP/1.1 '+self.status+'\r\nContent-Range: bytes '+str(a)+'-'+str(b)+'/'+str(len(self.payload)+int(self.wrong_total))+'\r\n\r\n')
        return types.SimpleNamespace(returncode=0,stdout=self.status,stderr='')

    def run_fetch(self,workers=2,pin=None):
        with patch('official_range_download.subprocess.run',side_effect=self.curl):
            return fetch(self.root,'https://quay.io/v2/ascend/vllm-ascend/','blobs',pin or self.pin,len(self.payload),workers=workers,chunk_size=4)

    def test_complete_digest_and_cache_reuse(self):
        p=self.run_fetch();self.assertEqual(p.read_bytes(),self.payload)
        count=len(self.calls);self.assertEqual(self.run_fetch(),p);self.assertEqual(len(self.calls),count)

    def test_interruption_resumes_only_missing_ranges(self):
        self.bad={8}
        with self.assertRaises(OSError):self.run_fetch(workers=1)
        self.bad.clear();self.calls=[]
        self.assertEqual(self.run_fetch(workers=1).read_bytes(),self.payload)
        self.assertEqual(self.calls,[(8,11)])

    def test_committed_corruption_is_refused_before_network(self):
        self.bad={8}
        with self.assertRaises(OSError):self.run_fetch(workers=1)
        p=self.root/(self.pin.replace(':','-')+'.range.partial')
        with p.open('r+b') as f:f.write(b'X')
        self.bad.clear();self.calls=[]
        with self.assertRaisesRegex(ValueError,'Committed range bytes changed'):self.run_fetch()
        self.assertFalse(self.calls)

    def test_status_and_total_size_must_match_exactly(self):
        self.status='200'
        with self.assertRaises(OSError):self.run_fetch(workers=1)
        self.status='206';self.wrong_total=True
        with self.assertRaises(OSError):self.run_fetch(workers=1)
        self.assertFalse((self.root/self.pin.replace(':','-')).exists())

    def test_whole_sha_is_still_required_after_valid_ranges(self):
        with self.assertRaisesRegex(ValueError,'SHA256 differs'):
            self.run_fetch(pin='sha256:'+'a'*64)
        self.assertFalse((self.root/('sha256-'+'a'*64)).exists())

    def test_owner_state_can_precede_partial_allocation(self):
        target=self.root/self.pin.replace(':','-')
        state=dict(schema='deeplinkengine-official-range-state/v1',digest=self.pin,bytes=len(self.payload),chunk_size=4,source='https://quay.io/v2/ascend/vllm-ascend/blobs/'+self.pin,chunks={})
        target.with_suffix('.range-state.json').write_text(json.dumps(state))
        self.assertEqual(self.run_fetch().read_bytes(),self.payload)

    def test_worker_limit_is_enforced(self):
        with self.assertRaises(ValueError):self.run_fetch(workers=5)

    def test_pure_download_facade_dispatches_only_when_explicitly_requested(self):
        import official_download
        payload=b'x'*(2*1024**2);blob=self.root/'fixture';blob.write_bytes(payload)
        identity='sha256:'+hashlib.sha256(payload).hexdigest()
        row=dict(registry='https://quay.io/v2/ascend/vllm-ascend/',kind='blobs',digest=identity,bytes=len(payload))
        with patch.dict('os.environ',DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS='4'), \
             patch('official_download.artifacts',return_value={identity:row}), \
             patch('official_download.shutil.which',return_value='curl'), \
             patch('official_range_download.fetch',return_value=blob) as ranged:
            self.assertEqual(official_download.run(dict(versions=['0.23.0'],cache=str(self.root/'cache'),out=str(self.root/'out')),artifact=identity,transport='curl'),0)
            self.assertEqual(ranged.call_args.kwargs['workers'],4)
        value=json.loads((self.root/'out/results.json').read_bytes())
        self.assertFalse(value['model_execution_authorized']);self.assertFalse(value['extraction_performed'])
        self.assertEqual(value['range_workers_requested'],4)

    def test_invalid_environment_workers_refused_before_output(self):
        import official_download
        with patch.dict('os.environ',DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS='5'):
            with self.assertRaises(ValueError):
                official_download.run(dict(versions=['0.23.0'],cache=str(self.root/'cache'),out=str(self.root/'out')))
        self.assertFalse((self.root/'out').exists())

    def test_full_image_preparation_fetch_uses_the_same_bounded_range_admission(self):
        import official_image
        self.payload=b'x'*(2*1024**2+7);self.pin='sha256:'+hashlib.sha256(self.payload).hexdigest()
        with patch.dict('os.environ',DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS='4'), \
             patch('official_range_download.subprocess.run',side_effect=self.curl):
            path=official_image.fetch(self.root,'https://quay.io/v2/ascend/vllm-ascend/','blobs',self.pin,len(self.payload))
        self.assertEqual(path.read_bytes(),self.payload);self.assertEqual(len(self.calls),3)

    def test_full_image_rejects_invalid_range_geometry_before_creating_sdk(self):
        import official_image
        lock=Path(__file__).resolve().parents[1]/'engine/official-baselines/0.23.0.json'
        with patch.dict('os.environ',DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS='5'):
            with self.assertRaisesRegex(ValueError,'between 1 and 4'):
                official_image.prepare(lock,self.root/'cache',self.root/'sdk')
        self.assertFalse((self.root/'sdk').exists())


if __name__=='__main__':unittest.main()
