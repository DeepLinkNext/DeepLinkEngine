from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_sdk_state as state

HOST=dict(hostname='fixture-host',boot_id='11111111-1111-1111-1111-111111111111')


def owner(**changes):
    return dict(schema='deeplinkengine-sdk-preparation-owner/v1',**HOST,pid=123,start_ticks=456,**changes)


class StateTests(unittest.TestCase):
    def check(self, process, **changes):
        value=owner();value.update(changes)
        with patch.object(state,'_host_identity',return_value=HOST), patch.object(state,'_process_identity',return_value=process):
            return state.refuse_active_owner(dict(owner=value))

    def test_active_owner_refused(self):
        with self.assertRaisesRegex(ValueError,'still active'):
            self.check(dict(state='S',start_ticks=456))

    def test_dead_and_reused_pid_do_not_block_recovery(self):
        self.assertEqual(self.check(None)['reason'],'process_absent')
        self.assertEqual(self.check(dict(state='S',start_ticks=789))['reason'],'pid_reused')
        self.assertEqual(self.check(dict(state='Z',start_ticks=456))['reason'],'zombie')

    def test_legacy_missing_and_foreign_owner_are_not_silently_moved(self):
        for value in ({},None,dict(owner={})): 
            with self.assertRaises(ValueError):state.refuse_active_owner(value)
        with self.assertRaisesRegex(ValueError,'Foreign'):
            self.check(None,hostname='another-host')

    def test_previous_boot_is_distinct_from_live_pid(self):
        self.assertEqual(self.check(dict(state='S',start_ticks=456),boot_id='22222222-2222-2222-2222-222222222222')['reason'],'different_boot')

    def test_unreadable_process_identity_stays_unknown(self):
        with patch.object(state,'_host_identity',return_value=HOST),patch.object(state,'_process_identity',side_effect=PermissionError):
            with self.assertRaisesRegex(ValueError,'Cannot verify'):state.refuse_active_owner(dict(owner=owner()))

    @unittest.skipUnless(sys.platform.startswith('linux'),'SDK preparation uses Linux /proc')
    def test_real_owner_info_and_same_process_refusal(self):
        current=state.owner_info();self.assertEqual(current['pid'],os.getpid())
        self.assertGreater(current['start_ticks'],0)
        with self.assertRaisesRegex(ValueError,'still active'):state.refuse_active_owner(dict(owner=current))

    @unittest.skipUnless(sys.platform.startswith('linux'),'SDK preparation uses Linux /proc')
    def test_sdk_level_lock_excludes_independent_process(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=Path(state.__file__).resolve().parent
            code="""from pathlib import Path
import sys,time
sys.path.insert(0,sys.argv[1]);from official_sdk_state import sdk_preparation
r=Path(sys.argv[2]);name=sys.argv[3]
with sdk_preparation(r/'sdk'):
 (r/(name+'-begin')).write_text(str(time.monotonic()))
 time.sleep(.3)
 (r/(name+'-end')).write_text(str(time.monotonic()))
"""
            first=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(source),str(root),'one'])
            deadline=time.monotonic()+10
            while not (root/'one-begin').exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue((root/'one-begin').exists())
            second=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(source),str(root),'two'])
            self.assertEqual(first.wait(timeout=10),0);self.assertEqual(second.wait(timeout=10),0)
            self.assertGreaterEqual(float((root/'two-begin').read_text()),float((root/'one-end').read_text()))


if __name__=='__main__':unittest.main()
