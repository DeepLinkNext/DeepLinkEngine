"""Idle cache recovery preserves the original guard and releases every optional page."""
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import checkpoint_cache as cache
import warmup_memory_recovery as recovery
G=2**30


class Memory:
    def __init__(self,fail=None):self.held=0;self.closed=0;self.checks=0;self.fail=fail
    def sample(self):
        return dict(current=48*G+self.held,limit=64*G,events=dict(oom=0,oom_kill=0),stat={})
    def allocate(self,n):
        owner=self
        class Buffer:
            def close(self):owner.held-=n;owner.closed+=n
        self.held+=n;return Buffer()
    def check(self):
        self.checks+=1
        if self.held and self.fail:raise RuntimeError(self.fail)
        return self.sample()


def plan(*a):return dict(eligible=True,temporary_bytes=3*64*2**20)


def test_every_private_page_is_released_before_return(tmp_path):
    m=Memory()
    with patch.object(cache,'reclaim_plan',side_effect=plan),patch.object(cache,'allocate_private_chunk',side_effect=m.allocate):
        result=recovery.reclaim_idle_headroom(m.sample,m.check,24*G,tmp_path/'proof.json')
    assert result['status']=='completed' and result['temporary_memory_released'] is True
    assert m.held==0 and m.closed==3*64*2**20 and m.checks==5
    assert result['formal_client_started'] is False and result['global_drop_caches'] is False
    assert result['memory_limit_changed'] is False


@pytest.mark.parametrize('message',['4GiB conservative working-set memory reserve reached','New cgroup oom event','unrelated model failure'])
def test_original_working_gate_releases_only_own_pages_and_other_errors_stay_fatal(tmp_path,message):
    m=Memory(message)
    with patch.object(cache,'reclaim_plan',side_effect=plan),patch.object(cache,'allocate_private_chunk',side_effect=m.allocate):
        if message.startswith('4GiB'):
            result=recovery.reclaim_idle_headroom(m.sample,m.check,24*G,tmp_path/'proof.json')
            assert result['status']=='stopped-at-original-working-set-reserve'
        else:
            with pytest.raises(RuntimeError,match=message):recovery.reclaim_idle_headroom(m.sample,m.check,24*G,tmp_path/'proof.json')
    assert m.held==0 and m.checks==3


def test_invalid_cache_budget_cannot_allocate_pages(tmp_path):
    m=Memory()
    with patch.object(cache,'reclaim_plan',return_value=dict(eligible=False)),patch.object(cache,'allocate_private_chunk',side_effect=AssertionError('No valid cache credit')):
        result=recovery.reclaim_idle_headroom(m.sample,m.check,24*G,tmp_path/'proof.json')
    assert result['status']=='declined' and result['allocated_bytes']==0


def test_generated_warmup_admission_requires_observed_raw_space_after_all_cleanup(tmp_path):
    from test_context_baseline_reuse import combined
    files=combined();namespace={};exec(files['scripts/checkpoint_cache.py'],namespace)
    cls=namespace['StartupMemoryReserve'];value=cls.__new__(cls)
    value.path=tmp_path/'reserve.json';value.record={'released':True};value.buffers=[]
    value.sample=lambda:dict(current=48*G,limit=64*G)
    value.check=lambda:None
    with patch.object(recovery,'reclaim_idle_headroom',return_value=dict(temporary_memory_released=True)),patch.dict(sys.modules,warmup_memory_recovery=recovery):
        with pytest.raises(RuntimeError,match='Less than24GiB'):value.before_warmup()
    assert value.record['required_warmup_raw_bytes']==24*G
    assert value.buffers==[]


def test_new_oom_cannot_be_mistaken_for_an_optional_working_set_release(tmp_path):
    m=Memory('4GiB conservative working-set memory reserve reached')
    original=m.sample;seen=[False]
    def sample():
        if m.held:seen[0]=True
        d=original();d['events']['oom']=int(seen[0]);return d
    with patch.object(cache,'reclaim_plan',side_effect=plan),patch.object(cache,'allocate_private_chunk',side_effect=m.allocate):
        with pytest.raises(RuntimeError,match='New cgroup OOM'):
            recovery.reclaim_idle_headroom(sample,m.check,24*G,tmp_path/'proof.json')
    assert m.held==0


def test_already_sufficient_observed_headroom_never_runs_pressure_allocation(tmp_path):
    from test_context_baseline_reuse import combined
    files=combined();namespace={};exec(files['scripts/checkpoint_cache.py'],namespace)
    cls=namespace['StartupMemoryReserve'];value=cls.__new__(cls)
    value.path=tmp_path/'reserve.json';value.record={'released':True};value.buffers=[]
    value.sample=lambda:dict(current=39*G,limit=64*G);value.check=lambda:None
    with patch.object(recovery,'reclaim_idle_headroom',side_effect=AssertionError('Already enough raw space')),patch.dict(sys.modules,warmup_memory_recovery=recovery):value.before_warmup()
    assert value.record['warmup_admission']['current']==39*G
