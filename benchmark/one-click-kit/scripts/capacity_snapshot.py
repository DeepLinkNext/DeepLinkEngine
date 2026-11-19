"""Two idle-boundary native geometry observations; no per-step hooks or arithmetic changes."""
import hashlib
import json
from pathlib import Path
import urllib.request
from runtime_geometry import BINS, captures, RELEASE


def snapshot(worker):
    from runtime_identity import identity
    from betterscale.models.qwen35 import count_policy,capacity,state_backend,cache_config,prefill_round_robin
    from betterscale.patches.qwen_gdn import graphs
    cfg,runner=worker.vllm_config,worker.model_runner
    root=runner._live_state_root
    cap=root.capacity
    closure=dict(zip(graphs.capacity.__code__.co_freevars,
        [cell.cell_contents for cell in graphs.capacity.__closure__ or ()]))
    if 'decode_capacities' not in closure:
        raise ValueError('Native GDN capacity closure missing')
    extra=cache_config.resolve(cfg.additional_config)
    mode=cfg.compilation_config.cudagraph_mode
    return dict(rank=int(worker.rank),runtime_identity=identity(worker),
        capacity=dict(execution_seats=cfg.scheduler_config.max_num_seqs,
        resident_seats=cap.resident_seats,state_root_execution_seats=cap.execution_seats,
        state_root_identity=id(root),speculative_tokens=cap.speculative_tokens,
        prefill_tokens=cap.prefill_tokens,scheduler_prefill_tokens=cfg.scheduler_config.max_num_batched_tokens,
        graph_mode=getattr(mode,'name',str(mode)),graph_captures=sorted(cfg.compilation_config.cudagraph_capture_sizes),
        configured_host_bytes=extra['state_cache_host_bytes'],
        state_flags={name:extra[name] for name in ('using_live_runtime','state_cache_incremental',
            'state_cache_streaming','state_cache_policy','state_cache_partial_reclaim')},
        native_gdn_decode_closure=list(closure['decode_capacities']),
        spec=list(count_policy.spec_capacities(cfg.scheduler_config.max_num_seqs)),
        loaded_sources={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (count_policy,capacity,state_backend,cache_config,prefill_round_robin,graphs)},observed_at_idle_boundary_only=True))


def admit(rows,profile,bindings):
    if len(rows)!=2 or {r['rank'] for r in rows}!={0,1}:
        raise ValueError('Both unique native ranks required')
    root=Path(profile['root']).resolve()
    execution,resident=(36,54) if profile['release'] == RELEASE else (24,36)
    for row in rows:
        c=row['capacity']
        if (c['execution_seats'],c['state_root_execution_seats'],c['resident_seats'],c['prefill_tokens'],
                c['scheduler_prefill_tokens'],c['speculative_tokens'])!=(execution,execution,resident,4096,4096,2):
            raise ValueError('Actual scheduler/State capacity drift')
        expected_flags={'using_live_runtime','state_cache_incremental','state_cache_streaming','state_cache_policy','state_cache_partial_reclaim'}
        spec=[n for n in BINS if n<execution*3]+[next(n for n in BINS if n>=execution*3)]
        if (set(c['state_flags'])!=expected_flags or c['spec']!=spec or c['configured_host_bytes']!=4*2**30 or c['graph_mode']!='FULL'
                or c['graph_captures']!=captures(execution) or not all(v is True for v in c['state_flags'].values())
                or c['native_gdn_decode_closure']!=c['spec'] or not c['observed_at_idle_boundary_only']
                or row['runtime_identity']['worker']!='corrected_serving_runtime.CorrectedWorker'
                or row['runtime_identity']['extra_host_observer']):
            raise ValueError('Actual native policy drift')
        if len(c['loaded_sources'])!=6:
            raise ValueError('Missing native source observations')
        for name,pin in c['loaded_sources'].items():
            relative=Path(name).resolve().relative_to(root).as_posix()
            if pin!=bindings[relative]:
                raise ValueError('Actual loaded source drift: '+relative)
    return True


def capture(out,suffix):
    out=Path(out)
    plan=json.loads((out/'plan.json').read_bytes())
    deployment=Path(plan['deployment'])
    profile=json.loads((deployment/'deployment.json').read_bytes())
    if profile['release'] != RELEASE or 'runtime_geometry' not in plan:
        return None
    bindings=json.loads((deployment/'provenance/installed-files.json').read_bytes())
    request=urllib.request.Request('http://127.0.0.1:18390/collective_rpc',
        json.dumps(dict(method='hust_capacity_snapshot',timeout=120)).encode(),{'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=150) as response:
        rows=json.load(response)['results']
    admit(rows,profile,bindings)
    if suffix=='after-measurement':
        before=json.loads((out/'capacity-before-warmup.json').read_bytes())
        if {r['rank']:r['capacity'] for r in before}!={r['rank']:r['capacity'] for r in rows}:
            raise ValueError('StateRoot/source/geometry changed across measurement')
    with (out/('capacity-'+suffix+'.json')).open('x') as stream:
        json.dump(rows,stream,indent=2)
        stream.write('\n')


def audit(root):
    root=Path(root)
    plan=json.loads((root/'service/plan.json').read_bytes())
    if 'runtime_geometry' not in plan:
        return None
    profile=json.loads((root/'deployment.json').read_bytes())
    bindings=json.loads((root/'installed-files.json').read_bytes())
    before=json.loads((root/'service/capacity-before-warmup.json').read_bytes())
    after=json.loads((root/'service/capacity-after-measurement.json').read_bytes())
    admit(before,profile,bindings)
    admit(after,profile,bindings)
    if {r['rank']:r['capacity'] for r in before}!={r['rank']:r['capacity'] for r in after}:
        raise ValueError('Sealed StateRoot/source/geometry drift')
    return dict(both_ranks_verified=True,unchanged_through_measurement=True)
