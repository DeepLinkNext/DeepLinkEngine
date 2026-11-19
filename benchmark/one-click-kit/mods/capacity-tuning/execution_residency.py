"""Explicit E/R proposals for the two actually measured C04 operating points."""

import argparse
import hashlib
import json
from pathlib import Path

from capacity_tuning import BASE, BINS, configure

PAIRS = ((24, 36), (36, 54))
WARMUP = 'observed-torch-triton-sealed-task-warmup-v5-capacity36'
CONCURRENCIES = [1, 2, 4, 8, 16, 24, 32, 36]


def graph_sizes(execution):
    if type(execution) is not int or execution not in (24, 36):
        raise ValueError('Paired capture geometry is measured only at E24 and E36')
    maximum = next(n for n in BINS if n >= execution * 3)
    return sorted(set(BASE) | {n for n in BINS if n <= maximum})


def configure_pair(baseline, execution, resident):
    if (type(execution) is not int or type(resident) is not int
            or (execution, resident) not in PAIRS):
        raise ValueError('Measured E/R pairs are24/36 and36/54; no inferred intermediate profile')
    # Reuse all existing r6 FULL/MTP2/CP1/budget/State/graph checks and its deepcopy.
    proposal = configure(baseline, 24)
    command = proposal['command']
    extra_index = command.index('--additional-config') + 1
    graph_index = command.index('--compilation-config') + 1
    state, graph = json.loads(command[extra_index]), json.loads(command[graph_index])
    state['state_resident_seats'] = resident
    graph['cudagraph_capture_sizes'] = graph_sizes(execution)
    command[command.index('--max-num-seqs') + 1] = str(execution)
    # Preserve the entire baseline command for the common all-off point.
    if (execution, resident) != PAIRS[0]:
        command[extra_index] = json.dumps(state)
        command[graph_index] = json.dumps(graph)
    config = dict(schema='C04-full-answer-capacity/v1', id=f'E{execution}-R{resident}',
        execution_seats=execution, resident_seats=resident, diagnostic_only=True)
    proposal['capacity_tuning'] = dict(schema='capacity-execution-residency/v1',
        execution_seats=execution, resident_seats=resident,
        native_controller_config=config,
        common_warmup=dict(policy=WARMUP, dense_decode_concurrencies=CONCURRENCIES,
            formal_clock_excludes_warmup=True),
        actual_admission_required=['Scheduler and StateRoot execution/resident seats on both ranks.',
            'Native GDN decode closure, graph capture keys, source hashes and fixed4GiB/rank State budget.',
            'Both-rank post-ready compile seal and idle-boundary pre/post StateRoot identity.',
            'Same original200,8192/EOS budget and scorer; retain correctness, truncation and all pairings.'],
        native_executor='experiments/optimization-20261115/C04-full-answer-capacity36-20261116-r1/runner/scripts/optimization_config.py',
        diagnostic_only=True, proposal_only=True, inference_started=False,
        default_changed=False, quality_qualified=False,
        current_r7_combination_measured=False,
        coupling='E and R change together; C04 cannot attribute either percentage to E or R alone.')
    return proposal


def evidence():
    here = Path(__file__).resolve().parent
    pins = json.loads((here / 'mod.json').read_text())['paired_capacity_evidence']
    path = here / pins['path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins['sha256']:
        raise ValueError('Frozen paired capacity analysis differs')
    result = json.loads(raw)
    if not result['complete'] or len(result['rows']) != 4 or len(result['pairs']) != 2:
        raise ValueError('Both complete repeated pairs must remain visible')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--execution-seats', type=int, required=True)
    parser.add_argument('--resident-seats', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    proposal = configure_pair(json.loads(args.baseline.read_text()), args.execution_seats, args.resident_seats)
    observed = evidence()
    proposal['paired_evidence'] = dict(pairs=observed['pairs'], all_four_trials=observed['rows'],
        note='Repeated full-model history on r6, not a new deployment or an r7 gain.')
    with args.out.open('x') as stream:
        json.dump(proposal, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
