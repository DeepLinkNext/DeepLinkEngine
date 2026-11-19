"""Keep frozen historical claims separate from a current service identity."""

import copy
import re


def current_b1_plan(original, *, release, logical_devices, physical_cards,
                    host_budget_per_rank_bytes):
    if not isinstance(original, dict):
        raise ValueError('Expected the preserved original plan object')
    if not isinstance(release, str) or not release.strip():
        raise ValueError('Bind this run to an explicit deployment release')
    if not isinstance(logical_devices, str) or not re.fullmatch(r'[0-9]+,[0-9]+', logical_devices):
        raise ValueError('Bind exactly two logical devices')
    logical = [int(item) for item in logical_devices.split(',')]
    if len(set(logical)) != 2:
        raise ValueError('Logical devices must differ')
    if (not isinstance(physical_cards, list) or len(physical_cards) != 2 or
            any(type(card) is not int or card < 0 for card in physical_cards) or
            len(set(physical_cards)) != 2):
        raise ValueError('Bind exactly two distinct physical cards')
    if type(host_budget_per_rank_bytes) is not int or host_budget_per_rank_bytes != 4 * 2**30:
        raise ValueError('The current B1 entry requires its explicit4GiB/rank budget')
    return dict(schema='current-b1-deployment-plan/v2', role='B1',
                engine_release=release, logical_devices=logical_devices,
                physical_cards=list(physical_cards), reserved_cards=2,
                host_budget_per_rank_bytes=host_budget_per_rank_bytes,
                historical_claims=copy.deepcopy(original),
                historical_claims_scope='Exact preserved original plan; its comparison, arrival/warmup, input, device and recovery statements describe that historical run only. Current service identity and selected execution bindings are recorded separately.')
