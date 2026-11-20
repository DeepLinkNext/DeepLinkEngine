"""A reused MMLU pair can bind an arm outside its own output directory."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from test_context_baseline_reuse import combined
import evaluation_client


def fixture(tmp_path):
    files = combined()
    tree = ast.parse(files['scripts/reuse_campaign.py'])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'verify_weights')
    namespace = dict(Path=Path, read=lambda p: json.loads(p.read_text()))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<generated>', 'exec'), namespace)
    # Use the real arm resolver and its original prospective-binding recheck.
    resolver = ModuleType('arm_reuse')
    exec(files['scripts/arm_reuse.py'], resolver.__dict__)
    old = tmp_path / 'original-B0'
    new = tmp_path / 'campaign/mmlu-pair/B1'
    for arm in (old, new):
        arm.mkdir(parents=True)
        (arm / 'model-weights.json').write_text(json.dumps({'weight': {'sha256': 'a'*64, 'bytes': 17}}))
        for name in ('checksums.json', 'receipt.json', 'controller-source.json'):
            (arm / name).write_text('{}')
    binding = dict(run=str(old), role='B0', conditions={}, original_started=123,
                   original_release='original-companion', weights={'fixed': True},
                   original_quality={'correct': 139})
    for name, key in [('checksums.json', 'seal_sha256'), ('receipt.json', 'receipt_sha256'),
                      ('controller-source.json', 'controller_sources_sha256')]:
        binding[key] = hashlib.sha256((old / name).read_bytes()).hexdigest()
    pair = new.parent
    (pair / 'frozen-pair-plan.json').write_text(json.dumps({'reused_arms': {'B0': binding}}))
    # Admission's full raw/source validation has its own tests. Keep the actual
    # recheck seal gate here, and avoid requiring an NPU or a 151k-file image.
    resolver.admit = lambda root, role, conditions: dict(binding)
    return namespace['verify_weights'], resolver, pair, old


def test_completed_pair_resolves_original_b0_without_a_placeholder_directory(tmp_path):
    verify, resolver, pair, old = fixture(tmp_path)
    assert not (pair / 'B0').exists()
    with patch.dict(sys.modules, arm_reuse=resolver):
        result = verify({'pair': dict(source_root=str(pair.parent), stage='mmlu-pair')})
    assert result == {'weight': {'sha256': 'a'*64, 'bytes': 17}}
    assert not (pair / 'B0').exists()


def test_baseline_only_selection_from_a_composed_pair_resolves_the_same_binding(tmp_path):
    verify, resolver, pair, old = fixture(tmp_path)
    with patch.dict(sys.modules, arm_reuse=resolver):
        result = verify({'baseline': dict(source_root=str(pair.parent), stage='mmlu-pair/B0')})
    assert result['weight']['bytes'] == 17


@pytest.mark.parametrize('name', ['checksums.json', 'receipt.json', 'controller-source.json'])
def test_original_bound_arm_change_is_rejected_instead_of_retried(tmp_path, name):
    verify, resolver, pair, old = fixture(tmp_path)
    (old / name).write_text('{"changed":true}')
    with patch.dict(sys.modules, arm_reuse=resolver):
        with pytest.raises(ValueError, match='changed after prospective admission'):
            verify({'pair': dict(source_root=str(pair.parent), stage='mmlu-pair')})


def test_different_model_weights_still_reject_the_pair(tmp_path):
    verify, resolver, pair, old = fixture(tmp_path)
    (old / 'model-weights.json').write_text(json.dumps({'weight': {'sha256': 'b'*64, 'bytes': 17}}))
    with patch.dict(sys.modules, arm_reuse=resolver):
        with pytest.raises(ValueError, match='different model weight bytes'):
            verify({'pair': dict(source_root=str(pair.parent), stage='mmlu-pair')})


def test_standalone_task_weight_records_are_unchanged(tmp_path):
    verify, resolver, pair, old = fixture(tmp_path)
    (old / 'frontier-solver-B0').mkdir()
    (old / 'frontier-solver-B0/model-weights.json').write_bytes((old / 'model-weights.json').read_bytes())
    with patch.dict(sys.modules, arm_reuse=resolver):
        result = verify({'task': dict(source_root=str(old), stage='frontier-solver-B0')})
    assert result['weight']['sha256'] == 'a'*64


def test_bridge_changes_only_weight_resolution_and_single_arm_origin():
    key = 'scripts/reuse_campaign.py'
    before = {key: (ROOT / 'benchmark/one-click-kit' / key).read_bytes()}
    after = dict(before)
    evaluation_client.campaign_pair_weights_bridge(after)
    old, new = [ast.parse(value[key]) for value in (before, after)]
    old_weights = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_weights')
    new_weights = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_weights')
    assert ast.dump(old_weights) != ast.dump(new_weights)
    old.body.remove(old_weights)
    new.body.remove(new_weights)
    old_prepare = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    new_prepare = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    # Normalize precisely the one old literal path assignment and its reviewed
    # resolver import, leaving every source/raw/weight/compatibility gate intact.
    class NormalizeReviewedOrigin(ast.NodeTransformer):
        removed = replaced = 0

        def visit_ImportFrom(self, node):
            if (node.module == 'arm_reuse' and len(node.names) == 1
                    and node.names[0].name == 'resolve'
                    and node.names[0].asname == 'resolve_pair_arm'):
                self.removed += 1
                return None
            return node

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'arm'
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'resolve_pair_arm'):
                expected = ast.parse("arm = resolve_pair_arm(root / 'mmlu-pair', 'B0', read(root / 'mmlu-pair/frozen-pair-plan.json'))").body[0]
                assert ast.dump(node) == ast.dump(expected)
                self.replaced += 1
                return ast.parse("arm = root / 'mmlu-pair/B0'").body[0]
            return self.generic_visit(node)

    normalizer = NormalizeReviewedOrigin()
    normalized = normalizer.visit(ast.parse(ast.unparse(new_prepare)).body[0])
    assert normalizer.removed == normalizer.replaced == 1
    assert ast.dump(normalized) == ast.dump(old_prepare)
    old.body.remove(old_prepare)
    new.body.remove(new_prepare)
    assert ast.dump(old) == ast.dump(new)
