"""Bind one frozen C8 tokenizer path when mount namespaces are unavailable."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'deeplinkengine-capless-c8-model-path/v1'
ORIGINAL_PATH = '/models/Qwen3.5-35B-A3B'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prefix(raw):
    nodes = []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            break
        nodes.append(node)
    return ast.Module(body=nodes, type_ignores=[])


def calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'from_pretrained'
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'AutoTokenizer']


def verify_binding(original, derived, model, expected_sha256):
    before = prefix(original)
    old = calls(before)
    if (len(old) != 1 or not old[0].args or not isinstance(old[0].args[0], ast.Constant)
            or old[0].args[0].value != ORIGINAL_PATH):
        raise ValueError('C8 path binding requires exactly one original tokenizer path')
    if sha(original) != expected_sha256:
        raise ValueError('Original C8 recipe source SHA differs')
    after = ast.parse(derived)
    new = calls(after)
    if (len(new) != 1 or not new[0].args or not isinstance(new[0].args[0], ast.Constant)
            or new[0].args[0].value != str(Path(model).resolve())):
        raise ValueError('C8 path binding must change only the selected model literal')
    new[0].args[0].value = ORIGINAL_PATH
    if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError('C8 model path adapter changed another AST node')
    return True


def bind(original, model, expected_sha256):
    tree = prefix(original)
    selected = calls(tree)
    if (len(selected) != 1 or not selected[0].args or not isinstance(selected[0].args[0], ast.Constant)
            or selected[0].args[0].value != ORIGINAL_PATH):
        raise ValueError('C8 path binding requires exactly one original tokenizer path')
    selected[0].args[0].value = str(Path(model).resolve())
    derived = (ast.unparse(tree) + '\n').encode()
    verify_binding(original, derived, model, expected_sha256)
    return derived


def validate_model(model, pin, admission):
    manifest = ROOT / 'engine/models/qwen3.5-35b-a3b.json'
    if sha(manifest.read_bytes()) != pin['model_manifest_sha256']:
        raise ValueError('Model manifest SHA differs')
    rows = json.loads(manifest.read_bytes())['files']
    if (admission.get('manifest_sha256') != pin['model_manifest_sha256']
            or admission.get('files_verified') != len(rows) or set(admission.get('files', {})) != set(rows)):
        raise ValueError('C8 requires the complete preceding model SHA admission')
    for name, expected in rows.items():
        actual = admission['files'][name]
        path = Path(model) / name
        info = path.stat()
        identity = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        if (path.is_symlink() or actual.get('sha256') != expected['sha256']
                or actual.get('bytes') != expected['bytes'] or actual.get('revision') != expected['revision']
                or identity != actual.get('stat_identity')):
            raise ValueError('Model changed or differs from its full SHA admission: ' + name)


def prepare(deployment, out, model, pin, admission):
    validate_model(model, pin, admission)
    source = Path(deployment) / 'runtime/entrypoints/prepare_continuous.py'
    original = source.read_bytes()
    derived = bind(original, model, pin['continuous_recipe_sha256'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    original_file = out / source.name
    bound_file = out / 'prepare_continuous.bound.py'
    original_file.write_bytes(original)
    bound_file.write_bytes(derived)
    verify_binding(original_file.read_bytes(), bound_file.read_bytes(), model, pin['continuous_recipe_sha256'])
    exec(compile(derived, str(bound_file), 'exec'), {'__file__': str(bound_file)})
    validate_model(model, pin, admission)
    output = out / 'continuous-inputs-r1.json'
    if sha(output.read_bytes()) != pin['continuous_input_sha256']:
        raise ValueError('Capless C8 output differs from the original input bytes')
    binding = dict(schema=VERSION, original_path=ORIGINAL_PATH, bound_path=str(Path(model).resolve()),
        original_source=str(original_file), original_source_sha256=sha(original),
        derived_source=str(bound_file), derived_source_sha256=sha(derived),
        adapter_source_sha256=sha(Path(__file__).read_bytes()),
        only_one_path_constant_bound=True, remaining_ast_identical=True,
        original_source_file_modified=False, model_manifest_sha256=pin['model_manifest_sha256'],
        full_model_sha_admission_rechecked=True, namespace_or_global_alias_created=False,
        model_started=False, formal_measurement=False)
    return dict(path=str(output), sha256=sha(output.read_bytes()), recipe_sha256=sha(original),
                scope='Original C8 algorithm; declared single tokenizer filesystem path binding',
                path_binding_adapter=binding)


def verify_record(record):
    value = record['path_binding_adapter']
    if value.get('schema') != VERSION or value.get('remaining_ast_identical') is not True:
        raise ValueError('Unknown C8 path-binding adapter')
    old, new = Path(value['original_source']), Path(value['derived_source'])
    parent = Path(record['path']).resolve().parent
    if old.resolve().parent != parent or new.resolve().parent != parent:
        raise ValueError('C8 adapter sources escaped their own input directory')
    if old.is_symlink() or new.is_symlink() or sha(old.read_bytes()) != value['original_source_sha256'] or sha(new.read_bytes()) != value['derived_source_sha256']:
        raise ValueError('Recorded C8 path-binding source bytes differ')
    verify_binding(old.read_bytes(), new.read_bytes(), value['bound_path'], record['recipe_sha256'])
