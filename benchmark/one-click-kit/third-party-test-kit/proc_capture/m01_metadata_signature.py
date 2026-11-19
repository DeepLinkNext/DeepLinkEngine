"""Read the loaded runner class's declared ABI without trusting wrapper signatures."""

import ast
import hashlib
import inspect
from pathlib import Path


def declaration_signature(node):
    parameters = []
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        if argument.arg == 'self':
            if index != 0:
                raise ValueError('Unexpected metadata receiver position')
            continue
        value = inspect.Parameter.empty if default is inspect.Parameter.empty else ast.literal_eval(default)
        kind = inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(inspect.Parameter(argument.arg, kind, default=value))
    if args.vararg:
        parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for argument, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        value = inspect.Parameter.empty if default is None else ast.literal_eval(default)
        parameters.append(inspect.Parameter(argument.arg, inspect.Parameter.KEYWORD_ONLY, default=value))
    if args.kwarg:
        parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def resolve(runner):
    for cls in type(runner).__mro__:
        if cls is object:
            continue
        path = Path(inspect.getsourcefile(cls)).resolve()
        raw = path.read_bytes()
        tree = ast.parse(raw)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__]
        if len(classes) != 1:
            raise ValueError('Cannot uniquely locate the loaded runner class declaration')
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and
                   n.name == '_build_attention_metadata']
        if not methods:
            continue
        if len(methods) != 1:
            raise ValueError('Ambiguous metadata declaration')
        signature = declaration_signature(methods[0])
        if 'num_reqs' not in signature.parameters:
            continue  # The CorrectedRunner forwarding method is generic.
        if 'for_cudagraph_capture' not in signature.parameters:
            raise ValueError('The actual metadata declaration lacks its graph-capture boundary')
        return signature, dict(class_name=cls.__module__ + '.' + cls.__name__,
            path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            signature=str(signature), method_line=methods[0].lineno,
            source='Current loaded class declaration; wrapper __signature__/__wrapped__ is not an ABI source')
    raise ValueError('Cannot resolve the current runner metadata declaration')
