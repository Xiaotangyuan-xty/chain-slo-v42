"""Remove detailed C audit construction while retaining every scheduling statement."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def stripped_tree(source):
    tree = ast.parse(source)
    removed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_schedule':
            for part in ast.walk(node):
                if not isinstance(part, ast.If):
                    continue
                keep = []
                for statement in part.body:
                    call = statement.value if isinstance(statement, ast.Expr) else None
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr == 'append'
                            and isinstance(call.func.value, ast.Attribute)
                            and isinstance(call.func.value.value, ast.Name)
                            and call.func.value.value.id == 'self'
                            and call.func.value.attr == 'reservation_events'):
                        removed += 1
                    else:
                        keep.append(statement)
                part.body = keep or [ast.Pass()]
    if removed != 1:
        raise RuntimeError(f'Unexpected reservation audit source: {removed}')
    return ast.fix_missing_locations(tree)


def install(stage):
    stage = Path(stage)
    source = stage / 'programs/native_chain.py'
    if 'native_chain' in sys.modules:
        raise RuntimeError('C audit adapter must precede native_chain import')
    expected = json.loads((stage / 'FROZEN_V45_CORE.json').read_text())['files']['programs/native_chain.py']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError('Frozen native chain source changed')
    spec = importlib.util.spec_from_file_location('native_chain', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules['native_chain'] = module
    tree = stripped_tree(source.read_text())
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    return {'mode': 'minimal_common', 'frozen_source_sha256': digest,
            'runtime_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
            'removed_statements': ['reservation_events.append(event_snapshot)'],
            'last_members_and_selected_updates_preserved': True}
