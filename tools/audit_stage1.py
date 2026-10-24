#!/usr/bin/env python3
"""Static Stage 1 inventory. Does not import or execute the package.

Inventories all top-level Python and C runtime sources, including quarantined
legacy code (explicitly tagged). C records are comment-stripped source sites;
they do not claim reachability. Psi4's own option schema is outside this scope.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/audit/stage1/inventory.json')
    args = parser.parse_args()
    data = dict(scope='src/*.py and src/*.c; source sites, not runtime reachability',
                sources={}, cavity_options=[], environment_reads=[], dictionary_reads=[],
                output_and_file_sites=[], cli_arguments=[], public_signatures=[], tests=[], documentation=[])
    output_names = {'print', 'printf', 'fprintf', 'puts', 'fputs', 'print_out',
                    'set_output_file', 'be_quiet', 'savetxt', 'save', 'savez',
                    'savez_compressed', 'loadtxt', 'load', 'open', 'to_file',
                    'from_file', 'exit', 'warn', 'NamedTemporaryFile', 'mkstemp'}
    for path in sorted((ROOT / 'src').glob('*')):
        if path.suffix not in {'.py', '.c'}:
            continue
        text = path.read_text()
        rel = str(path.relative_to(ROOT))
        data['sources'][rel] = hashlib.sha256(text.encode()).hexdigest()
        role = 'quarantined' if path.name == 'legacy_python_ci.py' else 'runtime_or_driver'
        def record(line, expression, **extra):
            return dict(file=rel, line=line, role=role, expression=expression, **extra)
        if path.suffix == '.c':
            clean = re.sub(r'/\*.*?\*/|//[^\n]*',
                           lambda m: '\n' * m.group().count('\n'), text, flags=re.S)
            for match in re.finditer(r'getenv\(\s*"([^"]+)"\s*\)', clean):
                line = clean.count('\n', 0, match.start()) + 1
                data['environment_reads'].append(record(line, clean.splitlines()[line-1].strip(),
                                                         name=match[1], timing='C runtime; may be cached'))
            for match in re.finditer(r'\b(printf|fprintf|puts|fputs|fopen|exit)\s*\(', clean):
                line = clean.count('\n', 0, match.start()) + 1
                data['output_and_file_sites'].append(record(line, clean.splitlines()[line-1].strip(), kind=match[1]))
            continue
        tree = ast.parse(text)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        def timing(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return 'call time'
            return 'import time'
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                owner = parents.get(node)
                if isinstance(owner, ast.ClassDef) and owner.name in {'PFHamiltonianGenerator', 'nuclear_grad'}:
                    data['public_signatures'].append(record(node.lineno, ast.unparse(node.args),
                                                           name=owner.name + '.' + node.name))
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                leaf = name.split('.')[-1]
                if leaf == 'add_argument':
                    data['cli_arguments'].append(record(node.lineno, ast.unparse(node)))
                if leaf == 'get' and any(x in name for x in ('options', 'cavity_dictionary')):
                    data['dictionary_reads'].append(record(node.lineno, ast.unparse(node), dictionary=name.rsplit('.', 1)[0],
                        key=ast.unparse(node.args[0]) if node.args else None))
                if leaf in output_names:
                    data['output_and_file_sites'].append(record(node.lineno, ast.unparse(node), kind=leaf))
                if (name.endswith('.getenv') or name.endswith('.environ.get')) and node.args:
                    key = node.args[0]
                    data['environment_reads'].append(record(node.lineno, ast.unparse(node),
                        name=key.value if isinstance(key, ast.Constant) else ast.unparse(key), timing=timing(node)))
            if isinstance(node, ast.Subscript):
                name = ast.unparse(node.value)
                if name.endswith('.environ'):
                    data['environment_reads'].append(record(node.lineno, ast.unparse(node),
                        name=ast.literal_eval(node.slice) if isinstance(node.slice, ast.Constant) else ast.unparse(node.slice),
                        timing=timing(node)))
                if isinstance(node.slice, ast.Constant) and any(x in name for x in ('options', 'cavity_dictionary')):
                    data['dictionary_reads'].append(record(node.lineno, ast.unparse(node), dictionary=name, key=node.slice.value))
        if path.name == 'helper_PFCI.py':
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'parseCavityOptions')
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                test = node.test
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Constant)
                        and len(test.comparators) == 1 and ast.unparse(test.comparators[0]) == 'cavity_dictionary'):
                    data['cavity_options'].append(record(node.lineno, ast.unparse(node), name=test.left.value,
                        default_branch=[ast.unparse(n) for n in node.orelse],
                        note='Default may be conditional or overridden later; expression is authoritative.'))
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        tree = ast.parse(path.read_text())
        data['tests'].append(dict(file=str(path.relative_to(ROOT)),
            functions=[n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]))
    data['documentation'] = [str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*.md'))
                             if not any(part in {'.git', 'build', '.codex'} for part in p.relative_to(ROOT).parts)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Wrote {args.output}: {len(data["cavity_options"])} cavity options, '
          f'{len(data["environment_reads"])} environment read sites, '
          f'{len(data["output_and_file_sites"])} output/file sites.')


if __name__ == '__main__':
    main()
