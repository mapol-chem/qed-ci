#!/usr/bin/env python3
"""Check user-documentation links and documented configuration fields.

This is intentionally a small, dependency-free check rather than a documentation
framework. It verifies that relative Markdown links resolve and that every public
field of the configuration dataclasses appears in the authoritative option
reference.
"""
import ast
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / 'README.md',
    ROOT / 'docs/README.md',
    ROOT / 'docs/installation.md',
    ROOT / 'docs/methods.md',
    ROOT / 'docs/development.md',
    ROOT / 'docs/configuration-and-output.md',
    ROOT / 'examples/quickstart/README.md',
    ROOT / 'examples/canonical/README.md',
    ROOT / 'docs/audit/stage5/README.md',
]
OPTION_REFERENCE = ROOT / 'docs/configuration-and-output.md'
CONFIG_MODULE = ROOT / 'src/qed_ci/config.py'
EXAMPLES = [ROOT / 'examples/quickstart/methods.py',
            ROOT / 'examples/quickstart/gradient.py']
INTERNAL_FIELDS = {'legacy', 'diagnostics', '_legacy_options'}


def relative_links(path):
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        target = match.group(1).strip().split()[0].strip('<>')
        if target.startswith('#'):
            continue
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        link = unquote(parsed.path)
        if not link:
            continue
        yield target, link


def check_links(errors):
    count = 0
    for path in DOC_FILES:
        if not path.is_file():
            errors.append(f'missing documentation file: {path.relative_to(ROOT)}')
            continue
        for original, link in relative_links(path):
            count += 1
            destination = (path.parent / link).resolve()
            if not destination.exists():
                errors.append(f'{path.relative_to(ROOT)}: broken link {original!r}')
    return count


def public_fields():
    tree = ast.parse(CONFIG_MODULE.read_text(encoding='utf-8'))
    fields = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in {
                'CQEDConfig', 'CISolverOptions', 'CASSCFOptions'}:
            continue
        names = []
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                name = statement.target.id
                if not name.startswith('_') and name not in INTERNAL_FIELDS:
                    names.append(name)
        fields[node.name] = names
    return fields


def check_option_reference(errors):
    text = OPTION_REFERENCE.read_text(encoding='utf-8')
    fields = public_fields()
    expected = {'CQEDConfig': 'CQEDConfig', 'CISolverOptions': 'CISolverOptions',
                'CASSCFOptions': 'CASSCFOptions'}
    for cls, name in expected.items():
        if cls not in fields:
            errors.append(f'could not find dataclass {cls} in src/qed_ci/config.py')
            continue
        for field in fields[cls]:
            if not re.search(rf'\b{re.escape(field)}\b', text):
                errors.append(f'{cls}.{field} is not documented in '
                              f'{OPTION_REFERENCE.relative_to(ROOT)}')
    return sum(len(names) for names in fields.values())


def check_examples(errors):
    for path in EXAMPLES:
        if not path.is_file():
            errors.append(f'missing example: {path.relative_to(ROOT)}')
    methods = (ROOT / 'examples/quickstart/methods.py')
    if methods.is_file():
        text = methods.read_text(encoding='utf-8')
        for method in ('cis', 'fci', 'casci', 'casscf', 'sa'):
            if method not in text:
                errors.append(f'methods.py does not mention example method {method!r}')


def main():
    errors = []
    links = check_links(errors)
    options = check_option_reference(errors)
    check_examples(errors)
    if errors:
        for error in errors:
            print('ERROR: ' + error, file=sys.stderr)
        return 1
    print(f'Documentation checks passed: {links} relative links, '
          f'{options} configuration fields.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
