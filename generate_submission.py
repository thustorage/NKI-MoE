#!/usr/bin/env python3
"""Build a source-expanded single-file submission."""

from __future__ import annotations

import argparse
import ast
import io
import sys
import textwrap
import tokenize
from collections import defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

LOCAL_PACKAGE_ROOTS = {"kernels"}
LOCAL_ROOT_MODULES = {}
ENTRY_OR_TOOL_FILES = {}


def module_name_for_path(path: Path) -> str:
    rel = path.resolve().relative_to(PROJECT_ROOT)
    if rel.name == "__init__.py":
        return ".".join(rel.parent.parts)
    return ".".join(rel.with_suffix("").parts)


def package_name_for_module(module_name: str) -> str:
    return module_name.rsplit(".", 1)[0] if "." in module_name else ""


def resolve_module_to_path(module_name: str) -> Path | None:
    parts = module_name.split(".")
    module_path = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    if module_path.exists():
        return module_path

    package_path = PROJECT_ROOT.joinpath(*parts) / "__init__.py"
    if package_path.exists():
        return package_path

    return None


def is_allowed_local_module(module_name: str, entry: Path) -> bool:
    top = module_name.split(".", 1)[0]
    if top in LOCAL_PACKAGE_ROOTS or top in LOCAL_ROOT_MODULES:
        return True
    if module_name == module_name_for_path(entry):
        return True

    root_file = PROJECT_ROOT / f"{top}.py"
    if root_file.exists() and root_file.resolve() != entry.resolve():
        return root_file.name not in ENTRY_OR_TOOL_FILES

    return False


def resolve_relative_module(current_module: str, module: str | None, level: int) -> str:
    current_package = package_name_for_module(current_module)
    package_parts = current_package.split(".") if current_package else []
    base_parts = package_parts[: len(package_parts) - level + 1]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(part for part in base_parts if part)


def resolve_relative_import_path(path: Path, module: str | None, level: int) -> Path | None:
    current_module = module_name_for_path(path)
    module_name = resolve_relative_module(current_module, module, level)
    if not module_name:
        return None
    return resolve_module_to_path(module_name)


def collect_local_dependencies(path: Path, entry: Path) -> set[Path]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    used_names = collect_used_names(tree)
    deps: set[Path] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_local_module(alias.name, entry):
                    continue
                if import_alias_bound_name(alias) not in used_names:
                    continue
                dep_path = resolve_module_to_path(alias.name)
                if dep_path is not None:
                    deps.add(dep_path.resolve())

        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            used_aliases = [
                alias for alias in node.names
                if alias.name == "*" or import_alias_bound_name(alias) in used_names
            ]
            if not used_aliases:
                continue

            if level:
                dep_path = resolve_relative_import_path(path, node.module, level)
                if dep_path is not None:
                    deps.add(dep_path.resolve())

                base_module = resolve_relative_module(module_name_for_path(path), node.module, level)
                for alias in used_aliases:
                    if alias.name == "*":
                        continue
                    dep_path = resolve_module_to_path(f"{base_module}.{alias.name}")
                    if dep_path is not None:
                        deps.add(dep_path.resolve())

            elif node.module and is_allowed_local_module(node.module, entry):
                dep_path = resolve_module_to_path(node.module)
                if dep_path is not None:
                    deps.add(dep_path.resolve())
                for alias in used_aliases:
                    if alias.name == "*":
                        continue
                    dep_path = resolve_module_to_path(f"{node.module}.{alias.name}")
                    if dep_path is not None:
                        deps.add(dep_path.resolve())

    return deps


def discover_modules(entry: Path) -> tuple[dict[Path, set[Path]], dict[Path, str]]:
    entry = entry.resolve()
    deps_by_path: dict[Path, set[Path]] = {}
    sources: dict[Path, str] = {}
    queue: deque[Path] = deque([entry])

    while queue:
        path = queue.popleft().resolve()
        if path in sources:
            continue
        source = path.read_text()
        sources[path] = source

        deps = collect_local_dependencies(path, entry)
        deps_by_path[path] = deps
        for dep in sorted(deps):
            if dep not in sources:
                queue.append(dep)

    return deps_by_path, sources


def topological_order(deps_by_path: dict[Path, set[Path]]) -> list[Path]:
    all_paths = set(deps_by_path)
    for deps in deps_by_path.values():
        all_paths.update(deps)

    in_degree = {path: 0 for path in all_paths}
    reverse_edges: dict[Path, set[Path]] = defaultdict(set)
    for path, deps in deps_by_path.items():
        in_degree[path] += len(deps)
        for dep in deps:
            reverse_edges[dep].add(path)

    queue = deque(sorted(path for path, degree in in_degree.items() if degree == 0))
    order: list[Path] = []
    while queue:
        path = queue.popleft()
        order.append(path)
        for dependent in sorted(reverse_edges[path]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(all_paths):
        remaining = sorted(all_paths - set(order))
        print("WARNING: circular local dependencies detected; appending remaining files.", file=sys.stderr)
        for path in remaining:
            print(f"  {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        order.extend(remaining)

    return order


def is_local_import_node(node: ast.AST, path: Path, entry: Path) -> bool:
    if isinstance(node, ast.Import):
        return any(is_allowed_local_module(alias.name, entry) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return True
        return bool(node.module and is_allowed_local_module(node.module, entry))
    return False


def imported_name_is_local_submodule(path: Path, node: ast.ImportFrom, name: str) -> bool:
    level = node.level or 0
    if level:
        base_module = resolve_relative_module(module_name_for_path(path), node.module, level)
    else:
        base_module = node.module or ""
    if not base_module:
        candidate = name
    else:
        candidate = f"{base_module}.{name}"
    return resolve_module_to_path(candidate) is not None


def import_alias_bound_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".", 1)[0]


def collect_used_names(tree: ast.AST) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return used


def sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value]
    sanitized = "".join(chars).strip("_")
    return sanitized or "module"


def collect_top_level_class_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def build_class_rename_maps(entry: Path, sources: dict[Path, str]) -> dict[Path, dict[str, str]]:
    class_paths: dict[str, list[Path]] = defaultdict(list)
    existing_names: set[str] = set()

    for path, source in sources.items():
        for class_name in collect_top_level_class_names(source):
            class_paths[class_name].append(path)
            existing_names.add(class_name)

    rename_maps: dict[Path, dict[str, str]] = {}
    for class_name, paths in sorted(class_paths.items()):
        if len(paths) <= 1:
            continue
        for path in sorted(paths):
            if path.resolve() == entry.resolve():
                continue
            module_name = sanitize_identifier(module_name_for_path(path))
            candidate = f"_Bundled_{module_name}_{class_name}"
            suffix = 2
            while candidate in existing_names:
                candidate = f"_Bundled_{module_name}_{class_name}_{suffix}"
                suffix += 1
            existing_names.add(candidate)
            rename_maps.setdefault(path.resolve(), {})[class_name] = candidate

    return rename_maps


def collect_import_rewrite_maps(
    source: str,
    path: Path,
    entry: Path,
    class_rename_maps: dict[Path, dict[str, str]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    tree = ast.parse(source, filename=str(path))
    direct_name_renames: dict[str, str] = {}
    module_alias_renames: dict[str, dict[str, str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_local_module(alias.name, entry):
                    continue
                dep_path = resolve_module_to_path(alias.name)
                if dep_path is None:
                    continue
                rename_map = class_rename_maps.get(dep_path.resolve())
                if rename_map:
                    module_alias_renames[import_alias_bound_name(alias)] = rename_map

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                dep_path = resolve_relative_import_path(path, node.module, node.level)
                base_module = resolve_relative_module(module_name_for_path(path), node.module, node.level)
            else:
                if not node.module or not is_allowed_local_module(node.module, entry):
                    continue
                dep_path = resolve_module_to_path(node.module)
                base_module = node.module

            module_rename_map = class_rename_maps.get(dep_path.resolve(), {}) if dep_path is not None else {}
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported_path = resolve_module_to_path(f"{base_module}.{alias.name}") if base_module else None
                rename_map = (
                    class_rename_maps.get(imported_path.resolve(), {})
                    if imported_path is not None
                    else module_rename_map
                )
                if alias.asname and rename_map:
                    module_alias_renames[alias.asname] = rename_map
                renamed = rename_map.get(alias.name)
                if renamed:
                    direct_name_renames[import_alias_bound_name(alias)] = renamed

    return direct_name_renames, module_alias_renames


class ClassReferenceRenamer(ast.NodeTransformer):
    def __init__(
        self,
        local_class_renames: dict[str, str],
        direct_name_renames: dict[str, str],
        module_alias_renames: dict[str, dict[str, str]],
    ):
        self.local_class_renames = local_class_renames
        self.direct_name_renames = direct_name_renames
        self.module_alias_renames = module_alias_renames

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name in self.local_class_renames:
            node.name = self.local_class_renames[node.name]
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            renamed = self.local_class_renames.get(node.id) or self.direct_name_renames.get(node.id)
            if renamed:
                return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            rename_map = self.module_alias_renames.get(node.value.id)
            if rename_map and node.attr in rename_map:
                return ast.copy_location(ast.Name(id=rename_map[node.attr], ctx=node.ctx), node)
        return node


def rename_class_references(
    source: str,
    path: Path,
    entry: Path,
    class_rename_maps: dict[Path, dict[str, str]],
) -> str:
    direct_name_renames, module_alias_renames = collect_import_rewrite_maps(
        source, path, entry, class_rename_maps
    )
    local_class_renames = class_rename_maps.get(path.resolve(), {})
    if not local_class_renames and not direct_name_renames and not module_alias_renames:
        return source

    tree = ast.parse(source, filename=str(path))
    tree = ClassReferenceRenamer(
        local_class_renames=local_class_renames,
        direct_name_renames=direct_name_renames,
        module_alias_renames=module_alias_renames,
    ).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def strip_local_imports(source: str, path: Path, entry: Path) -> str:
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    replacements: dict[int, str | None] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not is_local_import_node(node, path, entry):
            continue

        start = node.lineno
        end = node.end_lineno or node.lineno
        shim_lines: list[str] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_local_module(alias.name, entry):
                    shim_lines.append(ast.get_source_segment(source, node) + "\n")
                    continue
                if alias.asname:
                    shim_lines.append(f"{alias.asname} = _self_module\n")

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                if alias.asname:
                    if imported_name_is_local_submodule(path, node, alias.name):
                        shim_lines.append(f"{alias.asname} = _self_module\n")
                    else:
                        shim_lines.append(f"{alias.asname} = {alias.name}\n")
                elif imported_name_is_local_submodule(path, node, alias.name):
                    shim_lines.append(f"{alias.name} = _self_module\n")

        replacements[start] = "".join(shim_lines) if shim_lines else None
        for line_no in range(start + 1, end + 1):
            replacements[line_no] = None

    output: list[str] = []
    for line_no, line in enumerate(lines, 1):
        if line_no not in replacements:
            output.append(line)
        elif replacements[line_no] is not None:
            output.append(replacements[line_no])
    return "".join(output)


def format_import_alias(alias: ast.alias) -> str:
    if alias.asname:
        return f"{alias.name} as {alias.asname}"
    return alias.name


def format_import_node(node: ast.Import | ast.ImportFrom, aliases: list[ast.alias]) -> str:
    if isinstance(node, ast.Import):
        return "import " + ", ".join(format_import_alias(alias) for alias in aliases) + "\n"
    module = "." * (node.level or 0) + (node.module or "")
    return f"from {module} import " + ", ".join(format_import_alias(alias) for alias in aliases) + "\n"


def prune_unused_imports(source: str) -> str:
    tree = ast.parse(source)
    used_names = collect_used_names(tree)
    lines = source.splitlines(keepends=True)
    replacements: dict[int, str | None] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            continue

        kept_aliases = [
            alias for alias in node.names
            if import_alias_bound_name(alias) in used_names
        ]
        if len(kept_aliases) == len(node.names):
            continue

        start = node.lineno
        end = node.end_lineno or node.lineno
        replacements[start] = format_import_node(node, kept_aliases) if kept_aliases else None
        for line_no in range(start + 1, end + 1):
            replacements[line_no] = None

    if not replacements:
        return source

    output: list[str] = []
    for line_no, line in enumerate(lines, 1):
        if line_no not in replacements:
            output.append(line)
        elif replacements[line_no] is not None:
            output.append(replacements[line_no])
    return "".join(output)


def validate_sources(sources: dict[Path, str]) -> None:
    for path, source in sorted(sources.items()):
        ast.parse(source, filename=str(path))
        if "neuronxcc.nki" in source:
            print(
                f"WARNING: {path.relative_to(PROJECT_ROOT)} contains 'neuronxcc.nki'. "
                "This may violate the NKI v2 rule if that code path is used.",
                file=sys.stderr,
            )


def build_output(
    entry: Path,
    output: Path,
    order: list[Path],
    sources: dict[Path, str],
    class_rename_maps: dict[Path, dict[str, str]],
) -> None:
    parts = [
        textwrap.dedent(
            f"""\
            import sys as _sys
            _self_module = _sys.modules[__name__]
            """
        )
    ]

    def generated_file_header(entry: Path) -> str:
        return textwrap.dedent(
            f"""\
            # This file is generated by generate_submission.py from {entry.relative_to(PROJECT_ROOT)}.
            # Do not edit this file directly.
            # Update the source modules and regenerate instead.

            """
        )

    for path in order:
        source = rename_class_references(sources[path], path, entry, class_rename_maps)
        source = prune_unused_imports(strip_local_imports(source, path, entry))
        parts.append("\n\n")
        parts.append(source.rstrip() + "\n")

    bundled_source = strip_comments_and_docstrings(dedupe_top_level_imports("".join(parts)))
    output.write_text(generated_file_header(entry) + bundled_source)


def is_docstring_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def collect_standalone_string_lines(tree: ast.AST) -> set[int]:
    string_lines: set[int] = set()

    for node in ast.walk(tree):
        if is_docstring_expr(node):
            for line_no in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                string_lines.add(line_no)
    return string_lines


def remove_standalone_strings(source: str) -> str:
    tree = ast.parse(source)
    string_lines = collect_standalone_string_lines(tree)
    if not string_lines:
        return source
    return "".join(
        line for line_no, line in enumerate(source.splitlines(keepends=True), 1)
        if line_no not in string_lines
    )


def strip_comments(source: str) -> str:
    tokens = []
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.COMMENT:
            continue
        tokens.append(token)
    return tokenize.untokenize(tokens)


def strip_comments_and_docstrings(source: str) -> str:
    stripped = collapse_blank_lines(strip_comments(remove_standalone_strings(source)))
    ast.parse(stripped)
    return stripped


def collapse_blank_lines(source: str) -> str:
    output: list[str] = []
    previous_blank = False
    for line in source.splitlines(keepends=True):
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        output.append("\n" if is_blank else line)
        previous_blank = is_blank
    return "".join(output)


def dedupe_top_level_imports(source: str) -> str:
    """Remove repeated top-level imports with identical AST semantics."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    seen: set[str] = set()
    remove_lines: set[int] = set()

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        key = ast.dump(node, include_attributes=False)
        if key not in seen:
            seen.add(key)
            continue
        for line_no in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            remove_lines.add(line_no)

    if not remove_lines:
        return source

    return "".join(
        line for line_no, line in enumerate(lines, 1)
        if line_no not in remove_lines
    )


def print_report(entry: Path, order: list[Path], deps_by_path: dict[Path, set[Path]]) -> None:
    print(f"Entry:  {entry.relative_to(PROJECT_ROOT)}")
    print(f"Files:  {len(order)}")
    print()
    for index, path in enumerate(order, 1):
        tag = " (entry)" if path.resolve() == entry.resolve() else ""
        dep_count = len(deps_by_path.get(path, set()))
        module_name = module_name_for_path(path)
        print(f"{index:>3}. {module_name:<55} {dep_count:>3} deps  {path.relative_to(PROJECT_ROOT)}{tag}")


def print_class_rename_report(class_rename_maps: dict[Path, dict[str, str]]) -> None:
    if not class_rename_maps:
        print()
        print("Class renames: none")
        return

    print()
    print("Class renames:")
    for path in sorted(class_rename_maps):
        for old_name, new_name in sorted(class_rename_maps[path].items()):
            print(f"  {path.relative_to(PROJECT_ROOT)}: {old_name} -> {new_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand modular python project and local dependencies into one normal Python file."
    )
    parser.add_argument(
        "-e",
        "--entry",
        default="kernels/qwen_with_nki.py",
        help="Entry file to bundle. Default: kernels/qwen_with_nki.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="qwen_with_nki.py",
        help="Output file. Default: qwen_with_nki.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dependency order without writing the output file.",
    )
    args = parser.parse_args()

    entry = (PROJECT_ROOT / args.entry).resolve()
    output = (PROJECT_ROOT / args.output).resolve()

    if not entry.exists():
        print(f"ERROR: entry file not found: {entry}", file=sys.stderr)
        return 1

    deps_by_path, sources = discover_modules(entry)
    validate_sources(sources)
    order = topological_order(deps_by_path)
    class_rename_maps = build_class_rename_maps(entry, sources)
    print_report(entry, order, deps_by_path)
    print_class_rename_report(class_rename_maps)

    if args.dry_run:
        return 0

    build_output(entry, output, order, sources, class_rename_maps)
    merged = output.read_text()
    ast.parse(merged, filename=str(output))
    print()
    print(
        f"Written {output.relative_to(PROJECT_ROOT)} "
        f"({len(order)} files, {len(merged)} bytes, {merged.count(chr(10))} lines)"
    )
    print("Syntax check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
