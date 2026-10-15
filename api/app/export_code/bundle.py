"""Copy app functions into one standalone source file.

`Bundle.add(module, name)` takes a module-level def, class or constant of an app module and, transitively, every
module-level name it reaches: other definitions of the same module, names it imports from other app modules (`from
.profiling import coerce_numeric`), and functions it reaches through an app module (`bands.assign`, emitted as a
`SimpleNamespace`). Each is copied verbatim, comments and docstrings included, so the bundle runs the app's own code
rather than a rewrite of it. Third-party and standard-library imports are collected and emitted once at the top.

The bundle is flat: every copied name lives in one namespace, so two different things bound to the same name (two
definitions, a definition and an import, two imports) refuse to bundle (rename one in the app). A name can be
`provided`: the `provider` module's placeholders for names the caller defines elsewhere in the file, which no other
module may define or use. A definition can be `override`n with replacement source (for the odd helper that reaches
into the web framework).
"""
from __future__ import annotations

import ast
import builtins
import copy
import importlib
import importlib.metadata
import importlib.util
import inspect
import symtable
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType

APP = "app"
# Distribution names where they differ from the import name.
DISTRIBUTIONS = {"sklearn": "scikit-learn"}


class BundleError(Exception):
    pass


@dataclass
class _Import:
    module: str  # absolute module the name comes from
    name: str | None  # None for `import x [as y]`
    local: str  # the name it binds


@dataclass
class _Module:
    name: str
    source: str
    lines: list[str]
    defs: dict[str, ast.stmt]  # top-level def / class / assignment binding each name
    imports: dict[str, _Import]
    table: symtable.SymbolTable


@dataclass
class _Unit:
    module: str
    names: tuple[str, ...]  # the names the statement binds
    source: str
    lineno: int
    calls: set[tuple[str, str]] = field(default_factory=set)  # (module, name) needed once it runs
    defines: set[tuple[str, str]] = field(default_factory=set)  # needed while the statement itself executes


@lru_cache(maxsize=None)
def _module(name: str) -> _Module:
    mod = importlib.import_module(name)
    source = inspect.getsource(mod)
    tree = ast.parse(source)
    defs: dict[str, ast.stmt] = {}
    imports: dict[str, _Import] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defs[n.id] = node
        elif isinstance(node, ast.Import):
            for a in node.names:
                # `import pyarrow.parquet as pq` binds pq; `import os.path` binds os.
                local = a.asname or a.name.split(".")[0]
                imports[local] = _Import(a.name, None, local)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(name, node.module, node.level, mod)
            for a in node.names:
                imports[a.asname or a.name] = _Import(base, a.name, a.asname or a.name)
    return _Module(name, source, source.splitlines(), defs, imports, symtable.symtable(source, name, "exec"))


def _absolute(module: str, target: str | None, level: int, mod: ModuleType) -> str:
    if level == 0:
        return target or ""
    parts = (mod.__package__ or "").split(".")
    base = parts[: len(parts) - (level - 1)]
    return ".".join(base + ([target] if target else []))


def _segment(m: _Module, node: ast.stmt) -> tuple[str, int]:
    """The statement's source, with its decorators and the comment lines directly above it."""
    start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    while start > 1 and m.lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return "\n".join(m.lines[start - 1: node.end_lineno]), start


def _names(nodes) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        if node is None:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
    return out


def _exec_nodes(node: ast.stmt) -> list[ast.AST]:
    """The parts of a top-level statement that run when the statement itself runs (not when it's later called)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [*node.decorator_list, *node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]
    if isinstance(node, ast.ClassDef):
        out: list[ast.AST] = [*node.decorator_list, *node.bases, *[k.value for k in node.keywords]]
        for stmt in node.body:
            out += _exec_nodes(stmt) if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) else [stmt]
        return out
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else node
    return [_without_lambda_bodies(value)] if value is not None else []


def _without_lambda_bodies(node: ast.AST) -> ast.AST:
    class Strip(ast.NodeTransformer):
        def visit_Lambda(self, n: ast.Lambda):
            return ast.Constant(None)

    return Strip().visit(copy.deepcopy(node))


def _scope_globals(table: symtable.SymbolTable) -> set[str]:
    out = {s.get_name() for s in table.get_symbols() if s.is_global()}
    for child in table.get_children():
        out |= _scope_globals(child)
    return out


def _call_names(m: _Module, node: ast.stmt) -> set[str]:
    """Module-level names the statement reaches when it runs or is called."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        tables = [t for t in m.table.get_children() if t.get_name() == node.name and t.get_lineno() == node.lineno]
        if len(tables) != 1:
            raise BundleError(f"{m.name}.{node.name}: can't find its scope")
        local = {s.get_name() for s in tables[0].get_symbols() if s.is_local()} \
            if isinstance(node, ast.ClassDef) else set()
        return _scope_globals(tables[0]) | (_exec_names(node) - local)
    return _names([node.value] if isinstance(node, (ast.Assign, ast.AnnAssign)) else [node])


def _exec_names(node: ast.stmt) -> set[str]:
    return _names(_exec_nodes(node))


def _module_attrs(node: ast.AST, local: str) -> set[str]:
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == local}


def _is_app(module: str) -> bool:
    return module == APP or module.startswith(APP + ".")


class Bundle:
    def __init__(self, provided: set[str] | None = None, overrides: dict[tuple[str, str], str] | None = None,
                 provider: str | None = None):
        self.provided = set(provided or ())
        self.provider = provider  # the module whose uses of `provided` names the caller supplies (None: any)
        self.overrides = dict(overrides or {})
        self.units: dict[tuple[str, str], _Unit] = {}  # keyed by (module, first name)
        self.by_name: dict[str, tuple[str, str]] = {}
        self.imports: dict[str, _Import] = {}  # local name -> external import
        self.packages: set[str] = set()
        self.namespaces: dict[str, tuple[str, set[str]]] = {}  # local name -> (app module, attributes)
        self._order: list[str] = []  # modules in the order they were reached
        self.seen: set[tuple[str, str]] = set()

    # ----------------------------------------------------------------- collecting
    def add(self, module: str, name: str) -> None:
        self._need(module, name, frm=f"{module}.{name}")

    def _is_provided(self, module: str, name: str) -> bool:
        return name in self.provided and (self.provider is None or module == self.provider)

    def _need(self, module: str, name: str, frm: str) -> None:
        if self._is_provided(module, name):
            return
        key = (module, name)
        if key in self.seen:
            return
        self.seen.add(key)
        m = _module(module)
        if module not in self._order:
            self._order.append(module)
        if key in self.overrides:
            node = ast.parse(self.overrides[key]).body[0]
            src, lineno = self.overrides[key].strip("\n"), getattr(m.defs.get(name), "lineno", 0)
            unit = _Unit(module, (name,), src, lineno)
            self._claim(unit)
            table = symtable.symtable(self.overrides[key], f"{module}.{name}", "exec")
            for n in set().union(*[_scope_globals(t) for t in table.get_children()], _exec_names(node)) - {name}:
                self._resolve(m, n, unit, node, defines=False, frm=frm)
            return
        if name in m.imports:
            self._through_import(m, m.imports[name], frm)
            return
        node = m.defs.get(name)
        if node is None:
            if hasattr(builtins, name):
                return
            raise BundleError(f"{frm}: {module} has no top-level {name!r}")
        names = tuple(n for n, d in m.defs.items() if d is node)
        src, lineno = _segment(m, node)
        unit = _Unit(module, names, src, lineno)
        self._claim(unit)
        self._imports_inside(m, node, frm)
        strict = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        exec_names = _exec_names(node)
        for n in _call_names(m, node) - set(names):
            self._resolve(m, n, unit, node, defines=n in exec_names, frm=frm, strict=strict)

    def _claim(self, unit: _Unit) -> None:
        for n in unit.names:
            if n in self.provided or n in self.imports or n in self.namespaces:
                raise BundleError(f"{unit.module} defines {n!r}, which the bundle already binds to something else; "
                                  "rename it")
            other = self.by_name.get(n)
            if other is not None and other[0] != unit.module:
                theirs = self.units[other]
                if theirs.source.strip() != unit.source.strip():
                    raise BundleError(f"{n!r} is defined in both {other[0]} and {unit.module}; the bundle is one "
                                      "namespace, so rename one of them")
                return
            self.by_name[n] = (unit.module, unit.names[0])
        self.units[(unit.module, unit.names[0])] = unit

    def _resolve(self, m: _Module, n: str, unit: _Unit, node: ast.AST, defines: bool, frm: str,
                 strict: bool = True) -> None:
        if self._is_provided(m.name, n) or n == "__name__":
            return
        dep: tuple[str, str] | None = None
        if n in m.defs:
            self._need(m.name, n, frm)
            dep = self.by_name.get(n)
        elif n in m.imports:
            imp = m.imports[n]
            if _is_app(imp.module):
                if imp.name is None or _is_module(imp.module, imp.name):
                    target = imp.module if imp.name is None else f"{imp.module}.{imp.name}"
                    attrs = _module_attrs(node, n)
                    if n in self.by_name or n in self.imports or n in self.provided:
                        raise BundleError(f"{m.name} uses {n!r} for {target}, which the bundle already binds to "
                                          "something else")
                    ns = self.namespaces.setdefault(n, (target, set()))
                    if ns[0] != target:
                        raise BundleError(f"{n!r} names both {ns[0]} and {target}")
                    ns[1].update(attrs)
                    for a in attrs:
                        self._need(target, a, frm)
                    dep = ("<namespace>", n)
                else:
                    if imp.local != imp.name:
                        raise BundleError(f"{m.name} imports {imp.name} as {imp.local}; bundles keep app names")
                    self._need(imp.module, imp.name, frm)
                    dep = self.by_name.get(imp.name)
            else:
                self._external(imp)
        elif hasattr(builtins, n):
            return
        elif strict:
            raise BundleError(f"{frm}: {m.name} uses {n!r}, which it neither defines nor imports")
        if dep is not None:
            (unit.defines if defines else unit.calls).add(dep)

    def _through_import(self, m: _Module, imp: _Import, frm: str) -> None:
        if not _is_app(imp.module):
            self._external(imp)
        elif imp.name is not None:
            self._need(imp.module, imp.name, frm)

    def _external(self, imp: _Import) -> None:
        if imp.module == "__future__":
            return
        had = self.imports.get(imp.local)
        if (had is not None and (had.module, had.name) != (imp.module, imp.name)) or imp.local in self.by_name \
                or imp.local in self.namespaces or imp.local in self.provided:
            raise BundleError(f"{imp.local!r} is imported from {imp.module} but the bundle already binds it to "
                              "something else; rename one in the app")
        self.imports[imp.local] = imp
        self.packages.add(imp.module.split(".")[0])

    def _imports_inside(self, m: _Module, node: ast.AST, frm: str) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.ImportFrom) and n is not node:
                if n.level:
                    raise BundleError(f"{frm}: a relative import inside {m.name} can't be bundled")
                self.packages.add((n.module or "").split(".")[0])
            elif isinstance(n, ast.Import):
                self.packages.update(a.name.split(".")[0] for a in n.names)

    # ----------------------------------------------------------------- output
    def import_lines(self) -> list[str]:
        plain, froms = set(), {}
        if self.namespaces:
            froms.setdefault("types", set()).add("SimpleNamespace")
        for imp in self.imports.values():
            if imp.name is None:
                top = imp.module.split(".")[0]
                plain.add(f"import {imp.module}" if imp.local == top else f"import {imp.module} as {imp.local}")
            else:
                froms.setdefault(imp.module, set()).add(imp.name if imp.local == imp.name else
                                                        f"{imp.name} as {imp.local}")
        lines = sorted(plain) + [f"from {mod} import {', '.join(sorted(names))}" for mod, names in froms.items()]
        stdlib = [ln for ln in lines if _top(ln) in sys.stdlib_module_names]
        third = [ln for ln in lines if _top(ln) not in sys.stdlib_module_names]
        return sorted(stdlib, key=_import_key) + ([""] if stdlib and third else []) + sorted(third, key=_import_key)

    def requirements(self) -> list[str]:
        """Pinned third-party distributions, at the versions this app runs."""
        out = []
        for pkg in sorted(p for p in self.packages if p and p not in sys.stdlib_module_names and not _is_app(p)):
            dist = DISTRIBUTIONS.get(pkg, pkg)
            out.append(f"{dist}=={importlib.metadata.version(dist)}")
        return out

    def body(self) -> str:
        """The copied definitions, each after what its own execution needs, in the order the app modules were
        reached and their source order otherwise."""
        rank = {m: i for i, m in enumerate(self._order)}
        units = list(self.units.values())
        for local, (target, attrs) in sorted(self.namespaces.items()):
            src = f"{local} = SimpleNamespace(" + ", ".join(f"{a}={a}" for a in sorted(attrs)) + ")"
            units.append(_Unit("<namespace>", (local,), src, 10**9,
                               defines={self.by_name[a] for a in attrs if a in self.by_name}))
        keyed = {(u.module, u.names[0]): u for u in units}
        pending = sorted(keyed, key=lambda k: _sort(keyed[k], rank))
        placed: set[tuple[str, str]] = set()
        done: list[_Unit] = []
        while pending:
            k = next((k for k in pending if {d for d in keyed[k].defines if d in keyed and d != k} <= placed), None)
            if k is None:
                raise BundleError(f"definitions need each other to run: {pending}")
            pending.remove(k)
            placed.add(k)
            done.append(keyed[k])
        out, current = [], None
        for u in done:
            text = u.source
            if u.module != current:
                current = u.module
                label = "namespaces" if u.module == "<namespace>" else u.module.replace(".", "/") + ".py"
                text = f"# {'─' * 4} {label} {'─' * max(4, 110 - len(label))}\n{text}"
            out.append(text)
        return "\n\n\n".join(out)


def _sort(u: _Unit, rank: dict[str, int]) -> tuple:
    return (rank.get(u.module, len(rank)), u.lineno)


def _top(line: str) -> str:
    return line.split()[1].split(".")[0]


def _import_key(line: str) -> tuple:
    return (line.split()[1], line)


def undefined_names(source: str) -> set[str]:
    """Global names some function, class or lambda of `source` uses but nothing at its top level binds: what a bundle
    would fail on only once that code ran."""
    top = symtable.symtable(source, "bundle", "exec")
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    used: set[str] = set()
    for child in top.get_children():
        used |= _scope_globals(child)
    return {n for n in used - bound if not hasattr(builtins, n) and n not in ("__name__", "__file__")}


def _is_module(package: str, name: str) -> bool:
    try:
        return isinstance(getattr(importlib.import_module(package), name, None), ModuleType) or \
            importlib.util.find_spec(f"{package}.{name}") is not None
    except (ImportError, ValueError):
        return False
