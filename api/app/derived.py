"""Derived fields: columns computed from a dataset's own columns and added to its data, so review, export and Explore
treat them like any other field. Three kinds:

- formula: arithmetic over numeric fields, e.g. ``water / (cement + blast_furnace_slag + fly_ash)``. A name that isn't
  a plain identifier goes in backticks (```glyburide-metformin```). Functions: log, log1p, exp, sqrt, abs, min, max,
  clip(x, lo, hi). A result that isn't a finite number (0/0) is missing.
- groups: a field's values put into named groups, by exact values or numeric ranges (ICD-9 codes into chapters:
  ranges 390–459 and 785 → "Circulatory"); everything else is `other`, missing stays missing. All-numeric group labels
  make a number (age brackets to their midpoints).
- count: how many of the given fields hold one of the given values (medication columns marked Up or Down).
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any

import polars as pl
from fastapi import HTTPException

KINDS = ("formula", "groups", "count")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .-]{0,62}$")
FUNCS = {"log": lambda e: e.log(), "log1p": lambda e: e.log1p(), "exp": lambda e: e.exp(), "sqrt": lambda e: e.sqrt(),
         "abs": lambda e: e.abs()}


def _bad(msg: str) -> HTTPException:
    return HTTPException(422, msg)


def _formula(expr: str, columns: list[str]) -> tuple[pl.Expr, list[str]]:
    """A polars expression for `expr`, and the fields it reads. Only arithmetic, numbers, field names and FUNCS."""
    names: dict[str, str] = {}

    def quoted(m: re.Match) -> str:
        key = f"__f{len(names)}"
        names[key] = m.group(1)
        return key

    try:
        tree = ast.parse(re.sub(r"`([^`]+)`", quoted, expr.strip()), mode="eval")
    except SyntaxError as exc:
        raise _bad(f"The formula doesn't parse: {exc.msg}") from exc
    used: list[str] = []

    def walk(n: ast.AST) -> pl.Expr:
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return pl.lit(float(n.value))
        if isinstance(n, ast.Name):
            col = names.get(n.id, n.id)
            if col not in columns:
                raise _bad(f"No field named {col!r} (put a name with spaces or dashes in backticks)")
            used.append(col)
            return pl.col(col).cast(pl.Float64, strict=False)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -walk(n.operand) if isinstance(n.op, ast.USub) else walk(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            ops = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b, ast.Pow: a.pow(b)}
            if type(n.op) not in ops:
                raise _bad("Only + - * / and ** are allowed")
            return ops[type(n.op)]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            f, args = n.func.id, [walk(a) for a in n.args]
            if f in FUNCS and len(args) == 1:
                return FUNCS[f](args[0])
            if f in ("min", "max") and len(args) == 2:
                return pl.min_horizontal(args) if f == "min" else pl.max_horizontal(args)
            if f == "clip" and len(args) == 3:
                return args[0].clip(args[1], args[2])
            raise _bad(f"Unknown function {f}() or wrong number of arguments (use {', '.join(FUNCS)}, min, max, clip)")
        raise _bad("The formula may use field names, numbers, + - * / **, parentheses and functions only")

    out = walk(tree)
    return pl.when(out.is_finite()).then(out).otherwise(None), sorted(set(used))


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def check_spec(kind: str, spec: dict, columns: list[str]) -> list[str]:
    """The fields a spec reads; 422 for anything malformed."""
    if kind not in KINDS:
        raise _bad(f"kind must be one of {', '.join(KINDS)}")
    if kind == "formula":
        if not isinstance(spec.get("expr"), str) or not spec["expr"].strip():
            raise _bad("Give the formula")
        return _formula(spec["expr"], columns)[1]
    if kind == "groups":
        src, groups = spec.get("source"), spec.get("groups")
        if src not in columns:
            raise _bad(f"No field named {src!r}")
        if not isinstance(groups, list) or not groups or len(groups) > 100:
            raise _bad("Give 1 to 100 groups")
        for g in groups:
            if not isinstance(g, dict) or not str(g.get("label", "")).strip():
                raise _bad("Every group needs a label")
            values, ranges = g.get("values") or [], g.get("ranges") or []
            if not values and not ranges:
                raise _bad(f"Group {g['label']!r} needs values or ranges")
            for r in ranges:
                if not (isinstance(r, list) and len(r) == 2 and None not in map(_num, r) and _num(r[0]) <= _num(r[1])):
                    raise _bad(f"Group {g['label']!r}: a range is [low, high] with low ≤ high")
        return [src]
    fields, values = spec.get("fields"), spec.get("values")
    if not isinstance(fields, list) or not fields or (unknown := [f for f in fields if f not in columns]):
        raise _bad(f"Unknown field(s): {', '.join(map(repr, unknown))}" if fields else "Choose the fields to count across")
    if not isinstance(values, list) or not values:
        raise _bad("Give the values to count")
    return list(fields)


def compute(df: pl.DataFrame, kind: str, spec: dict) -> pl.Series:
    """The derived column for every row of `df`."""
    if kind == "formula":
        return df.select(_formula(spec["expr"], df.columns)[0]).to_series()
    if kind == "count":
        values = [str(v) for v in spec["values"]]
        return df.select(pl.sum_horizontal([pl.col(f).cast(pl.Utf8).is_in(values).fill_null(False).cast(pl.Int64)
                                            for f in spec["fields"]])).to_series()
    src = df[spec["source"]]
    text, num = src.cast(pl.Utf8), src.cast(pl.Utf8).cast(pl.Float64, strict=False)
    labels = [str(g["label"]).strip() for g in spec["groups"]]
    other = str(spec.get("other") or "Other").strip()
    expr = pl.lit(other)
    for g, label in reversed(list(zip(spec["groups"], labels))):
        hit = pl.lit(False)
        if g.get("values"):
            hit = hit | pl.col("t").is_in([str(v) for v in g["values"]])
        for lo, hi in g.get("ranges") or []:
            hit = hit | pl.col("n").is_between(_num(lo), _num(hi))
        expr = pl.when(hit.fill_null(False)).then(pl.lit(label)).otherwise(expr)
    out = pl.DataFrame({"t": text, "n": num}).select(pl.when(pl.col("t").is_null()).then(None).otherwise(expr)).to_series()
    if all(_num(v) is not None for v in [*labels, other]):
        out = out.cast(pl.Float64)
    return out


def check_name(name: str, existing: list[str]) -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise _bad("A field name starts with a letter and uses letters, digits, spaces, dots, dashes or underscores "
                   "(at most 63 characters)")
    if name.lower() in {n.lower() for n in existing}:
        raise _bad(f"There is already a field named {name!r}")
    return name
