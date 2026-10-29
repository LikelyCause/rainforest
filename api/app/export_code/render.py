"""Write an export's script: the header, the decisions as code someone can read and edit, the driver (`script.py`)
and the app functions it runs (`bundle`), then what the app read and exported, for the script's checks.

Everything that comes from people (field names, notes, dataset and experiment names, values) is written as a Python
literal with repr() or as a comment line by line, never into a docstring or code, so no name or note can turn into
code in the script.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import keyword
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..semantics import DEFAULT_PARAMS, NUMERIC, PARAM_RULES
from ..transform import FieldSpec
from . import script as driver
from .bundle import Bundle

DRIVER = "app.export_code.script"
DRIVER_ENTRIES = (
    "main",
    "prepare",
    "load_clean",
    "load_model_ready",
    "xy",
    "keep",
    "predict",
    "read_source",
    "compare_files",
)
# The derived-field parser answers malformed formulas with an HTTP 422; a script raises a plain error.
OVERRIDES = {
    ("app.derived", "_bad"): '''def _bad(msg: str) -> ValueError:
    """A malformed derived field (the app answers these with an HTTP 422)."""
    return ValueError(msg)
'''
}
# Pinned with the rest though only reached through others: they decide numbers too (scikit-learn's solvers and
# thread pools), so a script runs them at the app's versions.
ALSO_PINNED = ("scipy", "joblib", "threadpoolctl")
# Polars reads the Concrete strength archive's .xls with it. The app doesn't install it (sample_library/build.py needs
# it only to rebuild the library), so this is the version the rebuild was checked against: identical to the app's file.
FASTEXCEL = "fastexcel==0.21.0"
WIDTH = 118


@dataclass
class ExportInfo:
    """Everything a script is written from (`gather` reads it from the database)."""

    export_id: int
    dataset_name: str
    family: str
    specs: list[FieldSpec]
    targets: dict[str, dict]  # chain order -> {"task", "bins", "decimal_comma"}
    row_rules: dict[str, Any]
    split: dict[str, Any]
    fit_sample_rows: int | None
    chunk_rows: int
    source: dict[str, Any]  # {"kind": "upload"|"library"|"generated", ...}
    fit_mb: int | None = None  # the fit sample's memory budget (None: no cap beyond fit_sample_rows)
    chunk_mb: int | None = None  # the transform's batches' memory budget (None: chunk_rows as given)
    derived: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)  # field -> the team's notes
    adjusted: dict[str, str] = field(default_factory=dict)  # field -> what the experiment changed
    targets_changed: str | None = None
    left_out: list[tuple[str, str, str]] = field(default_factory=list)  # (field, type, why)
    expected_input: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] = field(default_factory=dict)
    baseline: str | None = None
    experiment: str | None = None
    adjustments: int = 0
    created_at: str | None = None
    created_by: str | None = None
    split_counts: dict[str, int] | None = None
    n_features: int | None = None
    made_at: str | None = None
    app_version: str | None = None
    warnings: list[str] = field(default_factory=list)


def file_name(info: ExportInfo) -> str:
    slug = re.sub(
        r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", info.dataset_name).encode("ascii", "ignore").decode().lower()
    ).strip("_")[:40]
    return f"export{info.export_id}" + (f"_{slug}" if slug else "") + ".py"


def bundle_for(info: ExportInfo) -> Bundle:
    b = Bundle(provided=driver.PROVIDED, overrides=OVERRIDES, provider=DRIVER)
    for name in DRIVER_ENTRIES:
        b.add(DRIVER, name)
    kind = info.source["kind"]
    b.add(
        *{
            "upload": (DRIVER, "read_upload"),
            "library": (DRIVER, "read_library"),
            "generated": ("app.sample_data", "generate"),
        }[kind]
    )
    return b


def render(info: ExportInfo) -> str:
    b = bundle_for(info)
    name = file_name(info)
    pins = b.requirements() + _pins(ALSO_PINNED) + ([FASTEXCEL] if _needs_excel(info) else [])
    parts = [
        _header(info, name, sorted(set(pins))),
        "from __future__ import annotations\n\n" + "\n".join(b.import_lines()),
        _section("The data"),
        _data(info, name),
        _section("The decisions"),
        _decisions(info),
        _section("Rebuilding the export: Rainforest's own code, copied from the app (no need to edit below here)"),
        b.body(),
        _section("What the app read and exported, for the checks"),
        f"EXPECTED_INPUT = {_literal(info.expected_input)}\n\nEXPECTED_OUTPUT = {_literal(info.expected_output)}",
        'if __name__ == "__main__":\n    sys.exit(main())',
    ]
    return _no_stray_blocks("\n\n\n".join(p.strip("\n") for p in parts) + "\n")


def _no_stray_blocks(text: str) -> str:
    """Indent any top-level comment line after the header's own that reads like PEP 723's `# ///` marker: one would
    extend or duplicate the script's metadata block, and `uv run` would refuse the file."""
    lines = text.split("\n")
    end = lines.index("# ///")
    for i in range(end + 1, len(lines)):
        if lines[i] == "# ///" or lines[i].startswith("# /// "):
            lines[i] = "#  " + lines[i][2:]
    return "\n".join(lines)


# ---- sections -------------------------------------------------------------------------------------------------------
def _header(info: ExportInfo, name: str, pins: list[str]) -> str:
    deps = "\n".join(f'#     "{p}",' for p in pins)
    rows = sum((info.split_counts or {}).values()) if info.split_counts else None
    exported = " ".join(
        filter(
            None,
            [
                info.created_at,
                f"by {info.created_by}" if info.created_by else None,
                f"· {rows:,} rows" if rows is not None else None,
                "({train:,} train, {valid:,} valid, {test:,} test)".format(**info.split_counts)
                if info.split_counts
                else None,
                f"· {info.n_features:,} features" if info.n_features is not None else None,
            ],
        )
    )
    baseline = info.baseline or "?"
    if info.experiment:
        baseline += f", experiment {info.experiment}"
    if info.adjustments:
        baseline += f" ({info.adjustments} field{'s' if info.adjustments != 1 else ''} adjusted)"
    src = info.source
    data = {
        "upload": f"{info.dataset_name} ({src.get('file') or 'uploaded file'})",
        "library": f"{info.dataset_name} (the app's sample from {src.get('source_url')})",
        "generated": f"{info.dataset_name} (made by the app's sample generator)",
    }[src["kind"]]
    arg = {"upload": " DATA_FILE", "library": " [DATA_FILE]", "generated": ""}[src["kind"]]
    lines = [
        f"Rainforest export #{info.export_id}, as a script: it rebuilds the export's model-ready data outside the app, "
        "from the original data and the decisions below, and checks the result against the app's.",
        "",
        f"  Dataset    {_one_line(data)}",
        f"  Baseline   {_one_line(baseline)}",
        f"  Exported   {_one_line(exported)}" if exported else None,
        "  Script     written " + " ".join(filter(None, [info.made_at, "by Rainforest", info.app_version])),
        "",
        "Run it with uv, which installs the versions listed above (or in an environment that has them):",
        f"  uv run {name}{arg}",
        "Options: --out model_ready.parquet writes the result (as the app's model_ready.parquet); --clean clean.parquet "
        "writes the rows with the decisions applied and nothing fitted; --compare FILE compares the result value by "
        f"value with the app's export{info.export_id}_model_ready.parquet; --no-check goes on when the data doesn't "
        "read as the app read it (the result is still checked).",
        "Exit status: 0 the result matches the app's export, 1 it differs, 2 the command line is wrong, 3 the data "
        "doesn't read as the app read it, 4 the rebuild failed.",
        "Or from Python, next to this file:",
        f"  from {name[:-3]} import load_model_ready, load_clean, xy",
        f"  data = load_model_ready({'DATA_FILE' if src['kind'] == 'upload' else ''})",
        '  X_train, y_train = xy(data, "train")',
    ]
    lines = [ln for ln in lines if ln is not None]
    if src["kind"] == "upload":
        lines += [
            "",
            "DATA_FILE is the file that was uploaded to Rainforest; the script checks that it reads the same.",
        ]
    elif src["kind"] == "library":
        lines += [
            "",
            "Without a file it downloads the dataset from its public source and rebuilds it as the app's "
            "copy was built; or give it that copy (api/app/sample_library/data/ in Rainforest's code).",
        ]
    else:
        lines += ["", "It makes the same data the app made, with the app's generator and the same seed."]
    lines += [f"Note: {_one_line(w)}" for w in info.warnings]
    return (
        '# /// script\n# requires-python = ">=3.10,<3.14"\n# dependencies = [\n'
        + deps
        + "\n# ]\n# ///\n#\n"
        + "\n".join(_comment(ln, wrap=True) for ln in lines)
    )


def _data(info: ExportInfo, name: str) -> str:
    src = info.source
    script = {"export_id": info.export_id, "file": name}
    if src["kind"] == "upload":
        how = (
            f"# The file read as the app read it when it was uploaded ({src['format']}).\n"
            f"def raw_frame(path=None):\n    return read_upload(path, {src['format']!r})"
        )
    elif src["kind"] == "library":
        how = (
            "# The sample dataset, from the copy given or rebuilt from its public source.\n"
            f"def raw_frame(path=None):\n    return read_library(path, {src['key']!r})"
        )
    else:
        s = src["settings"]
        how = (
            "# The app's synthetic sample, made again with the settings it was made with.\n"
            "def raw_frame(path=None):\n"
            "    if path is not None:\n"
            "        raise UsageError(\"This export's data is the app's generated sample, made again here: it takes "
            'no file.")\n'
            f"    return generate(n_rows={s['rows']!r}, n_fields={s['fields']!r}, seed={s['seed']!r}, "
            f"target_kind={s['target_kind']!r}, numeric_target={bool(s.get('numeric_target'))!r})"
        )
    derived = (
        "# Fields the app computed from other fields, in the order they were added (only the ones this export "
        "uses).\nDERIVED = " + _literal(info.derived)
    )
    return "\n\n".join([f"SCRIPT = {_literal(script)}", f"SOURCE = {_literal(src)}", how, derived])


def _decisions(info: ExportInfo) -> str:
    out = [
        _comment(
            f"{info.baseline or 'The baseline'}"
            + (f" as experiment {info.experiment} adjusted it" if info.adjustments else "")
            + ". Edit them to try other choices: the script then builds what the app would export with them, "
            "and its check against the app's export says what changed.",
            wrap=True,
        )
    ]
    out.append(f"FAMILY = {info.family!r}  # the model family the features are prepared for")
    calls = []
    for spec in info.specs:
        for label, text in (
            ("adjusted in this experiment", info.adjusted.get(spec.name)),
            ("notes", info.notes.get(spec.name)),
        ):
            if text:
                calls.append(_comment(f"{label}: {text}", indent=8, wrap=True))
        calls.append(" " * 8 + field_call(spec) + ",")
    out.append(
        'def kept_fields():\n    """The kept fields in column order, and how each is prepared."""\n    return [\n'
        + "\n".join(calls)
        + "\n    ]"
    )
    if info.left_out:
        width = min(max(len(n) for n, _, _ in info.left_out), 40)
        out.append(
            _comment("Left out:")
            + "\n"
            + "\n".join(_comment(f"  {n:<{width}}  {t} · {why}") for n, t, why in info.left_out)
        )
    tlines = []
    if info.targets_changed:
        tlines.append(_comment(f"targets changed in this experiment: {info.targets_changed}", indent=8, wrap=True))
    for t, task in info.targets.items():
        for label, text in (("adjusted in this experiment", info.adjusted.get(t)), ("notes", info.notes.get(t))):
            if text:
                tlines.append(_comment(f"{label}: {text}", indent=8, wrap=True))
        tlines.append(f"        {t!r}: {target_call(task)},")
    out.append(
        'def targets():\n    """The targets in chain order, and how each is predicted."""\n    return {\n'
        + "\n".join(tlines)
        + "\n    }"
    )
    out.append(
        "\n".join(
            [
                "# Rows the export leaves out before it splits, and the field whose rows stay in one split.",
                f"ROW_RULES = {_literal(info.row_rules)}",
                "# The shares of the labelled rows held out for validation and test, and the seed that drew them.",
                f"SPLIT = {_literal(info.split)}",
                "# The preprocessing (fill values, categories, scalers) is fitted on at most this many training rows (Max rows",
                "# to fit on), drawn with the seed, and on fewer when the features wouldn't fit in FIT_MB megabytes, as the app",
                "# does. None fits on every training row; FIT_MB None sets no memory budget.",
                f"FIT_SAMPLE_ROWS = {info.fit_sample_rows!r}",
                f"FIT_MB = {info.fit_mb!r}",
                "# Rows transformed at a time (fewer when they wouldn't fit in CHUNK_MB megabytes), as the app did; it only",
                "# matters to a date field whose format the app didn't record, which each batch then works out for itself.",
                f"CHUNK_ROWS = {info.chunk_rows!r}",
                f"CHUNK_MB = {info.chunk_mb!r}",
            ]
        )
    )
    return "\n\n\n".join(out)


def _section(title: str) -> str:
    return f"# {'═' * (WIDTH - 2)}\n# {title}\n# {'═' * (WIDTH - 2)}"


# ---- the decisions as calls -----------------------------------------------------------------------------------------
def field_call(spec: FieldSpec) -> str:
    """`keep.<type>(...)` with what differs from that builder's defaults, if it rebuilds exactly this spec; else
    `keep.field(...)` with every setting."""
    t = spec.semantic_type
    candidates: list[tuple[str, dict[str, Any]]] = []
    if t in NUMERIC:
        candidates.append(
            (
                "numeric",
                {
                    "discrete": t == "numeric_discrete",
                    "imputation": spec.imputation,
                    "impute_value": spec.impute_value,
                    "add_missing_indicator": spec.add_missing_indicator,
                    "transform": spec.transform,
                    "scaling": spec.scaling,
                    "decimal_comma": spec.decimal_comma,
                },
            )
        )
    elif t == "categorical":
        candidates += [
            (
                "categorical",
                {"encoding": spec.encoding, "imputation": spec.imputation, "impute_value": spec.impute_value, **p},
            )
            for p in _param_args(spec)
        ]
    elif t == "binary":
        candidates.append(("binary", {"imputation": spec.imputation, "impute_value": spec.impute_value}))
    elif t == "text":
        candidates += [
            ("text", {"encoding": spec.encoding, "impute_value": spec.impute_value, **p}) for p in _param_args(spec)
        ]
    elif t == "datetime":
        candidates.append(
            (
                "datetime",
                {"datetime_format": spec.datetime_format, "imputation": spec.imputation, "scaling": spec.scaling},
            )
        )
    for builder, kwargs in candidates:
        fn = getattr(driver.keep, builder)
        defaults = _defaults(fn)
        args = {k: v for k, v in kwargs.items() if k not in defaults or defaults[k] != v or k == "encoding"}
        try:
            built = fn(spec.name, **args)
        except TypeError:
            continue
        if built == spec:
            return (
                f"keep.{builder}("
                + ", ".join([repr(spec.name)] + [f"{k}={_literal(v)}" for k, v in args.items()])
                + ")"
            )
    every = {k: v for k, v in spec.__dict__.items() if k not in ("name", "semantic_type")}
    return (
        "keep.field(" + ", ".join([repr(spec.name), repr(t)] + [f"{k}={_literal(v)}" for k, v in every.items()]) + ")"
    )


def _param_args(spec: FieldSpec) -> list[dict[str, Any]]:
    """Ways to pass a spec's encoding parameters, shortest first: none (the defaults), the ones that differ from the
    defaults, or all of them as `params`."""
    params = spec.params or {}
    default = DEFAULT_PARAMS.get(spec.encoding, {})
    differ = {k: v for k, v in params.items() if default.get(k, object()) != v}
    out: list[dict[str, Any]] = [{}]
    # Keyword arguments only for the settings the encoding reads; anything else goes in `params`, so no stored key
    # can be a Python keyword or a look-alike that Python reads as another name.
    known = set(default) | set(PARAM_RULES.get(spec.encoding, {}))
    if differ and all(
        k in known
        and k.isascii()
        and k.isidentifier()
        and not keyword.iskeyword(k)
        and k not in ("name", "encoding", "imputation", "impute_value", "params")
        for k in differ
    ):
        out.append(differ)
    out.append({"params": dict(params)})
    return out


def _defaults(fn) -> dict[str, Any]:
    return {
        k: p.default for k, p in inspect.signature(fn).parameters.items() if p.default is not inspect.Parameter.empty
    }


def target_call(task: dict) -> str:
    """`predict.<how>(...)` if it rebuilds exactly this task, else the task as a dict."""
    comma = {"decimal_comma": True} if task.get("decimal_comma") else {}
    bins = task.get("bins") or {}
    options: list[tuple[str, tuple, dict]] = [("classes", (), {}), ("number", (), {})]
    if "positive" in bins:
        options.append(("one_vs_rest", (bins["positive"],), {}))
    if "k" in bins:
        options.append(("bands", (bins["k"],), {}))
    if "edges" in bins:
        options.append(("bands", (), {"edges": bins["edges"]}))
    for how, args, kwargs in options:
        kwargs = {**kwargs, **comma}
        if getattr(driver.predict, how)(*args, **kwargs) == task:
            return (
                f"predict.{how}("
                + ", ".join([_literal(a) for a in args] + [f"{k}={_literal(v)}" for k, v in kwargs.items()])
                + ")"
            )
    return _literal(task)


# ---- literals and comments ------------------------------------------------------------------------------------------
def _literal(value: Any, indent: int = 0) -> str:
    """A Python literal for JSON-like data, dicts in their order: on one line when it fits, else one item a line."""
    _check_finite(value)
    flat = repr(value)
    if len(flat) + indent <= WIDTH - 4 or not isinstance(value, (dict, list)) or not value:
        return flat
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        items = [f"{pad}{k!r}: {_literal(v, indent + 4)}," for k, v in value.items()]
        return "{\n" + "\n".join(items) + "\n" + " " * indent + "}"
    return "[\n" + "\n".join(f"{pad}{_literal(v, indent + 4)}," for v in value) + "\n" + " " * indent + "]"


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"can't write {value!r} as a literal")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_finite(k)
            _check_finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check_finite(v)


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x85  ]")


def _comment(text: str, indent: int = 0, wrap: bool = False) -> str:
    """`text` as comment lines: every line break (of any kind) starts a new comment line, so nothing in it can end
    the comment."""
    pad = " " * indent
    out = []
    for line in str(text).splitlines() or [""]:
        line = _CONTROL.sub(" ", line).rstrip()
        width = WIDTH - indent - 2
        chunks = _wrap(line, width) if wrap else [line]
        out += [f"{pad}# {c}".rstrip() for c in chunks]
    return "\n".join(out)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _wrap(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]
    lead = len(line) - len(line.lstrip())
    words, out, cur = line.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = " " * lead + w if lead else w
        else:
            cur = f"{cur} {w}" if cur else w
    return out + [cur]


def _pins(packages: tuple[str, ...]) -> list[str]:
    out = []
    for p in packages:
        try:
            out.append(f"{p}=={importlib.metadata.version(p)}")
        except importlib.metadata.PackageNotFoundError:
            out.append(p)
    return out


def _needs_excel(info: ExportInfo) -> bool:
    return info.source["kind"] == "library" and info.source.get("key") == "concrete_strength"
