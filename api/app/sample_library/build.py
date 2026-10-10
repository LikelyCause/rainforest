"""Rebuild the sample library's Parquet files from their public sources:
    python -m app.sample_library.build [key ...]
Downloads each UCI archive, reads it the way its file is laid out, and writes data/<key>.parquet. Concrete strength
ships as an .xls file, so rebuilding it needs `pip install fastexcel` (a build-time dependency only)."""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile

import polars as pl

from . import DATA_DIR, LIBRARY, path_of

ARCHIVES = {
    "diabetes_readmission": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "student_grades": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "student_dropout": "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip",
    "steel_faults": "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip",
    "concrete_strength": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
}
CONCRETE_COLUMNS = ["cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate",
                    "fine_aggregate", "age_days", "strength_mpa"]


def _zip(url: str) -> zipfile.ZipFile:
    with urllib.request.urlopen(url, timeout=120) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))


def _read(raw: bytes, **kw) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(raw.decode("utf-8-sig").encode()), infer_schema_length=None, **kw)


def build(key: str) -> pl.DataFrame:
    z = _zip(ARCHIVES[key])
    if key == "diabetes_readmission":
        # "?" marks a missing value; "None" in the lab-result columns means the test wasn't run, a real category.
        return _read(z.read("diabetic_data.csv"), null_values=["?"])
    if key == "student_grades":
        inner = zipfile.ZipFile(io.BytesIO(z.read("student.zip")))
        return _read(inner.read("student-por.csv"), separator=";")
    if key == "student_dropout":
        df = _read(z.read("data.csv"), separator=";")
        return df.rename({c: c.strip() for c in df.columns})
    if key == "steel_faults":
        names = z.read("Faults27x7_var").decode().split()
        return _read(z.read("Faults.NNA"), separator="\t", has_header=False, new_columns=names)
    if key == "concrete_strength":
        df = pl.read_excel(io.BytesIO(z.read("Concrete_Data.xls")))
        df = df.rename(dict(zip(df.columns, CONCRETE_COLUMNS)))
        return df.with_columns(pl.exclude("age_days").cast(pl.Float64), pl.col("age_days").cast(pl.Int64))
    raise KeyError(key)


def main(keys: list[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for key in keys or list(LIBRARY):
        df = build(key)
        missing = [t for t in LIBRARY[key]["targets"] if t not in df.columns]
        if missing:
            raise SystemExit(f"{key}: targets not in the data: {missing}")
        df.write_parquet(path_of(key), compression="zstd", compression_level=10)
        print(f"{key}: {df.height:,} rows x {df.width} columns, {path_of(key).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
