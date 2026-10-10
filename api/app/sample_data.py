"""Synthetic mixed-type dataset with realistic warts and learnable structure.

Every feature is driven by a handful of shared latent factors, so there are real
correlation clusters, and the two targets depend on those factors (plus some
categorical / text effects) so the ClassifierChain has signal to find:

    column_a  <- latent factors + region + review sentiment
    column_b  <- latent factors + column_a          (the chain dependency)

Warts included on purpose: missingness (MCAR and MAR), skew and outliers, integer
IDs, a UUID column, numbers stored as text, dates stored as text in two formats,
high-cardinality categoricals, free text, a constant column, a 97%-empty column
and a near-copy of column_a (target leakage) for the UI to flag.

CLI:  python -m app.sample_data --rows 50000 --out sample.csv
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

N_LATENT = 6

POS_WORDS = "great excellent love fast reliable helpful friendly easy smooth recommend happy perfect quality amazing".split()
NEG_WORDS = "slow broken terrible rude refund confusing expensive late disappointed awful buggy cancel poor waiting".split()
NEUTRAL_WORDS = (
    "account order service product delivery support team app price plan update store call email "
    "package week month item website billing manager option feature"
).split()
REGIONS = ["north", "south", "east", "west", "central", "overseas"]
STATES = [f"S{i:02d}" for i in range(50)]
OCCUPATIONS = [
    "engineer", "teacher", "nurse", "sales", "manager", "clerk", "driver", "artist", "chef", "analyst",
    "student", "retired", "lawyer", "doctor", "technician", "consultant", "farmer", "pilot", "writer",
    "designer", "mechanic", "scientist", "banker", "agent", "other",
]
TIERS = ["bronze", "silver", "gold", "platinum"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _with_missing(rng, values: np.ndarray | list, frac: float, mask_extra: np.ndarray | None = None) -> list:
    arr = list(values)
    n = len(arr)
    miss = rng.random(n) < frac
    if mask_extra is not None:
        miss |= mask_extra
    for i in np.flatnonzero(miss):
        arr[i] = None
    return arr


def _numeric_missing(rng, x: np.ndarray, frac: float, mar: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(float).copy()
    miss = rng.random(x.size) < frac
    if mar is not None:
        miss |= mar
    x[miss] = np.nan
    return x


def _texts(rng, sentiment: np.ndarray, min_words: int, max_words: int) -> list[str]:
    """Sentences whose positive/negative word mix tracks `sentiment` in [0, 1]."""
    n = sentiment.size
    lengths = rng.integers(min_words, max_words + 1, size=n)
    out: list[str] = []
    pos = np.array(POS_WORDS)
    neg = np.array(NEG_WORDS)
    neu = np.array(NEUTRAL_WORDS)
    u = rng.random((n, max_words))
    pick = rng.integers(0, 10_000, size=(n, max_words))
    for i in range(n):
        p = sentiment[i]
        words = []
        for k in range(lengths[i]):
            r = u[i, k]
            if r < 0.45:
                words.append(neu[pick[i, k] % neu.size])
            elif r < 0.45 + 0.55 * p:
                words.append(pos[pick[i, k] % pos.size])
            else:
                words.append(neg[pick[i, k] % neg.size])
        s = " ".join(words)
        out.append(s[0].upper() + s[1:] + ".")
    return out


def generate(n_rows: int = 50_000, n_fields: int = 150, seed: int = 42, target_kind: str = "mixed") -> pl.DataFrame:
    """Build the synthetic frame. `target_kind`: mixed | binary | multiclass."""
    rng = np.random.default_rng(seed)
    n = n_rows
    Z = rng.standard_normal((n, N_LATENT))
    cols: dict[str, object] = {}

    # ---- named, realistic block --------------------------------------------------
    region_idx = np.clip(((Z[:, 3] + 2.5) / 5 * len(REGIONS)).astype(int), 0, len(REGIONS) - 1)
    region_idx = np.where(rng.random(n) < 0.3, rng.integers(0, len(REGIONS), n), region_idx)
    region_effect = np.array([0.6, -0.4, 0.2, -0.2, 0.0, 0.9])[region_idx]
    sentiment = _sigmoid(1.4 * Z[:, 0] + 0.5 * rng.standard_normal(n))

    cols["customer_id"] = np.arange(100_001, 100_001 + n)
    cols["account_uuid"] = [str(uuid.UUID(int=int(x))) for x in rng.integers(0, 2**63, size=n, dtype=np.int64)]
    base = datetime(2018, 1, 1)
    signup_days = np.clip(1500 + 400 * Z[:, 1] + 200 * rng.standard_normal(n), 0, 2900).astype(int)
    cols["signup_date"] = _with_missing(
        rng, [(base + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in signup_days], 0.02
    )
    login_secs = rng.integers(0, 365 * 86400, size=n)
    cols["last_login_ts"] = [datetime(2025, 1, 1) + timedelta(seconds=int(s)) for s in login_secs]
    cols["birth_date_us"] = [
        (datetime(1950, 1, 1) + timedelta(days=int(d))).strftime("%m/%d/%Y")
        for d in rng.integers(0, 365 * 50, size=n)
    ]
    age = np.clip(np.round(45 + 14 * Z[:, 1] + 5 * rng.standard_normal(n)), 18, 90)
    cols["age"] = _numeric_missing(rng, age, 0.03)
    income = np.exp(10.8 + 0.5 * Z[:, 2] + 0.3 * Z[:, 0] + 0.25 * rng.standard_normal(n))
    cols["annual_income"] = _numeric_missing(rng, income, 0.06, mar=(Z[:, 1] > 1.6) & (rng.random(n) < 0.5))
    cols["credit_score"] = np.clip(np.round(680 + 60 * Z[:, 0] + 30 * rng.standard_normal(n)), 300, 850).astype(int)
    cols["num_dependents"] = rng.poisson(np.exp(0.2 + 0.3 * Z[:, 1].clip(-2, 2)))
    cols["tenure_months"] = np.clip(np.round(signup_days / 30 + rng.normal(0, 2, n)), 0, None).astype(int)
    cols["satisfaction_1to5"] = np.clip(np.round(3 + 1.1 * Z[:, 0] + 0.6 * rng.standard_normal(n)), 1, 5).astype(int)
    cols["region"] = _with_missing(rng, [REGIONS[i] for i in region_idx], 0.01)
    cols["state"] = [STATES[i] for i in rng.zipf(1.3, n) % len(STATES)]
    occ = np.clip(((Z[:, 2] + 3) / 6 * len(OCCUPATIONS)).astype(int), 0, len(OCCUPATIONS) - 1)
    occ = np.where(rng.random(n) < 0.5, rng.integers(0, len(OCCUPATIONS), n), occ)
    cols["occupation"] = _with_missing(rng, [OCCUPATIONS[i] for i in occ], 0.08)
    tier = np.digitize(Z[:, 2] + 0.5 * Z[:, 0], [-0.8, 0.4, 1.4])
    cols["product_tier"] = [TIERS[i] for i in tier]
    cols["is_premium"] = _with_missing(rng, np.where(tier >= 2, "Y", "N").tolist(), 0.02)
    cols["has_mobile_app"] = (Z[:, 4] + rng.standard_normal(n) > 0).tolist()
    cols["referral_code"] = [f"REF{c:05d}" for c in rng.zipf(1.15, n) % 3000]
    cols["customer_review"] = _with_missing(rng, _texts(rng, sentiment, 6, 28), 0.05)
    cols["support_notes"] = _with_missing(
        rng, _texts(rng, _sigmoid(0.8 * Z[:, 0] - 0.5 + 0.3 * rng.standard_normal(n)), 4, 18), 0.40
    )
    cols["schema_version"] = ["v1"] * n
    cols["legacy_score"] = _numeric_missing(rng, rng.normal(50, 10, n), 0.97)
    cols["unit_price_text"] = _with_missing(
        rng, [f"{v:.2f}" for v in np.exp(3 + 0.6 * Z[:, 2] + 0.2 * rng.standard_normal(n))], 0.02
    )

    # ---- generated blocks -----------------------------------------------------------
    remaining = max(n_fields - len(cols) - 3, 0)  # -3: two targets + leaky feature
    mix = {"cont": 0.42, "disc": 0.20, "cat": 0.22, "flag": 0.09, "txt": 0.03, "dt": 0.04}
    counts = {k: int(round(v * remaining)) for k, v in mix.items()}
    counts["cont"] += remaining - sum(counts.values())

    def latent_mix(k_factors: int = 2) -> np.ndarray:
        idx = rng.choice(N_LATENT, size=k_factors, replace=False)
        w = rng.normal(0, 1, size=k_factors)
        return Z[:, idx] @ w / np.sqrt((w**2).sum()) + rng.uniform(0.3, 1.5) * rng.standard_normal(n)

    miss_choices = [0, 0, 0, 0, 0.005, 0.02, 0.05, 0.15, 0.35]
    for i in range(counts["cont"]):
        base_sig = latent_mix()
        kind = rng.choice(["normal", "lognormal", "heavy", "bounded", "outliers"])
        if kind == "normal":
            x = rng.uniform(-50, 200) + rng.uniform(1, 40) * base_sig
        elif kind == "lognormal":
            x = np.exp(rng.uniform(0, 4) + rng.uniform(0.4, 1.2) * base_sig)
        elif kind == "heavy":
            x = base_sig + rng.standard_t(2.5, n)
        elif kind == "bounded":
            x = 100 * _sigmoid(base_sig)
        else:
            x = base_sig * 10
            spikes = rng.random(n) < 0.02
            x[spikes] *= rng.uniform(8, 30)
        cols[f"cont_{i:03d}"] = _numeric_missing(rng, x, float(rng.choice(miss_choices)))
    for i in range(counts["disc"]):
        sig = latent_mix()
        kind = rng.choice(["poisson", "ordinal", "count_wide"])
        if kind == "poisson":
            x = rng.poisson(np.exp(0.5 + 0.6 * sig.clip(-3, 3)))
        elif kind == "ordinal":
            x = np.digitize(sig, [-1.2, -0.4, 0.4, 1.2]) + 1
        else:
            x = np.round(np.exp(4 + 0.7 * sig.clip(-3, 3))).astype(int)
        frac = float(rng.choice(miss_choices))
        cols[f"disc_{i:03d}"] = _numeric_missing(rng, x, frac) if frac else x.astype(int)
    for i in range(counts["cat"]):
        sig = latent_mix()
        k = int(rng.choice([3, 4, 6, 8, 12, 20, 40, 150, 600]))
        if k <= 40:
            codes = np.clip(((sig + 3) / 6 * k).astype(int), 0, k - 1)
            codes = np.where(rng.random(n) < 0.35, rng.integers(0, k, n), codes)
        else:
            codes = (rng.zipf(1.25, n) + (sig > 0.5) * 7) % k
        prefix = "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz"), size=3))
        vals = [f"{prefix}_{c:03d}" for c in codes]
        cols[f"cat_{i:03d}"] = _with_missing(rng, vals, float(rng.choice(miss_choices)))
    for i in range(counts["flag"]):
        sig = latent_mix()
        cols[f"flag_{i:03d}"] = (sig > rng.uniform(-1, 1.5)).astype(int)
    for i in range(counts["txt"]):
        s_ = _sigmoid(latent_mix())
        cols[f"txt_{i:03d}"] = _with_missing(rng, _texts(rng, s_, 5, 20), float(rng.choice(miss_choices)))
    for i in range(counts["dt"]):
        sig = latent_mix()
        days = np.clip(1000 + 300 * sig, 0, 3000).astype(int)
        dates = [(base + timedelta(days=int(d))) for d in days]
        cols[f"dt_{i:03d}"] = [d.strftime("%Y-%m-%d") for d in dates] if i % 2 == 0 else dates

    # ---- targets ------------------------------------------------------------------
    logit_a = 1.1 * Z[:, 0] - 0.8 * Z[:, 1] + 0.5 * Z[:, 4] + region_effect + 1.2 * (sentiment - 0.5)
    logit_a += 0.6 * rng.standard_normal(n)
    score_b = 0.9 * Z[:, 2] + 0.7 * Z[:, 0] - 0.5 * Z[:, 5] + 0.6 * rng.standard_normal(n)

    if target_kind == "multiclass":
        a = np.digitize(logit_a, np.quantile(logit_a, [0.33, 0.66]))
        column_a = np.array(["low", "mid", "high"])[a]
        score_b += 0.8 * a
    else:
        a = (logit_a > np.quantile(logit_a, 0.6)).astype(int)
        column_a = a
        score_b += 1.2 * a
    if target_kind == "binary":
        column_b = (score_b > np.quantile(score_b, 0.5)).astype(int)
    else:
        b = np.digitize(score_b, np.quantile(score_b, [0.4, 0.75]))
        column_b = np.array(["bronze", "silver", "gold"])[b]

    flip = rng.random(n) < 0.005  # rare enough that the flag clears the 0.95 leak threshold with room to spare
    leak_base = a if target_kind != "multiclass" else (a >= 1).astype(int)
    cols["leaky_outcome_flag"] = np.where(flip, 1 - leak_base, leak_base)
    cols["column_a"] = column_a
    cols["column_b"] = column_b

    # Shuffle feature order (targets stay last) so types are interleaved like real data
    names = [c for c in cols if c not in ("column_a", "column_b")]
    order = list(rng.permutation(len(names)))
    ordered = {names[i]: cols[names[i]] for i in order}
    ordered["column_a"] = cols["column_a"]
    ordered["column_b"] = cols["column_b"]
    return pl.DataFrame({k: (v if isinstance(v, list) else np.asarray(v)) for k, v in ordered.items()},
                        strict=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--fields", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-kind", choices=["mixed", "binary", "multiclass"], default="mixed")
    ap.add_argument("--out", type=Path, default=Path("sample.csv"))
    args = ap.parse_args()
    df = generate(args.rows, args.fields, args.seed, args.target_kind)
    if args.out.suffix == ".parquet":
        df.write_parquet(args.out)
    else:
        df.write_csv(args.out)
    print(f"wrote {args.out}: {df.height:,} rows x {df.width} columns")


if __name__ == "__main__":
    main()
