"""
public_data.py
==============
Seed the instance generator from a PUBLIC dataset instead of pure synthetic
sampling, so the paper can say instances are drawn from real market data.

This keeps the study fully reproducible and free of any proprietary data:
you point it at a public CSV (or a public API pull you've saved to CSV) and
it builds Instance objects whose asset universe is real, while the constraint
limits are still sampled to create a spread of slack/binding cases.

WHY THIS MATTERS FOR THE PAPER
------------------------------
Reviewers discount pure-synthetic universes. A universe drawn from public data
(real yields, durations, ratings) answers "is this realistic?" without any
employer/IP exposure. The optimization and the faithfulness test are unchanged
either way -- only the source of (y, d, q) differs.

EXPECTED CSV SCHEMA (rename/adapt columns via `column_map`)
-----------------------------------------------------------
    objective : per-asset objective coefficient (yield or expected return, as a
                fraction, e.g. 0.045 for 4.5%). If your file has percent (4.5),
                set `objective_is_percent=True`.
    risk      : a per-asset risk measure (duration, beta, vol...). Any positive
                scale; used for the "risk ceiling" constraint.
    quality   : a per-asset quality/rating score in [0,1], higher = better.
                If you have letter ratings, map them to [0,1] first (helper
                `rating_to_score` below).

PUBLIC SOURCES YOU CAN USE (save any of them to a CSV, then load)
-----------------------------------------------------------------
  * FRED (fredapi / CSV export): treasury & corporate yield curves.
  * Kaggle bond / S&P 500 / corporate-bond datasets.
  * yfinance / OpenBB pulls: equity returns + betas (for an equity instance).
  * Any academic public sample (e.g. Fama-French style) for returns.

USAGE
-----
    from public_data import load_universe, make_instance_from_universe
    uni = load_universe("bonds.csv",
                        column_map={"objective":"ytm","risk":"duration","quality":"rating01"})
    inst = make_instance_from_universe(uni, rng, n=40, tier="tight_duration")

If pandas or the file is unavailable, `load_universe` raises a clear error;
the synthetic generator in ofs_core remains the default so the study always
runs offline.
"""
from __future__ import annotations
import numpy as np

from ofs_core import Instance

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# Optional helper: letter ratings -> [0,1] quality score
# ---------------------------------------------------------------------------
_RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                 "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
                 "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]

def rating_to_score(rating: str) -> float:
    """Map an S&P-style letter rating to a [0,1] quality score (AAA=1, D~0)."""
    r = str(rating).strip().upper()
    if r not in _RATING_ORDER:
        return 0.5
    idx = _RATING_ORDER.index(r)
    return 1.0 - idx / (len(_RATING_ORDER) - 1)


# ---------------------------------------------------------------------------
# Load a public CSV into a universe dict of arrays
# ---------------------------------------------------------------------------
def load_universe(csv_path: str, column_map: dict | None = None,
                  objective_is_percent: bool = False) -> dict:
    """Read a public CSV and return {'y','d','q'} arrays.

    column_map maps our names -> your columns, e.g.
        {"objective": "ytm", "risk": "duration", "quality": "rating01"}
    If a "quality" column holds letter ratings, pass the letter column and set
    it up first with rating_to_score, or pre-compute a [0,1] column.
    """
    if pd is None:
        raise RuntimeError("pandas is required for public data loading "
                           "(pip install pandas).")
    cm = {"objective": "objective", "risk": "risk", "quality": "quality"}
    cm.update(column_map or {})
    df = pd.read_csv(csv_path)
    for need in cm.values():
        if need not in df.columns:
            raise ValueError(f"column '{need}' not in CSV; columns are "
                             f"{list(df.columns)}")
    y = df[cm["objective"]].to_numpy(dtype=float)
    if objective_is_percent:
        y = y / 100.0
    d = df[cm["risk"]].to_numpy(dtype=float)
    q = df[cm["quality"]].to_numpy(dtype=float)
    # basic hygiene
    good = np.isfinite(y) & np.isfinite(d) & np.isfinite(q)
    y, d, q = y[good], d[good], q[good]
    q = np.clip(q, 0.0, 1.0)
    if len(y) < 10:
        raise ValueError("universe has <10 usable assets after cleaning.")
    return {"y": y, "d": d, "q": q}


# ---------------------------------------------------------------------------
# Build an Instance by sampling n assets from a real universe
# ---------------------------------------------------------------------------
def make_instance_from_universe(uni: dict, rng: np.random.Generator,
                                n: int = 40, tier: str = "mixed") -> Instance:
    """Sample n assets from a real universe and set constraint limits so that
    slack/binding status varies by tier -- exactly like the synthetic
    generator, but with a real (y, d, q) universe.
    """
    N = len(uni["y"])
    idx = rng.choice(N, size=min(n, N), replace=False)
    y, d, q = uni["y"][idx], uni["d"][idx], uni["q"][idx]

    cap = float(rng.uniform(0.03, 0.08))
    dmean = float(d.mean())
    if tier == "tight_duration":
        D_max = dmean * rng.uniform(0.80, 0.95)
    elif tier == "loose_duration":
        D_max = dmean * rng.uniform(1.15, 1.40)
    else:
        D_max = dmean * rng.uniform(0.90, 1.20)

    qmean = float(q.mean())
    if tier == "tight_quality":
        Q_min = qmean * rng.uniform(1.00, 1.15)
    else:
        Q_min = qmean * rng.uniform(0.60, 0.95)

    return Instance(y=y, d=d, q=q, cap=cap,
                    D_max=float(D_max), Q_min=float(Q_min))


# ---------------------------------------------------------------------------
# Demo: build a tiny public-style CSV offline so the path is testable here
# ---------------------------------------------------------------------------
def _write_demo_csv(path: str = "demo_universe.csv", n: int = 60, seed: int = 1):
    """Writes a small CSV that mimics a public bond export, so you can test the
    loader offline. Replace with a real download for the paper."""
    rng = np.random.default_rng(seed)
    ytm = rng.uniform(3.5, 6.5, n)                       # percent
    dur = np.clip(2 + 0.6*(ytm-3.5) + rng.normal(0, 1.2, n), 0.5, 12)
    ratings = rng.choice(["AAA","AA","A","BBB","BB","B"], size=n,
                         p=[.05,.15,.30,.30,.15,.05])
    q01 = np.array([rating_to_score(r) for r in ratings])
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ytm","duration","rating","rating01"])
        for i in range(n):
            w.writerow([round(float(ytm[i]),3), round(float(dur[i]),2),
                        ratings[i], round(float(q01[i]),3)])
    return path


if __name__ == "__main__":
    # Offline demonstration of the full public-data path.
    if pd is None:
        print("pandas not installed; install it to use public data.")
    else:
        path = _write_demo_csv()
        uni = load_universe(path, column_map={"objective": "ytm",
                            "risk": "duration", "quality": "rating01"},
                            objective_is_percent=True)
        rng = np.random.default_rng(7)
        inst = make_instance_from_universe(uni, rng, n=30, tier="tight_duration")
        from ofs_core import solve_with_certificate
        cert = solve_with_certificate(inst)
        print(f"Loaded universe of {len(uni['y'])} real-style assets.")
        print(f"Built instance with {inst.n} assets; solved objective = "
              f"{cert.objective*100:.2f} (units of the objective).")
        print(f"Binding: {cert.binding}")
