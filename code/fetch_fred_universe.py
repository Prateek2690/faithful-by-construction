"""
fetch_fred_universe.py
======================
Builds a PUBLIC-DATA asset universe for the study from FRED (no API key).

Each asset is a representative bond for one (rating, maturity) cell:
    yield     = US Treasury constant-maturity yield (maturity m)       [FRED DGSm]
              + ICE BofA US corporate option-adjusted spread (rating r) [FRED BAML...]
    duration  = modified duration of a par bond with coupon = yield, maturity m
    quality   = rating mapped to [0,1]  (UST/AAA = 1.0 ... CCC = 0.25)
8 credit buckets (UST, AAA, AA, A, BBB, BB, B, CCC) x 8 maturities = 64 assets
per observation date. Several month-end dates are kept so spreads/curve shape
vary across instances.

Honest description for the paper: "asset universes calibrated to public FRED
data (US Treasury constant-maturity yields and ICE BofA option-adjusted
spreads by rating) on K month-end dates"; assets are rating x maturity
representative bonds, not individual CUSIPs.

Usage:  python fetch_fred_universe.py --dates 6 --end 2026-10-14 --out fred_universe.csv
        (--end pins the last observation date; the paper uses 2026-10-14)
Writes fred_universe.csv (the universe) and fred_raw.csv (raw series, provenance).
"""
import argparse, io, sys, time
import numpy as np

try:
    import pandas as pd, requests
except ImportError:
    sys.exit("pip install pandas requests")

TSY = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5", 7: "DGS7",
       10: "DGS10", 20: "DGS20", 30: "DGS30"}
OAS = {"AAA": "BAMLC0A1CAAA", "AA": "BAMLC0A2CAA", "A": "BAMLC0A3CA",
       "BBB": "BAMLC0A4CBBB", "BB": "BAMLH0A1HYBB", "B": "BAMLH0A2HYB",
       "CCC": "BAMLH0A3HYC"}
QUALITY = {"UST": 1.00, "AAA": 1.00, "AA": 0.90, "A": 0.80, "BBB": 0.65,
           "BB": 0.50, "B": 0.38, "CCC": 0.25}


def fetch(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", series_id]
            df["date"] = pd.to_datetime(df["date"])
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            return df.set_index("date")[series_id]
        except Exception as e:
            last = e; time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {series_id}: {last}")


def par_mod_duration(y: float, m: float) -> float:
    """Modified duration of an annual-pay par bond (coupon = yield)."""
    if y <= 1e-9:
        return m
    return (1 - (1 + y) ** (-m)) / y


def build_universe(raw: pd.DataFrame, n_dates: int) -> pd.DataFrame:
    month_end = raw.dropna().resample("ME").last().dropna()
    if len(month_end) == 0:
        raise RuntimeError("no dates where all series are available")
    month_end = month_end.iloc[-n_dates:]
    rows = []
    for date, obs in month_end.iterrows():
        for bucket in ["UST"] + list(OAS):
            spread = 0.0 if bucket == "UST" else obs[OAS[bucket]]
            for m, sid in TSY.items():
                y_pct = obs[sid] + spread
                rows.append({"date": date.strftime("%Y-%m-%d"),
                             "asset": f"{bucket}_{m}y", "rating": bucket,
                             "maturity_y": m, "ytm": round(y_pct, 4),
                             "duration": round(par_mod_duration(y_pct / 100, m), 4),
                             "quality": QUALITY[bucket]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=6, help="number of month-end dates")
    ap.add_argument("--end", default=None, help="last observation date to use, YYYY-MM-DD (paper: 2026-10-14)")
    ap.add_argument("--out", default="fred_universe.csv")
    a = ap.parse_args()
    series = {}
    for sid in list(TSY.values()) + list(OAS.values()):
        print(f"downloading {sid} ...", flush=True)
        series[sid] = fetch(sid)
    raw = pd.DataFrame(series)
    if a.end:
        raw = raw.loc[:a.end]
    raw.to_csv("fred_raw.csv")
    uni = build_universe(raw, a.dates)
    uni.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(uni)} rows, {uni['date'].nunique()} dates "
          f"({uni['date'].min()} .. {uni['date'].max()}), "
          f"{uni.groupby('date').size().iloc[0]} assets per date")
    print(uni[uni["date"] == uni["date"].max()].pivot(index="rating", columns="maturity_y", values="ytm")
          .reindex(["UST"] + list(OAS)).round(2).to_string())
    print("\nraw series saved to fred_raw.csv (provenance)")
