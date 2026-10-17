"""
run_final_study.py  (v5 main experiment)
========================================
One command for the paper's main result.

  * Builds ONE fixed set of portfolio problems (paired design: every method
    explains the exact same problems), from the public FRED universe built by
    fetch_fred_universe.py (or synthetic with --universe synthetic).
  * Explainers:
        CG           certificate-grounded (reads the solver certificate)
        FF           free-form LLM: data + constraints + chosen weights
        FF_informed  free-form LLM: the above + realized avg duration, avg
                     quality, positions at cap (still NO duals)
        ALWAYS_CAP_DUR  trivial template baseline (always names cap+duration)
  * Metrics per explainer (unit-free first):
        hard-failure rate  = share of explanations naming >=1 constraint whose
                             exact re-solve effect is ZERO
        status faithfulness = share of named constraints that truly move the optimum
        recall             = share of truly-effective constraints that were named
        magnitude accuracy = among correctly named, bp within 35% (units stated)
        OFS (combined)     = status AND magnitude
  * Resumable: every reply is logged; re-running never re-bills a prompt.

Usage:
  python fetch_fred_universe.py --dates 6           # once (downloads public data)
  python run_final_study.py --dry_run               # free: checks pipeline, no API
  python run_final_study.py                         # the real run (asks to confirm)
  python run_final_study.py --models openai/gpt-4o-mini anthropic/claude-sonnet-4.5
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

from ofs_core import (generate_instance, solve_with_certificate, measured_shadow_price,
                      certificate_grounded_explanation, free_form_explanation,
                      score_explanation, is_degenerate, Claim, CONSTRAINTS, ZERO_TOL_BP)

TIERS = ["mixed", "tight_duration", "loose_duration", "tight_quality"]
LOG = "final_study_log.jsonl"   # overridden by --log


# ---------------------------------------------------------------- problems
def build_problems(n, seed, universe, fred_csv, require_slack=False, max_tries=20000):
    rng = np.random.default_rng(seed)
    unis, dates = None, None
    if universe == "fred":
        import pandas as pd
        from public_data import make_instance_from_universe
        if not os.path.exists(fred_csv):
            sys.exit(f"{fred_csv} not found -- run: python fetch_fred_universe.py --dates 6")
        df = pd.read_csv(fred_csv)
        dates = sorted(df["date"].unique())
        unis = {d: {"y": df[df.date == d]["ytm"].to_numpy() / 100.0,
                    "d": df[df.date == d]["duration"].to_numpy(),
                    "q": df[df.date == d]["quality"].to_numpy()} for d in dates}
    probs = []
    i = -1
    while True:
        i += 1
        if (require_slack and (len(probs) >= n or i >= max_tries)) or (not require_slack and i >= n):
            break
        tier = TIERS[i % 4]
        size = int(rng.integers(25, 60))
        if unis:
            date = dates[i % len(dates)]
            inst = make_instance_from_universe(unis[date], rng, n=size, tier=tier)
        else:
            date = None
            inst = generate_instance(rng, n=size, tier=tier)
        cert = solve_with_certificate(inst)
        if cert.status != "optimal":
            continue
        eff = {c: measured_shadow_price(inst, c)[0] for c in CONSTRAINTS}
        if require_slack and all(abs(eff[c]) > ZERO_TOL_BP for c in CONSTRAINTS):
            continue   # keep only problems where at least one rule has zero effect
        probs.append({"i": i, "tier": tier, "date": date, "inst": inst, "cert": cert,
                      "effect": eff, "degenerate": is_degenerate(inst, cert)})
    return probs


# ---------------------------------------------------------------- scoring
def score_one(p, claims):
    sc = score_explanation(p["inst"], claims, "x")
    effective = {c for c in CONSTRAINTS if abs(p["effect"][c]) > ZERO_TOL_BP}
    named = {cl.constraint for cl in claims}
    return {"i": p["i"], "named": sorted(named), "n_claims": len(claims), "empty": len(claims) == 0,
            "hard": sc.hard_failure, "status": sc.ofs_status if claims else None,
            "mag": sc.magnitude_acc, "ofs": sc.ofs if claims else None,
            "recall": (len(named & effective) / len(effective)) if effective else None}


def summarize(rows):
    def m(key):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return None, None
        v = np.array(v, dtype=float)
        return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    out = {"n": len(rows), "n_empty": int(sum(r["empty"] for r in rows)),
           "distinct_named_sets": len({tuple(r["named"]) for r in rows})}
    for k in ["hard", "status", "recall", "mag", "ofs"]:
        out[k], out[k + "_ci"] = m(k)
    return out


# ---------------------------------------------------------------- models
def pick_models(vendors):
    from llm_adapter import available_models, VENDOR_CANDIDATES
    avail = available_models()
    chosen = []
    for v in vendors:
        hit = next((mid for mid in VENDOR_CANDIDATES[v] if mid in avail), None)
        print(f"  {v:10s} -> {hit or 'NONE AVAILABLE (skipped)'}")
        if hit:
            chosen.append(hit)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_instances", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--universe", choices=["fred", "synthetic"], default="fred")
    ap.add_argument("--fred_csv", default="fred_universe.csv")
    ap.add_argument("--models", nargs="*", default=None, help="explicit OpenRouter ids")
    ap.add_argument("--vendors", nargs="*", default=["openai", "anthropic", "google"])
    ap.add_argument("--conditions", nargs="*", default=["FF", "FF_informed"])
    ap.add_argument("--dry_run", action="store_true", help="simulated narrator, no API calls")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--out", default="final_results.json")
    ap.add_argument("--log", default="final_study_log.jsonl")
    ap.add_argument("--neutral_example", action="store_true",
                    help="placeholder format line instead of an example with numbers (avoids anchoring)")
    ap.add_argument("--require_slack", action="store_true",
                    help="keep generating until n_instances problems each have >=1 zero-effect rule")
    a = ap.parse_args()
    global LOG
    LOG = a.log

    print(f"Building problems ({a.universe} universe, seed {a.seed}) ...")
    probs = build_problems(a.n_instances, a.seed, a.universe, a.fred_csv, require_slack=a.require_slack)
    print(f"  {len(probs)} solvable problems "
          f"({sum(p['degenerate'] for p in probs)} degenerate)")

    results = {"meta": {"universe": a.universe, "seed": a.seed, "n_requested": a.n_instances,
                        "n_problems": len(probs), "dry_run": a.dry_run,
                        "require_slack": a.require_slack,
                        "neutral_example": a.neutral_example,
                        "dates": sorted({p['date'] for p in probs if p['date']}),
                        "units": "bp per +1pp cap / +0.1y duration / -0.01 quality",
                        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
               "explainers": {}}
    per_problem = {}

    def record(name, rows):
        results["explainers"][name] = summarize(rows)
        per_problem[name] = rows

    # Certificate-grounded and trivial template baseline (no API)
    record("CG", [score_one(p, certificate_grounded_explanation(p["inst"], p["cert"])) for p in probs])
    tmpl = lambda p: [Claim("issuer_cap", True, 10.0), Claim("duration_ceiling", True, 5.0)]
    record("ALWAYS_CAP_DUR", [score_one(p, tmpl(p)) for p in probs])

    # LLM explainers
    if a.dry_run:
        rng = np.random.default_rng(0)
        for cond in a.conditions:
            record(f"SIMULATED/{cond}", [score_one(p, free_form_explanation(p["inst"], p["cert"], rng)) for p in probs])
    else:
        from llm_adapter import make_openrouter_ff, load_reply_cache
        print("Selecting models available on OpenRouter ...")
        models = a.models if a.models else pick_models(a.vendors)
        if not models:
            sys.exit("no models available")
        n_calls = len(models) * len(a.conditions) * len(probs)
        cache = load_reply_cache(LOG)
        print(f"\nPlan: {len(models)} models x {len(a.conditions)} conditions x {len(probs)} problems "
              f"= {n_calls} calls ({len(cache)} already cached, will not be re-billed).")
        print("Models:", ", ".join(models))
        if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")
        for model in models:
            for cond in a.conditions:
                ff = make_openrouter_ff(model, cache=cache, log_path=LOG,
                                        informed=(cond == "FF_informed"), condition=cond,
                                        neutral_example=a.neutral_example)
                rows = []
                for k, p in enumerate(probs, 1):
                    try:
                        rows.append(score_one(p, ff(p["inst"], p["cert"], None)))
                    except RuntimeError as e:
                        print(f"\n  ! {model} {cond} problem {p['i']}: {e}")
                        if "HTTP 40" in str(e):
                            print("  -> non-retryable error; skipping this model/condition")
                            break
                    if k % 10 == 0:
                        print(f"  {model} {cond}: {k}/{len(probs)}", end="\r", flush=True)
                print()
                record(f"{model}/{cond}", rows)

    # attach per-problem context and save
    by_i = {p["i"]: p for p in probs}
    results["per_problem"] = {name: [{**r, "tier": by_i[r["i"]]["tier"], "date": by_i[r["i"]]["date"],
                                      "effective": sorted(c for c in CONSTRAINTS
                                                          if abs(by_i[r["i"]]["effect"][c]) > ZERO_TOL_BP),
                                      "degenerate": by_i[r["i"]]["degenerate"]}
                                     for r in rows]
                              for name, rows in per_problem.items()}
    json.dump(results, open(a.out, "w"), indent=1)

    print("\n" + "=" * 96)
    print(f"{'explainer':44s}{'hard-fail':>11s}{'status':>9s}{'recall':>9s}{'magnit.':>9s}{'OFS':>8s}{'sets':>6s}")
    print("-" * 96)
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for name, s in results["explainers"].items():
        hf = "   -  " if s["hard"] is None else f"{100*s['hard']:5.1f}%"
        print(f"{name:44s}{hf:>11s}{fmt(s['status']):>9s}{fmt(s['recall']):>9s}"
              f"{fmt(s['mag']):>9s}{fmt(s['ofs']):>8s}{s['distinct_named_sets']:>6d}"
              + (f"   ({s['n_empty']} unparsed)" if s["n_empty"] else ""))
    print("=" * 96)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
