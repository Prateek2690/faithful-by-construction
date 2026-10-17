"""Are the true shadow prices determined by the information shown in the prompt?
Held bonds strictly between 0 and the cap are basic. Complementary slackness gives
    y_j = mu + lam_D d_j - lam_Q q_j      for each such bond,
with lam_D (lam_Q) present only if that limit binds. Solving these equations by least squares gives
mu, lam_D, lam_Q; the cap's shadow price follows from the bonds at the cap,
    sum_j (y_j - mu - lam_D d_j + lam_Q q_j).
We do this (a) with exact data and (b) with the values as displayed in the prompt
(yield to 0.01%, duration to 0.1y, quality to 0.01), convert to the paper's units, and compare
with the re-solve ground truth. Writes computability_v5.json and macros in numbers_v5_robust.tex."""
import json, numpy as np
from run_final_study import build_problems
from ofs_core import ZERO_TOL_BP, UNITS

def implied(y, d, q, x, cap, bind_D, bind_Q):
    frac = (x > 1e-7) & (x < cap - 1e-7); atcap = x >= cap - 1e-7
    cols = [np.ones(len(y))] + ([d] if bind_D else []) + ([-q] if bind_Q else [])
    A = np.column_stack(cols)[frac]
    if A.shape[0] < A.shape[1] or np.linalg.matrix_rank(A, tol=1e-9) < A.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(A, y[frac], rcond=None)
    mu = coef[0]; i = 1
    lD = coef[i] if bind_D else 0.0; i += bind_D
    lQ = coef[i] if bind_Q else 0.0
    nu = y[atcap] - mu - lD * d[atcap] + lQ * q[atcap]
    return {"issuer_cap": nu.sum() * UNITS["issuer_cap"] * 1e4,
            "duration_ceiling": lD * UNITS["duration_ceiling"] * 1e4,
            "quality_floor": lQ * UNITS["quality_floor"] * 1e4}

res = {}; exact_err = []; disp_ratio = []
for run, seed, slack in [("main", 7, False), ("slack", 11, True)]:
    probs = build_problems(100, seed, "fred", "fred_universe.csv", require_slack=slack)
    k = n = lk = ln = 0
    for p in probs:
        inst, x = p["inst"], p["cert"].x
        eff = p["effect"]; bD = abs(eff["duration_ceiling"]) > ZERO_TOL_BP; bQ = abs(eff["quality_floor"]) > ZERO_TOL_BP
        ex = implied(inst.y, inst.d, inst.q, x, inst.cap, bD, bQ)
        dp = implied(np.round(inst.y * 100, 2) / 100, np.round(inst.d, 1), np.round(inst.q, 2), x, inst.cap, bD, bQ)
        n += 1; k += int(ex is not None)
        for c, t in eff.items():
            if abs(t) <= ZERO_TOL_BP: continue
            ln += 1
            if ex is not None:
                lk += 1; exact_err.append(abs(ex[c] - t) / abs(t))
            if dp is not None:
                disp_ratio.append(dp[c] / t)
    res[run] = {"problems_determined": k, "problems": n, "binding_limits_determined": lk, "binding_limits": ln}
tot = {key: sum(r[key] for r in res.values()) for key in res["main"]}
dr = np.array(disp_ratio)
res["total"] = tot
ee=np.array(exact_err); res["exact_max_rel_err"] = float(ee.max()); res["exact_within_1e-6"] = float(np.mean(ee<=1e-6)); res["exact_within_1pct"] = float(np.mean(ee<=0.01)); res["exact_n_off"] = int(np.sum(ee>0.01))
res["displayed_within35"] = float(np.mean(np.abs(dr - 1) <= 0.35)); res["displayed_within10"] = float(np.mean(np.abs(dr - 1) <= 0.10))
res["displayed_n"] = int(len(dr))
json.dump(res, open("computability_v5.json", "w"), indent=1)
print(json.dumps(res, indent=1))
txt = open("numbers_v5_robust.tex").read().split("% computability")[0]
txt += "% computability (computability_v5.py)\n"
txt += f"\\newcommand{{\\compLimK}}{{{tot['binding_limits_determined']}}}\n\\newcommand{{\\compLimN}}{{{tot['binding_limits']}}}\n"
txt += f"\\newcommand{{\\compExactOne}}{{{100*res['exact_within_1pct']:.0f}}}\n"
txt += f"\\newcommand{{\\compDispThirtyFive}}{{{100*res['displayed_within35']:.0f}}}\n\\newcommand{{\\compDispTen}}{{{100*res['displayed_within10']:.0f}}}\n"
open("numbers_v5_robust.tex", "w").write(txt)
