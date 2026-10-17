"""
make_v5_numbers.py -- turns the two real runs into every number the v5 papers use.
Inputs : final_results.json (+ final_study_log.jsonl)            main run, seed 7
         final_results_slack.json (+ final_study_log_slack.jsonl) slack-focused run, seed 11
Outputs: v5_summary.json (for figures/slides), numbers_v5.tex (LaTeX macros)
"""
import json, numpy as np, collections
from run_final_study import build_problems
from llm_adapter import build_prompt, _stable_key, parse_reply_to_claims

C = ["issuer_cap", "duration_ceiling", "quality_floor"]
MODELS = ["openai/gpt-4o-mini", "google/gemini-2.5-flash", "anthropic/claude-sonnet-4.5"]
SHORT = {"openai/gpt-4o-mini": "GPT-4o-mini", "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5"}
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=["example", "neutral"], default="example")
VARIANT = ap.parse_args().variant
NEUTRAL = VARIANT == "neutral"
sfx = "_neutral" if NEUTRAL else ""
RUNS = {"main": dict(res=f"final_results{sfx}.json", log=f"final_study_log{sfx}.jsonl", seed=7, n=100, slack=False),
        "slack": dict(res=f"final_results_slack{sfx}.json", log=f"final_study_log_slack{sfx}.jsonl", seed=11, n=100, slack=True)}
for r in RUNS.values():
    if not os.path.exists(r["res"]):
        raise SystemExit(f"missing {r['res']} -- run the {VARIANT} experiments first")

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [c - h, c + h]

out = {"variant": VARIANT, "runs": {}, "explainers": {}, "magnitude": {}, "examples": []}
slack_tot = collections.defaultdict(lambda: [0, 0]); hard_tot = collections.defaultdict(lambda: [0, 0])
mag_tot = collections.defaultdict(lambda: [0, 0]); ratios = collections.defaultdict(list)
for run, cfg in RUNS.items():
    R = json.load(open(cfg["res"])); P = R["per_problem"]
    out["runs"][run] = {"n_problems": R["meta"]["n_problems"], "dates": R["meta"]["dates"]}
    for name, rows in P.items():
        k = n = 0
        for r in rows:
            sl = set(C) - set(r["effective"]); n += len(sl); k += len(sl & set(r["named"]))
        s = R["explainers"][name]
        out["explainers"].setdefault(name, {})[run] = {
            "slack_named": k, "slack_cases": n, "rate": k/n if n else None, "ci": wilson(k, n),
            "hard": s["hard"], "status": s["status"], "recall": s["recall"], "mag": s["mag"], "n": s["n"]}
        slack_tot[name][0] += k; slack_tot[name][1] += n
        hard_tot[name][0] += round(s["hard"] * s["n"]); hard_tot[name][1] += s["n"]
    # claim-level magnitude, rebuilt from the logs
    probs = build_problems(cfg["n"], cfg["seed"], "fred", "fred_universe.csv", require_slack=cfg["slack"])
    log = {json.loads(l)["key"]: json.loads(l)["reply"] for l in open(cfg["log"])}
    for m in MODELS:
        for cond in ["FF", "FF_informed"]:
            for p in probs:
                user = build_prompt(p["inst"], p["cert"].x, informed=(cond == "FF_informed"),
                                    neutral_example=NEUTRAL)[1]
                rep = log.get(_stable_key(m, cond, user))
                if rep is None: continue
                cl = parse_reply_to_claims(rep)
                for c_ in cl:
                    t = p["effect"][c_.constraint]
                    if abs(t) > 1e-6:
                        r_ = c_.asserted_sp_bp / t
                        ratios[(m, c_.constraint)].append(r_)
                        mag_tot[(m, cond)][0] += abs(r_ - 1) <= 0.35; mag_tot[(m, cond)][1] += 1
                # worked-example candidates: informed prompt, slack quality floor named
                if run == "slack" and cond == "FF_informed":
                    named = {x.constraint: x.asserted_sp_bp for x in cl}
                    slackset = [c for c in C if abs(p["effect"][c]) <= 1e-6]
                    hit = [c for c in slackset if c in named]
                    if hit:
                        x = p["cert"].x; inst = p["inst"]
                        out["examples"].append({"model": m, "i": p["i"], "date": p["date"],
                            "true_bp": {c: round(p["effect"][c], 3) for c in C},
                            "claimed_bp": named, "slack_named": hit,
                            "avg_dur": float(inst.d @ x), "dur_cap": inst.D_max,
                            "avg_q": float(inst.q @ x), "q_floor": inst.Q_min,
                            "at_cap": int(np.sum(np.isclose(x, inst.cap, atol=1e-6))),
                            "objective_pct": p["cert"].objective * 100, "reply": rep})
for name, (k, n) in slack_tot.items():
    out["explainers"][name]["pooled"] = {"slack_named": k, "slack_cases": n, "rate": k/n, "ci": wilson(k, n)}
for (m, cond), (k, n) in mag_tot.items():
    out["magnitude"][f"{m}/{cond}"] = {"within35": k/n, "n_claims": n, "ci": wilson(k, n)}
out["ratios"] = {f"{m}|{c}": v for (m, c), v in ratios.items()}
out["median_ratio"] = {f"{m}|{c}": float(np.median(v)) for (m, c), v in ratios.items()}
# anchoring side-result: how often the example-prompt runs copied the example numbers
anch = {}
for lf in ["final_study_log.jsonl", "final_study_log_slack.jsonl"]:
    if os.path.exists(lf):
        for l in open(lf):
            e = json.loads(l); cl = {c.constraint: c.asserted_sp_bp for c in parse_reply_to_claims(e["reply"])}
            a = anch.setdefault(e["model"], [0, 0])
            if "issuer_cap" in cl: a[1] += 1; a[0] += cl["issuer_cap"] == 12.0
out["anchoring_cap12"] = {m: k / n for m, (k, n) in anch.items() if n}
json.dump(out, open(f"v5_summary{sfx}.json", "w"), indent=1)

# ------------------------------------------------------------------ LaTeX macros
def pct(x): return f"{100*x:.0f}"
L = ["%% AUTO-GENERATED by make_v5_numbers.py -- do not edit by hand"]
def mac(n, v): L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
mac("nMain", out["runs"]["main"]["n_problems"]); mac("nSlack", out["runs"]["slack"]["n_problems"])
mac("nProblems", out["runs"]["main"]["n_problems"] + out["runs"]["slack"]["n_problems"])
E = out["explainers"]
mac("nSlackCases", E["CG"]["pooled"]["slack_cases"])
mac("nSlackCasesMain", E["CG"]["main"]["slack_cases"]); mac("nSlackCasesSlack", E["CG"]["slack"]["slack_cases"])
tag = {"openai/gpt-4o-mini": "Gpt", "google/gemini-2.5-flash": "Gem", "anthropic/claude-sonnet-4.5": "Son"}
for m, t in tag.items():
    for cond, ct in [("FF", "Ff"), ("FF_informed", "Inf")]:
        e = E[f"{m}/{cond}"]["pooled"]
        mac(f"sn{t}{ct}", pct(e["rate"])); mac(f"sn{t}{ct}Lo", pct(e["ci"][0])); mac(f"sn{t}{ct}Hi", pct(e["ci"][1]))
        mg = out["magnitude"][f"{m}/{cond}"]; mac(f"mag{t}{ct}", pct(mg["within35"]))
        mac(f"hardMain{t}{ct}", pct(E[f"{m}/{cond}"]["main"]["hard"]))
mac("snTmpl", pct(E["ALWAYS_CAP_DUR"]["pooled"]["rate"])); mac("snCG", pct(E["CG"]["pooled"]["rate"]))
mac("snCGHi", pct(E["CG"]["pooled"]["ci"][1]))
mac("hardMainTmpl", pct(E["ALWAYS_CAP_DUR"]["main"]["hard"]))
mac("magCG", f"{100*min(E['CG']['main']['mag'], E['CG']['slack']['mag']):.1f}")
allmag = [v["within35"] for v in out["magnitude"].values()]
mac("magMin", pct(min(allmag))); mac("magMax", pct(max(allmag)))
mac("nMagClaims", sum(v["n_claims"] for v in out["magnitude"].values()))
for c, ct in [("issuer_cap", "Cap"), ("duration_ceiling", "Dur"), ("quality_floor", "Qual")]:
    vals = np.concatenate([out["ratios"][f"{m}|{c}"] for m in MODELS])
    mac(f"ratio{ct}", f"{np.median(vals):.2f}")
for m, t in tag.items():
    if m in out["anchoring_cap12"]: mac(f"anchor{t}", pct(out["anchoring_cap12"][m]))
open(f"numbers_v5{sfx}.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nexamples:", len(out["examples"]))
