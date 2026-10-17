"""Independent audit: recompute the paper's reported numbers from the raw result files and
check that each appears in the compiled PDF text. Usage: python audit_numbers.py <paper.pdf>"""
import json, sys, subprocess, re, statistics, collections
pdf = sys.argv[1]
txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
txt = re.sub(r"\s+", " ", txt.replace("–", "-").replace("−", "-"))
lay = re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout)
C = ["issuer_cap", "duration_ceiling", "quality_floor"]
R = {r: json.load(open(f))["per_problem"] for r, f in [("main", "final_results.json"), ("slack", "final_results_slack.json")]}
S = json.load(open("v5_summary.json")); RB = json.load(open("robustness_v5.json")); CP = json.load(open("computability_v5.json"))
M = {"Son": "anthropic/claude-sonnet-4.5", "Gem": "google/gemini-2.5-flash", "Gpt": "openai/gpt-4o-mini"}
checks = []
def want(label, s): checks.append((label, s, s in txt))
pc = lambda x: f"{100*x:.0f}%"

nprob = sum(len(R[r]["CG"]) for r in R); want("problems", f"{nprob} bond-portfolio problems")
want("natural solvable", f"{len(R['main']['CG'])} solvable problems")
def zero(name, run):
    k = n = 0
    for r in R[run][name]:
        z = set(C) - set(r["effective"]); n += len(z); k += len(z & set(r["named"]))
    return k, n
zc = sum(zero("CG", r)[1] for r in R); want("zero cases", f"{zc} cases in which a limit had exactly zero")
allbind = sum(len(r["effective"]) == 3 for r in R["main"]["CG"]); want("all bind", f"{allbind} of {len(R['main']['CG'])} in our first set")
cap_always = all("issuer_cap" in r["effective"] for r in R["main"]["CG"] + R["slack"]["CG"])
checks.append(("cap binds in every problem", str(cap_always), cap_always))
lim = collections.Counter(c for r in R["main"]["CG"] + R["slack"]["CG"] for c in set(C) - set(r["effective"]))
want("slack durations", f"duration ceiling ({lim['duration_ceiling']})"); want("slack quality", f"quality floor ({lim['quality_floor']})")
rows = {"Claude Sonnet 4.5, basic": f"{M['Son']}/FF", "Claude Sonnet 4.5, with facts": f"{M['Son']}/FF_informed",
        "Gemini 2.5 Flash, basic": f"{M['Gem']}/FF", "Gemini 2.5 Flash, with facts": f"{M['Gem']}/FF_informed",
        "GPT-4o-mini*, basic": f"{M['Gpt']}/FF", "GPT-4o-mini*, with facts": f"{M['Gpt']}/FF_informed",
        "Template: always cap + duration": "ALWAYS_CAP_DUR", "Certificate-grounded": "CG"}
for lab, key in rows.items():
    a, b = zero(key, "main"), zero(key, "slack"); p = (a[0] + b[0]) / (a[1] + b[1])
    s_ = f"{a[0]}/{a[1]} {b[0]}/{b[1]} {pc(p)}"; checks.append((f"table {lab}", s_, s_ in lay))
    e = RB["explainers"][key]; want(f"binding named {lab}", f"{pc(e['p_named_given_binding'])}")
# headline sentences
z = lambda k: pc(sum(zero(k, r)[0] for r in R) / zc)
want("abstract Claude/Gemini", f"driver in {z(rows['Claude Sonnet 4.5, basic'])} (Claude) and {z(rows['Gemini 2.5 Flash, basic'])} (Gemini)")
want("abstract binding", f"binding limits in {pc(RB['explainers'][rows['Claude Sonnet 4.5, basic']]['p_named_given_binding'])} and {pc(RB['explainers'][rows['Gemini 2.5 Flash, basic']]['p_named_given_binding'])}")
want("all three Claude", f"all three limits in {pc(RB['explainers'][rows['Claude Sonnet 4.5, basic']]['share_name_all3'])}")
# hard failure, natural set
for s, lab in [("Son", "for Claude"), ("Gem", "for Gemini")]:
    rr = R["main"][f"{M[s]}/FF"]; h = sum(bool(set(r["named"]) - set(r["effective"])) for r in rr) / len(rr)
    want(f"hard {s}", f"{pc(h)} {lab}")
rr = R["main"]["ALWAYS_CAP_DUR"]; want("hard template", f"{pc(sum(bool(set(r['named'])-set(r['effective'])) for r in rr)/len(rr))} for the template")
# magnitudes
mk = [k for k in S["magnitude"] if not k.startswith("openai")]
nm = sum(S["magnitude"][k]["n_claims"] for k in mk); w = sum(S["magnitude"][k]["within35"] * S["magnitude"][k]["n_claims"] for k in mk) / nm
want("magnitude", f"{pc(w)} of {nm:,} claims")
for c, lab in [("issuer_cap", "cap"), ("duration_ceiling", "dur"), ("quality_floor", "qual")]:
    pool = [x for k, v in S["ratios"].items() if not k.startswith("openai") and k.endswith(c) for x in v if x is not None]
    want(f"median ratio {lab}", f"{statistics.median(pool):.2f}")
for s, lab in [("Gpt", "GPT"), ("Gem", "Gemini"), ("Son", "Claude")]:
    want(f"anchor {lab}", f"{100*S['anchoring_cap12'][M[s]]:.0f}%")
t = RB["tolerance"]["pooled_son_gem"]; want("tolerance", f"{pc(t['tol0.2'])}, {pc(t['tol0.5'])} and {pc(t['within2x'])}")
want("computability", f"in all {CP['total']['binding_limits']} cases"); want("displayed 10%", f"{pc(CP['displayed_within10'])} of binding limits")
want("exact 1%", f"within 1% for {pc(CP['exact_within_1pct'])}")
for k, lab in [("Son_basic_vs_facts", "0.57)"), ("Gem_facts_vs_template", "p = 0.84"), ("Son_vs_Gem_basic", "p = 0.013")]:
    want(f"mcnemar {k}", lab)
b = RB["cluster_boot_ci"]; want("bootstrap", f"{100*b[M['Son']+'/FF'][0]:.0f}-{100*b[M['Son']+'/FF'][1]:.0f} for Claude")
ex = [e for e in S["examples"] if e["i"] == 248 and e["model"] == M["Gem"]][0]
for lab, s in [("ex obj", f"{ex['objective_pct']:.2f}%"), ("ex cap", f"{ex['true_bp']['issuer_cap']:.1f}"), ("ex dur", f"{ex['true_bp']['duration_ceiling']:.2f}"),
               ("ex avgq", f"{ex['avg_q']:.3f}"), ("ex floor", f"{ex['q_floor']:.3f}")]:
    want(lab, s)
bad = [c for c in checks if not c[2]]
for lab, s, ok in checks: print(("OK  " if ok else "FAIL"), lab, "|", s)
print(f"\n{len(checks)-len(bad)}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
