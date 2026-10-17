"""Writes table_v5.tex (main results table) and adds check macros to numbers_v5_example.tex."""
import json
d = json.load(open("v5_summary.json")); E = d["explainers"]; M = d["magnitude"]
RB = json.load(open("robustness_v5.json"))["explainers"]   # run robustness_v5.py first
rows = [("Certificate-grounded", "CG", None), ("Template: always cap + duration", "ALWAYS_CAP_DUR", None),
        ("Claude Sonnet 4.5, basic", "anthropic/claude-sonnet-4.5/FF", "anthropic/claude-sonnet-4.5/FF"),
        ("Claude Sonnet 4.5, with facts", "anthropic/claude-sonnet-4.5/FF_informed", "anthropic/claude-sonnet-4.5/FF_informed"),
        ("Gemini 2.5 Flash, basic", "google/gemini-2.5-flash/FF", "google/gemini-2.5-flash/FF"),
        ("Gemini 2.5 Flash, with facts", "google/gemini-2.5-flash/FF_informed", "google/gemini-2.5-flash/FF_informed"),
        ("GPT-4o-mini$^{*}$, basic", "openai/gpt-4o-mini/FF", "openai/gpt-4o-mini/FF"),
        ("GPT-4o-mini$^{*}$, with facts", "openai/gpt-4o-mini/FF_informed", "openai/gpt-4o-mini/FF_informed")]
L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r"& \multicolumn{2}{c}{Zero-effect rules named} & Pooled & Binding rules & Size within \\",
     r"Explainer & natural & slack-focused & (95\% CI) & named & $\pm$35\% \\", r"\midrule"]
for lab, key, mk in rows:
    e = E[key]; a, b, p = e["main"], e["slack"], e["pooled"]
    if mk: mag = f"{100*M[mk]['within35']:.0f}\\%"
    elif key == "CG": mag = f"{100*min(a['mag'], b['mag']):.1f}\\%"
    else: mag = f"{100*(a['mag']*a['n']+b['mag']*b['n'])/(a['n']+b['n']):.0f}\\%"
    bind = f"{100*RB[key]['p_named_given_binding']:.0f}\\%"
    L.append(f"{lab} & {a['slack_named']}/{a['slack_cases']} & {b['slack_named']}/{b['slack_cases']} & "
             f"{100*p['rate']:.0f}\\% [{100*p['ci'][0]:.0f}, {100*p['ci'][1]:.0f}] & {bind} & {mag} \\\\")
    if key == "ALWAYS_CAP_DUR": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open("table_v5.tex", "w").write("\n".join(L) + "\n")
# degeneracy / breakpoint flags, read from the results files (no hard-coded values)
deg = {}
for run, f in [("Main", "final_results.json"), ("Slack", "final_results_slack.json")]:
    rows_ = json.load(open(f))["per_problem"]["CG"]
    deg[run] = sum(1 for r in rows_ if r.get("degenerate"))
s = open("numbers_v5_example.tex").read().splitlines()
s = [l for l in s if not any(k in l for k in ("computableK", "computableN", "degMain", "degSlack", "ratioMain", "nMagMain", "magMain"))]
s += [f"\\newcommand{{\\degMain}}{{{deg['Main']}}}", f"\\newcommand{{\\degSlack}}{{{deg['Slack']}}}"]
open("numbers_v5_example.tex", "w").write("\n".join(s) + "\n")
print("\n".join(L))
