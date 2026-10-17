"""
make_v5_figures.py -- the v5 paper figures, drawn from v5_summary[_neutral].json.
  fig_v5_pipeline.pdf     the test procedure
  fig_v5_slack.pdf        main result: how often each explainer blames a rule with zero effect (95% CI)
  fig_v5_magnitude.pdf    claimed / true effect size, per rule and model (log scale)
Palette: reference categorical slots 1-3 (validated all-pairs, light); certificate + template in neutral ink.
Every series is direct-labelled and has its own marker shape (aqua is < 3:1 on white).
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

variant = sys.argv[1] if len(sys.argv) > 1 else "example"
S = json.load(open("v5_summary_neutral.json" if variant == "neutral" else "v5_summary.json"))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# GPT-4o-mini copied the prompt's example numbers in ~97% of answers -> asterisked, listed last,
# excluded from the paper's main claims.
MODELS = [("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", "#1baf7a", "D"),
          ("google/gemini-2.5-flash", "Gemini 2.5 Flash", "#eb6834", "s"),
          ("openai/gpt-4o-mini", "GPT-4o-mini*", "#2a78d6", "o")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.06})

# ---------------------------------------------------------------- main result
def fig_slack():
    E = S["explainers"]
    rows = [("Certificate-grounded", E["CG"]["pooled"], INK, "*", True),
            ("Template (cap + duration)", E["ALWAYS_CAP_DUR"]["pooled"], MUTED, "X", True)]
    for mid, lab, col, mk in MODELS:
        rows.append((f"{lab}, basic", E[f"{mid}/FF"]["pooled"], col, mk, False))
        rows.append((f"{lab}, with facts", E[f"{mid}/FF_informed"]["pooled"], col, mk, True))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, e, col, mk, filled) in zip(y, rows):
        lo, hi = e["ci"]; r = e["rate"]
        ax.plot([100*lo, 100*hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(100*r, yi, marker=mk, ms=7, mfc=col if filled else "white", mec=col, mew=1.5, ls="none")
        ax.text(100*hi + 2, yi, f"{100*r:.0f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(-2, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("zero-effect rules named as drivers (%, 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ["top", "right", "left"]: ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig("fig_v5_slack.pdf"); plt.close(fig)

# ---------------------------------------------------------------- magnitude
def fig_magnitude():
    cons = [("issuer_cap", "single-issuer cap"), ("duration_ceiling", "duration ceiling"),
            ("quality_floor", "quality floor")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axvspan(0.65, 1.35, color="#f0efec", zorder=0)
    ax.axvline(1, color=INK2, lw=0.8, zorder=1)
    ylabels, yt, pos = [], [], 0
    for ci, (c, clab) in enumerate(cons):
        for mi, (mid, lab, col, mk) in enumerate(MODELS):
            v = np.array(S["ratios"].get(f"{mid}|{c}", []))
            v = v[v > 0]
            if len(v) == 0: continue
            q1, med, q3 = np.percentile(v, [25, 50, 75]); p10, p90 = np.percentile(v, [10, 90])
            ax.plot([p10, p90], [pos, pos], color=col, lw=1, zorder=2)
            ax.plot([q1, q3], [pos, pos], color=col, lw=4, solid_capstyle="butt", zorder=3)
            ax.plot(med, pos, marker=mk, ms=6, mfc="white", mec=col, mew=1.5, zorder=4)
            yt.append(pos); ylabels.append(f"{clab}: {lab}" if mi == 0 else lab)
            pos -= 1
        pos -= 0.6
    ax.set_xscale("log"); ax.set_xlim(0.03, 60)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.1×", "0.3×", "1×", "3×", "10×", "30×"])
    ax.set_yticks(yt); ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("claimed ÷ true effect (log; shaded ±35%)")
    for s_ in ["top", "right", "left"]: ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig("fig_v5_magnitude.pdf"); plt.close(fig)

# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.4, 2.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    def box(x, y, w, h, t, fc, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    lw=0.5, ec=INK2, fc=fc))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=6.0, color=tc)
    def arr(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.6, color=INK2))
    box(0.2, 5.6, 4.2, 1.1, "Problem built from\nFRED bond data", "#52514e")
    box(5.6, 5.6, 4.2, 1.1, "Solver: portfolio\n+ certificate", "#0b0b0b")
    arr((4.4, 6.15), (5.6, 6.15))
    box(0.2, 3.2, 3.0, 1.2, "Certificate-\ngrounded", "#0b0b0b")
    box(3.5, 3.2, 3.0, 1.2, "LLM explains\nthe portfolio", "#2a78d6")
    box(6.8, 3.2, 3.0, 1.2, "LLM + facts\n(no certificate)", "#2a78d6")
    for x in (1.7, 5.0, 8.3): arr((7.7, 5.6), (x, 4.4))
    box(1.6, 0.6, 6.8, 1.4, "Check each claim: relax the rule, re-solve.\nNo change = not a driver.", "#f0efec", INK)
    for x in (1.7, 5.0, 8.3): arr((x, 3.2), (min(max(x, 2.2), 7.8), 2.0))
    fig.savefig("fig_v5_pipeline.pdf"); plt.close(fig)

fig_slack(); fig_magnitude(); fig_pipeline()
print("wrote fig_v5_pipeline.pdf, fig_v5_slack.pdf, fig_v5_magnitude.pdf (variant:", variant + ")")
