"""
ofs_core.py
===========
Core machinery for the paper "Faithful by Construction? Measuring the
Faithfulness of Explanations for Portfolio Optimization Decisions."

This module implements, with no external solver dependencies beyond SciPy/HiGHS:

  1. A synthetic constrained-portfolio-optimization instance generator.
     (The objective is a generic per-asset "portfolio objective" c_i; insurance
     book yield is one instantiation, but nothing here is insurance-specific.)
  2. An LP solver that returns the *optimality certificate* (binding set + duals).
  3. The Optimization Faithfulness Score (OFS) via exact re-solve interventions.
  4. Two explanation methods:
        - Certificate-Grounded (CG): reads reasons from the certificate.
        - Free-form (FF): a *simulated* LLM narrator whose unfaithfulness rate
          is controllable, so the harness is fully reproducible offline. In a
          live study, replace `free_form_explanation()` with a real LLM call
          (see `llm_adapter.py`).

All randomness is seeded. Every number the paper reports is produced by
`run_study()` in `run_experiment.py`.

Problem (constrained portfolio optimization, LP form):
    maximize    sum_i  y_i * x_i                      (portfolio objective)
    subject to  sum_i x_i = 1                          (fully invested)
                sum_i d_i x_i <= D_max                 (duration ceiling)
                x_i <= cap                             (single-issuer cap)
                sum_i q_i x_i >= Q_min                 (credit-quality floor)
                x_i >= 0
where y_i = objective coefficient (e.g. yield/return), d_i = a risk measure
(e.g. duration), q_i = a quality/rating score of asset i.

We express everything as <= for a clean dual sign convention, then read
shadow prices from HiGHS marginals.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linprog


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Instance:
    y: np.ndarray          # objective coefficients, e.g. yield/return (n,)
    d: np.ndarray          # durations (n,)
    q: np.ndarray          # quality scores (n,)
    cap: float             # single-issuer cap
    D_max: float           # duration ceiling
    Q_min: float           # quality floor
    seed: int = 0

    @property
    def n(self) -> int:
        return len(self.y)


# Named constraints, in a fixed order, each with a human-readable label.
CONSTRAINTS = ["issuer_cap", "duration_ceiling", "quality_floor"]
CONSTRAINT_LABELS = {
    "issuer_cap": "single-issuer cap",
    "duration_ceiling": "duration ceiling",
    "quality_floor": "credit-quality floor",
}

# v5: every shadow price is reported in bp of the objective per ONE defined
# unit of relaxation, stated explicitly to any explainer (incl. the LLM).
UNITS = {"issuer_cap": 0.01,        # +1 percentage point on the cap
         "duration_ceiling": 0.1,   # +0.1 years on the duration ceiling
         "quality_floor": 0.01}     # -0.01 on the quality-score floor
UNIT_TEXT = {"issuer_cap": "per +1 percentage point increase in the single-issuer cap",
             "duration_ceiling": "per +0.1 year increase in the duration ceiling",
             "quality_floor": "per -0.01 decrease in the quality floor"}
# A constraint whose re-solve effect is below this is "no effect" (slack).
ZERO_TOL_BP = 1e-6


@dataclass
class Certificate:
    objective: float
    x: np.ndarray
    binding: dict            # name -> bool
    shadow_price: dict       # name -> float (bp of book yield per unit relax)
    status: str


@dataclass
class Claim:
    """A single assertion inside an explanation."""
    constraint: str          # one of CONSTRAINTS
    asserted_binding: bool
    asserted_sp_bp: float     # asserted shadow price in bp (0 if not a driver)


@dataclass
class ScoredExplanation:
    method: str
    claims: list
    faithful_flags: list      # bool per claim (status AND magnitude correct)
    hard_failure_flags: list  # bool per claim (named a zero-effect constraint a driver)
    status_flags: list = None     # bool per claim: named constraint truly has an effect
    magnitude_flags: list = None  # bool per status-correct claim: magnitude within tol

    @property
    def ofs_status(self) -> float:
        f = self.status_flags or []
        return float(np.mean(f)) if f else 1.0

    @property
    def magnitude_acc(self):
        f = [m for m in (self.magnitude_flags or []) if m is not None]
        return float(np.mean(f)) if f else None

    @property
    def ofs(self) -> float:
        return float(np.mean(self.faithful_flags)) if self.faithful_flags else 1.0

    @property
    def hard_failure(self) -> bool:
        return any(self.hard_failure_flags)


# ----------------------------------------------------------------------------
# Instance generator
# ----------------------------------------------------------------------------
def generate_instance(rng: np.random.Generator, n: int = 40,
                      tier: str = "mixed") -> Instance:
    """Sample a realistic-ish constrained-portfolio instance.

    'tier' shifts how tight constraints are, so slack vs binding status
    varies across the corpus (this is what makes the FF failure mode appear).
    """
    # Book yields ~ 3.5%-6.5%, higher yield correlates with longer duration
    # and (mildly) lower quality -- the usual risk premia.
    base = rng.uniform(0.035, 0.065, size=n)
    d = np.clip(2.0 + 60.0 * (base - 0.035) + rng.normal(0, 1.2, n), 0.5, 12.0)
    # quality score in [0,1], 1 = highest; negatively related to yield
    q = np.clip(0.9 - 8.0 * (base - 0.035) + rng.normal(0, 0.08, n), 0.05, 1.0)
    y = base

    cap = rng.uniform(0.03, 0.08)
    # Duration ceiling: sometimes binding, sometimes not, by tier
    dmean = float(d.mean())
    if tier == "tight_duration":
        D_max = dmean * rng.uniform(0.80, 0.95)
    elif tier == "loose_duration":
        D_max = dmean * rng.uniform(1.15, 1.4)
    else:
        D_max = dmean * rng.uniform(0.9, 1.2)

    qmean = float(q.mean())
    if tier == "tight_quality":
        Q_min = qmean * rng.uniform(1.0, 1.15)
    else:
        Q_min = qmean * rng.uniform(0.6, 0.95)   # usually slack

    return Instance(y=y, d=d, q=q, cap=float(cap),
                    D_max=float(D_max), Q_min=float(Q_min))


# ----------------------------------------------------------------------------
# Solver + certificate
# ----------------------------------------------------------------------------
def _solve(inst: Instance):
    """Solve the LP; return (objective, x, res). Maximize the objective."""
    n = inst.n
    # linprog minimizes c^T x  ->  minimize -y^T x
    c = -inst.y
    # Inequalities A_ub x <= b_ub
    #   duration:      d^T x <= D_max
    #   quality floor: -q^T x <= -Q_min   (i.e. q^T x >= Q_min)
    A_ub = np.vstack([inst.d, -inst.q])
    b_ub = np.array([inst.D_max, -inst.Q_min])
    # Equality: sum x = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    # Bounds: 0 <= x_i <= cap  (single-issuer cap as a variable bound)
    bounds = [(0.0, inst.cap)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return res


def solve_with_certificate(inst: Instance, tol: float = 1e-6) -> Certificate:
    """Solve and extract the optimality certificate.

    Shadow prices are reported in basis points of the objective per unit
    relaxation of the constraint's right-hand side. For the issuer cap
    (a variable upper bound) we aggregate the bound duals of the assets
    sitting at the cap, which is the marginal value of raising the cap.
    """
    res = _solve(inst)
    if res.status != 0:
        return Certificate(objective=np.nan, x=np.zeros(inst.n),
                           binding={c: False for c in CONSTRAINTS},
                           shadow_price={c: 0.0 for c in CONSTRAINTS},
                           status="infeasible_or_unbounded")

    x = res.x
    obj = float(-res.fun)

    # Duals for the two A_ub rows (duration, quality).
    # HiGHS marginals for ineqlin are <= 0 at an active <= constraint.
    m_ub = res.ineqlin.marginals  # length 2
    sp_duration_frac = abs(m_ub[0])          # d/dD_max of objective
    sp_quality_frac = abs(m_ub[1])           # d/d(-Q_min); tie to floor below

    # Bound duals give the issuer-cap shadow price: sum of upper-bound
    # marginals over assets pinned at the cap.
    m_upper = res.upper.marginals            # length n, nonzero where x_i = cap
    sp_cap_frac = float(np.sum(np.abs(m_upper[np.isclose(x, inst.cap, atol=1e-6)])))

    # Binding status from slack, not from duals (robust to degeneracy).
    dur_slack = inst.D_max - float(inst.d @ x)
    qual_slack = float(inst.q @ x) - inst.Q_min
    cap_binding = bool(np.any(np.isclose(x, inst.cap, atol=1e-6)))

    binding = {
        "issuer_cap": cap_binding,
        "duration_ceiling": bool(dur_slack <= tol),
        "quality_floor": bool(qual_slack <= tol),
    }
    # Convert fractional objective to bp per 1 percentage-point
    # (0.01) of RHS relaxation, matching measured_shadow_price()'s scale.
    shadow_price = {
        "issuer_cap": sp_cap_frac * 1e4 * UNITS["issuer_cap"] if binding["issuer_cap"] else 0.0,
        "duration_ceiling": sp_duration_frac * 1e4 * UNITS["duration_ceiling"] if binding["duration_ceiling"] else 0.0,
        "quality_floor": sp_quality_frac * 1e4 * UNITS["quality_floor"] if binding["quality_floor"] else 0.0,
    }
    return Certificate(objective=obj, x=x, binding=binding,
                       shadow_price=shadow_price, status="optimal")


# ----------------------------------------------------------------------------
# Ground-truth via exact re-solve (the heart of OFS)
# ----------------------------------------------------------------------------
def measured_shadow_price(inst: Instance, constraint: str,
                          delta: float = 1e-3) -> tuple[float, bool]:
    """Perturb one constraint's RHS by +delta, re-solve, and return the
    realized marginal effect on the objective (in bp) plus true binding status.

    This is the exact intervention that makes OFS ground-truth-based.
    """
    base = solve_with_certificate(inst)
    z0 = base.objective

    pert = Instance(**{**inst.__dict__})
    if constraint == "issuer_cap":
        pert.cap = inst.cap + delta
    elif constraint == "duration_ceiling":
        pert.D_max = inst.D_max + delta
    elif constraint == "quality_floor":
        pert.Q_min = inst.Q_min - delta   # relaxing a floor = lowering it
    else:
        raise ValueError(constraint)

    res = _solve(pert)
    z1 = float(-res.fun) if res.status == 0 else z0
    # Report bp of objective per 1 percentage-point (0.01) of RHS relaxation
    # for duration/cap-style constraints. This keeps all constraints on a
    # comparable, readable scale (single-digit-to-teens bp), matching how a
    # PM reads a shadow price ("relax the cap 1pp -> +X bp yield").
    realized_bp = (z1 - z0) / delta * UNITS[constraint] * 1e4
    true_binding = base.binding[constraint]
    return realized_bp, true_binding


# ----------------------------------------------------------------------------
# Degeneracy detection (task A3)
# ----------------------------------------------------------------------------
def is_degenerate(inst: Instance, cert: Certificate,
                  probe_ratio_tol: float = 0.25) -> bool:
    """Flag primal degeneracy, where the LP dual (shadow price) may be
    non-unique and the 'exact ground truth' is one of several valid values.

    Practical proxy: for each *binding* constraint, compare the forward and
    backward one-sided shadow-price probes. If they disagree by more than
    probe_ratio_tol (relative), the dual is direction-sensitive -> the vertex
    is (near-)degenerate for that constraint. This is exactly the case the
    paper must quantify and exclude from strong claims.
    """
    if cert.status != "optimal":
        return False
    for cname in CONSTRAINTS:
        if not cert.binding[cname]:
            continue
        base = solve_with_certificate(inst)
        z0 = base.objective
        # forward and backward probes
        fwd = _probe(inst, cname, +1e-3, z0)
        bwd = _probe(inst, cname, +5e-4, z0)
        denom = max(abs(fwd), abs(bwd), 1e-9)
        if abs(fwd - bwd) / denom > probe_ratio_tol:
            return True
    return False


def _probe(inst: Instance, constraint: str, delta: float, z0: float) -> float:
    pert = Instance(**{**inst.__dict__})
    if constraint == "issuer_cap":
        pert.cap = inst.cap + delta
    elif constraint == "duration_ceiling":
        pert.D_max = inst.D_max + delta
    elif constraint == "quality_floor":
        pert.Q_min = inst.Q_min - delta
    res = _solve(pert)
    z1 = float(-res.fun) if res.status == 0 else z0
    return (z1 - z0) / delta * UNITS[constraint] * 1e4


# ----------------------------------------------------------------------------
# Explanation methods
# ----------------------------------------------------------------------------
def certificate_grounded_explanation(inst: Instance, cert: Certificate,
                                     driver_threshold_bp: float = ZERO_TOL_BP) -> list:
    """CG: emit one claim per constraint that is binding *and materially so*
    (certificate shadow price above the driver threshold), with its true dual.
    Faithful by construction: it never asserts a non-material constraint drives
    the allocation, so it cannot commit the hard failure that FF commits."""
    claims = []
    for cname in CONSTRAINTS:
        if cert.binding[cname] and cert.shadow_price[cname] > driver_threshold_bp:
            claims.append(Claim(constraint=cname, asserted_binding=True,
                                asserted_sp_bp=cert.shadow_price[cname]))
    return claims


def free_form_explanation(inst: Instance, cert: Certificate,
                          rng: np.random.Generator,
                          p_hallucinate_slack: float = 0.35,
                          p_miss_binding: float = 0.15,
                          sp_noise: float = 0.4) -> list:
    """Simulated free-form LLM narrator (offline, reproducible).

    Models three realistic failure modes documented in the LLM-explanation
    literature:
      * naming a *slack* constraint as a driver (hard failure),
      * omitting a genuinely binding constraint,
      * mis-stating shadow-price magnitude.

    Replace with a real LLM in `llm_adapter.py` for the live study; the
    scoring in `score_explanation` is identical either way.
    """
    claims = []
    # Sometimes assert a slack constraint drives the portfolio (hallucination).
    for cname in CONSTRAINTS:
        if cert.binding[cname]:
            if rng.random() > p_miss_binding:
                noisy = cert.shadow_price[cname] * (1 + rng.normal(0, sp_noise))
                claims.append(Claim(cname, True, max(0.1, noisy)))
        else:
            if rng.random() < p_hallucinate_slack:
                # invents a plausible magnitude for a non-driver
                claims.append(Claim(cname, True, rng.uniform(2, 15)))
    if not claims:  # never emit nothing; assert the most plausible
        claims.append(Claim("quality_floor", True, rng.uniform(2, 10)))
    return claims


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_explanation(inst: Instance, claims: list, method: str,
                      driver_threshold_bp: float = ZERO_TOL_BP,
                      rel_tol: float = 0.35,
                      abs_tol_bp: float = 0.25,
                      delta: float = 1e-3) -> ScoredExplanation:
    """Score each claim by exact re-solve. A claim is faithful iff:
        - asserted binding status matches the re-solved truth, AND
        - asserted shadow price is within tol of the realized dual.
    A HARD failure is asserting a slack constraint (true sp ~ 0) is a driver.

    The re-solve uses a symmetric two-sided probe to stay robust at
    degenerate vertices, where a one-sided perturbation can report a
    spurious zero (or jump) for a genuinely binding constraint.
    """
    faithful, hard, status, mag = [], [], [], []
    for cl in claims:
        fwd, _ = measured_shadow_price(inst, cl.constraint, delta)
        bwd, _ = measured_shadow_price(inst, cl.constraint, delta / 2)
        realized_bp = fwd if abs(fwd) >= abs(bwd) else bwd  # informative side
        is_driver = abs(realized_bp) > driver_threshold_bp
        status_ok = (cl.asserted_binding == is_driver)
        mag_ok = abs(cl.asserted_sp_bp - realized_bp) <= max(abs_tol_bp, rel_tol * abs(realized_bp))
        faithful.append(bool(status_ok and mag_ok))
        hard.append(bool(cl.asserted_binding and not is_driver))
        status.append(bool(status_ok))
        mag.append(bool(mag_ok) if (status_ok and is_driver) else None)
    return ScoredExplanation(method=method, claims=claims,
                             faithful_flags=faithful, hard_failure_flags=hard,
                             status_flags=status, magnitude_flags=mag)
