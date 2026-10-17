"""
llm_adapter.py
==============
Real free-form (FF) explanation generation via OpenRouter, for task A1.

Design principles that keep the experiment defensible:
  * The model is shown the problem data, the constraint list, and the solved
    weights x* -- but NEVER the certificate, the duals, or the word "binding".
    Leaking any of those would inflate FF's faithfulness score and invalidate
    the comparison. This is the single most important methodological guard.
  * The model must end its reply with a strict JSON block naming each
    constraint it considers a *driver* and its estimated book-yield cost in bp.
    We parse only that block, so scoring is deterministic and identical to the
    simulated narrator's scoring.
  * temperature is fixed (default 0) and recorded; model id + timestamp are
    stored on every Claim set for the methods/appendix section.

USAGE
-----
    export OPENROUTER_API_KEY="sk-or-..."
    # then in run_experiment.py select engine="openrouter", model=<id>

Requires: `requests` (pip install requests).  No other dependency.

Swap-in point: run_experiment.py passes a callable ff_fn(inst, cert, rng) ->
list[Claim].  `make_openrouter_ff(model_id)` returns exactly such a callable.
"""
from __future__ import annotations
import os, json, time, re
import numpy as np

from ofs_core import Claim, CONSTRAINTS, CONSTRAINT_LABELS, UNIT_TEXT
import hashlib

try:
    import requests
except ImportError:
    requests = None

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompt construction (no certificate leakage)
# ---------------------------------------------------------------------------
def build_prompt(inst, x, informed: bool = False, neutral_example: bool = False) -> tuple[str, str]:
    """Return (system, user) messages. x = solved weights (shown to model).

    informed=False : data + constraints + chosen weights (the v4 run's setup).
    informed=True  : additionally states the portfolio's realized average
                     duration, average quality and how many positions sit at
                     the cap -- everything an analyst needs to tell which
                     limits are tight -- but still NO shadow prices/certificate.
    """
    n = len(x)
    # Show only the held bonds (weight > 0) plus a sample of the universe,
    # to keep the prompt compact but honest.
    lines = []
    held = np.where(x > 1e-6)[0]
    for i in held:
        lines.append(
            f"  bond {i:>3}: weight={x[i]*100:5.2f}%  "
            f"yield={inst.y[i]*100:4.2f}%  duration={inst.d[i]:4.1f}y  "
            f"quality_score={inst.q[i]:.2f}"
        )
    example_line = ('Format of the final line (placeholders, not values): '
                    '{"<constraint_key>": <your bp estimate>, ...}') if neutral_example else \
                   'Example final line: {"issuer_cap": 12.0, "duration_ceiling": 4.5}'
    facts = ""
    if informed:
        avg_d = float(inst.d @ x); avg_q = float(inst.q @ x)
        at_cap = int(np.sum(np.isclose(x, inst.cap, atol=1e-6)))
        facts = (f"Portfolio facts: average duration = {avg_d:.3f}y (ceiling {inst.D_max:.3f}y); "
                 f"average quality_score = {avg_q:.3f} (floor {inst.Q_min:.3f}); "
                 f"{at_cap} of {len(held)} held bonds are exactly at the {inst.cap*100:.2f}% cap.\n\n")
    universe_note = (
        f"The investable universe has {n} bonds; {len(held)} are held "
        f"(shown below). Higher quality_score = higher credit quality."
    )
    system = (
        "You are a fixed-income portfolio analyst. You explain, in plain "
        "language, why an optimizer chose a given bond portfolio. Be specific "
        "about which constraints most shaped the allocation and roughly how "
        "much each one costs in book yield. Base your explanation only on the "
        "information provided."
    )
    user = f"""A portfolio optimizer maximized book yield subject to these constraints:
  - single-issuer cap: no bond may exceed {inst.cap*100:.2f}% of the portfolio
  - duration ceiling: portfolio average duration must be <= {inst.D_max:.2f} years
  - credit-quality floor: portfolio average quality_score must be >= {inst.Q_min:.2f}
  - fully invested: weights sum to 100%

{universe_note}

Chosen portfolio (book yield = {float(inst.y @ x)*100:.2f}%):
{chr(10).join(lines)}

{facts}Explain in 3-5 sentences why this portfolio looks the way it does: which
constraints actively shaped it, and for each of those, approximately how many
basis points of book yield could be gained by relaxing it, measured in these
units:
  - issuer_cap: bp {UNIT_TEXT["issuer_cap"]}
  - duration_ceiling: bp {UNIT_TEXT["duration_ceiling"]}
  - quality_floor: bp {UNIT_TEXT["quality_floor"]}
A constraint that is not tight would gain 0 bp and is NOT a driver.

Then, on the LAST line, output ONLY a JSON object of the constraints you judge
to be DRIVERS of the allocation, mapping each to your estimated gain in bp in
the units above. Use exactly these keys where applicable:
"issuer_cap", "duration_ceiling", "quality_floor".
{example_line}
Include a constraint only if you believe relaxing it would raise book yield.
Output nothing after the JSON."""
    return system, user


# ---------------------------------------------------------------------------
# Parsing the model reply into Claims
# ---------------------------------------------------------------------------
def parse_reply_to_claims(text: str) -> list:
    """Extract the final JSON object and convert to Claim list.
    Robust to code fences and trailing whitespace."""
    # find the last {...} block in the reply
    matches = re.findall(r"\{[^{}]*\}", text.replace("\n", " "))
    claims = []
    if not matches:
        return claims
    blob = matches[-1]
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        # try to salvage: replace single quotes
        try:
            d = json.loads(blob.replace("'", '"'))
        except Exception:
            return claims
    for k, v in d.items():
        key = k.strip().lower()
        if key in CONSTRAINTS:
            try:
                bp = float(v)
            except (TypeError, ValueError):
                bp = 1.0
            claims.append(Claim(constraint=key, asserted_binding=True,
                                asserted_sp_bp=max(0.0, bp)))
    return claims


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------
def call_openrouter(system: str, user: str, model: str,
                    temperature: float = 0.0, max_retries: int = 4) -> str:
    if requests is None:
        raise RuntimeError("pip install requests")
    key = os.environ.get("OPENROUTER_API_KEY") or _key_from_dotenv()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment.")
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 700,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    last = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              data=json.dumps(payload), timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"].get("content") or ""
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 401, 402, 403, 404):
                break   # not retryable (bad key / model / credits)
        except Exception as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))  # backoff for rate limits
    raise RuntimeError(f"OpenRouter failed after {max_retries} tries: {last}")


# ---------------------------------------------------------------------------
# Factory: returns an ff_fn(inst, cert, rng) -> list[Claim]
# ---------------------------------------------------------------------------
def _key_from_dotenv():
    for p in (".env", "../.env"):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _stable_key(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def make_openrouter_ff(model: str, temperature: float = 0.0,
                       cache: dict | None = None, log_path: str | None = None,
                       informed: bool = False, condition: str | None = None,
                       neutral_example: bool = False):
    """Build a free-form explainer backed by a real model.

    cache    : dict of stable-hash -> reply. Load it from the log with
               load_reply_cache() so re-runs never re-bill a prompt.
    log_path : appends every prompt+reply (with model + condition) for audit.
    informed : use the 'informed' prompt (portfolio facts, still no duals).
    """
    cond = condition or ("FF_informed" if informed else "FF")

    def ff_fn(inst, cert, rng):
        # cert.x (the chosen weights) is shown; duals/binding flags are NOT.
        system, user = build_prompt(inst, cert.x, informed=informed,
                                    neutral_example=neutral_example)
        k = _stable_key(model, cond, user)
        if cache is not None and k in cache:
            reply = cache[k]
        else:
            reply = call_openrouter(system, user, model, temperature)
            if cache is not None:
                cache[k] = reply
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"model": model, "condition": cond,
                                        "key": k, "prompt": user, "reply": reply,
                                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        claims = parse_reply_to_claims(reply)
        ff_fn.n_calls += 1
        ff_fn.n_empty += int(len(claims) == 0)
        return claims

    ff_fn.model = model
    ff_fn.condition = cond
    ff_fn.temperature = temperature
    ff_fn.n_calls = 0
    ff_fn.n_empty = 0
    return ff_fn


def load_reply_cache(log_path: str) -> dict:
    """Rebuild the reply cache from a log so interrupted runs resume for free."""
    cache = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            try:
                e = json.loads(line)
                if "key" in e:
                    cache[e["key"]] = e["reply"]
            except json.JSONDecodeError:
                pass
    return cache


def available_models() -> set:
    """Model ids OpenRouter currently serves (no key needed)."""
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    return {m["id"] for m in r.json().get("data", [])}


# ---------------------------------------------------------------------------
# Recommended models to run (edit freely). Chosen to span vendors/sizes so a
# gap that holds across all of them is a strong, model-agnostic result.
# ---------------------------------------------------------------------------
RECOMMENDED_MODELS = [
    "openai/gpt-4o-mini",              # the v4 run's model (continuity)
    "openai/gpt-4o",                   # frontier OpenAI
    "anthropic/claude-3.5-sonnet",     # frontier Anthropic
    "google/gemini-flash-1.5",         # cheap Google
    "meta-llama/llama-3.1-70b-instruct",  # strong open-weight
]
# One model per vendor; run_final_study picks the first id in each list that
# OpenRouter currently serves, so stale ids are skipped instead of failing.
VENDOR_CANDIDATES = {
    "openai":    ["openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-5-mini"],
    "anthropic": ["anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4",
                  "anthropic/claude-3.7-sonnet", "anthropic/claude-3.5-sonnet"],
    "google":    ["google/gemini-2.5-flash", "google/gemini-2.0-flash-001",
                  "google/gemini-flash-1.5"],
    "meta":      ["meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-70b-instruct"],
}


if __name__ == "__main__":
    # Offline self-test of the parser (no API call).
    sample = 'The cap and duration mostly shaped it.\n{"issuer_cap": 12.0, "duration_ceiling": 8.5}'
    print("parsed:", parse_reply_to_claims(sample))
