# Faithful by Construction? An Exact Test of LLM Explanations for Constrained Portfolio Optimization Decisions

Code, model replies and scored results for the working paper by **Prateek Bhatnagar** (Independent Researcher), October 2026.
SSRN: [abstract 7625878](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7625878) (posted 19 October 2026; a working paper, not yet peer-reviewed).

## What this repository contains
| Path | Contents |
|---|---|
| `paper/` | LaTeX sources and the PDF of the paper |
| `code/ofs_core.py` | Linear-programming portfolio model, solver certificate, re-solve ground truth, claim scoring |
| `code/llm_adapter.py` | Prompt construction, OpenRouter calls, reply parsing |
| `code/run_final_study.py` | Builds the problems and queries the models (needs an API key) |
| `code/fetch_fred_universe.py` | Downloads the FRED series and builds the asset universes |
| `code/make_v5_*.py`, `robustness_v5.py`, `computability_v5.py` | Produce every number, the table and the figures |
| `code/audit_numbers.py` | Recomputes the reported numbers and checks them against the PDF |
| `code/final_results*.json` | Scored results for both problem sets |
| `code/final_study_log*.jsonl` | Every model reply (1,176 replies), keyed by a hash of the prompt |

The prompts themselves are not stored in the logs because they contain values derived from ICE BofA index data (see *Data* below). They are rebuilt exactly from the downloaded data and the fixed random seeds.

## Reproducing the paper (no API key needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code
./reproduce.sh
```
`reproduce.sh` downloads the FRED series, pinned to the paper's last observation date (2026-10-14), and checks the universe file against this SHA-256:
```
0251c43155c72fb485d3b5eac8216127242096bc752607ff394d1313f20f12b7  fred_universe.csv
```
It then rebuilds all numbers (`numbers_v5*.tex`), the table (`table_v5.tex`) and the figures (`fig_v5_*.pdf`) from the saved replies. To check the result against the paper:
```bash
python audit_numbers.py ../paper/Bhatnagar_Faithful-by-Construction_SSRN.pdf
```
If FRED has revised a series since October 2026, the checksum warns you and some numbers may differ slightly.

## Re-running the models (costs money)
Set `OPENROUTER_API_KEY` in your environment and see the options in `run_final_study.py`. The paper's runs were made on 15 October 2026 at temperature 0 with:
- `anthropic/claude-sonnet-4.5`
- `google/gemini-2.5-flash`
- `openai/gpt-4o-mini`

Model providers update their models, so new runs may not match the saved replies.

## Data
- US Treasury constant-maturity yields: Board of Governors of the Federal Reserve System, retrieved from FRED.
- ICE BofA option-adjusted spreads: © ICE Data Indices, LLC, retrieved from FRED. ICE's terms prohibit redistribution of these series, so **they are not included here**. `fetch_fred_universe.py` downloads them for your own use.

## Licence
- Code: MIT (see `LICENSE`).
- Results and model replies: CC BY 4.0.
- FRED/ICE data: subject to their providers' terms.

## Citation
See `CITATION.cff`. Cite as:
> Bhatnagar, P. (2026). *Faithful by Construction? An Exact Test of LLM Explanations for Constrained Portfolio Optimization Decisions.* SSRN Working Paper 7625878. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7625878

## AI disclosure
The author used Claude (Anthropic) for coding assistance, literature search and editing. An Anthropic model, Claude Sonnet 4.5, is among the models evaluated.
