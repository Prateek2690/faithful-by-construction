#!/usr/bin/env bash
# Rebuild every number, the table and the figures from the saved model replies (no API calls).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f fred_universe.csv ]; then
  python fetch_fred_universe.py --dates 6 --end 2026-10-14 --out fred_universe.csv
fi
echo "0251c43155c72fb485d3b5eac8216127242096bc752607ff394d1313f20f12b7  fred_universe.csv" | shasum -a 256 -c - \
  || echo "WARNING: fred_universe.csv differs from the paper's data (FRED may have revised a series)."
python make_v5_numbers.py
python make_v5_example.py
python robustness_v5.py
python computability_v5.py
python make_v5_table.py
python make_v5_figures.py
echo "Done. Optional check against the PDF: python audit_numbers.py ../paper/Bhatnagar_Faithful-by-Construction_SSRN.pdf"
