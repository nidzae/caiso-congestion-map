"""A4 — inventory the queryable scope.

What we need to confirm:
1. Distinct pricing-node count (reported, not assumed).
2. The queryable RTM date range covers the seasons we want.

For (2) we probe a few sample dates rather than scraping the full availability
window — we just need to know summer + spring of a recent year work.
"""
import sys
import datetime as dt

import gridstatus
from gridstatus import Markets
import pandas as pd

import config

print("== A4: scope inventory ==")

caiso = gridstatus.CAISO()

print("\n-- pnode catalog (get_pnodes) --")
pnodes = caiso.get_pnodes()
print(f"rows: {len(pnodes)}")
print(f"columns: {list(pnodes.columns)}")
print("\nfirst 5 rows:")
print(pnodes.head().to_string())

# Distinct node count — depend on the right column. Look for a "node"/"name" column.
name_col_candidates = [c for c in pnodes.columns if any(k in c.lower() for k in ["node", "name", "pnode", "apnode"])]
print(f"\ncandidate identifier columns: {name_col_candidates}")

# Pick the most-distinct string column as the node identifier
best_col, best_n = None, 0
for c in name_col_candidates or pnodes.select_dtypes(include="object").columns:
    n = pnodes[c].nunique(dropna=True)
    if n > best_n:
        best_col, best_n = c, n
print(f"selected identifier column: {best_col!r}  ({best_n} unique)")

assert best_n > 0, "BLOCKER: pnode catalog empty"

# Per-type breakdown if a type column is exposed
type_col_candidates = [c for c in pnodes.columns if "type" in c.lower()]
if type_col_candidates:
    tc = type_col_candidates[0]
    print(f"\nbreakdown by {tc!r}:")
    print(pnodes[tc].value_counts(dropna=False).head(20).to_string())

# ---- Date range probe ----
print("\n-- date-range probe (RTM 5-min on PROBE_FLAT) --")
SAMPLES = [
    "2023-06-15",  # 3 years back
    "2024-01-15",  # winter ~2 years back
    "2025-04-15",  # spring 1 year back
    "2025-08-15",  # summer 1 year back
    "2025-12-15",  # winter 6 months back
    "2026-03-15",  # spring this year
    "2026-06-15",  # already used (sanity)
]

for date in SAMPLES:
    try:
        df = caiso.get_lmp(
            date=date,
            market=Markets.REAL_TIME_5_MIN,
            locations=[config.PROBE_FLAT],
        )
        ok = (not df.empty) and 250 <= len(df) <= 320
        status = "OK " if ok else "BAD"
        print(f"  {date}: {status}  rows={len(df)}")
    except Exception as e:
        print(f"  {date}: ERROR  {type(e).__name__}: {e}")

print("\nA4 — done. Read the printed node count and confirm sample dates cover all four seasons.")
