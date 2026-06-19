"""Probe the bulk OASIS LMP endpoint to scope Phase C run time.

We try a few candidate parameterizations of the lmp_real_time_5_min dataset
to find the fastest way to pull all generator pnodes for one day.

Goal: get a working bulk-pull command + measured throughput so we can
estimate the full month run time before committing.
"""
import time
from pathlib import Path

import gridstatus
import pandas as pd

DATE = "2025-08-01"
END = "2025-08-02"

caiso = gridstatus.CAISO()


def try_call(label: str, params: dict | None):
    print(f"\n-- {label} --")
    print(f"params: {params}")
    t0 = time.time()
    try:
        df = caiso.get_oasis_dataset(
            dataset="lmp_real_time_5_min",
            date=DATE,
            end=END,
            params=params,
        )
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return None
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")
    # Find the node-identifier column
    cand = [c for c in df.columns if "node" in c.lower() or c.lower() == "node_id"]
    if cand:
        col = cand[0]
        uniq = df[col].nunique()
        print(f"  distinct {col}: {uniq:,}")
        print(f"  first 5 nodes: {df[col].dropna().unique()[:5].tolist()}")
    return df


# Candidates — based on the schema hint in earlier debug output:
# 'params': {'market_run_id': 'RTM', 'node': None, 'grp_type': [None, 'ALL', 'ALL_APNODES']}
candidates = [
    ("ALL_APNODES",          {"market_run_id": "RTM", "grp_type": "ALL_APNODES"}),
    ("ALL",                  {"market_run_id": "RTM", "grp_type": "ALL"}),
]

results = {}
for label, params in candidates:
    df = try_call(label, params)
    if df is not None and not df.empty:
        results[label] = df

print("\n== summary ==")
for label, df in results.items():
    print(f"\n[{label}]  rows={len(df):,}")
    # Estimate disk footprint
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  memory: ~{size_mb:.1f} MB in RAM")
