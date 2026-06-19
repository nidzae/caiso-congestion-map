"""B4 — bankability concentration (marker style channel).

Per brief §3e:
  conc_node = sum(|MCC| in top 1% of intervals) / sum(|MCC| over all intervals)
  hollow (spike-driven) if conc > threshold (default 0.5); else filled

Runbook says use native 5-min resolution for spike detection (not hourly),
so a 31-day month gives 8928 intervals and the top 1% is ~89 intervals.

|MCC| is the magnitude — positive and negative MCC both represent
congestion value (one is import-side rent, the other export-side discount
the battery can charge into). Sign-agnostic concentration is the right
metric for "how spike-driven is this node?".
"""
import sys
from pathlib import Path

import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"
TOP_FRAC = 0.01


def fivemin_path(node: str) -> Path:
    return DATA_DIR / f"rtm_5min_{node}_{SEASON_TAG}.parquet"


def concentration(mcc: pd.Series, top_frac: float) -> dict:
    abs_mcc = mcc.abs()
    n = len(abs_mcc)
    k = max(1, int(round(top_frac * n)))
    sorted_desc = abs_mcc.sort_values(ascending=False)
    top_sum = float(sorted_desc.iloc[:k].sum())
    total_sum = float(abs_mcc.sum())
    conc = top_sum / total_sum if total_sum > 0 else float("nan")
    return {
        "intervals": n,
        "top_k": k,
        "top_sum": top_sum,
        "total_sum": total_sum,
        "conc": conc,
        "peak_abs_mcc": float(abs_mcc.max()),
    }


print("== B4: bankability concentration ==")
print(f"native resolution: 5-min")
print(f"top fraction: {TOP_FRAC:.2%}")
print(f"conc threshold (hollow vs filled): {config.CONCENTRATION_THRESHOLD}")

rows = []
for label, node in [("PROBE_IMPORT", config.PROBE_IMPORT),
                    ("PROBE_FLAT",   config.PROBE_FLAT)]:
    p = fivemin_path(node)
    if not p.exists():
        print(f"BLOCKER: missing 5-min cache {p}; rerun B1.")
        sys.exit(1)
    df = pd.read_parquet(p)
    mcc = df["Congestion"]
    stats = concentration(mcc, TOP_FRAC)
    marker = "hollow" if stats["conc"] > config.CONCENTRATION_THRESHOLD else "filled"
    rows.append({"probe": label, "node": node, "marker": marker, **stats})

table = pd.DataFrame(rows)
print()
print(table.to_string(index=False))

# Sanity assertion: conc in [0, 1]
for r in rows:
    assert 0 <= r["conc"] <= 1, f"BLOCKER: conc for {r['probe']} = {r['conc']:.4f} out of [0,1]"

print()
print("B4 OK — both conc values in [0,1].")
print()
print("Interpretation:")
for r in rows:
    print(f"  {r['probe']} ({r['node']}): conc={r['conc']:.3f} -> {r['marker']}")
    print(f"      peak |MCC| = ${r['peak_abs_mcc']:.2f}; top {r['top_k']} of {r['intervals']} intervals = ${r['top_sum']:.0f}; total = ${r['total_sum']:.0f}")
