"""G2 — duration sweep on existing summer 2025 data.

Recompute round-trip MCC spread for D ∈ {2, 4, 8} hours, η = 0.85,
using the already-pulled mcc_wide_summer2025.parquet.

What we expect (runbook inspect):
  longer duration → different nodes light up, because longer-binding
  constraints reward storage that can hold for more contiguous hours.
  Short-D arbitrage rewards spike-heavy nodes; long-D rewards
  persistently-spread nodes.

Outputs:
  data/duration_sweep_summer2025.csv  — per-node spread at D=2,4,8 + rank shifts
  Printed comparison + top-20 by each D
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"
DURATIONS = [2, 4, 8]
ETA = config.ROUND_TRIP_EFFICIENCY


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def daily_spread_matrix(hourly: pd.DataFrame, D: int, eta: float) -> pd.DataFrame:
    """Returns (n_days × n_nodes) of daily-spread values."""
    idx_date = hourly.index.normalize()
    unique_days = idx_date.unique()
    N = hourly.shape[1]
    out = np.full((len(unique_days), N), np.nan)
    for i, day in enumerate(unique_days):
        mask = idx_date == day
        block = hourly.loc[mask].values
        if block.shape[0] < 2 * D:
            continue
        sorted_asc = np.sort(block, axis=0)
        valid_counts = np.sum(~np.isnan(block), axis=0)
        enough = valid_counts >= 2 * D
        charge = sorted_asc[:D, :].sum(axis=0)
        discharge = np.full(N, np.nan)
        for j in np.where(enough)[0]:
            vc = valid_counts[j]
            discharge[j] = sorted_asc[vc - D:vc, j].sum()
        ds = discharge - (1.0 / eta) * charge
        out[i, :] = np.where(enough, ds, np.nan)
    return pd.DataFrame(out, columns=hourly.columns)


log("loading mcc_wide...")
mcc_wide = pd.read_parquet(DATA_DIR / f"mcc_wide_{SEASON_TAG}.parquet")
if mcc_wide.index.tz is None:
    mcc_wide.index = pd.to_datetime(mcc_wide.index, utc=True)
else:
    mcc_wide.index = mcc_wide.index.tz_convert("US/Pacific")
hourly = mcc_wide.resample("1h").mean()
log(f"hourly: {hourly.shape[0]:,} × {hourly.shape[1]:,}")

# Compute spread per duration
spread_by_d = {}
for D in DURATIONS:
    log(f"computing spread for D={D}h...")
    ds = daily_spread_matrix(hourly, D, ETA)
    spread_by_d[D] = ds.median(axis=0)

result = pd.DataFrame({f"spread_D{D}": spread_by_d[D] for D in DURATIONS})
# Also rank per duration so we can see shifts
for D in DURATIONS:
    result[f"rank_D{D}"] = result[f"spread_D{D}"].rank(ascending=False, method="min")
result["rank_shift_2_vs_8"] = (result["rank_D8"] - result["rank_D2"]).abs()
result.index.name = "node"

out = DATA_DIR / f"duration_sweep_{SEASON_TAG}.csv"
result.to_csv(out)
log(f"saved {out}")

# --- comparison ---
print()
print("== summary statistics ==")
for D in DURATIONS:
    s = result[f"spread_D{D}"]
    print(f"  D={D}h:  median ${s.median():.0f}   p90 ${s.quantile(0.9):.0f}   "
          f"max ${s.max():.0f}")

print()
print("== correlation of node-level spread between durations ==")
print(result[[f"spread_D{D}" for D in DURATIONS]].corr().round(3).to_string())

print()
print("== top 20 by spread_D2 (short-duration storage favors spike nodes) ==")
top_d2 = result.sort_values("spread_D2", ascending=False).head(20)
print(top_d2[["spread_D2","spread_D4","spread_D8","rank_D2","rank_D4","rank_D8"]].to_string())

print()
print("== top 20 by spread_D8 (long-duration storage favors persistent spread) ==")
top_d8 = result.sort_values("spread_D8", ascending=False).head(20)
print(top_d8[["spread_D2","spread_D4","spread_D8","rank_D2","rank_D4","rank_D8"]].to_string())

print()
print("== biggest rank shifts (top 15) — nodes whose ranking changes most between D=2 and D=8 ==")
big_shifts = (result[result["spread_D2"].notna() & result["spread_D8"].notna()]
                .sort_values("rank_shift_2_vs_8", ascending=False).head(15))
print(big_shifts[["spread_D2","spread_D8","rank_D2","rank_D8","rank_shift_2_vs_8"]].to_string())

# Probe sanity
print()
print(f"PROBE_IMPORT {config.PROBE_IMPORT}:")
if config.PROBE_IMPORT in result.index:
    print(result.loc[config.PROBE_IMPORT, [f"spread_D{D}" for D in DURATIONS] +
                                            [f"rank_D{D}" for D in DURATIONS]].to_string())
