"""D1 — build the shadow-price panel for August 2025.

Two OASIS datasets (both 5-min RTM):
  - interval_intertie_constraint_shadow_prices  (PRC_RTM_FLOWGATE)
  - interval_nomogram_branch_shadow_prices      (PRC_RTM_NOMOGRAM)

A single nomogram can bind under multiple contingencies (CONSTRAINT_CAUSE),
each of which is a distinct constraint instance for the regression.
Composite keys:
  nomogram: f"NOM:{NOMOGRAM_ID}|{CONSTRAINT_CAUSE}"
  intertie: f"ITC:{TI_ID}|{TI_DIRECTION}|{CONSTRAINT_CAUSE}"

Output:
  data/shadow_panel_summer2025.parquet  (long-format: time, constraint_key, mu)
  data/shadow_panel_wide_summer2025.parquet  (wide: time x constraint, NaN -> 0)
  data/constraint_summary_summer2025.csv (one row per constraint, with rankings)
"""
import sys
import sys
import time
from pathlib import Path

import gridstatus
import numpy as np
import pandas as pd

import config

# CLI: python3 d1_shadow_panel.py <DATE_START> <DATE_END> <SEASON_TAG>
DATE_START = "2025-08-01"
DATE_END = "2025-09-01"
SEASON_TAG = "summer2025"
if len(sys.argv) >= 4:
    DATE_START, DATE_END, SEASON_TAG = sys.argv[1], sys.argv[2], sys.argv[3]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CACHE_ITC = DATA_DIR / f"itc_raw_{SEASON_TAG}.parquet"
CACHE_NOM = DATA_DIR / f"nom_raw_{SEASON_TAG}.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


caiso = gridstatus.CAISO()


def pull_or_load(dataset_name: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        log(f"using cache: {cache}")
        return pd.read_parquet(cache)
    log(f"pulling {dataset_name} {DATE_START} -> {DATE_END}")
    t0 = time.time()
    df = caiso.get_oasis_dataset(dataset=dataset_name, date=DATE_START, end=DATE_END)
    log(f"  rows: {len(df):,}  elapsed: {time.time()-t0:.0f}s")
    df.to_parquet(cache)
    return df


log("== D1: shadow-price panel ==")
itc = pull_or_load("interval_intertie_constraint_shadow_prices", CACHE_ITC)
nom = pull_or_load("interval_nomogram_branch_shadow_prices", CACHE_NOM)

# Normalize and build composite keys
itc["constraint_key"] = ("ITC:" + itc["TI_ID"].astype(str)
                         + "|" + itc["TI_DIRECTION"].astype(str)
                         + "|" + itc["CONSTRAINT_CAUSE"].astype(str))
nom["constraint_key"] = ("NOM:" + nom["NOMOGRAM_ID"].astype(str)
                         + "|" + nom["CONSTRAINT_CAUSE"].astype(str))

# Common columns: interval_start, constraint_key, mu (= PRC)
itc_long = itc[["INTERVALSTARTTIME_GMT", "constraint_key", "PRC"]].rename(
    columns={"INTERVALSTARTTIME_GMT": "interval_start", "PRC": "mu"})
nom_long = nom[["INTERVALSTARTTIME_GMT", "constraint_key", "PRC"]].rename(
    columns={"INTERVALSTARTTIME_GMT": "interval_start", "PRC": "mu"})
panel_long = pd.concat([itc_long, nom_long], ignore_index=True)
panel_long["interval_start"] = pd.to_datetime(panel_long["interval_start"], utc=True)

log(f"combined long rows: {len(panel_long):,}")
log(f"distinct constraint keys: {panel_long['constraint_key'].nunique():,}")

# Save long panel
long_out = DATA_DIR / f"shadow_panel_{SEASON_TAG}.parquet"
panel_long.to_parquet(long_out)
log(f"saved {long_out}")

# Build wide panel (time x constraint), filling absent intervals with 0
# (a constraint is non-binding -> shadow price 0)
log("pivoting to wide...")
panel_wide = panel_long.pivot_table(
    index="interval_start", columns="constraint_key",
    values="mu", aggfunc="mean",
).fillna(0.0)
# Reindex to the full 5-min grid for August 2025 (in Pacific)
full_grid = pd.date_range(
    start=pd.Timestamp(DATE_START, tz="US/Pacific"),
    end=pd.Timestamp(DATE_END, tz="US/Pacific"),
    freq="5min", inclusive="left",
).tz_convert("UTC")
panel_wide = panel_wide.reindex(full_grid, fill_value=0.0)
log(f"wide panel: {panel_wide.shape[0]:,} intervals × {panel_wide.shape[1]:,} constraints")

wide_out = DATA_DIR / f"shadow_panel_wide_{SEASON_TAG}.parquet"
panel_wide.to_parquet(wide_out)
log(f"saved {wide_out}")

# Per-constraint summary
log("computing per-constraint rankings...")
summary = pd.DataFrame({
    "n_binding_intervals": (panel_wide != 0).sum(axis=0),
    "sum_mu":              panel_wide.sum(axis=0),
    "sum_abs_mu":          panel_wide.abs().sum(axis=0),
    "max_mu":              panel_wide.max(axis=0),
    "min_mu":              panel_wide.min(axis=0),
})
summary["abs_sum_mu"] = summary["sum_mu"].abs()
summary = summary.sort_values("sum_abs_mu", ascending=False)

# Self-check
n_constraints_raw = summary.shape[0]
n_all_zero = (summary["sum_abs_mu"] == 0).sum()
if n_all_zero:
    zero_keys = summary.index[summary["sum_abs_mu"] == 0].tolist()
    print(f"NOTE: dropping {n_all_zero} constraints that appeared in the report with PRC=0 only "
          f"(OASIS publication artifacts, not real binding constraints):")
    for k in zero_keys:
        print(f"  - {k}")
    summary = summary[summary["sum_abs_mu"] > 0]
    panel_wide = panel_wide.drop(columns=zero_keys)
    log(f"after dropping all-zero columns, panel: {panel_wide.shape[1]:,} constraints")
n_constraints = summary.shape[0]
assert n_constraints > 0, "BLOCKER: panel has 0 constraints after cleanup"

# Type tag for readability
summary["type"] = summary.index.str.split(":").str[0]

summary_out = DATA_DIR / f"constraint_summary_{SEASON_TAG}.csv"
summary.to_csv(summary_out)
log(f"saved {summary_out}")

print()
print(f"== panel summary ==")
print(f"  total constraints: {n_constraints:,}")
print(f"  intertie:    {(summary['type']=='ITC').sum():,}")
print(f"  nomogram:    {(summary['type']=='NOM').sum():,}")
print(f"  intervals in panel: {panel_wide.shape[0]:,}")

print()
print("== top 15 by Σ|μ| (volatility-weighted) ==")
top_abs = summary.head(15).copy()
print(top_abs[["type", "n_binding_intervals", "sum_abs_mu", "sum_mu", "max_mu", "min_mu"]].to_string())

print()
print("== top 15 by |Σμ| (directional, signed-rent style) ==")
top_dir = summary.sort_values("abs_sum_mu", ascending=False).head(15)
print(top_dir[["type", "n_binding_intervals", "abs_sum_mu", "sum_mu", "max_mu", "min_mu"]].to_string())

print()
print("== top 15 by binding-interval count (most persistent constraints) ==")
top_persist = summary.sort_values("n_binding_intervals", ascending=False).head(15)
print(top_persist[["type", "n_binding_intervals", "sum_abs_mu", "sum_mu"]].to_string())

log("D1 OK.")
