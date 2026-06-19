"""C1 — scale archetype, spread, concentration to all generator pnodes.

Strategy:
1. Target pnodes = the ~2,277 generator pnodes from get_pnodes().
2. For each day in August 2025, pull RTM 5-min LMP in batches of 200 locations.
   Cache each (day, batch_index) result as parquet so the run is resumable.
3. After all daily pulls complete, concat -> wide pivot on Congestion.
4. Compute per-node: archetype (B2), spread (B3), conc (B4).
5. Outputs: node_metrics_raw.csv, spread_histogram.png, archetype counts,
   coverage breakdown.

Throughput: ~10 nodes/sec sustained from OASIS.
2,277 nodes / 200 = 12 batches/day * 31 days = ~372 OASIS requests.
Estimated wall time: ~2 hours.
"""
import sys
import time
from pathlib import Path

import gridstatus
from gridstatus import Markets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

DATE_START = "2025-08-01"
DATE_END = "2025-09-01"   # exclusive
SEASON_TAG = "summer2025"
BATCH_SIZE = 200
INTER_BATCH_SLEEP = 1.0

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "mcc_raw" / SEASON_TAG
RAW_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Phase 1: bulk pull (resumable)
# -----------------------------------------------------------------------------
log("== C1 starting ==")
caiso = gridstatus.CAISO()

log("loading pnode catalog...")
pnodes = caiso.get_pnodes()
target_pnodes = sorted(pnodes["PNode ID"].dropna().unique().tolist())
log(f"target generator pnodes: {len(target_pnodes):,}")

batches = [target_pnodes[i:i + BATCH_SIZE]
           for i in range(0, len(target_pnodes), BATCH_SIZE)]
log(f"batches: {len(batches)} of up to {BATCH_SIZE} nodes")

days = pd.date_range(DATE_START, DATE_END, freq="D", inclusive="left")
log(f"days: {len(days)}  ({DATE_START} -> {DATE_END} exclusive)")

total_requests = len(batches) * len(days)
log(f"total requests planned: {total_requests}")

t_pull_start = time.time()
done_count = 0
skipped_count = 0

for di, d in enumerate(days, 1):
    day_str = d.strftime("%Y-%m-%d")
    next_day = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    day_dir = RAW_DIR / day_str
    day_dir.mkdir(exist_ok=True)

    for bi, batch in enumerate(batches):
        cache = day_dir / f"batch_{bi:03d}.parquet"
        if cache.exists():
            skipped_count += 1
            continue

        attempt = 0
        while True:
            attempt += 1
            try:
                df = caiso.get_lmp(
                    date=day_str,
                    market=Markets.REAL_TIME_5_MIN,
                    locations=batch,
                )
                break
            except Exception as e:
                if attempt >= 3:
                    log(f"  FAILED {day_str} batch {bi} after {attempt} attempts: {e}")
                    df = None
                    break
                wait = 10 * attempt
                log(f"  retry {day_str} batch {bi} attempt {attempt}: {e}; sleeping {wait}s")
                time.sleep(wait)

        if df is None or df.empty:
            # Save an empty marker so we don't retry forever
            pd.DataFrame(columns=["Interval Start", "Location", "Congestion"]).to_parquet(cache)
            done_count += 1
            continue

        keep = df[["Interval Start", "Location", "Congestion"]].copy()
        keep.to_parquet(cache)
        done_count += 1

        if done_count % 10 == 0 or bi == len(batches) - 1:
            elapsed = time.time() - t_pull_start
            done_total = done_count + skipped_count
            remain = total_requests - done_total
            rate = done_count / elapsed if elapsed > 0 else 0
            eta_s = remain / rate if rate > 0 else 0
            log(f"  day {di}/{len(days)} {day_str}  batch {bi+1}/{len(batches)}  "
                f"done={done_count} skipped={skipped_count}  "
                f"elapsed={elapsed/60:.1f}m  eta={eta_s/60:.1f}m")

        time.sleep(INTER_BATCH_SLEEP)

log(f"pull complete: {done_count} new, {skipped_count} skipped from cache, "
    f"elapsed={(time.time()-t_pull_start)/60:.1f}m")


# -----------------------------------------------------------------------------
# Phase 2: concat all cached files into a wide MCC frame
# -----------------------------------------------------------------------------
log("loading cached data...")
parts = []
for d in days:
    day_str = d.strftime("%Y-%m-%d")
    day_dir = RAW_DIR / day_str
    for p in sorted(day_dir.glob("batch_*.parquet")):
        df = pd.read_parquet(p)
        if not df.empty:
            parts.append(df)

mcc_long = pd.concat(parts, ignore_index=True)
log(f"long rows: {len(mcc_long):,}")

mcc_long["Interval Start"] = pd.to_datetime(mcc_long["Interval Start"])
# Pivot wide: rows=interval, cols=node, vals=Congestion
mcc_wide = mcc_long.pivot_table(
    index="Interval Start", columns="Location", values="Congestion", aggfunc="mean",
)
mcc_wide = mcc_wide.sort_index()
# Ensure tz is US/Pacific so hour-of-day windows work correctly
if mcc_wide.index.tz is None:
    mcc_wide.index = mcc_wide.index.tz_localize("US/Pacific")
else:
    mcc_wide.index = mcc_wide.index.tz_convert("US/Pacific")
log(f"wide: {mcc_wide.shape[0]:,} intervals × {mcc_wide.shape[1]:,} nodes")

mcc_wide_path = DATA_DIR / f"mcc_wide_{SEASON_TAG}.parquet"
mcc_wide.to_parquet(mcc_wide_path)
log(f"saved {mcc_wide_path}")

# -----------------------------------------------------------------------------
# Phase 3: metrics
# -----------------------------------------------------------------------------
log("resampling to hourly...")
hourly = mcc_wide.resample("1h").mean()
log(f"hourly: {hourly.shape[0]:,} hours × {hourly.shape[1]:,} nodes")

# --- archetype ---
def window_mean(df: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    lo, hi = window
    mask = (df.index.hour >= lo) & (df.index.hour < hi)
    return df.loc[mask].mean(axis=0)

mcc_mid = window_mean(hourly, config.MIDDAY_WINDOW)
mcc_eve = window_mean(hourly, config.EVENING_WINDOW)
import_signal = mcc_eve
export_signal = -mcc_mid

eps = config.EPSILON_FLATNESS
imp = import_signal >= eps
exp = export_signal >= eps
arch = pd.Series("WHITE", index=hourly.columns, dtype=object)
arch[imp & exp]   = "PURPLE"
arch[imp & ~exp]  = "BLUE"
arch[~imp & exp]  = "RED"

# --- spread (D=4, eta=0.85) ---
D = config.BATTERY_DURATION_HOURS
eta = config.ROUND_TRIP_EFFICIENCY
log(f"computing daily spread D={D}h eta={eta}...")

# For each calendar day, sort each node's 24 hourly values and take top/bottom D.
hourly_idx_date = hourly.index.normalize()
unique_days = hourly_idx_date.unique()
n_days = len(unique_days)
n_nodes = hourly.shape[1]
daily_spread = np.full((n_days, n_nodes), np.nan)

for i, day in enumerate(unique_days):
    mask = hourly_idx_date == day
    block = hourly.loc[mask].values  # (h, n)
    if block.shape[0] < 2 * D:
        continue
    # Sort ascending, NaN goes to end. We need to handle NaN carefully.
    sorted_asc = np.sort(block, axis=0)  # NaN at end
    # Count non-NaN per column
    valid_counts = np.sum(~np.isnan(block), axis=0)
    enough = valid_counts >= 2 * D
    if not enough.any():
        continue
    # bottom-D (charge): first D
    charge = sorted_asc[:D, :].sum(axis=0)
    # top-D (discharge) but skipping NaN: take values at indices [valid_count-D : valid_count]
    discharge = np.full(n_nodes, np.nan)
    for j in np.where(enough)[0]:
        vc = valid_counts[j]
        discharge[j] = sorted_asc[vc - D:vc, j].sum()
    ds = discharge - (1.0 / eta) * charge
    daily_spread[i, :] = np.where(enough, ds, np.nan)

spread_node = pd.Series(np.nanmedian(daily_spread, axis=0), index=hourly.columns)

# --- concentration (5-min, top 1%) ---
log("computing concentration on 5-min...")
abs_mcc = mcc_wide.abs().values
n_intervals = abs_mcc.shape[0]
top_k = max(1, int(round(0.01 * n_intervals)))
total_per_node = np.nansum(abs_mcc, axis=0)
# partial sort: top_k largest per column
top_per_node = np.full(mcc_wide.shape[1], np.nan)
for j in range(mcc_wide.shape[1]):
    col = abs_mcc[:, j]
    col_valid = col[~np.isnan(col)]
    if len(col_valid) >= top_k:
        top_per_node[j] = np.partition(col_valid, -top_k)[-top_k:].sum()
conc_node = pd.Series(np.where(total_per_node > 0, top_per_node / total_per_node, np.nan),
                      index=mcc_wide.columns)

marker = np.where(conc_node > config.CONCENTRATION_THRESHOLD, "hollow", "filled")

# --- coverage ---
hours_per_node = hourly.notna().sum(axis=0)

result = pd.DataFrame({
    "archetype": arch,
    "MCC_mid": mcc_mid,
    "MCC_eve": mcc_eve,
    "import_signal": import_signal,
    "export_signal": export_signal,
    "spread": spread_node,
    "conc": conc_node,
    "marker": marker,
    "hours_covered": hours_per_node,
})
result.index.name = "node"

out_csv = DATA_DIR / f"node_metrics_raw_{SEASON_TAG}.csv"
result.sort_values("spread", ascending=False).to_csv(out_csv)
log(f"saved {out_csv}")

# --- reports ---
print()
print("== coverage ==")
target_n = len(target_pnodes)
present_n = len(result)
missing_n = target_n - present_n
full_n = (hours_per_node >= 700).sum()
partial_n = ((hours_per_node > 0) & (hours_per_node < 700)).sum()
print(f"  target generator pnodes: {target_n:,}")
print(f"  present in pull:         {present_n:,}")
print(f"  missing entirely:        {missing_n:,}")
print(f"  full coverage (>=700h):  {full_n:,}")
print(f"  partial coverage:        {partial_n:,}")

print()
print("== archetype counts ==")
print(result["archetype"].value_counts().to_string())

print()
print("== top 10 by spread ==")
print(result.sort_values("spread", ascending=False).head(10).to_string())

# --- spread histogram ---
fig, ax = plt.subplots(figsize=(10, 5))
spreads = result["spread"].dropna()
ax.hist(spreads, bins=80, color="tab:blue", edgecolor="white")
ax.set_xlabel("spread ($/MWh)")
ax.set_ylabel("node count")
ax.set_title(f"Round-trip MCC spread distribution — {SEASON_TAG}  "
             f"(D={D}h, eta={eta})")
ax.grid(alpha=0.3)
fig.tight_layout()
hist_png = DATA_DIR / f"spread_histogram_{SEASON_TAG}.png"
fig.savefig(hist_png, dpi=150)
log(f"saved {hist_png}")

# Quick sanity: where do our two probes sit?
print()
print("== probe sanity ==")
for label, node in [("PROBE_IMPORT", config.PROBE_IMPORT),
                    ("PROBE_FLAT",   config.PROBE_FLAT)]:
    if node in result.index:
        r = result.loc[node]
        print(f"  {label} {node}: archetype={r['archetype']}, spread=${r['spread']:.2f}, conc={r['conc']:.3f}, marker={r['marker']}")
    else:
        print(f"  {label} {node}: NOT in pulled set (this is expected if it's not in the get_pnodes catalog)")

log("C1 OK.")
