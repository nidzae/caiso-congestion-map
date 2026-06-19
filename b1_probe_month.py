"""B1 — pull one summer month of RTM 5-min for both probes; resample hourly.

Caches the 5-min CSVs as parquet so reruns are free.
Self-checks: ~720 hourly rows per probe, prints NaN counts.
Inspect: probe_mcc_by_hour.png — mean Congestion vs. hour-of-day.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import gridstatus
from gridstatus import Markets

import config

START = "2025-08-01"
END = "2025-09-01"   # gridstatus end is exclusive of the trailing day in practice
SEASON_TAG = "summer2025"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def pull_or_load(node: str) -> pd.DataFrame:
    cache = DATA_DIR / f"rtm_5min_{node}_{SEASON_TAG}.parquet"
    if cache.exists():
        print(f"[{node}] using cache: {cache}")
        return pd.read_parquet(cache)

    print(f"[{node}] pulling RTM 5-min {START} -> {END} (this takes a few minutes)")
    caiso = gridstatus.CAISO()
    df = caiso.get_lmp(
        date=START,
        end=END,
        market=Markets.REAL_TIME_5_MIN,
        locations=[node],
        sleep=2,
    )
    print(f"[{node}] rows raw: {len(df)}")
    df.to_parquet(cache)
    return df


print("== B1: one-month probe pull + hourly resample ==")
print(f"window: {START} -> {END}")
print(f"probes: import={config.PROBE_IMPORT}  flat={config.PROBE_FLAT}")

raw = {}
for node in (config.PROBE_IMPORT, config.PROBE_FLAT):
    raw[node] = pull_or_load(node)

# --- Resample to hourly on the Congestion (MCC) column ---
hourly = {}
for node, df in raw.items():
    # Use Interval Start as the timestamp index (tz-aware from gridstatus)
    s = df[["Interval Start", "Congestion"]].copy()
    s = s.set_index("Interval Start").sort_index()
    # Aggregate to hourly mean MCC
    h = s["Congestion"].resample("1h").mean()
    hourly[node] = h
    print(f"\n[{node}]")
    print(f"  5-min rows:  {len(df)}")
    print(f"  hourly rows: {len(h)}")
    print(f"  hourly NaN:  {int(h.isna().sum())}")
    print(f"  MCC mean:    {h.mean():.4f}")
    print(f"  MCC std:     {h.std():.4f}")
    print(f"  MCC min,max: {h.min():.4f}, {h.max():.4f}")
    assert 700 <= len(h) <= 750, f"BLOCKER: expected ~720 hourly rows, got {len(h)}"
    assert not h.isna().all(), f"BLOCKER: all-NaN hourly series for {node}"

# Save hourly for downstream B2-B4
for node, h in hourly.items():
    out = DATA_DIR / f"rtm_hourly_{node}_{SEASON_TAG}.parquet"
    h.to_frame("Congestion").to_parquet(out)
    print(f"saved {out}")

# --- Hour-of-day mean for the chart ---
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"import": "tab:blue", "flat": "tab:gray"}
for node, label, color in [
    (config.PROBE_IMPORT, f"PROBE_IMPORT  ({config.PROBE_IMPORT})", colors["import"]),
    (config.PROBE_FLAT,   f"PROBE_FLAT    ({config.PROBE_FLAT})",   colors["flat"]),
]:
    h = hourly[node]
    by_hour = h.groupby(h.index.hour).mean()
    ax.plot(by_hour.index, by_hour.values, marker="o", label=label, color=color)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("hour of day (local)")
ax.set_ylabel("mean MCC ($/MWh)")
ax.set_title(f"Mean MCC by hour — {SEASON_TAG}")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

png = DATA_DIR / "probe_mcc_by_hour.png"
fig.savefig(png, dpi=150)
print(f"\nsaved {png}")
print("\nB1 OK — eyeball probe_mcc_by_hour.png: import should rise in evening, flat should hug zero.")
