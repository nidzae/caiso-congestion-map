"""Aggregate per-month metrics into a single Full Year view (tag = 2025-full).

This is an aggregation, not a re-fit:
  - spread, size, conc, spread_D{2,4,8}: median across the 12 months
    (median chosen for robustness — a single quiet/noisy month doesn't
    skew the annual story).
  - archetype: re-classified from the MEAN of monthly MCC_mid / MCC_eve.
  - controlling line k*: the most-frequent monthly k* for that node.
  - k* rating: the rating recorded in any month that picked that k*.
  - marker: derived from the annual mean conc and the same threshold.

Outputs (under tag 2025-full):
  data/node_metrics_with_size_2025-full.csv
  data/duration_sweep_2025-full.csv
  data/node_coordinates_2025-full.csv  (symlink to the global coords file)
"""
import os
from pathlib import Path

import pandas as pd

import config

DATA = Path("data")
TAGS = [f"2025-{m:02d}" for m in range(1, 13)]
OUT_TAG = "2025-full"
EPS = config.EPSILON_FLATNESS


def classify(import_signal: float, export_signal: float, eps: float) -> str:
    imp = import_signal >= eps
    exp = export_signal >= eps
    if imp and exp:
        return "PURPLE"
    if imp:
        return "BLUE"
    if exp:
        return "RED"
    return "WHITE"


# Load per-month tables
metrics, sweep = {}, {}
for t in TAGS:
    mp = DATA / f"node_metrics_with_size_{t}.csv"
    sp = DATA / f"duration_sweep_{t}.csv"
    if not (mp.exists() and sp.exists()):
        print(f"SKIP {t} (missing metric/sweep files)")
        continue
    metrics[t] = pd.read_csv(mp, index_col=0)
    sweep[t] = pd.read_csv(sp, index_col=0)

print(f"loaded {len(metrics)} months")
if not metrics:
    raise SystemExit("no monthly data available — nothing to aggregate")

# Build long-format frames
m_long = pd.concat([df.assign(month=t) for t, df in metrics.items()])
s_long = pd.concat([df.assign(month=t) for t, df in sweep.items()])
print(f"per-month metric rows: {len(m_long):,}  sweep rows: {len(s_long):,}")

# Per-node aggregation
print("aggregating per-node ...")
g_m = m_long.groupby(m_long.index)
g_s = s_long.groupby(s_long.index)

agg = pd.DataFrame(index=g_m.size().index)
agg.index.name = "node"

agg["MCC_mid"] = g_m["MCC_mid"].mean()
agg["MCC_eve"] = g_m["MCC_eve"].mean()
agg["import_signal"] = agg["MCC_eve"]
agg["export_signal"] = -agg["MCC_mid"]
agg["archetype"] = [classify(i, e, EPS) for i, e in zip(agg["import_signal"], agg["export_signal"])]
agg["spread"] = g_m["spread"].median()
agg["size_node"] = g_m["size_node"].median()
agg["conc"] = g_m["conc"].mean()
agg["marker"] = ["hollow" if c > config.CONCENTRATION_THRESHOLD else "filled"
                  for c in agg["conc"]]

# Controlling line — most-frequent monthly k* per node
def mode_or_none(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]

agg["kstar_physical_line"] = g_m["kstar_physical_line"].apply(mode_or_none)
# Rating + source: pull from the monthly row whose k* matches the annual k*
def rating_for_kstar(node):
    rows = m_long.loc[node]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    k = agg.loc[node, "kstar_physical_line"]
    if pd.isna(k) or k is None:
        return (None, None)
    match = rows[rows["kstar_physical_line"] == k]
    if match.empty:
        return (None, None)
    return (match["kstar_rating_MW"].iloc[0], match["kstar_rating_source"].iloc[0])

ratings = [rating_for_kstar(n) for n in agg.index]
agg["kstar_rating_MW"] = [r[0] for r in ratings]
agg["kstar_rating_source"] = [r[1] for r in ratings]

# Hours-covered: total over the year
if "hours_covered" in m_long.columns:
    agg["hours_covered"] = g_m["hours_covered"].sum()

# Optional auxiliary columns mirrored from monthly outputs (top-5 lines etc.)
for col in ["kstar_contribution"]:
    if col in m_long.columns:
        agg[col] = g_m[col].median()

# Write metric output (columns shaped like a monthly output so f1 picks it up)
out_metrics = DATA / f"node_metrics_with_size_{OUT_TAG}.csv"
cols_in_monthly = pd.read_csv(DATA / f"node_metrics_with_size_{TAGS[0]}.csv",
                                index_col=0, nrows=1).columns.tolist()
# Restrict to the union of common columns + things we computed
keep = [c for c in cols_in_monthly if c in agg.columns]
extras = [c for c in agg.columns if c not in keep and c in ("MCC_mid","MCC_eve",
            "import_signal","export_signal","spread","size_node","conc","marker",
            "archetype","kstar_physical_line","kstar_rating_MW",
            "kstar_rating_source","hours_covered","kstar_contribution")]
final_cols = list(dict.fromkeys(keep + extras))
agg_out = agg[final_cols].copy()

# Join in per-node persistence + per-line TPP overlay BEFORE writing.
persist_path = DATA / "persistence_2025.csv"
if persist_path.exists():
    p = pd.read_csv(persist_path, index_col=0)
    keep_persist = [c for c in ["n_months_active", "size_cv",
                                  "size_max_share", "same_kstar_months",
                                  "persistence_label"] if c in p.columns]
    agg_out = agg_out.join(p[keep_persist], how="left")
    print(f"joined {len(keep_persist)} persistence columns from {persist_path.name}")

tpp_path = DATA / "tpp_crosswalk.csv"
if tpp_path.exists():
    tpp = pd.read_csv(tpp_path).set_index("physical_line")
    tpp = tpp[["n_tpp_projects", "earliest_isd_active",
                "oldest_plan_year", "max_slip_years",
                "projects_summary"]]
    # Join on the controlling-line key
    if "kstar_physical_line" in agg_out.columns:
        joined = agg_out.merge(tpp, how="left",
                                 left_on="kstar_physical_line",
                                 right_index=True)
        # Preserve original index (was lost by merge)
        joined.index = agg_out.index
        agg_out = joined
        print(f"joined TPP columns from {tpp_path.name}")

agg_out.to_csv(out_metrics)
print(f"saved {out_metrics}  ({len(agg_out)} nodes, {len(agg_out.columns)} cols)")

# Duration-sweep aggregate
sweep_agg = pd.DataFrame(index=g_s.size().index)
sweep_agg.index.name = "node"
for D in (2, 4, 8):
    col = f"spread_D{D}"
    if col in s_long.columns:
        sweep_agg[col] = g_s[col].median()
out_sweep = DATA / f"duration_sweep_{OUT_TAG}.csv"
sweep_agg.to_csv(out_sweep)
print(f"saved {out_sweep}")

# Coords symlink
coord_link = DATA / f"node_coordinates_{OUT_TAG}.csv"
if not coord_link.exists():
    src = "node_coordinates_summer2025.csv"
    os.symlink(src, coord_link)
    print(f"symlinked {coord_link} -> {src}")

print("\narchetype counts (annual):")
print(agg["archetype"].value_counts().to_string())
