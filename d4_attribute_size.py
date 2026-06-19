"""D4 — constraint rent and node size attribution.

Per brief §3d:
  constraint_rent_k = rating_k × Σ_t |μ_k(t)|   (using |μ| so size ≥ 0)
  size_node         = constraint_rent_{k*}
  k* = controlling physical line for the node

For each node n:
  1. Ridge-fit MCC_n(t) ≈ Σ_k β_{n,k} μ_k(t)   (multi-output, one solve)
  2. Aggregate contributions by physical line (sum scenarios for same wire)
  3. k* = argmax over physical lines of mean_t |β·μ|
  4. size_node = rating[k*] × Σ|μ_{k*}|

Multi-output Ridge runs in <1 minute for 2,172 nodes × 287 constraints.
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config

# CLI: python3 d4_attribute_size.py <SEASON_TAG>
DATA_DIR = Path("data")
SEASON_TAG = "summer2025"
if len(sys.argv) >= 2:
    SEASON_TAG = sys.argv[1]


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def physical_line(key: str) -> str:
    if "|" not in key:
        return key
    return key.partition("|")[0]


log("== D4: constraint rent + size attribution ==")

log("loading data...")
mcc_wide = pd.read_parquet(DATA_DIR / f"mcc_wide_{SEASON_TAG}.parquet")
panel = pd.read_parquet(DATA_DIR / f"shadow_panel_wide_{SEASON_TAG}.parquet")
crosswalk = pd.read_csv(DATA_DIR / f"constraint_rating_crosswalk_{SEASON_TAG}.csv")
metrics = pd.read_csv(DATA_DIR / f"node_metrics_raw_{SEASON_TAG}.csv", index_col=0)

# Drop the two all-zero columns from panel (consistent with D2)
panel = panel.loc[:, panel.abs().sum() > 0]

# Align timestamps (both should be tz-aware UTC after pivot)
if mcc_wide.index.tz is None:
    mcc_wide.index = pd.to_datetime(mcc_wide.index, utc=True)
else:
    mcc_wide.index = mcc_wide.index.tz_convert("UTC")
if panel.index.tz is None:
    panel.index = pd.to_datetime(panel.index, utc=True)

common = mcc_wide.index.intersection(panel.index)
log(f"common intervals: {len(common):,}")

mcc_aligned = mcc_wide.loc[common]
panel_aligned = panel.loc[common]

# The 2025-08-20 batch-0 failure left ~200 nodes (those with first-batch IDs,
# alphabetically early — including Alamitos) with NaN for that whole day.
# Drop those 288 intervals from BOTH X and Y so every node keeps full coverage.
nan_per_row = mcc_aligned.isna().sum(axis=1)
bad_threshold = max(50, int(0.05 * mcc_aligned.shape[1]))
bad_rows = nan_per_row > bad_threshold
log(f"dropping {int(bad_rows.sum()):,} rows with >{bad_threshold} NaN nodes (the 8/20 outage window)")
mcc_aligned = mcc_aligned.loc[~bad_rows]
panel_aligned = panel_aligned.loc[~bad_rows]

# Any node that STILL has NaN after the row drop has scattered gaps; drop those.
n_before = mcc_aligned.shape[1]
mcc_aligned = mcc_aligned.dropna(axis=1)
n_after = mcc_aligned.shape[1]
if n_before != n_after:
    log(f"dropped {n_before - n_after} nodes with scattered NaN gaps")
log(f"regression Y shape: {mcc_aligned.shape}")

X = panel_aligned.values
Y = mcc_aligned.values
T, K = X.shape
N = Y.shape[1]
node_ids = mcc_aligned.columns.tolist()
constraint_keys = panel.columns.tolist()
log(f"regression: T={T:,} samples, K={K} constraints, N={N:,} nodes")

# ---- Ridge multi-output fit ----
log("Ridge alpha=10 multi-output fit...")
ridge = Ridge(alpha=10.0, fit_intercept=True)
ridge.fit(X, Y)
B = ridge.coef_  # shape (N, K)
log(f"coefficients: {B.shape}")

# ---- Per-(node, constraint) contribution: mean_t |β * μ| = |β| * mean_t|μ| ----
mean_abs_X = np.abs(X).mean(axis=0)              # (K,)
sum_abs_X  = np.abs(X).sum(axis=0)               # (K,)
contrib = np.abs(B) * mean_abs_X[None, :]        # (N, K)

# ---- Aggregate by physical line ----
phys_lines = [physical_line(k) for k in constraint_keys]
unique_phys = sorted(set(phys_lines))
phys_idx = {p: i for i, p in enumerate(unique_phys)}
M = len(unique_phys)

contrib_by_phys = np.zeros((N, M))
sum_abs_mu_per_phys = np.zeros(M)
for k_idx, phys in enumerate(phys_lines):
    j = phys_idx[phys]
    contrib_by_phys[:, j] += contrib[:, k_idx]
    sum_abs_mu_per_phys[j] += sum_abs_X[k_idx]

# Per-node controlling physical line
kstar_idx = contrib_by_phys.argmax(axis=1)
kstar = [unique_phys[i] for i in kstar_idx]

# ---- Rating lookup ----
rating_lookup = dict(zip(crosswalk["physical_line"], crosswalk["rating_MW"]))
rating_source_lookup = dict(zip(crosswalk["physical_line"], crosswalk["rating_source"]))
rating_per_phys = np.array([rating_lookup.get(p, np.nan) for p in unique_phys])
constraint_rent_per_phys = rating_per_phys * sum_abs_mu_per_phys

# size_node = constraint_rent[k*]
size_per_node = constraint_rent_per_phys[kstar_idx]
rating_per_node = rating_per_phys[kstar_idx]
sum_abs_mu_per_node = sum_abs_mu_per_phys[kstar_idx]
kstar_contribution = contrib_by_phys[np.arange(N), kstar_idx]
rating_source_per_node = [rating_source_lookup.get(p, "unknown") for p in kstar]

# Top-5 controlling lines per node, for richer reporting
top5_lines = []
top5_shares = []
for i in range(N):
    contribs = contrib_by_phys[i, :]
    total = contribs.sum()
    order = np.argsort(contribs)[::-1][:5]
    lines = [unique_phys[j] for j in order]
    shares = (contribs[order] / total) if total > 0 else np.zeros(5)
    top5_lines.append(" | ".join(lines))
    top5_shares.append(" | ".join(f"{s:.0%}" for s in shares))

attribution_df = pd.DataFrame({
    "kstar_physical_line":      kstar,
    "kstar_rating_MW":          rating_per_node,
    "kstar_rating_source":      rating_source_per_node,
    "kstar_sum_abs_mu":         sum_abs_mu_per_node,
    "size_node":                size_per_node,
    "kstar_contribution":       kstar_contribution,
    "top5_lines":               top5_lines,
    "top5_shares":              top5_shares,
}, index=node_ids)
attribution_df.index.name = "node"

# Merge with metrics
result = metrics.join(attribution_df, how="left")

# Reorder columns
front = ["archetype", "spread", "conc", "marker", "size_node",
         "kstar_physical_line", "kstar_rating_MW", "kstar_rating_source"]
back = [c for c in result.columns if c not in front]
result = result[front + back]

# Sort by size
result = result.sort_values("size_node", ascending=False, na_position="last")

# Self-check
size_neg = int((result["size_node"] < 0).sum())
size_nan = int(result["size_node"].isna().sum())
log(f"size < 0: {size_neg}   size NaN: {size_nan} (rating unknown for k*)")
assert size_neg == 0, f"BLOCKER: {size_neg} nodes have negative size"

# Save
out = DATA_DIR / f"node_metrics_with_size_{SEASON_TAG}.csv"
result.to_csv(out)
log(f"saved {out}")

# ---- Reports ----
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 240)

print()
print("== TOP 10 NODES BY SIZE ==")
print(result[["archetype", "spread", "conc",
              "kstar_physical_line", "kstar_rating_MW", "size_node"]].head(10).to_string())

print()
print("== distribution of k* (top 10 most-attributed physical lines) ==")
kstar_counts = result["kstar_physical_line"].value_counts().head(10)
print(kstar_counts.to_string())

print()
print("== PROBE_IMPORT sanity ==")
if config.PROBE_IMPORT in result.index:
    r = result.loc[config.PROBE_IMPORT]
    print(f"  node:          {config.PROBE_IMPORT}")
    print(f"  archetype:     {r['archetype']}")
    print(f"  spread:        ${r['spread']:.2f}")
    print(f"  size:          ${r['size_node']:,.0f}")
    print(f"  k*:            {r['kstar_physical_line']}")
    print(f"  k* rating:     {r['kstar_rating_MW']} MW ({r['kstar_rating_source']})")
    print(f"  top 5 lines:   {r['top5_lines']}")
    print(f"  top 5 shares:  {r['top5_shares']}")

# Histogram of size
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
sizes = result["size_node"].dropna()
ax.hist(sizes, bins=80, color="tab:blue", edgecolor="white")
ax.set_xlabel("size_node ($/MWh * MW = $)")
ax.set_ylabel("node count")
ax.set_title(f"Distribution of size_node — {SEASON_TAG}")
ax.grid(alpha=0.3)
fig.tight_layout()
hist_path = DATA_DIR / f"size_histogram_{SEASON_TAG}.png"
fig.savefig(hist_path, dpi=150)
log(f"saved {hist_path}")

log("D4 OK.")
