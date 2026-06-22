"""Per-node rent-persistence indicators across the 12 monthly metrics files.

Outputs data/persistence_2025.csv with:
  node, n_months_active, n_months_blue_or_purple,
  size_cv, size_median_x12, size_max_share, same_kstar_months,
  persistence_label   (plain-language: stable / moderate / variable / spike-driven)

The renderer joins on `node` index to surface these fields in hovers and
bar-chart labels.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = DATA / "persistence_2025.csv"
TAGS = [f"2025-{m:02d}" for m in range(1, 13)]
SIZE_ACTIVE_THRESHOLD = 1_000_000   # treat size_node ≥ $1M as "active"


def persistence_label(cv: float | None, n_active: int) -> str:
    """Plain-language stability descriptor for the rent.

    Two-axis: presence (n_active) and variability (CV). Thresholds chosen
    empirically from the 12-month CAISO 2025 dataset where median CV ≈ 1.2:
    constraint attribution can switch month-to-month so even "real" sites
    show moderate-to-high CV.
    """
    if n_active <= 1:
        return "single month"
    if n_active <= 4:
        return f"sporadic ({n_active}/12 mo)"
    if cv is None or pd.isna(cv):
        return f"{n_active}/12 mo · unknown"
    if cv < 0.50:
        return f"{n_active}/12 mo · stable"
    if cv < 1.00:
        return f"{n_active}/12 mo · moderate"
    if cv < 2.00:
        return f"{n_active}/12 mo · variable"
    return f"{n_active}/12 mo · volatile"


def main():
    frames = []
    for tag in TAGS:
        p = DATA / f"node_metrics_with_size_{tag}.csv"
        if not p.exists():
            print(f"SKIP {tag} (missing)")
            continue
        df = pd.read_csv(p, index_col=0)[
            ["archetype", "size_node", "kstar_physical_line"]
        ].assign(month=tag)
        frames.append(df)
    if not frames:
        raise SystemExit("no monthly metrics files found")

    long = pd.concat(frames)
    print(f"loaded {len(long):,} per-(node, month) rows from {len(frames)} months")

    g = long.groupby(long.index)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "node"

    # n_months_active: count of months where size_node >= threshold
    out["n_months_active"] = g["size_node"].apply(
        lambda s: int((s.fillna(0) >= SIZE_ACTIVE_THRESHOLD).sum())
    )

    # n_months_blue_or_purple: count of months with import-side classification
    out["n_months_blue_or_purple"] = g["archetype"].apply(
        lambda s: int(s.isin(["BLUE", "PURPLE"]).sum())
    )

    # Coefficient of variation of size_node across months (ignoring NaN)
    def cv(s):
        s2 = s.dropna()
        if len(s2) < 2:
            return np.nan
        mean = s2.mean()
        if mean == 0:
            return np.nan
        return float(s2.std() / mean)

    out["size_cv"] = g["size_node"].apply(cv)

    # Median monthly size × 12 — proxy for annual rent
    out["size_median_x12"] = g["size_node"].median() * 12

    # Share of total annual size coming from the single biggest month
    def max_share(s):
        s2 = s.dropna()
        tot = s2.sum()
        if tot <= 0:
            return np.nan
        return float(s2.max() / tot)

    out["size_max_share"] = g["size_node"].apply(max_share)

    # How many months share the same controlling line as the most-common one
    def same_kstar(s):
        s2 = s.dropna()
        if s2.empty:
            return 0
        mode = s2.mode()
        if mode.empty:
            return 0
        return int((s2 == mode.iloc[0]).sum())

    out["same_kstar_months"] = g["kstar_physical_line"].apply(same_kstar)

    # Plain-language label
    out["persistence_label"] = [
        persistence_label(cv, n)
        for cv, n in zip(out["size_cv"], out["n_months_active"])
    ]

    out.to_csv(OUT)
    print(f"saved {OUT}  ({len(out)} nodes)")

    print()
    print("=== distribution of persistence labels ===")
    print(out["persistence_label"].value_counts().to_string())
    print()
    print("=== distribution of n_months_active ===")
    print(out["n_months_active"].value_counts().sort_index().to_string())

    print()
    print("=== probe sanity ===")
    for n in ["ALAMT1G_7_B1", "KRNSNST_7_N001", "RIOBRVQF_6_N001"]:
        if n in out.index:
            r = out.loc[n]
            print(f"  {n}: {r['n_months_active']}/12 active, "
                  f"CV={r['size_cv']:.2f} -> {r['persistence_label']!r}")


if __name__ == "__main__":
    main()
