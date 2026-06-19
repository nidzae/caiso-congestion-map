"""F2 — emit canonical audit outputs.

Per brief §8 output #2 — node_metrics.csv with one row per node:
  name, lat, lon, archetype, spread, size, controlling_constraint,
  rating, rating_source, conc, placed_flag

Joins:
  - D4 per-node metrics  (node_metrics_with_size_<season>.csv)
  - E1/E2 coordinates    (node_coordinates_<season>.csv)

Useful auxiliary columns also retained so the CSV can be sorted and audited
independently of the visual.
"""
from pathlib import Path

import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"


def main():
    metrics = pd.read_csv(DATA_DIR / f"node_metrics_with_size_{SEASON_TAG}.csv", index_col=0)
    coords = pd.read_csv(DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv").rename(
        columns={"PNode ID": "name"}).set_index("name")

    df = metrics.join(coords[["Latitude", "Longitude", "placement",
                                "Plant Name", "abbrev", "match_score",
                                "Aggregate PNode ID"]], how="left")

    # Brief §8 canonical columns first, then auxiliary columns for audit
    canonical_cols = {
        "Latitude":              "lat",
        "Longitude":             "lon",
        "archetype":             "archetype",
        "spread":                "spread",
        "size_node":             "size",
        "kstar_physical_line":   "controlling_constraint",
        "kstar_rating_MW":       "rating",
        "kstar_rating_source":   "rating_source",
        "conc":                  "conc",
        "placement":             "placed_flag",
    }
    aux_cols = {
        "marker":                "marker",
        "MCC_mid":               "MCC_mid",
        "MCC_eve":               "MCC_eve",
        "import_signal":         "import_signal",
        "export_signal":         "export_signal",
        "hours_covered":         "hours_covered",
        "Aggregate PNode ID":    "aggregate_hub",
        "abbrev":                "abbrev",
        "Plant Name":            "eia_plant_name",
        "match_score":           "eia_match_score",
        "kstar_contribution":    "kstar_contribution",
        "top5_lines":            "top5_lines",
        "top5_shares":           "top5_shares",
    }

    rename_map = {**canonical_cols, **aux_cols}
    keep = [c for c in rename_map if c in df.columns]
    out = df[keep].rename(columns=rename_map).copy()
    out.index.name = "name"

    # Place canonical columns first
    canonical_in_out = [v for v in canonical_cols.values() if v in out.columns]
    aux_in_out = [v for v in aux_cols.values() if v in out.columns]
    out = out[canonical_in_out + aux_in_out]

    out = out.sort_values("size", ascending=False, na_position="last")

    out_path = DATA_DIR / f"node_metrics.csv"
    out.to_csv(out_path)
    print(f"saved {out_path}  ({len(out):,} rows, {len(out.columns)} columns)")

    print()
    print("== canonical columns (brief §8) ==")
    for c in canonical_in_out:
        n_nan = int(out[c].isna().sum())
        print(f"  {c:<30}  {n_nan:>5} NaN")

    print()
    print("== aux columns (audit support) ==")
    for c in aux_in_out:
        n_nan = int(out[c].isna().sum())
        print(f"  {c:<30}  {n_nan:>5} NaN")

    # Per-archetype breakdown summary
    print()
    print("== rows by archetype × placed_flag ==")
    summary = (out.groupby(["archetype", "placed_flag"], dropna=False)
                   .size()
                   .unstack(fill_value=0))
    print(summary.to_string())

    # Probe sanity row from the canonical CSV
    print()
    print("== PROBE_IMPORT canonical row ==")
    if config.PROBE_IMPORT in out.index:
        r = out.loc[config.PROBE_IMPORT]
        print(r[canonical_in_out].to_string())

    print()
    print(f"coverage report (parameters + gates) already at: "
          f"{DATA_DIR / f'coverage_report_{SEASON_TAG}.txt'}")


if __name__ == "__main__":
    main()
