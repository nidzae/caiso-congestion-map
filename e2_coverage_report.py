"""E2 — placement status, coverage report.

Per the runbook E2:
  - placement = 'precise'   if EIA-860 matched lat/lon assigned in E1
  - placement = 'regional'  for DLAPs / sub-LAPs / trading hubs at centroids
  - placement = 'unplaced'  for everything else (substation-only pnodes, etc.)
  - assert: precise + regional + unplaced == total

User chose earlier to skip aggregation/DLAP nodes from the bulk pipeline, so
regional = 0 in this run. Centroid coordinates are listed in the report for
reference if we add aggregation nodes back in v2.
"""
from pathlib import Path

import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"

# Reference centroids — not used in this run (aggregations excluded per user
# choice), but published in the coverage report so we can add them later.
AGG_CENTROIDS = {
    "TH_NP15_GEN-APND":  (38.5, -121.5),  # Sacramento area
    "TH_SP15_GEN-APND":  (34.0, -118.0),  # LA Basin
    "TH_ZP26_GEN-APND":  (36.7, -119.8),  # Fresno area
    "TH_PACE_GEN-APND":  (40.8, -111.9),  # Salt Lake City
    "TH_PACW_GEN-APND":  (45.5, -122.7),  # Portland
    "DLAP_SCE-APND":     (34.5, -117.5),  # SCE territory centroid
    "DLAP_PGAE-APND":    (38.5, -121.5),  # PG&E territory
    "DLAP_SDGE-APND":    (32.9, -117.0),  # SDG&E (San Diego)
    "DLAP_VEA-APND":     (36.6, -116.5),  # Valley Electric Assoc (NV/CA border)
}


print("== E2: placement status + coverage report ==")

# Load E1 placements
coords = pd.read_csv(DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv")

# Add placement status
coords["placement"] = coords["Latitude"].notna().map({True: "precise", False: "unplaced"})

# Save the enriched coords
out_csv = DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv"
coords.to_csv(out_csv, index=False)
print(f"updated {out_csv}")

# Counts
total = len(coords)
precise = int((coords["placement"] == "precise").sum())
regional = int((coords["placement"] == "regional").sum())
unplaced = int((coords["placement"] == "unplaced").sum())

print()
print(f"total:    {total:,}")
print(f"precise:  {precise:,}  ({100*precise/total:.1f}%)")
print(f"regional: {regional:,}")
print(f"unplaced: {unplaced:,}  ({100*unplaced/total:.1f}%)")
assert precise + regional + unplaced == total, "BLOCKER: counts don't sum to total"

# Per-aggregate-hub breakdown
print()
print("by aggregate hub:")
breakdown = coords.groupby("Aggregate PNode ID")["placement"].value_counts().unstack(fill_value=0)
print(breakdown.to_string())

# Write coverage report
lines = []
lines.append("CAISO Congestion-Relief Node Map — Coverage Report")
lines.append("=" * 56)
lines.append("")
lines.append(f"Season:           {SEASON_TAG}")
lines.append(f"Data window:      2025-08-01 .. 2025-09-01 (RTM 5-min)")
lines.append(f"Market:           {config.MARKET}")
lines.append("")
lines.append("Parameters (config.py):")
lines.append(f"  Battery duration:     {config.BATTERY_DURATION_HOURS} hours")
lines.append(f"  Round-trip eff (η):   {config.ROUND_TRIP_EFFICIENCY}")
lines.append(f"  Midday window:        {config.MIDDAY_WINDOW}")
lines.append(f"  Evening window:       {config.EVENING_WINDOW}")
lines.append(f"  Flatness threshold ε: ${config.EPSILON_FLATNESS}/MWh")
lines.append(f"  Concentration thresh: {config.CONCENTRATION_THRESHOLD}")
lines.append(f"  PROBE_IMPORT:         {config.PROBE_IMPORT}")
lines.append(f"  PROBE_FLAT:           {config.PROBE_FLAT}")
lines.append("")
lines.append("Geolocation coverage:")
lines.append(f"  Total generator pnodes:   {total:,}")
lines.append(f"  Precise (EIA-860 match):  {precise:,}  ({100*precise/total:.1f}%)")
lines.append(f"  Regional (aggregations):  {regional}  (excluded from this run)")
lines.append(f"  Unplaced (drop from map): {unplaced:,}  ({100*unplaced/total:.1f}%)")
lines.append("")
lines.append("Per-aggregate hub:")
for agg, row in breakdown.iterrows():
    p = int(row.get("precise", 0))
    u = int(row.get("unplaced", 0))
    t = p + u
    rate = 100 * p / t if t else 0
    lines.append(f"  {agg:<22} {p:>4}/{t:<4} placed  ({rate:.1f}%)")
lines.append("")
lines.append("Aggregation centroids (reference only — not in v1 pipeline):")
for k, (lat, lon) in AGG_CENTROIDS.items():
    lines.append(f"  {k:<22} ({lat:.2f}, {lon:.2f})")
lines.append("")
lines.append("Trust gates passed:")
lines.append("  A1/A2 — MCC component exposed separately + decomposition identity holds")
lines.append("  B2    — PROBE_IMPORT (Alamitos) classifies BLUE")
lines.append("  D2    — PROBE_IMPORT attribution k* = Path 26 (Midway-Vincent 500, 4,000 MW WECC)")
lines.append("  D3/D4 — Rating crosswalk + size attribution working; top-size nodes on big paths")
lines.append("  E1    — Probe nodes (Alamitos, Harbor) placed at correct Long Beach coords")
lines.append("")
lines.append("Known unplaced node categories (not in EIA-860, by design):")
lines.append("  - Substation-only pnodes (Vincent, Lugo, Mira Loma, Devers, Sycamore)")
lines.append("  - Aggregator pnodes (TH_*-APND, DLAP_*-APND, SLAP_*) — see centroids above")
lines.append("  - Some retired or proposed plants")
lines.append("")
lines.append("Next phase: F (Plotly render).")
out_txt = DATA_DIR / f"coverage_report_{SEASON_TAG}.txt"
out_txt.write_text("\n".join(lines) + "\n")
print()
print(f"saved {out_txt}")
print()
print("E2 OK.")
