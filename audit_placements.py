"""Audit script — find suspicious EIA-860 plant matches.

Two signals of likely false matches:
  A) Many distinct CAISO abbreviations all map to the same EIA plant
     (sibling units share an abbreviation — distinct abbrevs hitting one plant
     usually means the fuzzy matcher reached for the same long name).
  B) Low subseq-density / short-prefix matches — borderline hits.

We print review-worthy plants and abbreviation→plant pairs.
"""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"


def main():
    coords = pd.read_csv(DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv")
    coords = coords[coords["placement"] == "precise"].copy()
    print(f"placed nodes: {len(coords):,}")
    print(f"distinct abbreviations placed: {coords['abbrev'].nunique():,}")
    print(f"distinct plants used: {coords['Plant Name'].nunique():,}")

    # ---- A) Plants with many DISTINCT abbreviations pointing to them ----
    suspect = (coords.groupby("Plant Name")
                    .agg(n_nodes=("PNode ID", "size"),
                         n_distinct_abbrev=("abbrev", "nunique"),
                         abbrevs=("abbrev", lambda s: sorted(s.unique())),
                         agg_hubs=("Aggregate PNode ID",
                                   lambda s: sorted(s.unique())),
                         lat=("Latitude", "first"),
                         lon=("Longitude", "first")))
    multi = suspect[suspect["n_distinct_abbrev"] >= 2].sort_values(
        "n_distinct_abbrev", ascending=False)

    print()
    print(f"plants with 2+ distinct abbreviations: {len(multi)}")
    print()
    print("== TOP 25 SUSPICIOUS PLANTS (multiple distinct abbrevs → same plant) ==")
    for plant, row in multi.head(25).iterrows():
        abbr_str = ", ".join(row["abbrevs"][:10])
        if len(row["abbrevs"]) > 10:
            abbr_str += f" ... (+{len(row['abbrevs'])-10})"
        print(f"\n  {plant!r}  ({row['lat']:.4f}, {row['lon']:.4f})")
        print(f"    nodes:    {row['n_nodes']}")
        print(f"    abbrevs:  {abbr_str}")
        print(f"    hubs:     {', '.join(row['agg_hubs'])}")

    # ---- B) Low-density matches that still passed (borderline) ----
    print()
    print("== B) lowest-confidence matches (small score, non-alias) ==")
    non_alias = coords[coords["match_score"] < 99]
    # match_score is the primary metric; treat short matches as borderline
    borderline = non_alias.sort_values("match_score").head(30)
    print(borderline[["PNode ID", "abbrev", "Plant Name", "State",
                       "match_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
