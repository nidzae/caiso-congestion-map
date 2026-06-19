"""Audit mixed-archetype coordinate stacks.

For each coord with 2+ nodes spanning 2+ archetypes, dump the node list with
archetype, signals, spread, and EIA plant. Classify each as likely-legit
(near ε boundary, alias-driven) vs likely-false (clearly different abbrevs
forced into same plant).
"""
from pathlib import Path
import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"


def main():
    metrics = pd.read_csv(DATA_DIR / f"node_metrics_with_size_{SEASON_TAG}.csv", index_col=0)
    coords = pd.read_csv(DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv").rename(
        columns={"PNode ID": "node"}).set_index("node")
    df = metrics.join(coords[["Latitude", "Longitude", "placement", "Plant Name",
                                "abbrev", "match_score", "Aggregate PNode ID"]])
    df = df[df["placement"] == "precise"].dropna(subset=["archetype"])

    df["coord_key"] = (df["Latitude"].round(4).astype(str) + ","
                       + df["Longitude"].round(4).astype(str))

    stacks = df.groupby("coord_key").agg(
        n=("archetype", "size"),
        n_arch=("archetype", "nunique"),
        archetypes=("archetype", lambda s: ",".join(sorted(s.unique()))),
        plant=("Plant Name", "first"),
        lat=("Latitude", "first"),
        lon=("Longitude", "first"),
    )
    mixed = stacks[(stacks["n"] >= 2) & (stacks["n_arch"] >= 2)].sort_values(
        "n", ascending=False)
    print(f"=== {len(mixed)} coord-stacks with 2+ nodes and 2+ archetypes ===")
    print()

    eps = config.EPSILON_FLATNESS

    def classification_hint(rows):
        """Heuristic: is the mix likely a legit borderline case or a false match?"""
        # All same EIA plant + similar signals → legit
        # Different signals far above/below ε → likely real difference
        # Multiple distinct abbreviations + all near ε → could go either way
        abbrevs = rows["abbrev"].nunique()
        all_aliased = (rows["match_score"] == 99).all()
        any_aliased = (rows["match_score"] == 99).any()
        # Are import/export signals all very close to ε?
        sig_min_dist = rows.apply(
            lambda r: min(abs(r["import_signal"] - eps),
                          abs(r["export_signal"] - eps)), axis=1).min()
        if all_aliased:
            return "LEGIT (all alias-mapped, expected at same plant)"
        if abbrevs == 1 and sig_min_dist < 1.0:
            return "LIKELY LEGIT (one abbrev, near ε boundary)"
        if abbrevs == 1:
            return "POSSIBLE (one abbrev — sibling unit-level pnodes)"
        if any_aliased:
            return "MIXED (some alias-mapped, some fuzzy — verify)"
        return "SUSPICIOUS (multiple distinct abbrevs, no alias anchor)"

    # Walk through each mixed stack and print full detail
    for key, srow in mixed.iterrows():
        sub = df[df["coord_key"] == key].sort_values("spread", ascending=False)
        hint = classification_hint(sub)
        print(f"\n--- {srow['plant']!r} @ ({srow['lat']:.4f}, {srow['lon']:.4f}) ---")
        print(f"    {srow['n']} nodes, {srow['n_arch']} archetypes: {srow['archetypes']}")
        print(f"    classification hint: {hint}")
        cols = ["abbrev", "archetype", "import_signal", "export_signal",
                "spread", "marker", "match_score"]
        print(sub[cols].to_string())

    # Aggregate counts
    print()
    print("=== aggregate hint counts ===")
    hints = []
    for key in mixed.index:
        sub = df[df["coord_key"] == key]
        hints.append(classification_hint(sub))
    hint_counts = pd.Series(hints).value_counts()
    print(hint_counts.to_string())


if __name__ == "__main__":
    main()
