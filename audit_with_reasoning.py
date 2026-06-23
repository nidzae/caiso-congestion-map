"""LLM-reasoned audit of EIA-860 plant placements.

The fuzzy matcher in e1_place_gen_nodes.py picks the EIA plant whose name
best matches a CAISO pnode's abbreviation by subsequence density. That
heuristic can be fooled by unrelated long names (e.g. IVGEN matching
"rIVer coGenEratioN") even when the abbreviation refers to a substation
or facility that isn't in EIA-860 at all.

This script asks an LLM agent to reason about each placed (abbreviation,
plant) pair using regional context — pnode hub (SP15/NP15/ZP26/...),
plant BA + state + coordinates, and the top-3 alternative candidates the
matcher could have picked — and emit one of three verdicts per pair:

  KEEP    — the current match is plausible
  REJECT  — no candidate plant is plausibly the right one; leave unplaced
  REPLACE — current is wrong but one of the alternatives is correct

Run order:
  1. python3 audit_with_reasoning.py prep
       → writes data/eia/audit_batches/batch_NN.json files
       → prints the batch IDs and how many pairs each holds
  2. For each batch_NN.json file, the orchestrator invokes an Agent with
     the batch contents and saves the verdicts to
     data/eia/audit_verdicts/batch_NN.json
  3. python3 audit_with_reasoning.py apply
       → reads all verdict files, builds placement_overrides.py and
         data/eia/placement_audit.csv

Re-running with the verdict files already present produces no LLM cost.
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path("data")
EIA = DATA / "eia"
COORDS = DATA / "node_coordinates_summer2025.csv"
BATCH_DIR = EIA / "audit_batches"
VERDICT_DIR = EIA / "audit_verdicts"
AUDIT_CSV = EIA / "placement_audit.csv"
OVERRIDES_PY = Path("placement_overrides.py")

BATCH_SIZE = 25
RANDOM_SEED = 42
SAMPLE_FRACTION_MEDIUM = 0.20  # of match_score 6-8


def normalize(s):
    return re.sub(r"[^A-Z]+", "", str(s).upper()) if pd.notna(s) else ""


def best_subseq_density(short: str, long: str) -> float:
    if not short or not long:
        return 0.0
    n = len(short)
    best = 0.0
    for start in range(len(long) - n + 1):
        if long[start] != short[0]:
            continue
        i = 0
        last = start
        first = start
        for j in range(start, len(long)):
            if long[j] == short[i]:
                if i == 0:
                    first = j
                last = j
                i += 1
                if i == n:
                    break
        if i == n:
            span = last - first + 1
            density = n / span
            if density > best:
                best = density
    return best


def load_eia_plants() -> pd.DataFrame:
    """Same filtering pipeline e1 uses, so candidate alternatives match the
    pool e1 was selecting from."""
    plants = pd.read_excel(EIA / "2___Plant_Y2025_Early_Release.xlsx",
                            sheet_name=0, header=2)
    plants = plants[plants["State"].isin(
        ["CA", "AZ", "NV", "OR", "ID", "UT", "WY", "MT", "NM", "WA", "CO"])].copy()
    plants["Latitude"] = pd.to_numeric(plants["Latitude"], errors="coerce")
    plants["Longitude"] = pd.to_numeric(plants["Longitude"], errors="coerce")
    plants = plants.dropna(subset=["Latitude", "Longitude"])
    plants["name_norm"] = plants["Plant Name"].apply(normalize)
    # Capacity filter — copy from e1
    gen = pd.read_excel(EIA / "3_1_Generator_Y2025_Early_Release.xlsx",
                         sheet_name=0, header=2)
    gen["Nameplate Capacity (MW)"] = pd.to_numeric(
        gen["Nameplate Capacity (MW)"], errors="coerce")
    plant_mw = gen.groupby("Plant Code")["Nameplate Capacity (MW)"].sum()
    plants["total_mw"] = plants["Plant Code"].map(plant_mw).fillna(0.0)
    plants = plants[plants["total_mw"] >= 10.0].copy()
    return plants.reset_index(drop=True)


HUB_BA_PREF = {
    "TH_SP15_GEN-APND": ["CISO", "LDWP", "BANC", "IID", "TIDC"],
    "TH_NP15_GEN-APND": ["CISO", "BANC", "TIDC"],
    "TH_ZP26_GEN-APND": ["CISO", "BANC", "TIDC"],
    "TH_PACE_GEN-APND": ["PACE", "WACM", "NEVP", "IPCO", "PSCO"],
    "TH_PACW_GEN-APND": ["PACW", "BPAT", "PGE", "CISO"],
}
HUB_REGION_NOTE = {
    "TH_SP15_GEN-APND": "SP15 = Southern California (LA Basin, Inland Empire, Imperial Valley, Desert)",
    "TH_NP15_GEN-APND": "NP15 = Northern California (Bay Area, North Coast, Sacramento)",
    "TH_ZP26_GEN-APND": "ZP26 = Central California (San Joaquin Valley, Central Coast)",
    "TH_PACE_GEN-APND": "PACE = PacifiCorp East (Utah, Wyoming, Idaho, parts of Oregon/CA)",
    "TH_PACW_GEN-APND": "PACW = PacifiCorp West (Oregon, parts of WA/CA)",
}


def top_alternatives(abbrev: str, hub: str, plants: pd.DataFrame,
                      current_code: int, n: int = 3) -> list[dict]:
    """Rank EIA plants in the hub's preferred BAs by subseq density of
    abbrev within plant name, returning the top N other than current."""
    pref_bas = set(HUB_BA_PREF.get(hub, ["CISO"]))
    pool = plants[plants["Balancing Authority Code"].fillna("").isin(pref_bas)]
    scored = []
    for _, p in pool.iterrows():
        if int(p["Plant Code"]) == current_code:
            continue
        d = best_subseq_density(abbrev, p["name_norm"])
        if d >= 0.4:  # slightly looser than e1's floor; we want alternatives
            scored.append((d, p))
    scored.sort(key=lambda t: -t[0])
    out = []
    for d, p in scored[:n]:
        out.append({
            "plant_code": int(p["Plant Code"]),
            "plant_name": p["Plant Name"],
            "ba": p.get("Balancing Authority Code") or "",
            "state": p["State"],
            "lat": round(float(p["Latitude"]), 4),
            "lon": round(float(p["Longitude"]), 4),
            "subseq_density": round(d, 3),
            "total_mw": round(float(p["total_mw"]), 1),
        })
    return out


def cmd_prep():
    """Build per-pair audit records and write them as batches of JSON."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)

    coords = pd.read_csv(COORDS)
    placed = coords[coords["placement"] == "precise"].copy()
    print(f"loaded {len(coords)} pnodes, {len(placed)} placed")

    # Distinct (abbrev, plant) pairs — what we're really auditing
    pairs = (placed.groupby(["abbrev", "Plant Code"])
                    .agg(plant_name=("Plant Name", "first"),
                          ba=("Aggregate PNode ID", lambda s: s.iloc[0]),  # hub
                          state=("State", "first"),
                          lat=("Latitude", "first"),
                          lon=("Longitude", "first"),
                          match_score=("match_score", "first"),
                          n_pnodes=("PNode ID", "size"),
                          pnode_ids=("PNode ID", lambda s: sorted(s.tolist())))
                    .reset_index())
    pairs = pairs.rename(columns={"ba": "hub"})
    print(f"distinct (abbrev, plant) pairs: {len(pairs)}")
    print(f"  score==99 (alias-pinned, will skip): "
          f"{(pairs.match_score == 99).sum()}")
    print(f"  score <= 5 (audit all):              "
          f"{(pairs.match_score <= 5).sum()}")
    print(f"  score 6-8 (sample 20%):              "
          f"{((pairs.match_score >= 6) & (pairs.match_score <= 8)).sum()}")

    # Bucket
    random.seed(RANDOM_SEED)
    must_audit = pairs[(pairs.match_score != 99) & (pairs.match_score <= 5)]
    sample_pool = pairs[(pairs.match_score >= 6) & (pairs.match_score <= 8)]
    sample_n = int(len(sample_pool) * SAMPLE_FRACTION_MEDIUM)
    sampled = sample_pool.sample(n=sample_n, random_state=RANDOM_SEED)
    to_audit = pd.concat([must_audit, sampled], ignore_index=True)
    print(f"\ntotal pairs to audit: {len(to_audit)}")

    # Build records with top-3 alternatives
    print("loading EIA plant pool to score alternatives...")
    plants = load_eia_plants()

    records = []
    for _, p in to_audit.iterrows():
        hub = p["hub"]
        alts = top_alternatives(
            p["abbrev"], hub, plants, current_code=int(p["Plant Code"]), n=3)
        records.append({
            "abbrev": p["abbrev"],
            "hub": hub,
            "hub_region_note": HUB_REGION_NOTE.get(
                hub, "(non-standard CAISO hub)"),
            "n_pnodes_using_this_abbrev": int(p["n_pnodes"]),
            "example_pnode_ids": p["pnode_ids"][:3],
            "current_match": {
                "plant_code": int(p["Plant Code"]),
                "plant_name": p["plant_name"],
                "state": p["state"],
                "lat": round(float(p["lat"]), 4),
                "lon": round(float(p["lon"]), 4),
                "fuzzy_match_score": int(p["match_score"]),
            },
            "top_alternatives": alts,
        })

    # Write batches
    n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nwriting {n_batches} batches of up to {BATCH_SIZE} pairs each "
          f"to {BATCH_DIR}/")
    # Clear any old batches first so re-prep is deterministic
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()
    for i in range(n_batches):
        chunk = records[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        path = BATCH_DIR / f"batch_{i:02d}.json"
        path.write_text(json.dumps(chunk, indent=2))
        print(f"  {path.name}: {len(chunk)} pairs")
    print()
    print("next: orchestrator runs an Agent per batch, saving verdicts to "
          f"{VERDICT_DIR}/batch_NN.json.")
    print("Then: python3 audit_with_reasoning.py apply")


def cmd_apply():
    """Read all verdict files and produce placement_overrides.py +
    placement_audit.csv."""
    verdicts = []
    files = sorted(VERDICT_DIR.glob("batch_*.json"))
    if not files:
        sys.exit(f"no verdict files in {VERDICT_DIR}/")
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            sys.exit(f"bad JSON in {f}: {e}")
        if not isinstance(data, list):
            sys.exit(f"{f} should contain a JSON list, got {type(data)}")
        for v in data:
            v["_source_file"] = f.name
            verdicts.append(v)
    print(f"loaded {len(verdicts)} verdicts from {len(files)} batch files")

    df = pd.DataFrame(verdicts)
    # Tidy columns
    for col in ("abbrev", "verdict", "reason"):
        if col not in df.columns:
            sys.exit(f"verdict rows missing required field: {col}")
    if "suggested_plant_code" not in df.columns:
        df["suggested_plant_code"] = None
    df.to_csv(AUDIT_CSV, index=False)
    print(f"saved {AUDIT_CSV} ({len(df)} rows)")
    print("\nverdict distribution:")
    print(df["verdict"].value_counts().to_string())

    # Build placement_overrides.py
    rejects = sorted(df.loc[df.verdict == "REJECT", "abbrev"].dropna().unique().tolist())
    replaces_df = df.loc[df.verdict == "REPLACE"].dropna(
        subset=["suggested_plant_code"])
    replace_map = {}
    for _, r in replaces_df.iterrows():
        abbrev = str(r["abbrev"]).strip()
        try:
            code = int(r["suggested_plant_code"])
        except (TypeError, ValueError):
            continue
        replace_map[abbrev] = code

    body = ['"""Auto-generated by audit_with_reasoning.py. Do not edit by hand."""',
            "",
            "# Abbreviations the LLM audit asked to be REPLACE'd with a specific plant.",
            "AUDIT_ALIAS_ADDITIONS = {"]
    for k in sorted(replace_map):
        body.append(f"    {k!r}: {replace_map[k]},")
    body.append("}")
    body.append("")
    body.append("# Abbreviations the LLM audit asked to be REJECT'd (left unplaced).")
    body.append("AUDIT_REJECTIONS = {")
    for r in rejects:
        body.append(f"    {r!r},")
    body.append("}")
    body.append("")
    OVERRIDES_PY.write_text("\n".join(body))
    print(f"\nsaved {OVERRIDES_PY}  "
          f"({len(replace_map)} replacements, {len(rejects)} rejections)")
    print("\nnext: python3 e1_place_gen_nodes.py")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) >= 2 else "help"
    if cmd == "prep":
        cmd_prep()
    elif cmd == "apply":
        cmd_apply()
    else:
        print(__doc__)
