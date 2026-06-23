"""E1 — geolocate generator pnodes via EIA-860 fuzzy match.

Strategy:
  1. Load EIA-860 plant table (2025 early release). Filter to WECC states.
  2. For each CAISO pnode, extract the leading alphabetic token as the plant
     abbreviation (e.g., ALAMT1G_7_B1 -> ALAMT).
  3. For each abbreviation, find the EIA plant whose normalized name has the
     longest common prefix; prefer CA over other states.
  4. Attach lat/long. Report match rate and a coverage CSV.
  5. Assert all matched coordinates lie inside a CA/WECC bounding box.
  6. Plot a coverage chart.

The brief expects partial coverage — substation-only pnodes (e.g., VINCENT,
MIDWAY) won't match because EIA-860 is plants, not substations.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gridstatus

DATA_DIR = Path("data")
EIA_PLANT_FILE = DATA_DIR / "eia" / "2___Plant_Y2025_Early_Release.xlsx"
EIA_GENERATOR_FILE = DATA_DIR / "eia" / "3_1_Generator_Y2025_Early_Release.xlsx"
SEASON_TAG = "summer2025"

# Minimum plant nameplate capacity to be a fuzzy-match candidate.
# CAISO pnodes correspond to utility-scale plants; the EIA-860 file has many
# tiny commercial / behind-the-meter installations that aren't real CAISO
# pricing points and only generate false fuzzy matches.
MIN_PLANT_MW = 10.0

WECC_STATES = ["CA", "AZ", "NV", "OR", "ID", "UT", "WY", "MT", "NM", "WA", "CO"]
WECC_BBOX = {"lat_min": 30.0, "lat_max": 50.0, "lon_min": -125.0, "lon_max": -100.0}
MIN_MATCH_LEN = 4

# Aggregate hub → preferred Balancing Authority codes (much tighter than
# state, since CISO ≠ all-of-California). Falls back to broader state pool
# only if no candidate plant in the preferred BAs matches.
AGG_BA_PREF = {
    "TH_SP15_GEN-APND": ["CISO", "LDWP", "BANC", "IID", "TIDC"],
    "TH_NP15_GEN-APND": ["CISO", "BANC", "TIDC"],
    "TH_ZP26_GEN-APND": ["CISO", "BANC", "TIDC"],
    "TH_PACE_GEN-APND": ["PACE", "WACM", "NEVP", "IPCO", "PSCO"],
    "TH_PACW_GEN-APND": ["PACW", "BPAT", "PGE", "CISO"],
}
# Final state-fallback pool if nothing in preferred BAs matches.
AGG_STATE_FALLBACK = {
    "TH_SP15_GEN-APND": ["CA"],
    "TH_NP15_GEN-APND": ["CA"],
    "TH_ZP26_GEN-APND": ["CA"],
    "TH_PACE_GEN-APND": ["UT", "WY", "ID", "OR", "CA", "CO", "NV"],
    "TH_PACW_GEN-APND": ["OR", "WA", "CA", "ID"],
}

# Hand-curated abbreviation -> EIA Plant Code aliases.
# CAISO uses cryptic 4-6 char abbreviations; for the well-known SP15 plants
# whose fuzzy match would be ambiguous (e.g., ALAMT → Alamo vs AES Alamitos),
# the alias map is authoritative. Codes verified from EIA-860 2025 ER.
ABBREV_ALIAS = {
    "ALAMT":    315,    # AES Alamitos LLC, Long Beach (33.77, -118.10)
    "ALAMITOS": 315,
    "HARBORG":  399,    # Harbor, Wilmington (33.77, -118.27)
    "ETIWANDA": 745,    # Etiwanda (34.10, -117.53)
    "ETIWAND":  745,
    "ETIWND":   745,
    "HUNTBCH":  335,    # AES Huntington Beach LLC (33.64, -117.98)
    "HNTBCH":   335,
    "ORMND":    350,    # Ormond Beach (34.13, -119.17)
    "ELSEG":    57901,  # El Segundo Energy Center (33.91, -118.42)
    "ELSEGNDO": 57901,
    "SENTL":    57482,  # Sentinel Energy Center, Desert Hot Springs
    "SENTI":    57482,
}

# Abbreviations the matcher must REFUSE to place — even if the fuzzy
# matcher finds something. Use this when the abbreviation has no plausible
# EIA-860 plant (e.g., it's a substation, an aggregator, or a generic name
# whose letters happen to subseq-match an unrelated plant elsewhere in the
# state). The IVGEN case (Imperial Valley generic, was matching "rIVer
# coGenEratioN" 300mi away) is the canonical seed; the audit script
# populates the rest.
REJECTED_ABBREVS: set[str] = {
    "IVGEN",  # "Imperial Valley generator generic" — no single EIA plant
}

# The LLM audit emits a companion file placement_overrides.py with two
# names: AUDIT_ALIAS_ADDITIONS (dict, merges into ABBREV_ALIAS) and
# AUDIT_REJECTIONS (set, unions into REJECTED_ABBREVS). Kept in a
# separate file so hand-curated entries above stay distinct from
# audit-derived ones.
try:
    import placement_overrides  # type: ignore
    ABBREV_ALIAS.update(getattr(placement_overrides, "AUDIT_ALIAS_ADDITIONS", {}))
    REJECTED_ABBREVS |= getattr(placement_overrides, "AUDIT_REJECTIONS", set())
    log_msg = (
        f"[overrides] +{len(getattr(placement_overrides, 'AUDIT_ALIAS_ADDITIONS', {}))}"
        f" alias entries, +{len(getattr(placement_overrides, 'AUDIT_REJECTIONS', set()))}"
        f" rejections from placement_overrides.py"
    )
    print(log_msg, flush=True)
except ImportError:
    pass


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(s):
    return re.sub(r"[^A-Z]+", "", str(s).upper()) if pd.notna(s) else ""


def extract_abbrev(pnode_id: str) -> str | None:
    """Leading alphabetic token, uppercased."""
    tok = pnode_id.split("_")[0]
    m = re.match(r"^([A-Za-z]+)", tok)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
log("loading CAISO pnodes...")
pnodes = gridstatus.CAISO().get_pnodes()
pnodes = pnodes[["Aggregate PNode ID", "PNode ID", "DESCRIPTION"]].dropna(subset=["PNode ID"])
pnodes = pnodes.drop_duplicates(subset=["PNode ID"]).reset_index(drop=True)
log(f"CAISO generator pnodes: {len(pnodes):,}")

pnodes["abbrev"] = pnodes["PNode ID"].apply(extract_abbrev)
unique_abbrevs = pnodes["abbrev"].dropna().unique().tolist()
log(f"unique abbreviations: {len(unique_abbrevs)}")

log("loading EIA-860 plant table...")
plants = pd.read_excel(EIA_PLANT_FILE, sheet_name=0, header=2)
plants = plants[plants["State"].isin(WECC_STATES)].copy()
plants["Latitude"] = pd.to_numeric(plants["Latitude"], errors="coerce")
plants["Longitude"] = pd.to_numeric(plants["Longitude"], errors="coerce")
plants = plants.dropna(subset=["Latitude", "Longitude"])
plants["name_norm"] = plants["Plant Name"].apply(normalize)
log(f"WECC plants with coords (pre capacity filter): {len(plants):,}")

# Capacity filter via generator-sheet aggregation
log("loading EIA-860 generator sheet for nameplate-capacity aggregation...")
gen = pd.read_excel(EIA_GENERATOR_FILE, sheet_name=0, header=2)
gen["Nameplate Capacity (MW)"] = pd.to_numeric(gen["Nameplate Capacity (MW)"], errors="coerce")
plant_mw = gen.groupby("Plant Code")["Nameplate Capacity (MW)"].sum()
plants["total_mw"] = plants["Plant Code"].map(plant_mw).fillna(0.0)
n_before = len(plants)
plants = plants[plants["total_mw"] >= MIN_PLANT_MW].copy()
log(f"after dropping plants < {MIN_PLANT_MW} MW total nameplate: "
    f"{len(plants):,} (cut {n_before - len(plants):,})")
log("by state:")
for s, n in plants["State"].value_counts().items():
    log(f"  {s}: {n:,}")
log("by BA (top 10):")
for ba, n in plants["Balancing Authority Code"].value_counts().head(10).items():
    log(f"  {ba}: {n:,}")

# ---------------------------------------------------------------------------
log("building matcher...")

plants_sorted = plants.reset_index(drop=True).sort_values("name_norm").reset_index(drop=True)
names_norm = plants_sorted["name_norm"].tolist()
plant_states = plants_sorted["State"].tolist()
plant_bas = plants_sorted["Balancing Authority Code"].fillna("").tolist()

# Index plant rows by code for alias lookup
plant_by_code = {int(c): i for i, c in enumerate(plants_sorted["Plant Code"]) if pd.notna(c)}


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def best_subseq_density(short: str, long: str) -> float:
    """Highest matched-chars / span ratio of `short` as a subseq of `long`,
    over all valid starting positions. 1.0 means a contiguous substring
    match; 0.5 means the abbreviation's chars are spread over 2× their
    length in the plant name; 0 means no subseq match.
    Penalizes "long plant name where my letters happen to appear scattered."
    """
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


# Minimum subseq density to count as a fuzzy hit. 0.5 = abbreviation
# spans at most 2x its length in the plant name.
SUBSEQ_DENSITY_FLOOR = 0.5


def score_match(abbrev: str, plant_name: str) -> tuple[float, int, int]:
    """Composite score (higher better):
      (subseq_density_if_above_floor_else_0, prefix_len, -name_len)
    A dense subseq beats a long prefix at a sprawling unrelated plant;
    a sparse subseq is rejected.
    """
    pl = common_prefix_len(abbrev, plant_name)
    density = best_subseq_density(abbrev, plant_name)
    eff_density = density if density >= SUBSEQ_DENSITY_FLOOR else 0.0
    return (eff_density, pl, -len(plant_name))


def best_match(abbrev: str, preferred_bas: list[str], state_fallback: list[str]):
    """Best plant match. Tries (a) preferred BAs first, (b) state fallback,
    (c) all WECC. Stops at the first layer with a qualifying match."""
    if not abbrev or len(abbrev) < MIN_MATCH_LEN:
        return None, 0, 0
    # Alias short-circuit
    if abbrev in ABBREV_ALIAS:
        code = ABBREV_ALIAS[abbrev]
        if code in plant_by_code:
            return plant_by_code[code], 99, 0
    # Rejection short-circuit — abbreviations the audit (or hand-curated
    # list) has explicitly flagged as "no good plant match exists; do not
    # place." Returning early prevents the fuzzy matcher from inventing a
    # placement just because the letters subseq-match an unrelated plant.
    if abbrev in REJECTED_ABBREVS:
        return None, 0, 0

    def scan(indices):
        best_key = (0.0, 0, 0)
        best_idx = None
        for i in indices:
            key = score_match(abbrev, names_norm[i])
            if key > best_key:
                best_key = key
                best_idx = i
        density, pl, _ = best_key
        primary = len(abbrev) if density >= SUBSEQ_DENSITY_FLOOR else pl
        return best_idx, primary, 0

    ba_set = set(preferred_bas)
    ba_indices = [i for i, ba in enumerate(plant_bas) if ba in ba_set]
    idx, score, second = scan(ba_indices)
    if idx is not None and score >= MIN_MATCH_LEN:
        return idx, score, second

    state_set = set(state_fallback)
    state_indices = [i for i, st in enumerate(plant_states) if st in state_set]
    idx, score, second = scan(state_indices)
    if idx is not None and score >= MIN_MATCH_LEN:
        return idx, score, second

    # No tertiary fallback — better to leave unplaced than place in the wrong
    # state. Restricting to BA + state-fallback keeps false matches geographically
    # plausible at the cost of coverage.
    return None, 0, 0


# ---------------------------------------------------------------------------
# Per-abbreviation results depend on the preferred-state set of the *pnode's
# aggregate*, so the same abbreviation under different aggregates could in
# principle resolve differently. We therefore index matches by (abbrev, agg).
log(f"matching {len(pnodes):,} pnodes (per-aggregate preference)...")

# Cache by (abbrev, agg) — same abbreviation under same aggregate gives same answer
cache: dict[tuple[str, str], dict | None] = {}
match_records = []
for _, row in pnodes.iterrows():
    abbrev = row["abbrev"]
    agg = row["Aggregate PNode ID"]
    key = (abbrev, agg)
    if key in cache:
        match_records.append(cache[key])
        continue
    pref_bas = AGG_BA_PREF.get(agg, ["CISO"])
    fallback_states = AGG_STATE_FALLBACK.get(agg, ["CA"])
    idx, score, second = best_match(abbrev, pref_bas, fallback_states)
    if idx is not None and score >= MIN_MATCH_LEN:
        r = plants_sorted.iloc[idx]
        m = {
            "Plant Code": int(r["Plant Code"]),
            "Plant Name": r["Plant Name"],
            "State": r["State"],
            "Latitude": float(r["Latitude"]),
            "Longitude": float(r["Longitude"]),
            "match_score": int(score),
            "second_best": int(second),
        }
    else:
        m = None
    cache[key] = m
    match_records.append(m)

unique_abbrev_match = len(set(k[0] for k, v in cache.items() if v))
log(f"abbreviations matched (across aggregates): {unique_abbrev_match}")
abbrev_match = {}  # for the field-population block below — keyed by (abbrev, agg)
abbrev_match_records = match_records

# Attach to pnodes (records are in pnodes row order)
for col in ["Plant Code", "Plant Name", "State", "Latitude", "Longitude",
            "match_score", "second_best"]:
    pnodes[col] = [r[col] if r else None for r in abbrev_match_records]

n_matched = int(pnodes["Latitude"].notna().sum())
log(f"nodes matched (pre-collision-pass): {n_matched:,} / {len(pnodes):,}")

# Post-pass: when ≥2 DISTINCT non-alias abbreviations both fuzzy-match to
# one plant, at least one is wrong (probably both). We can't tell which, so
# demote all such non-alias matches to unplaced. Alias-driven matches
# (score = 99) are exempt — they're hand-curated and can coexist with fuzzy
# matches at the same plant (e.g., AES Alamitos has ALAMT/ALAMITOS aliased
# plus ALAMOSC fuzzy — both legitimate at the same site).
matched_mask = pnodes["Latitude"].notna()
non_alias = pnodes[matched_mask & (pnodes["match_score"] != 99)]
plant_abbrev_counts = (non_alias.groupby("Plant Code")["abbrev"].nunique())
colliding_plants = plant_abbrev_counts[plant_abbrev_counts >= 2].index.tolist()
collide_mask = (pnodes["Plant Code"].isin(colliding_plants) & (pnodes["match_score"] != 99))
n_demoted = int(collide_mask.sum())
if n_demoted:
    log(f"demoting {n_demoted} nodes attached to {len(colliding_plants)} "
        f"plants that had 2+ distinct non-alias abbrevs (likely false matches)")
    for col in ["Plant Code", "Plant Name", "State", "Latitude",
                 "Longitude", "match_score", "second_best"]:
        pnodes.loc[collide_mask, col] = None
n_matched = int(pnodes["Latitude"].notna().sum())
log(f"nodes matched (final): {n_matched:,} / {len(pnodes):,} ({100*n_matched/len(pnodes):.1f}%)")

# Save BEFORE the assert so we can debug
out = DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv"
pnodes.to_csv(out, index=False)
log(f"saved {out}")

# Self-check: bbox — separately for matched-state and coords
matched = pnodes.dropna(subset=["Latitude", "Longitude"])
bad_bbox = matched[
    (matched["Latitude"] < WECC_BBOX["lat_min"]) | (matched["Latitude"] > WECC_BBOX["lat_max"]) |
    (matched["Longitude"] < WECC_BBOX["lon_min"]) | (matched["Longitude"] > WECC_BBOX["lon_max"])
]
log(f"out-of-WECC-bbox: {len(bad_bbox)}")
if len(bad_bbox):
    print("WARN: sample bad-bbox rows:")
    print(bad_bbox[["PNode ID", "abbrev", "Plant Name", "State",
                     "Latitude", "Longitude", "match_score"]].head(15).to_string())
assert len(bad_bbox) == 0, f"BLOCKER: {len(bad_bbox)} matched coordinates outside WECC bbox"

# Coverage breakdown by aggregate
log("\ncoverage by aggregate hub:")
agg_breakdown = pnodes.groupby("Aggregate PNode ID").agg(
    total=("PNode ID", "size"),
    matched=("Latitude", lambda s: s.notna().sum()),
)
agg_breakdown["match_rate"] = agg_breakdown["matched"] / agg_breakdown["total"]
print(agg_breakdown.to_string())

# Save
out = DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv"
pnodes.to_csv(out, index=False)
log(f"saved {out}")

# ---------------------------------------------------------------------------
# Plot — matched nodes on a simple CA bounding box
log("plotting...")
fig, ax = plt.subplots(figsize=(10, 11))
m = pnodes.dropna(subset=["Latitude", "Longitude"])
state_colors = {"CA": "tab:blue", "AZ": "tab:orange", "NV": "tab:green",
                "OR": "tab:red", "ID": "tab:purple", "UT": "tab:brown",
                "WY": "tab:pink", "MT": "tab:gray", "NM": "tab:olive",
                "WA": "tab:cyan", "CO": "k", "TX": "magenta"}
for state, color in state_colors.items():
    sub = m[m["State"] == state]
    if len(sub) == 0:
        continue
    ax.scatter(sub["Longitude"], sub["Latitude"], s=12, c=color, alpha=0.6,
               label=f"{state} ({len(sub)})")

# Reference frame: CAISO area bounding box (CA mainly)
ax.set_xlim(-125, -110)
ax.set_ylim(31, 43)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Placed generator pnodes — {n_matched:,} / {len(pnodes):,} "
             f"({100*n_matched/len(pnodes):.1f}%)")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
png = DATA_DIR / f"placed_gen_nodes_{SEASON_TAG}.png"
fig.savefig(png, dpi=150)
log(f"saved {png}")

# Probe sanity
log("\nprobe sanity:")
for label, node in [("PROBE_IMPORT", "ALAMT1G_7_B1"), ("HARBOR check", "HARBORG_7_B2")]:
    row = pnodes[pnodes["PNode ID"] == node]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {label}  {node}:  "
              f"plant={r['Plant Name']!r}, state={r['State']}, "
              f"lat={r['Latitude']:.4f}, lon={r['Longitude']:.4f}, "
              f"score={r['match_score']}")
log("E1 OK.")
