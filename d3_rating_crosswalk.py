"""D3 — constraint-to-rating crosswalk.

1. Parse the 2026 WECC Path Rating Catalog into a table of named paths.
2. For each binding constraint in our D1 panel, attempt to fuzzy-match the
   OASIS NOMOGRAM_ID / TI_ID against the WECC path definitions.
3. For unmatched constraints, infer voltage from the NOMOGRAM_ID and use the
   voltage-class proxy MVA from config.

Output: data/constraint_rating_crosswalk_summer2025.csv
        with columns: physical_line, constraint_key, kV, matched_path,
        rating_MW, rating_source, match_confidence
"""
import re
import sys
import sys
from pathlib import Path

import pandas as pd
import pypdf

import config

# CLI: python3 d3_rating_crosswalk.py <SEASON_TAG>
CATALOG_PDF = Path("data/wecc/2026_path_rating_catalog_public.pdf")
OUT_PATHS_CSV = Path("data/wecc/wecc_paths_2026.csv")
DATA_DIR = Path("data")
SEASON_TAG = "summer2025"
if len(sys.argv) >= 2:
    SEASON_TAG = sys.argv[1]

# Add a 70 kV proxy (CAISO uses _70.0_ for some sub-transmission); fold into config
VOLTAGE_TO_MVA = dict(config.VOLTAGE_TO_MVA)
VOLTAGE_TO_MVA.setdefault(70, 150)
VOLTAGE_TO_MVA.setdefault(60, 150)
VOLTAGE_TO_MVA.setdefault(345, 1300)


# ---------------------------------------------------------------------------
# 1. Parse the WECC catalog
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(r"^\s*(\d{1,3})\.\s+(\S.*?)\s*$", re.M)
MW_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)\s*MW", re.I)
HEADER_NOISE_RE = re.compile(r"\d{4}\s+PATH RATING CATALOG PUBLIC\s+\d+")


def parse_catalog(pdf_path: Path) -> pd.DataFrame:
    """Parse the catalog. One path entry per (num, name) tuple; entries can
    span multiple pages, and small "[Deleted]" entries can stack on one page."""
    r = pypdf.PdfReader(str(pdf_path))
    # Stitch all body text together to handle multi-page entries cleanly
    body = ""
    for i, page in enumerate(r.pages):
        if i < 5:
            continue
        text = page.extract_text() or ""
        cleaned = HEADER_NOISE_RE.sub(" ", text).replace("<Public>", " ")
        cleaned = cleaned.replace("–", "-")
        body += "\n" + cleaned

    # Find every header line "N. <text>" — locate their positions to slice the body.
    headers = []
    for m in HEADER_RE.finditer(body):
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        if num < 1 or num > 99:
            continue
        headers.append((num, m.start(), m.group(2).strip()))

    rows = []
    for idx, (num, start, header_line) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(body)
        block = body[start:end]
        name = re.sub(r"\[.*$", "", header_line).strip()
        if re.search(r"\bdeleted\b|\bsee path[s]?\b", header_line, re.I):
            rows.append({"path_number": num, "path_name": name,
                         "status": "deleted", "definitions": "",
                         "mw_values": [], "raw_block": header_line})
            continue

        mws = [int(x.replace(",", "")) for x in MW_RE.findall(block)]

        defn = block
        defn = re.sub(r"(?i)\b(accepted|existing|conditionally)\s+rating\b", " ", defn)
        defn = re.sub(r"(?i)\b(definitions|transfer limits)\b", " ", defn)
        defn = re.sub(r"(?i)\b(north|south|east|west|to)\b\s+", " ", defn)
        defn = re.sub(r"(?i)\b(simultaneous|non-simultaneous|w/?o?)\b", " ", defn)
        defn = MW_RE.sub(" ", defn)
        defn = re.sub(r"\s+", " ", defn).strip()
        defn = defn[:2000]

        rows.append({
            "path_number": num,
            "path_name": name,
            "status": "active",
            "definitions": defn,
            "mw_values": mws,
            "raw_block": block[:800],
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["path_number"]).sort_values("path_number")
    return df


print("== D3: rating crosswalk ==")
print(f"parsing catalog: {CATALOG_PDF}")
paths_df = parse_catalog(CATALOG_PDF)
print(f"parsed paths: {len(paths_df)}  ({(paths_df['status']=='active').sum()} active, "
      f"{(paths_df['status']=='deleted').sum()} deleted)")

# Compute a single rating value per path: max MW found (typical: pick the higher direction limit).
def pick_rating(mws):
    if not mws:
        return None
    return max(mws)

paths_df["rating_MW"] = paths_df["mw_values"].apply(pick_rating)
paths_df.to_csv(OUT_PATHS_CSV, index=False)
print(f"saved {OUT_PATHS_CSV}")

print("\nsample of parsed paths:")
print(paths_df[paths_df["status"]=="active"]
      [["path_number","path_name","rating_MW","definitions"]].head(15).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Build crosswalk for D1 constraints
# ---------------------------------------------------------------------------
constraint_summary = pd.read_csv(
    DATA_DIR / f"constraint_summary_{SEASON_TAG}.csv", index_col=0)
# Restrict to physical lines (drop contingency suffix), aggregate sum_abs_mu.
def physical_line(key: str) -> str:
    if "|" not in key:
        return key
    head, _, _ = key.partition("|")
    return head

constraint_summary["physical_line"] = constraint_summary.index.map(physical_line)
phys = constraint_summary.groupby("physical_line").agg(
    n_scenarios=("sum_abs_mu", "size"),
    sum_abs_mu=("sum_abs_mu", "sum"),
).sort_values("sum_abs_mu", ascending=False)
print(f"\ndistinct physical lines: {len(phys)}  (vs {len(constraint_summary)} scenario-instances)")


# Substation tokens from a NOMOGRAM_ID like
#   NOM:30060_MIDWAY  _500_24156_VINCENT _500_BR_2 _3
# After NOM: prefix, drop trailing branch element, keep substation tokens.
SUBNAME_RE = re.compile(r"_?(\d+)_([A-Z][A-Z0-9.\- ]{1,15}?)\s*_(\d{2,3}(?:\.0)?)")
# Intertie names like "Line_MA-OD_69KV" or "WMesaWT2_448 MVA" or "LDWP_IPP_NORTH"
ITC_VOLTAGE_RE = re.compile(r"_(\d{2,3})\s*KV", re.I)
ITC_MVA_RE = re.compile(r"_(\d{2,4})\s*MVA", re.I)
ITC_NAME_RE = re.compile(r"^(?:Line_)?([A-Za-z][A-Za-z0-9]*)-([A-Za-z][A-Za-z0-9]*)")


def extract_substations(line_key: str):
    """Return (sub_name_1, sub_name_2, voltage_kv, explicit_rating_mw_or_None)."""
    explicit_rating = None
    # ITC handling — different format from NOM
    if line_key.startswith("ITC:"):
        body = line_key[4:]
        kv_m = ITC_VOLTAGE_RE.search(body)
        mva_m = ITC_MVA_RE.search(body)
        if mva_m:
            explicit_rating = int(mva_m.group(1))
        kv = int(kv_m.group(1)) if kv_m else None
        # Strip Line_ prefix and trailing _<num>KV / _<num> MVA
        clean = re.sub(r"_\d{2,4}\s*(KV|MVA)", "", body, flags=re.I)
        clean = clean.replace("Line_", "")
        # Pull out hyphenated substation pair if present (e.g., MA-OD)
        name_m = ITC_NAME_RE.search(clean)
        if name_m:
            return name_m.group(1), name_m.group(2), kv, explicit_rating
        # Otherwise treat the whole stem as a single tag (e.g., LDWP_IPP_NORTH)
        tag = re.split(r"[_\s]", clean)[0]
        return tag, None, kv, explicit_rating

    if line_key.startswith("NOM:"):
        line_key = line_key[4:]
    matches = SUBNAME_RE.findall(line_key)
    if not matches:
        return None, None, None, None
    names = []
    voltages = set()
    for _node, name, kv in matches:
        n = name.strip().replace(" ", "")
        if n and n not in [x[0] for x in names]:
            names.append((n, kv))
        try:
            voltages.add(int(float(kv)))
        except ValueError:
            pass
    sub1 = names[0][0]
    sub2 = names[1][0] if len(names) > 1 else None
    voltage = max(voltages) if voltages else None
    return sub1, sub2, voltage, None


# Hand-curated aliases for OASIS-vs-WECC substation spelling drifts.
# Map OASIS form -> WECC form so the normalized match works either way.
ALIAS = {
    "WIRLWIND": "WHIRLWIND",
    "VINCNT":   "VINCENT",
    "MOSSLD":   "MOSSLANDING",
    "LASAGUIL": "LASAGUILA",
    "LOSBNS":   "LOSBANOS",
    "MIDWY":    "MIDWAY",
    "ELDORADO": "ELDORADO",
    "PALOVRD":  "PALOVERDE",
}


def normalize(s: str) -> str:
    if not s:
        return ""
    t = re.sub(r"[^A-Za-z]+", "", s).upper()
    return ALIAS.get(t, t)


# Voltage extraction fallback: handle names like "115kv SOC_BESO" or "_69KV"
LOOSE_VOLTAGE_RE = re.compile(r"(\d{2,3})\s*kv", re.I)


def match_to_wecc_path(sub1: str, sub2: str, paths: pd.DataFrame):
    """Return (path_number, path_name, rating_MW, confidence) or (None, None, None, 0)."""
    if not sub1 or not sub2:
        return None, None, None, 0
    s1, s2 = normalize(sub1), normalize(sub2)
    best = (None, None, None, 0)
    for _, p in paths[paths["status"] == "active"].iterrows():
        defn_norm = normalize(p["definitions"])
        # Need both substations present in the path definition.
        if s1 in defn_norm and s2 in defn_norm:
            # confidence: shorter substrings = weaker match
            conf = min(len(s1), len(s2))
            if conf > best[3]:
                best = (p["path_number"], p["path_name"], p["rating_MW"], conf)
    return best


print("\nbuilding crosswalk (all 176 physical lines)...")
rows = []
for line_key, srow in phys.iterrows():
    sub1, sub2, kv, explicit_rating = extract_substations(line_key)
    # If structured extraction missed kV, try a loose name scan
    if kv is None:
        loose = LOOSE_VOLTAGE_RE.search(line_key)
        if loose:
            kv = int(loose.group(1))
    pnum, pname, prating, conf = match_to_wecc_path(sub1, sub2, paths_df)
    if pnum is not None and prating is not None:
        rating_MW = prating
        source = f"WECC path {pnum}"
    elif explicit_rating is not None:
        rating_MW = explicit_rating
        source = "ITC name (explicit MVA)"
    elif kv is not None:
        rating_MW = VOLTAGE_TO_MVA.get(kv)
        source = f"voltage proxy {kv} kV" if rating_MW else f"voltage {kv} kV (no proxy)"
    else:
        rating_MW = None
        source = "unknown"
    rows.append({
        "physical_line": line_key,
        "n_scenarios": srow["n_scenarios"],
        "sum_abs_mu": srow["sum_abs_mu"],
        "sub1": sub1, "sub2": sub2, "kV": kv,
        "matched_path": pname,
        "rating_MW": rating_MW,
        "rating_source": source,
        "match_confidence": conf,
    })

crosswalk = pd.DataFrame(rows)
out = DATA_DIR / f"constraint_rating_crosswalk_{SEASON_TAG}.csv"
crosswalk.to_csv(out, index=False)
print(f"saved {out}")

print("\n== TOP 30 PHYSICAL LINES — RATING CROSSWALK (audit by eye) ==")
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 220)
print(crosswalk.head(30).to_string(index=False))

# Coverage summary
named = (crosswalk["rating_source"].str.startswith("WECC")).sum()
proxy = (crosswalk["rating_source"].str.startswith("voltage")).sum()
explicit = (crosswalk["rating_source"].str.startswith("ITC")).sum()
unknown = (crosswalk["rating_source"] == "unknown").sum()
print()
print(f"of {len(crosswalk)} physical lines:")
print(f"  named-path matches:      {named}")
print(f"  ITC explicit MVA:        {explicit}")
print(f"  voltage proxies:         {proxy}")
print(f"  unknown:                 {unknown}")

# Sanity check: where does our key constraint (Midway-Vincent / Path 26) land?
print()
print("== sanity check: Path 26 constraints (Midway-Vincent / Wirlwind / Whirlwind) ==")
path26_mask = crosswalk["physical_line"].str.contains("MIDWAY", case=False, na=False)
print(crosswalk[path26_mask].to_string(index=False))
