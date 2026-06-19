"""Search SP15 pnodes for LA Basin plant identifiers, pick a candidate."""
import re

import gridstatus
import pandas as pd

# Substring patterns that likely identify LA Basin gas plants in CAISO pnode IDs.
# CAISO uses abbreviated substation names — try several variants.
PATTERNS = [
    "ALAMIT", "ALAMT", "ALAM",            # Alamitos
    "HNTGBH", "HNTBCH", "HUNTBC", "HNT",   # Huntington Beach
    "ORMOND", "ORMD",                      # Ormond Beach
    "MANDAL", "MNDL", "MANDA",             # Mandalay
    "REDOND",                              # Redondo Beach
    "ELSEG", "ELSG",                       # El Segundo
    "WALNUT", "WLNT",                      # Walnut Creek
    "SENTL", "SENTI",                      # Sentinel
    "ETIWAN", "ETIWND",                    # Etiwanda
    "INLAND",                              # Inland Empire
    "HARBR", "HARBOR", "HRBR",             # Harbor (LADWP / nearby)
    "LCIEN", "LACIE",                      # La Cienega substation
    "LBASIN", "LABAS",                     # explicit la basin
    "WSCE", "SCEW",                        # SCE Western area code
    "VINCNT", "VINCEN",                    # Vincent substation
    "LUGO",                                # Lugo substation
    "MIRA",                                # Mira Loma
]

print("loading pnode catalog...")
pnodes = gridstatus.CAISO().get_pnodes()
sp15 = pnodes[pnodes["Aggregate PNode ID"] == "TH_SP15_GEN-APND"].copy()
print(f"SP15 generator pnodes: {len(sp15)}")

ids = sp15["PNode ID"].astype(str)

hits = {}
for pat in PATTERNS:
    m = ids[ids.str.contains(pat, case=False, na=False, regex=False)].unique()
    if len(m):
        hits[pat] = sorted(m.tolist())

print("\n== pattern matches (in SP15) ==")
for pat, matches in hits.items():
    print(f"\n[{pat}] ({len(matches)})")
    for n in matches[:15]:
        print(f"   {n}")
    if len(matches) > 15:
        print(f"   ... ({len(matches)-15} more)")
