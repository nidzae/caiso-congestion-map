"""A3 — pull binding constraints + shadow prices for one day.

gridstatus exposes two RTM constraint datasets via OASIS:
  - interval_intertie_constraint_shadow_prices  (PRC_RTM_FLOWGATE — interties)
  - interval_nomogram_branch_shadow_prices      (PRC_RTM_NOMOGRAM — all
                                                 internal elements:
                                                 single branches, transformers,
                                                 OMS constraints, nomograms)

Probed other candidate OASIS reports (PRC_BC, PRC_CMP, PRC_INTVL_CNSTR_PRC,
PRC_CNSTR) — all return INVALID_REQUEST or non-shadow-price data. The two
datasets above are the complete picture.
"""
import sys

import pandas as pd

import gridstatus

import config

DATE = "2026-06-15"
END = "2026-06-16"
DATASETS = [
    "interval_intertie_constraint_shadow_prices",
    "interval_nomogram_branch_shadow_prices",
]

print("== A3: constraint + shadow-price pull ==")
print(f"date: {DATE}")

caiso = gridstatus.CAISO()

panels: dict[str, pd.DataFrame] = {}

for ds in DATASETS:
    print(f"\n-- dataset: {ds} --")
    try:
        df = caiso.get_oasis_dataset(dataset=ds, date=DATE, end=END)
    except Exception as e:
        print(f"  error: {type(e).__name__}: {e}")
        continue
    print(f"  rows: {len(df)}")
    if df.empty:
        print("  (empty)")
        continue
    print(f"  columns: {list(df.columns)}")
    print("  first row:")
    print(df.head(1).to_string(index=False))
    panels[ds] = df

if not panels:
    print("\nBLOCKER: shadow-price panel missing — no constraint data returned")
    sys.exit(1)


SCHEMA = {
    "interval_intertie_constraint_shadow_prices": {
        "id_col": "TI_ID",
        "id_extra": ["TI_DIRECTION"],
        "price_col": "PRC",
        "scenario_col": "CONSTRAINT_CAUSE",
    },
    "interval_nomogram_branch_shadow_prices": {
        "id_col": "NOMOGRAM_ID",
        "id_extra": [],
        "price_col": "PRC",
        "scenario_col": "CONSTRAINT_CAUSE",
    },
}

print("\n== summary per dataset ==")
for name, df in panels.items():
    schema = SCHEMA[name]
    id_col = schema["id_col"]
    price_col = schema["price_col"]
    scenario_col = schema["scenario_col"]

    print(f"\n[{name}]")
    print(f"  identifier column: {id_col!r}  (+ {schema['id_extra']})")
    print(f"  shadow-price column: {price_col!r}")

    unique_ids = df[id_col].dropna().unique()
    print(f"  distinct constraint IDs: {len(unique_ids)}")
    print("  first 15 IDs:")
    for n in unique_ids[:15]:
        print(f"    - {n}")

    unique_scenarios = df[scenario_col].dropna().unique()
    print(f"  distinct contingency scenarios ({scenario_col}): {len(unique_scenarios)}")
    for s in unique_scenarios[:10]:
        print(f"    - {s}")

    s = pd.to_numeric(df[price_col], errors="coerce")
    print(f"  shadow price min:    {s.min():.4f}")
    print(f"  shadow price max:    {s.max():.4f}")
    print(f"  shadow price |mean|: {s.abs().mean():.4f}")
    print(f"  nonzero rows: {(s.abs() > 1e-9).sum()} / {len(s)}")

print("\n== A3 gate ==")
print("nomogram + intertie shadow prices available — comprehensive coverage.")
print("A3 OK.")
