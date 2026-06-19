"""A1 — pull one day of RTM 5-min LMP for PROBE_FLAT, inspect schema.

Gate: a separate congestion-component (MCC) field must exist in the returned
dataframe. If not, print BLOCKER and exit non-zero.
"""
import sys

import gridstatus
from gridstatus import Markets

import config

DATE = "2026-06-15"

print(f"== A1: one-day RTM pull ==")
print(f"node:   {config.PROBE_FLAT}")
print(f"date:   {DATE}")
print(f"market: REAL_TIME_5_MIN")
print()

caiso = gridstatus.CAISO()
df = caiso.get_lmp(
    date=DATE,
    market=Markets.REAL_TIME_5_MIN,
    locations=[config.PROBE_FLAT],
)

assert not df.empty, "BLOCKER: returned dataframe is empty"

n = len(df)
print(f"rows: {n}")
assert 250 <= n <= 320, f"BLOCKER: expected ~288 five-minute rows, got {n}"

print("\ncolumns:")
for col in df.columns:
    print(f"  - {col!r:40s}  dtype={df[col].dtype}")

cong_cols = [c for c in df.columns if "congest" in c.lower() or "mcc" in c.lower()]
energy_cols = [c for c in df.columns if "energy" in c.lower() or c.lower() == "mec"]
loss_cols = [c for c in df.columns if "loss" in c.lower() or c.lower() == "mlc"]

print(f"\ncongestion-component candidate columns: {cong_cols}")
print(f"energy-component candidate columns:     {energy_cols}")
print(f"loss-component candidate columns:       {loss_cols}")

if not cong_cols:
    print("\nBLOCKER: MCC component not exposed in returned schema")
    sys.exit(1)

print("\nfirst 3 rows:")
print(df.head(3).to_string())

print("\nA1 OK — congestion component is exposed; proceed to A2.")
