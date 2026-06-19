"""A2 — verify the LMP decomposition identity holds.

For every 5-min interval: LMP - sum(posted components) must be ~0.
Components in CAISO RTM: Energy (MEC), Congestion (MCC), Loss (MLC), GHG.
Assert max|residual| < $0.05.
"""
import sys

import gridstatus
from gridstatus import Markets
import numpy as np

import config

DATE = "2026-06-15"
COMPONENT_COLS = ["Energy", "Congestion", "Loss", "GHG"]
TOLERANCE = 0.05

print("== A2: decomposition identity ==")
print(f"node:  {config.PROBE_FLAT}")
print(f"date:  {DATE}")

caiso = gridstatus.CAISO()
df = caiso.get_lmp(
    date=DATE,
    market=Markets.REAL_TIME_5_MIN,
    locations=[config.PROBE_FLAT],
)

present = [c for c in COMPONENT_COLS if c in df.columns]
missing = [c for c in COMPONENT_COLS if c not in df.columns]
print(f"\ncomponents summed: {present}")
if missing:
    print(f"components absent (skipped): {missing}")

component_sum = df[present].sum(axis=1)
residual = df["LMP"] - component_sum

max_abs = float(np.abs(residual).max())
mean_abs = float(np.abs(residual).mean())

print(f"\nrows:           {len(df)}")
print(f"max |residual|: ${max_abs:.6f}")
print(f"mean|residual|: ${mean_abs:.6f}")

if max_abs >= TOLERANCE:
    worst = residual.abs().idxmax()
    print("\nworst row:")
    print(df.loc[[worst], ["Time", "LMP"] + present].to_string())
    print(f"\nBLOCKER: decomposition identity violated (max |residual| ${max_abs:.4f} >= ${TOLERANCE})")
    sys.exit(1)

print(f"\nA2 OK — identity holds within ${TOLERANCE}; we are using the real components.")
