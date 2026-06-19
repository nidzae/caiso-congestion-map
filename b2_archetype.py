"""B2 — archetype (hue) classification on the two probes.

Per brief §3b:
  MCC_mid = mean MCC over MIDDAY_WINDOW
  MCC_eve = mean MCC over EVENING_WINDOW
  import_signal = MCC_eve     (positive: importing into binding constraint at peak)
  export_signal = -MCC_mid    (positive: negative midday MCC = export/oversupply)

Classify with flatness threshold epsilon:
  both <  eps                       -> WHITE
  import >= eps, export <  eps      -> BLUE
  export >= eps, import <  eps      -> RED
  both   >= eps                     -> PURPLE
"""
import sys
from pathlib import Path

import pandas as pd

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"


def hourly_path(node: str) -> Path:
    return DATA_DIR / f"rtm_hourly_{node}_{SEASON_TAG}.parquet"


def window_mean(s: pd.Series, window: tuple[int, int]) -> float:
    lo, hi = window
    mask = (s.index.hour >= lo) & (s.index.hour < hi)
    return float(s[mask].mean())


def classify(import_signal: float, export_signal: float, eps: float) -> str:
    imp = import_signal >= eps
    exp = export_signal >= eps
    if imp and exp:
        return "PURPLE"
    if imp:
        return "BLUE"
    if exp:
        return "RED"
    return "WHITE"


print("== B2: archetype classification ==")
print(f"midday window:  {config.MIDDAY_WINDOW} (end-exclusive)")
print(f"evening window: {config.EVENING_WINDOW} (end-exclusive)")
print(f"epsilon:        ${config.EPSILON_FLATNESS}")

rows = []
for label, node in [("PROBE_IMPORT", config.PROBE_IMPORT),
                    ("PROBE_FLAT",   config.PROBE_FLAT)]:
    p = hourly_path(node)
    if not p.exists():
        print(f"BLOCKER: missing hourly cache {p}; rerun B1.")
        sys.exit(1)
    h = pd.read_parquet(p)["Congestion"]

    mcc_mid = window_mean(h, config.MIDDAY_WINDOW)
    mcc_eve = window_mean(h, config.EVENING_WINDOW)
    import_signal = mcc_eve
    export_signal = -mcc_mid
    arch = classify(import_signal, export_signal, config.EPSILON_FLATNESS)

    rows.append({
        "probe": label,
        "node": node,
        "MCC_mid": mcc_mid,
        "MCC_eve": mcc_eve,
        "import_signal": import_signal,
        "export_signal": export_signal,
        "archetype": arch,
    })

table = pd.DataFrame(rows)
print()
print(table.to_string(index=False))

probe_import_arch = next(r["archetype"] for r in rows if r["probe"] == "PROBE_IMPORT")

print()
if probe_import_arch != "BLUE":
    print(f"BLOCKER: PROBE_IMPORT classified {probe_import_arch}, expected BLUE.")
    print("Either the probe is too dilute (switch to a tighter sensor) or epsilon is too high.")
    sys.exit(1)

print(f"B2 OK — PROBE_IMPORT classified BLUE.")
