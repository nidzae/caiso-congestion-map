"""Render one HTML page per available month, then mirror the latest to index.html.

The per-page HTMLs share a month-nav bar at the top so the user can click
across months. GitHub Pages serves them all at the repo root URL.

Usage:
  python3 render_all_months.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path


DATA_DIR = Path("data")
pat = re.compile(r"node_metrics_with_size_(.+)\.csv$")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def discover_months():
    tags = []
    for p in DATA_DIR.iterdir():
        m = pat.match(p.name)
        if not m:
            continue
        tag = m.group(1)
        if not (DATA_DIR / f"duration_sweep_{tag}.csv").exists():
            continue
        if not (DATA_DIR / f"node_coordinates_{tag}.csv").exists():
            continue
        tags.append(tag)
    # canonical YYYY-MM sort, with summer2025 sorted as 2025-08
    def key(t): return "2025-08" if t == "summer2025" else t
    return sorted(set(tags), key=key)


if __name__ == "__main__":
    tags = discover_months()
    if not tags:
        sys.exit("no completed-month metric files found")
    log(f"rendering {len(tags)} months: {tags}")
    for tag in tags:
        log(f"--- rendering {tag} ---")
        rc = subprocess.run([sys.executable, "-u", "f1_render_map.py", tag]).returncode
        if rc != 0:
            log(f"!!! f1 failed for {tag} (exit {rc})")
    log("done")
