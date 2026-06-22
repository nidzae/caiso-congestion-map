"""Run d1 → d3 → d4 → g2 for one month tag.

Assumes c1 has already produced data/mcc_wide_<tag>.parquet and
data/node_metrics_raw_<tag>.csv.

Usage:
  python3 run_month_metrics.py 2025-01
"""
import subprocess
import sys
import time
from calendar import monthrange


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def month_dates(tag: str):
    """'2025-01' -> ('2025-01-01', '2025-02-01')"""
    y, m = map(int, tag.split("-"))
    last = monthrange(y, m)[1]
    start = f"{y:04d}-{m:02d}-01"
    nm = m + 1
    ny = y
    if nm > 12:
        nm = 1; ny = y + 1
    end = f"{ny:04d}-{nm:02d}-01"
    return start, end


def run(args):
    log(f"$ {' '.join(args)}")
    return subprocess.run([sys.executable, "-u", *args]).returncode


def ensure_coords_symlink(tag: str):
    """Point node_coordinates_<tag>.csv at the global summer2025 file so the
    renderer can find per-tag coords. Plant lat/lon doesn't change month
    to month, so e1 doesn't need to be re-run."""
    import os
    from pathlib import Path
    src = Path("data/node_coordinates_summer2025.csv")
    dst = Path(f"data/node_coordinates_{tag}.csv")
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        log(f"  WARN: {src} missing; multi-month render will fail for {tag}")
        return
    # relative symlink so it survives a folder move
    os.symlink(src.name, dst)
    log(f"  symlinked {dst} -> {src.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 run_month_metrics.py <SEASON_TAG e.g. 2025-01>")
        sys.exit(1)
    tag = sys.argv[1]
    start, end = month_dates(tag)
    log(f"=== month pipeline for {tag} ({start} -> {end}) ===")

    rc = run(["d1_shadow_panel.py", start, end, tag])
    if rc != 0: sys.exit(rc)
    rc = run(["d3_rating_crosswalk.py", tag])
    if rc != 0: sys.exit(rc)
    rc = run(["d4_attribute_size.py", tag])
    if rc != 0: sys.exit(rc)
    rc = run(["g2_duration_sweep.py", tag])
    if rc != 0: sys.exit(rc)
    ensure_coords_symlink(tag)
    # Always-on rebuild of the cross-month TPP crosswalk so the latest
    # tag's k* coverage is included. compute_persistence is cross-month
    # too — only rerun once after the LAST month's metrics finish.
    rc = run(["tpp_crosswalk.py"])
    if rc != 0:
        log("  (tpp_crosswalk failed but continuing)")
    log(f"=== {tag} metrics complete ===")
