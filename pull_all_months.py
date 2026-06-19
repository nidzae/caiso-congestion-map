"""Driver — pull RTM 5-min MCC for every month of 2025 by repeatedly invoking
c1_scale_metrics.py.

Sequential because OASIS rate-limits parallel requests anyway. Resumable —
c1 itself caches per-batch and skips done batches.

Estimated wall time: ~2 hr per month × 12 months = ~24 hr.
"""
import subprocess
import sys
import time
from calendar import monthrange

YEAR = 2025
MONTHS = list(range(1, 13))


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def month_range(year: int, month: int):
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year = year + 1
    start = f"{year:04d}-{month:02d}-01"
    end = f"{next_year:04d}-{next_month:02d}-01"
    tag = f"{year:04d}-{month:02d}"
    return start, end, tag


# Skip 2025-08 since it's already pulled (under SEASON_TAG=summer2025, but the
# raw batches under mcc_raw/summer2025 are good — we just need the wide
# parquet under the new tag. Simplest: rerun c1 with new tag, it'll re-pull.
# Decision: skip 2025-08 entirely (re-running would re-cache; instead we will
# rename in a follow-up if needed). Pull the other 11 months.
SKIP_TAGS = set()  # leave empty to do all 12; we'll dedup August at metrics time

t_global = time.time()
for m in MONTHS:
    start, end, tag = month_range(YEAR, m)
    if tag in SKIP_TAGS:
        log(f"SKIP {tag}")
        continue
    log(f"=== month {m}/12 — {tag} ({start} -> {end}) ===")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "c1_scale_metrics.py", start, end, tag],
        cwd=".",
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"!!! c1 failed for {tag} (exit {proc.returncode}); continuing anyway")
    log(f"=== {tag} done in {elapsed/60:.1f}m; total elapsed {(time.time()-t_global)/3600:.2f}h ===")

log("ALL MONTHS DONE")
