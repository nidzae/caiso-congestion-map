"""Auto-publish watcher: as each month's c1 finishes, run the metrics
pipeline, rerender every month's HTML so the month-nav reflects the new
month, and push to GitHub. Polls; idempotent.

Designed to run alongside pull_all_months.py in the background.

Usage:
  nohup python3 -u auto_publish.py > data/auto_publish.log 2>&1 &
"""
import re
import subprocess
import sys
import time
from pathlib import Path


DATA = Path("data")
POLL_INTERVAL = 90  # seconds between checks

RAW_PAT  = re.compile(r"node_metrics_raw_(.+)\.csv$")
SIZE_PAT = re.compile(r"node_metrics_with_size_(.+)\.csv$")
DUR_PAT  = re.compile(r"duration_sweep_(.+)\.csv$")
COORD_PAT = re.compile(r"node_coordinates_(.+)\.csv$")


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def find_tags(pattern: re.Pattern) -> set[str]:
    out = set()
    for p in DATA.iterdir():
        m = pattern.match(p.name)
        if m:
            out.add(m.group(1))
    return out


def run(label: str, *args: str) -> int:
    log(f"$ {label} :: {' '.join(args)}")
    r = subprocess.run(args, capture_output=True, text=True)
    tail = (r.stdout or "")[-600:]
    if tail.strip():
        log(tail.rstrip())
    if r.returncode != 0:
        err = (r.stderr or "")[-600:]
        log(f"  exit={r.returncode}  stderr={err.rstrip()}")
    return r.returncode


def git_publish(reason: str) -> bool:
    """Stage everything (gitignore filters cache), commit if non-empty, push.
    Returns True if a new commit was pushed."""
    subprocess.run(["git", "add", "-A"], check=False)
    # Detect if there's anything staged
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode
    if diff == 0:
        log("  (no staged changes; skipping commit)")
        return False
    msg = f"auto: refresh maps — {reason}"
    rc = subprocess.run(
        ["git", "commit", "-m", msg,
         "-m", "(automated by auto_publish.py after a month completed)"]
    ).returncode
    if rc != 0:
        log(f"  git commit failed (exit {rc})")
        return False
    rc = subprocess.run(["git", "push"]).returncode
    if rc != 0:
        log(f"  git push failed (exit {rc})")
        return False
    log(f"  pushed: {msg}")
    return True


last_published_size_tags: set[str] = set()
processed_metrics: set[str] = set()   # tags we've attempted run_month_metrics for

log("=== auto-publish watcher starting ===")
log(f"poll interval: {POLL_INTERVAL}s")
log(f"DATA dir: {DATA.resolve()}")

while True:
    try:
        raw  = find_tags(RAW_PAT)
        size = find_tags(SIZE_PAT)

        # Step 1: for any month with raw but no size, run the metrics pipeline.
        needs_metrics = (raw - size) - processed_metrics
        for tag in sorted(needs_metrics):
            log(f"--- new c1 output detected for {tag}; running metrics pipeline ---")
            processed_metrics.add(tag)
            rc = run(f"metrics[{tag}]", sys.executable, "run_month_metrics.py", tag)
            if rc != 0:
                log(f"  metrics pipeline failed for {tag}; will not retry this run")
                continue

        # Step 2: full set of months ready to render
        ready = find_tags(SIZE_PAT) & find_tags(DUR_PAT) & find_tags(COORD_PAT)

        # Step 3: if the ready set grew (or is non-empty + nothing published yet),
        # rerender every page and push.
        if ready and ready != last_published_size_tags:
            new_tags = ready - last_published_size_tags
            log(f"--- {len(new_tags)} new ready month(s): {sorted(new_tags)} — rerendering ---")
            rc = run("render-all", sys.executable, "render_all_months.py")
            if rc == 0:
                pushed = git_publish(reason=f"{len(ready)} months available "
                                            f"(+{sorted(new_tags)})")
                if pushed:
                    last_published_size_tags = ready
            else:
                log("  render_all_months failed; will retry next cycle")

        time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        log("interrupted, exiting")
        break
    except Exception as e:
        log(f"!!! unexpected error: {type(e).__name__}: {e}")
        time.sleep(POLL_INTERVAL)
