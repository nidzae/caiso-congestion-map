"""Download the CAISO Approved Projects TPP attachments.

Two artifacts:
  1. The cross-PTO XLSX (machine-readable, all 9 PTOs in separate sheets).
     This is the primary source after the Jan 2025 publication.
  2. The SCE-only PDF (older format; kept as a supplemental fallback so
     status text not present in the XLSX is still available).
  3. The 2024-2025 Board-Approved Transmission Plan Appendix H PDF
     (per-project narrative descriptions — name, objectives, ISD).

Idempotent: skips download if the cached file already exists.

Both the XLSX and the Appendix H PDF rotate at the Transmission Development
Forum. The XLSX probe tries the newest semi-annual vintages and falls back to
the last one we know is published. When a new edition appears, just update
LATEST_XLSX_TAG.
"""
import sys
from pathlib import Path
import urllib.request
import urllib.error

TPP_DIR = Path("data/tpp")

# Try these vintage tags in order until one returns 200. Keeps the script
# working through new TDF publications without manual edits.
XLSX_VINTAGE_PROBES = [
    "jul-2026", "apr-2026", "jan-2026",
    "oct-2025", "jul-2025", "apr-2025", "jan-2025",
]
XLSX_URL_FMT = (
    "https://www.caiso.com/documents/"
    "approved-projects-transmission-planning-process-{tag}.xlsx"
)

# SCE-only PDF (status text richer than XLSX in some cases)
PDF_URL = (
    "https://www.caiso.com/documents/"
    "attachment-1-approved-projects-transmission-planning-process-oct-2025.pdf"
)
PDF_FILE = TPP_DIR / "approved_projects_oct_2025.pdf"

# Board-Approved 2024-2025 Transmission Plan Appendix H (project narratives)
APPENDIX_H_URL = (
    "https://www.caiso.com/documents/"
    "appendix-h-board-approved-2024-2025-transmission-plan.pdf"
)
APPENDIX_H_FILE = TPP_DIR / "appendix_h_2024_2025_tplan.pdf"

# Where we record which XLSX vintage was actually fetched (for source-URL
# attribution in the UI hover).
SOURCE_LOG = TPP_DIR / "sources.json"


def _fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"cached: {dest}  ({dest.stat().st_size:,} bytes)")
        return True
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    dest.write_bytes(data)
    print(f"saved {dest}  ({len(data):,} bytes)")
    return True


def download_latest_xlsx() -> tuple[Path | None, str | None, str | None]:
    """Find the newest available XLSX. Returns (path, url, vintage_label)."""
    for tag in XLSX_VINTAGE_PROBES:
        url = XLSX_URL_FMT.format(tag=tag)
        dest = TPP_DIR / f"approved_projects_{tag.replace('-', '_')}.xlsx"
        if dest.exists():
            print(f"cached: {dest}  ({dest.stat().st_size:,} bytes)")
            return dest, url, tag
        if _fetch(url, dest):
            return dest, url, tag
        print(f"  not yet published: {tag}")
    return None, None, None


def main():
    TPP_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path, xlsx_url, xlsx_tag = download_latest_xlsx()
    if xlsx_path is None:
        print("WARN: no XLSX vintage available — TPP coverage will be SCE-PDF only")
    _fetch(PDF_URL, PDF_FILE)
    _fetch(APPENDIX_H_URL, APPENDIX_H_FILE)

    import json
    # Build the source manifest the UI hover will use to render hyperlinks
    # with descriptive anchor text ("CAISO Approved Projects XLSX (Jul 2025)").
    sources = {
        "xlsx": {
            "path": str(xlsx_path) if xlsx_path else None,
            "url": xlsx_url,
            "vintage": xlsx_tag,
            "label": (
                f"CAISO Approved Projects XLSX ({xlsx_tag.title()})"
                if xlsx_tag else None
            ),
        },
        "pdf_sce": {
            "path": str(PDF_FILE) if PDF_FILE.exists() else None,
            "url": PDF_URL,
            "label": "CAISO Approved Projects Attachment 1 (Oct 2025, SCE)",
        },
        "appendix_h": {
            "path": str(APPENDIX_H_FILE) if APPENDIX_H_FILE.exists() else None,
            "url": APPENDIX_H_URL,
            "label": "CAISO 2024-2025 Board-Approved Transmission Plan, Appendix H",
        },
    }
    SOURCE_LOG.write_text(json.dumps(sources, indent=2))
    print(f"saved source manifest -> {SOURCE_LOG}")


if __name__ == "__main__":
    main()
