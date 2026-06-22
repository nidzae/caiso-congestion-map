"""Download the CAISO 'Approved Projects' TPP attachment (vintage-agnostic
list of every active transmission project in the pipeline).

Idempotent: skips download if the cached PDF already exists.

Updated semi-annually at the Transmission Development Forum (TDF). When a
new edition is published, update TPP_URL below.
"""
import sys
from pathlib import Path
import urllib.request

TPP_DIR = Path("data/tpp")
TPP_URL = (
    "https://www.caiso.com/documents/"
    "attachment-1-approved-projects-transmission-planning-process-oct-2025.pdf"
)
TPP_PDF = TPP_DIR / "approved_projects_oct_2025.pdf"


def main():
    TPP_DIR.mkdir(parents=True, exist_ok=True)
    if TPP_PDF.exists():
        print(f"cached: {TPP_PDF}  ({TPP_PDF.stat().st_size:,} bytes)")
        return
    print(f"downloading {TPP_URL}")
    req = urllib.request.Request(TPP_URL,
                                  headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    TPP_PDF.write_bytes(data)
    print(f"saved {TPP_PDF}  ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
