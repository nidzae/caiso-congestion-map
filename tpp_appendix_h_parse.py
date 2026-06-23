"""Parse Appendix H of the CAISO Board-Approved Transmission Plan.

The appendix has one project per page (occasionally two) with a labelled
form layout:

  Name              <project name>
  Brief Description <one or two sentences>
  Type              Reliability | Economic | Public Policy
  Objectives        <bullet points or paragraph>
  Project Need Date <date or quarter>
  Expected In-service Date  <date or quarter>
  Interim Solution  <text>
  Project Cost      <e.g. $42M>
  Alternatives Considered but Rejected  <bullets>

We extract Name, Brief Description, Type, Objectives, Project Need Date,
and Expected In-service Date. The longer "Alternatives" text is too noisy
to surface in a hover; it's captured as `alternatives_considered` for the
audit trail but truncated.

Output: data/tpp/appendix_h.csv with one row per project.

The parser is best-effort: PDF extraction merges/splits whitespace
unpredictably, so we use forgiving regexes and trim to the next field
label. If a field is absent the column is left blank.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pypdf

TPP_DIR = Path("data/tpp")
SOURCE_MANIFEST = TPP_DIR / "sources.json"
OUT_CSV = TPP_DIR / "appendix_h.csv"

# Field labels appearing on their own in the form. The order here defines
# the splits — every label terminates the preceding field's text.
FIELD_LABELS = [
    "Name",
    "Brief\\s*Description",
    "Type",
    "Objectives",
    "Project Need\\s*Date",
    "Expected In-\\s*service Date",
    "Interim Solution",
    "Project Cost",
    "Alternatives\\s*Considered but\\s*Rejected",
]
# Compile a single regex that finds any label as a delimiter.
LABEL_RE = re.compile(
    r"(?P<label>" + "|".join(FIELD_LABELS) + r")\s*",
)

# Page-header chrome that we strip before parsing
HEADER_PATTERNS = [
    re.compile(r"ISO 20\d\d-20\d\d Transmission Plan\s+\w+\s+\d{1,2},\s+20\d\d"),
    re.compile(r"California ISO/I&OP\s+H-\d+"),
    re.compile(r"Intentionally left blank", re.I),
]


def _clean(text: str) -> str:
    for pat in HEADER_PATTERNS:
        text = pat.sub("", text)
    return text


def extract_full_text(pdf_path: Path) -> str:
    r = pypdf.PdfReader(str(pdf_path))
    chunks = []
    for p in r.pages:
        t = p.extract_text() or ""
        chunks.append(_clean(t))
    # Join pages with a clear delimiter so a project spanning a page boundary
    # is still recoverable.
    return "\n\n".join(chunks)


def parse(text: str) -> list[dict]:
    """Split on each 'Name <project>' header, then within each block use
    label boundaries to pull out the remaining fields."""
    # First locate every "Name " label that's the start of a record. To
    # avoid false positives where "Name" appears mid-sentence, require it
    # to be at start-of-line OR preceded by whitespace + a label terminator.
    name_re = re.compile(r"(?m)^\s*Name\s+(?P<title>[A-Z][^\n]{4,200})\s*$")
    matches = list(name_re.finditer(text))
    rows = []
    for i, m in enumerate(matches):
        title = m.group("title").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        rows.append({"name": title, **_parse_block(block)})
    return rows


def _parse_block(block: str) -> dict:
    """Walk the label sequence in order, capturing the text between each
    pair of labels. Unknown / missing fields are blank."""
    # Find every label occurrence in the block; record (start, end, label).
    label_re = re.compile(r"(?P<label>" + "|".join(FIELD_LABELS) + r")\s*",
                          re.MULTILINE)
    found = []
    for m in label_re.finditer(block):
        # Normalize the matched label to a canonical key.
        raw = m.group("label").lower().replace("\n", " ")
        raw = re.sub(r"\s+", " ", raw).strip()
        key = {
            "brief description": "brief_description",
            "type": "type",
            "objectives": "objectives",
            "project need date": "project_need_date",
            "expected in- service date": "expected_in_service_date",
            "expected in-service date": "expected_in_service_date",
            "interim solution": "interim_solution",
            "project cost": "project_cost",
            "alternatives considered but rejected":
                "alternatives_considered",
            "name": "name",  # ignored — used as record split, not field
        }.get(raw)
        if key:
            found.append((m.start(), m.end(), key))
    # Build per-field text by slicing between successive label ends.
    fields = {}
    for i, (start, end, key) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(block)
        val = block[end:next_start].strip()
        # Tidy: collapse repeated whitespace, drop bullet character noise.
        val = re.sub(r"[ ]+", " ", val)
        val = re.sub(r"[\t\r]+", " ", val)
        val = re.sub(r" +\n", "\n", val)
        val = re.sub(r"\n{3,}", "\n\n", val)
        if key and key not in fields:
            fields[key] = val
    return fields


def main():
    if not SOURCE_MANIFEST.exists():
        raise SystemExit(
            f"missing {SOURCE_MANIFEST}. Run download_tpp.py first."
        )
    sources = json.loads(SOURCE_MANIFEST.read_text())
    h_meta = sources.get("appendix_h") or {}
    pdf_path = Path(h_meta.get("path") or "")
    if not pdf_path.exists():
        raise SystemExit(
            f"Appendix H PDF missing at {pdf_path}. Re-run download_tpp.py."
        )

    text = extract_full_text(pdf_path)
    rows = parse(text)
    if not rows:
        raise SystemExit("no projects parsed from Appendix H")

    df = pd.DataFrame(rows)
    # Carry source metadata
    df["source_label"] = h_meta.get("label")
    df["source_url"] = h_meta.get("url")
    # Truncate long fields so the CSV stays readable
    for c in ("brief_description", "objectives", "alternatives_considered"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.slice(0, 500)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"parsed {len(df)} projects -> {OUT_CSV}")
    print()
    print("=== sample (first 3) ===")
    for _, r in df.head(3).iterrows():
        print(f"\n  {r['name']}")
        if pd.notna(r.get("brief_description")):
            print(f"    desc: {str(r['brief_description'])[:200]}")
        if pd.notna(r.get("expected_in_service_date")):
            print(f"    ISD: {r['expected_in_service_date']}")


if __name__ == "__main__":
    main()
