"""Parse the CAISO Approved Projects XLSX -> structured CSV.

Replaces the older SCE-only PDF parser. The XLSX has one sheet per PTO
(PG&E, SCE, SDG&E, VEA/GLW, DCRT, LS Power, Citizens, Lotus, HWT) with a
consistent column shape:

  col  0  = Project name
  col  1  = PTO
  col  2  = Transmission Plan Approved (e.g. '2022-2023')
  col  3  = In-Service Date at Approval (the "original" ISD)
  cols 4..N-5 = Successive TDF revisions (semi-annual; newest at the right)
  col  N-4 = Project Status
  col  N-3 = Expected CPUC Permit Application Filing
  col  N-2 = Expected Construction Start
  col  N-1 = Notes

The "latest ISD" is the rightmost non-empty cell among the TDF columns. The
original-vs-latest delta is reported as years_slipped.

Per-cell date format varies wildly:
  - pandas datetime (most common)
  - bare integer year like 2032
  - string like 'TBD', 'New', 'Pending', 'Exempt', 'Q4 2024', 'In-Service',
    'Completed', 'Cancelled', 'Close-out', 'In service', 'Dec-21', or even
    multi-line composites like 'VEA portion: Dec-21\\nGLW portion: 2023'.

The parser tolerates all of these; cells that can't be coerced to a year
are recorded as the original string in original_isd/latest_isd but contribute
None to original_year/latest_year.

The output schema is unchanged from the previous PDF parser so the rest of
the pipeline (tpp_crosswalk.py) doesn't need to know.
"""
import json
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd

TPP_DIR = Path("data/tpp")
SOURCE_MANIFEST = TPP_DIR / "sources.json"
OUT_CSV = TPP_DIR / "tpp_projects.csv"

# --- substation extraction (lifted from the PDF parser, unchanged) ----
SUB_PAIR_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s*[\-–]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b"
)
LEADING_SUB_RE = re.compile(r"^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b")
KV_RE = re.compile(r"\b(\d{2,3})\s*kV\b", re.I)
_TRAILING_NOISE_RE = re.compile(
    r"\s*(Line|Lines|Upgrade|Project|Reinforcement|Reconductor|Reconfiguration|"
    r"Mitigation|Replacement|Installation|Conversion|Capacitor|Reactor|Transformer)\b.*$",
    re.I,
)


def _strip_trailing_noise(s: str) -> str:
    return _TRAILING_NOISE_RE.sub("", s).strip()


def parse_substations(name: str) -> tuple[str | None, str | None, int | None]:
    if not isinstance(name, str):
        return None, None, None
    kv_m = KV_RE.search(name)
    kv = int(kv_m.group(1)) if kv_m else None
    cleaned = re.sub(
        r"^(?:Reconductor|Rebuild/build|Loop|Method of Service for|Add\s+\S+\s+\S+|New)\s+",
        "", name,
    )
    pair = SUB_PAIR_RE.search(cleaned)
    if pair:
        s1 = _strip_trailing_noise(pair.group(1).strip())
        s2 = _strip_trailing_noise(pair.group(2).strip())
        return s1 or None, s2 or None, kv
    lead = LEADING_SUB_RE.search(cleaned)
    if lead:
        return _strip_trailing_noise(lead.group(1).strip()) or None, None, kv
    return None, None, kv


# --- date cell coercion ---------------------------------------------------
_TERMINAL_STATES = {
    "complete", "completed", "in-service", "in service", "close-out",
    "closeout", "cancelled", "canceled", "exempt", "approved", "energized",
}
_PENDING_STATES = {"tbd", "new", "pending", "n/a", "na", "?"}

_DATE_STR_RE = re.compile(
    r"(?:(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\-\s]?(?P<yy2>\d{2})\b)"
    r"|(?P<full>(?<!\d)(?:19|20)\d{2}(?!\d))"
)


def coerce_year(cell) -> tuple[int | None, str | None, str]:
    """Return (year, display_str, kind) where kind is one of:
      'date', 'terminal' (Complete/In-Service/etc), 'pending' (TBD/New/etc),
      'empty'."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None, None, "empty"
    if isinstance(cell, (datetime, date, pd.Timestamp)):
        try:
            yr = cell.year
            disp = cell.strftime("%Y-%m") if hasattr(cell, "strftime") else str(yr)
            return yr, disp, "date"
        except Exception:
            return None, str(cell), "date"
    if isinstance(cell, (int,)):
        if 1990 <= cell <= 2050:
            return cell, str(cell), "date"
        return None, str(cell), "empty"
    if isinstance(cell, float):
        if pd.isna(cell):
            return None, None, "empty"
        if cell.is_integer() and 1990 <= int(cell) <= 2050:
            return int(cell), str(int(cell)), "date"
        return None, str(cell), "empty"
    s = str(cell).strip()
    if not s:
        return None, None, "empty"
    low = s.lower()
    if any(t in low for t in _TERMINAL_STATES):
        return None, s.split("\n")[0][:40], "terminal"
    if low in _PENDING_STATES:
        return None, s, "pending"
    # Find every year hint inside the cell, return the LATEST one (so the
    # PG&E split-cell "VEA portion: Dec-21\nGLW portion: 2023" reports the
    # GLW (later) year).
    years = []
    for m in _DATE_STR_RE.finditer(s):
        if m.group("full"):
            years.append(int(m.group("full")))
        else:
            yy = int(m.group("yy2"))
            years.append(2000 + yy if yy < 80 else 1900 + yy)
    if years:
        yr = max(years)
        return yr, s.split("\n")[0][:40], "date"
    return None, s[:40], "empty"


def first_plan_year(cell) -> int | None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    s = str(cell).strip()
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


# --- per-sheet parser -----------------------------------------------------
def parse_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0)
    n_cols = len(df.columns)
    if n_cols < 8:
        return pd.DataFrame()
    proj_col, pto_col = df.columns[0], df.columns[1]
    plan_col, orig_col = df.columns[2], df.columns[3]
    # The last 4 columns are always Status / CPUC / Construction / Notes.
    status_col, notes_col = df.columns[n_cols - 4], df.columns[n_cols - 1]
    tdf_cols = list(df.columns[4:n_cols - 4])

    rows = []
    for _, r in df.iterrows():
        name = r[proj_col]
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        pto = str(r[pto_col]).strip() if isinstance(r[pto_col], str) else None
        plan_year_str = str(r[plan_col]).strip() if r[plan_col] is not None else None
        orig_yr, orig_disp, orig_kind = coerce_year(r[orig_col])

        # Latest = rightmost meaningful TDF cell. Walk right-to-left; first
        # date-typed cell wins. If no date is found, fall back to the last
        # terminal cell (e.g. 'In-Service').
        latest_yr, latest_disp, latest_kind = None, None, "empty"
        for c in reversed(tdf_cols):
            yr, disp, kind = coerce_year(r[c])
            if kind == "date":
                latest_yr, latest_disp, latest_kind = yr, disp, kind
                break
            if latest_disp is None and kind in ("terminal", "pending"):
                latest_yr, latest_disp, latest_kind = yr, disp, kind
        # If still nothing, fall back to original
        if latest_disp is None:
            latest_yr, latest_disp, latest_kind = orig_yr, orig_disp, orig_kind

        status_raw = r[status_col]
        status = (
            str(status_raw).strip() if isinstance(status_raw, str) else None
        )
        notes_raw = r[notes_col]
        notes = (str(notes_raw).strip()[:300]
                  if isinstance(notes_raw, str) else "")

        years_slipped = None
        if orig_yr is not None and latest_yr is not None and latest_kind == "date":
            years_slipped = latest_yr - orig_yr

        sub1, sub2, kv = parse_substations(name)

        rows.append({
            "project_name": name,
            "pto": pto,
            "plan_year_approved": plan_year_str,
            "original_isd": orig_disp,
            "latest_isd": latest_disp,
            "original_year": orig_yr,
            "latest_year": latest_yr,
            "years_slipped": years_slipped,
            "status": status,
            "sub1": sub1,
            "sub2": sub2,
            "voltage_kV": kv,
            "notes": notes,
        })
    return pd.DataFrame(rows)


def parse_xlsx(xlsx_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xl.sheet_names:
        sub = parse_sheet(xlsx_path, sheet)
        if not sub.empty:
            frames.append(sub)
            print(f"  {sheet:<18}  {len(sub):>3} projects")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    if not SOURCE_MANIFEST.exists():
        raise SystemExit(
            f"missing {SOURCE_MANIFEST}. Run download_tpp.py first."
        )
    sources = json.loads(SOURCE_MANIFEST.read_text())
    xlsx_meta = sources.get("xlsx") or {}
    xlsx_path = Path(xlsx_meta.get("path") or "")
    if not xlsx_path.exists():
        raise SystemExit(
            f"XLSX not found at {xlsx_path}. Run download_tpp.py first."
        )

    print(f"parsing {xlsx_path.name} ...")
    df = parse_xlsx(xlsx_path)
    if df.empty:
        raise SystemExit("no rows parsed from XLSX")

    # Stamp every row with the source url + label so the renderer can build
    # a hyperlink with descriptive anchor text. All projects come from the
    # same XLSX here; the Appendix H parse adds extra rows with their own
    # source pair downstream.
    df["source_label"] = xlsx_meta.get("label")
    df["source_url"] = xlsx_meta.get("url")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nparsed {len(df)} projects -> {OUT_CSV}")
    print()
    print("=== by PTO ===")
    print(df["pto"].value_counts().to_string())
    print()
    print(f"projects with both substations parsed: {df['sub2'].notna().sum()}")
    print(f"projects with only one substation:     "
          f"{df[df['sub1'].notna() & df['sub2'].isna()].shape[0]}")
    print(f"projects with no substation parsed:    {df['sub1'].isna().sum()}")
    print(f"projects with measurable slippage:     "
          f"{df['years_slipped'].notna().sum()}")
    if df["years_slipped"].notna().any():
        print(f"  median slip (years): {df['years_slipped'].median()}")


if __name__ == "__main__":
    main()
