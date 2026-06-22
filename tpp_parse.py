"""Parse the CAISO TPP 'Approved Projects' PDF -> structured CSV.

Layout per project (single-page PDF, ~40 rows):
  <project name> <PTO> <plan_year> <ISD_at_approval> <2020-21 plan ISD>
  <Oct 2023 TDF> <Jan 2024 TDF> <July 2024 TDF> <Jan 2025 TDF>
  <July 2025 TDF> <status> <CPUC permit filing> <construction start> <notes>

PTO is currently always "SCE" in this attachment (LA-Basin focus). The PTO
marker is the cleanest split point between the variable-length project
name (sometimes wrapping multiple lines) and the structured columns.

The parser tolerates "Pending", "N/A", "New", "TBD", "Complete" in date
slots; project names wrapping multiple lines (joined on '\\n' from pypdf).
"""
import re
from pathlib import Path

import pandas as pd
import pypdf

TPP_PDF = Path("data/tpp/approved_projects_oct_2025.pdf")
OUT_CSV = Path("data/tpp/tpp_projects.csv")

# month-abbrev-year, e.g. 'Dec-27', 'Jun-29' (CAISO's date convention)
MONTH_ABBR = "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DATE_TOK = rf"(?:{MONTH_ABBR}-\d{{2}}|N/A|TBD|Pending|New|Complete|\d{{4}})"
# plan-year like '2009', '2012-2013', '2024-2025'
PLAN_TOK = r"(?:\d{4}(?:-\d{4})?)"

# split each project row on the PTO marker
PTO_RE = re.compile(r"\s+SCE\s+")
# Pull plan-year + first 2 date columns (ISD_at_approval + 2020-21 plan ISD)
HEAD_RE = re.compile(rf"^({PLAN_TOK})\s+(\S+)\s+(\S+)\s+(.*)$", re.S)

# Year extractor — pull a 4-digit year out of a date token like 'Dec-27' -> 2027
def year_of(tok: str) -> int | None:
    if not tok or tok in {"N/A", "TBD", "Pending", "New", "Complete"}:
        return None
    if re.fullmatch(r"\d{4}", tok):
        return int(tok)
    m = re.fullmatch(rf"{MONTH_ABBR}-(\d{{2}})", tok)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 80 else 1900 + yy
    return None


def first_plan_year(plan_tok: str) -> int | None:
    if not plan_tok: return None
    return int(plan_tok.split("-")[0])


# Substation-pair extractor for project names. CAISO uses both ASCII '-' and
# en-dash '–'. Captures "Lugo-Victorville", "Julian Hinds-Mirage",
# "Devers-Red Bluff", etc. Returns (sub1, sub2) or (sub1, None) for
# single-substation projects ("Inyo 230 kV Shunt Reactor" -> ("Inyo", None)).
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
    kv_m = KV_RE.search(name)
    kv = int(kv_m.group(1)) if kv_m else None
    cleaned = re.sub(r"^(?:Reconductor|Rebuild/build|Loop|Method of Service for|Add\s+\S+\s+\S+|New)\s+", "", name)
    pair = SUB_PAIR_RE.search(cleaned)
    if pair:
        s1 = _strip_trailing_noise(pair.group(1).strip())
        s2 = _strip_trailing_noise(pair.group(2).strip())
        return s1 or None, s2 or None, kv
    lead = LEADING_SUB_RE.search(cleaned)
    if lead:
        return _strip_trailing_noise(lead.group(1).strip()) or None, None, kv
    return None, None, kv


def parse_pdf(pdf_path: Path) -> pd.DataFrame:
    text = pypdf.PdfReader(str(pdf_path)).pages[0].extract_text() or ""
    # Strip the multi-line column header at the top (everything before the
    # first project row). The first project starts at "Alberhill ... SCE 2009".
    header_end = re.search(r"^.*?(?=^\S[^\n]*\sSCE\s)", text, re.S | re.M)
    body = text[header_end.end():] if header_end else text

    # Each project row begins at a non-whitespace char and contains " SCE ".
    # Greedy split: collect each chunk between successive " SCE " markers,
    # carry the trailing fragment forward as the next row's leading name.
    # Implementation: find every " SCE " offset, then for each, walk back to
    # the previous row boundary (a date or newline-after-notes), forward to
    # the next " SCE " (or end).
    pto_positions = [m.start() for m in PTO_RE.finditer(body)]
    rows = []
    for i, start in enumerate(pto_positions):
        # Project name = everything between the previous PTO's end-of-row and
        # this PTO. For the first row, name starts at body[0].
        if i == 0:
            name_start = 0
        else:
            # Previous row's end is somewhere in the previous segment; rough
            # rule — name starts after a newline near the start of this chunk.
            # We'll grab the slice between the previous PTO match and this one,
            # then take the LAST line-or-two as the name.
            prev_end = pto_positions[i - 1]
            # advance past the SCE token
            prev_post = body.find("SCE", prev_end) + 3
            interstitial = body[prev_post:start]
            # The interstitial is: " <plan_year> <ISD> ... <notes>\n<next project name>"
            # Cut at the LAST occurrence of a newline that's followed by text
            # which looks like a project-name start (capital letter, no digits).
            # Heuristic: the next-project-name is the part after the LAST
            # newline OR the last "  " (double-space) that introduces capital.
            split_at = max(interstitial.rfind("\n"), 0)
            name_text = interstitial[split_at:].strip()
            # If the chosen name fragment is empty/too short, fall back to the
            # last 80 chars before this PTO.
            if len(name_text) < 5:
                name_text = body[max(0, start - 80):start].strip().splitlines()[-1]
            name_start = None  # not used past here

        if i == 0:
            name = body[name_start:start].strip().replace("\n", " ")
        else:
            name = name_text.replace("\n", " ")

        # Columns after " SCE ": plan_year ISD_approval ISD_2020-21 Oct23 Jan24 Jul24 Jan25 Jul25 status ...
        post_start = start + len(" SCE ")  # account for spaces
        # Use the precise PTO_RE match length:
        m_pto = PTO_RE.match(body, start)
        post_start = m_pto.end() if m_pto else post_start
        next_start = pto_positions[i + 1] if i + 1 < len(pto_positions) else len(body)
        post = body[post_start:next_start]

        # Tokenize the post block. Status fields like "Final Engineering" or
        # "Preliminary Engineering" are multi-word — handle by splitting on
        # whitespace and rejoining the status segment heuristically. For our
        # purposes the FIRST eight whitespace-tokens are the dated columns,
        # then status starts.
        toks = post.split()
        # First token is plan_year
        if not toks:
            continue
        plan_year = toks[0]
        if not re.fullmatch(PLAN_TOK, plan_year):
            continue  # not a real row
        # Next ~8 tokens are date columns (some are 'Pending', 'New', 'N/A',
        # or wrapped multi-line — keep them as-is).
        # For our needs we want: original_isd (1st col after plan_year) and
        # latest_isd (last non-N/A among the next 7 cols).
        date_slots = toks[1:9] if len(toks) > 8 else toks[1:]
        original_isd = date_slots[0] if len(date_slots) >= 1 else None
        # Latest expected ISD: last meaningful date token in date_slots
        latest_isd = None
        for t in reversed(date_slots):
            if t in {"N/A", "TBD", "Pending", "New", "Complete"}:
                if latest_isd is None and t == "Complete":
                    latest_isd = "Complete"
                continue
            if year_of(t) is not None:
                latest_isd = t
                break

        # Status starts at token index 9 — typically 1 or 2 words. Known
        # statuses: Engineering, Construction, Design, Licensing, Initiation,
        # "Final Engineering", "Preliminary Engineering", "Engineering Design".
        status_idx = 9
        status_words = []
        KNOWN_STATUS_WORDS = {"Engineering", "Construction", "Design",
                                "Licensing", "Initiation", "Final",
                                "Preliminary"}
        while status_idx < len(toks):
            t = toks[status_idx]
            if t in KNOWN_STATUS_WORDS:
                status_words.append(t)
                status_idx += 1
                if len(status_words) >= 3:
                    break
            else:
                break
        status = " ".join(status_words) or None

        # Everything past status = CPUC + construction start + notes; we keep
        # just the notes as a single string for the hover.
        rest = " ".join(toks[status_idx:]).strip()
        notes = rest

        # Year-of conversion for hover
        orig_year = year_of(original_isd)
        latest_year = year_of(latest_isd) if latest_isd != "Complete" else None
        years_slipped = (latest_year - orig_year) if (orig_year and latest_year) else None

        sub1, sub2, kv = parse_substations(name)

        rows.append({
            "project_name": name,
            "pto": "SCE",
            "plan_year_approved": plan_year,
            "original_isd": original_isd,
            "latest_isd": latest_isd,
            "original_year": orig_year,
            "latest_year": latest_year,
            "years_slipped": years_slipped,
            "status": status,
            "sub1": sub1,
            "sub2": sub2,
            "voltage_kV": kv,
            "notes": notes[:300],  # cap to keep CSV tidy
        })

    df = pd.DataFrame(rows)
    return df


def supplement_with_other_pto():
    """The 'attachment-1-approved-projects' PDF is SCE-only. Other PTOs
    publish their projects in different formats (Appendix I narrative for
    competitive solicitations, scattered chapters of the main Transmission
    Plan for the rest). Until we have a machine-readable cross-PTO list,
    hand-add the high-profile PG&E projects we know about from Appendix I."""
    extras = [
        dict(
            project_name="Manning–Metcalf 500 kV Line",
            pto="PG&E (competitive)",
            plan_year_approved="2024-2025",
            original_isd="2032",
            latest_isd="2032",
            original_year=2032, latest_year=2032, years_slipped=0,
            status="Approved",
            sub1="Manning", sub2="Metcalf", voltage_kV=500,
            notes="Reliability-driven; ~100-mile 500 kV AC line, eligible for "
                  "competitive solicitation, per 2024-2025 ISO Transmission Plan Appendix I.",
        ),
        dict(
            project_name="NRS – San Jose B 230 kV Line",
            pto="SVP/PG&E (competitive)",
            plan_year_approved="2024-2025",
            original_isd="2030",
            latest_isd="2030",
            original_year=2030, latest_year=2030, years_slipped=0,
            status="Approved",
            sub1="NRS", sub2="San Jose B", voltage_kV=230,
            notes="Reliability-driven for high load forecast in San Jose area; "
                  "7-10 mile 230 kV line, est. $150-200M. Per Appendix I.",
        ),
    ]
    return pd.DataFrame(extras)


def main():
    if not TPP_PDF.exists():
        raise SystemExit(f"TPP PDF missing: {TPP_PDF}. Run download_tpp.py first.")
    df = parse_pdf(TPP_PDF)
    extras = supplement_with_other_pto()
    df = pd.concat([df, extras], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"parsed {len(df)} projects (incl. {len(extras)} hand-added non-SCE) -> {OUT_CSV}")
    print()
    print("=== sample (first 6) ===")
    print(df.head(6).to_string())
    print()
    print(f"projects with both substations parsed: {df['sub2'].notna().sum()}")
    print(f"projects with only one substation:     {df[df['sub1'].notna() & df['sub2'].isna()].shape[0]}")
    print(f"projects with no substation parsed:    {df['sub1'].isna().sum()}")
    print(f"projects with slippage:                {df['years_slipped'].notna().sum()}")
    print(f"  median slip (years): {df['years_slipped'].median()}")


if __name__ == "__main__":
    main()
