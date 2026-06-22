"""Build a constraint -> TPP-projects crosswalk.

Output: data/tpp_crosswalk.csv (one row per unique physical_line found in
any month's constraint_rating_crosswalk, with all matching active TPP
projects pre-summarized for the renderer's hover).

The crosswalk is time-invariant — TPP projects don't change month-to-month
and neither do constraint physical-line IDs — so a single file is reused
by every per-month render.

Match rule (mirrors d3_rating_crosswalk's pair-must-both-appear approach):
  - Both-substation TPP projects (sub1 AND sub2 present): match if BOTH
    substations appear in the constraint's normalized substation pair.
  - One-substation TPP projects (sub1 only): match if that substation is
    one of the constraint's two endpoints. Weaker signal but still useful
    (e.g., "Mira Loma 500 kV CB Upgrade" is relevant to any constraint
    that touches Mira Loma).
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
TPP_CSV = DATA / "tpp" / "tpp_projects.csv"
WECC_CSV = DATA / "wecc" / "wecc_paths_2026.csv"
OUT_CSV = DATA / "tpp_crosswalk.csv"


# Aliases from d3 (kept in sync; OASIS spellings <-> WECC/TPP spellings)
ALIAS = {
    "WIRLWIND": "WHIRLWIND",
    "VINCNT":   "VINCENT",
    "MOSSLD":   "MOSSLANDING",
    "LASAGUIL": "LASAGUILA",
    "LOSBNS":   "LOSBANOS",
    "MIDWY":    "MIDWAY",
    "PALOVRD":  "PALOVERDE",
    "ELDORADO": "ELDORADO",
    "MIRALOMA": "MIRALOMA",
    "JHINDS":   "JULIANHINDS",
    "JHINDS2":  "JULIANHINDS",
}

SUBNAME_RE = re.compile(r"_?(\d+)_([A-Z][A-Z0-9.\- ]{1,15}?)\s*_(\d{2,3}(?:\.0)?)")


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = re.sub(r"[^A-Za-z]+", "", s).upper()
    return ALIAS.get(t, t)


def constraint_subs(physical_line: str) -> tuple[str | None, str | None]:
    """Extract (sub1, sub2) from a NOMOGRAM_ID like
    'NOM:30060_MIDWAY  _500_24156_VINCENT _500_BR_2 _3'."""
    if not isinstance(physical_line, str):
        return None, None
    body = physical_line[4:] if physical_line.startswith(("NOM:", "ITC:")) else physical_line
    matches = SUBNAME_RE.findall(body)
    subs = []
    for _, name, _kv in matches:
        n = normalize(name)
        if n and n not in subs:
            subs.append(n)
    if len(subs) >= 2:
        return subs[0], subs[1]
    if len(subs) == 1:
        return subs[0], None
    return None, None


def main():
    if not TPP_CSV.exists():
        raise SystemExit(f"missing {TPP_CSV}. Run tpp_parse.py first.")
    tpp = pd.read_csv(TPP_CSV)
    print(f"loaded {len(tpp)} TPP projects")

    tpp["sub1_n"] = tpp["sub1"].map(normalize)
    tpp["sub2_n"] = tpp["sub2"].map(normalize)

    # Build WECC named-path -> set of substations bridge. Lets a constraint
    # that already matched a WECC path (e.g. MIDWAY-VINCENT -> Path 26 =
    # "Northern -Southern California") pick up TPP projects on any other
    # substation in that path's definition (e.g. Antelope-Whirlwind,
    # since Whirlwind is part of Path 26).
    path_subs: dict[str, set[str]] = {}
    if WECC_CSV.exists():
        wecc = pd.read_csv(WECC_CSV)
        for _, w in wecc.iterrows():
            name = w.get("path_name")
            defs = w.get("definitions")
            if not isinstance(name, str) or not isinstance(defs, str):
                continue
            # Split on whitespace, normalize each token, keep ones that look
            # like substation names (>=3 alpha chars). Pairs like "Midway-Vincent"
            # decompose into Midway and Vincent.
            tokens = re.split(r"[\s\-–/(),]+", defs)
            subs = set()
            for tok in tokens:
                t = normalize(tok)
                if t and len(t) >= 3 and not t.startswith("KV") and t != "DEFINITION":
                    subs.add(t)
            # Trim obvious noise words
            noise = {"DEFINITION", "OTHER", "RATING", "LINE", "LINES",
                      "TRANSFORMER", "BANK", "BANKS", "BUS", "AND",
                      "OPEN", "CLOSED", "NORMALLY", "NO", "NOT", "DEFINED",
                      "TBD", "FROM", "TO", "BPA", "BCH", "AVA", "PAC", "USBR",
                      "WINTER", "SPRING", "SUMMER", "FALL", "SEASONAL", "DEDC",
                      "RATING", "EXISTING", "ACCEPTED", "WITH", "WITHIN",
                      "INTO", "INTERTIE", "CIRCUIT", "BREAKER", "CB",
                      "CBS", "UPGRADE", "REINFORCEMENT", "RECONDUCTOR",
                      "INSTALL", "MITIGATION", "RECONFIGURATION", "SWITCHRACK",
                      "REPLACEMENT", "PROJECT", "PROJECTS", "METHOD",
                      "OF", "SERVICE", "FOR", "ADD", "NEW", "OUT",
                      "OPER", "OPERATING", "EQUIPMENT", "AT", "MOSSLD",
                      "AC", "DC", "TIE", "CAP", "REACTOR", "SHUNT",
                      "SOLAR", "WIND", "POWER", "PLANT", "STATION",
                      "GIS", "REBUILD", "BUILD", "LOOP", "SEGMENT",
                      "EXEMPT", "PENDING", "BY", "THE", "ARE", "OR",
                      "TAP", "JCT", "JT", "SUB", "VLY", "VLLY", "BR",
                      "XF", "MW", "MVA"}
            subs -= noise
            path_subs[name] = subs

    # Gather every unique physical_line + (optionally) its WECC path mapping.
    # For the WECC-path bridge we need both physical_line AND matched_path
    # from the per-month constraint_rating_crosswalk files.
    phys_to_path: dict[str, str | None] = {}
    cw_files = sorted(DATA.glob("constraint_rating_crosswalk_*.csv"))
    for f in cw_files:
        df = pd.read_csv(f, usecols=["physical_line", "matched_path"])
        for _, r in df.iterrows():
            pl = r["physical_line"]
            mp = r["matched_path"]
            if not isinstance(pl, str):
                continue
            if pl not in phys_to_path or (
                phys_to_path.get(pl) is None and isinstance(mp, str)
            ):
                phys_to_path[pl] = mp if isinstance(mp, str) else None
    phys_lines = set(phys_to_path)
    print(f"discovered {len(phys_lines)} distinct physical lines across "
          f"{len(cw_files)} per-month crosswalks")
    print(f"loaded {len(path_subs)} WECC named-path substation sets")

    rows = []
    for line in sorted(phys_lines):
        s1, s2 = constraint_subs(line)
        wecc_path = phys_to_path.get(line)
        path_substations = path_subs.get(wecc_path, set()) if wecc_path else set()
        matches = []
        # Pass 1: TPP projects with both substations equalling the constraint's
        # pair (most precise match).
        if s1 and s2:
            both = tpp[tpp["sub2_n"].notna()]
            cset = {s1, s2}
            for _, p in both.iterrows():
                tset = {p["sub1_n"], p["sub2_n"]}
                if tset == cset:
                    matches.append(p)
        # Pass 2: single-substation TPP projects that touch one of the
        # constraint's endpoints.
        if s1:
            singles = tpp[tpp["sub2_n"].isna() & tpp["sub1_n"].notna()]
            for _, p in singles.iterrows():
                if p["sub1_n"] == s1 or (s2 and p["sub1_n"] == s2):
                    matches.append(p)
        # Pass 3: WECC named-path bridge. If this constraint is part of a
        # named path (e.g., Path 26 / "Northern -Southern California"),
        # also match any TPP project whose substations appear in the
        # path's definition. This catches things like Antelope-Whirlwind
        # being relief for any Midway-Vincent-area constraint.
        if path_substations:
            for _, p in tpp.iterrows():
                ts1, ts2 = p["sub1_n"], p["sub2_n"]
                if ts1 in path_substations or (isinstance(ts2, str) and ts2 in path_substations):
                    matches.append(p)

        if not matches:
            rows.append(dict(
                physical_line=line, n_tpp_projects=0,
                earliest_isd_active=None, latest_isd_active=None,
                oldest_plan_year=None, max_slip_years=None,
                projects_summary=""))
            continue

        m_df = pd.DataFrame(matches).drop_duplicates(subset=["project_name"])
        # Drop already-complete projects (their relief is already in our data)
        active = m_df[m_df["latest_isd"].astype(str) != "Complete"]
        if active.empty:
            active = m_df  # keep complete as a fallback so user sees something

        earliest = active["latest_year"].dropna().min()
        latest_yr = active["latest_year"].dropna().max()
        oldest_plan = active["original_year"].dropna().min()
        max_slip = active["years_slipped"].dropna().max()

        # Pre-format the per-project summary the hover will show.
        def fmt(p):
            name = p["project_name"]
            ply = p.get("plan_year_approved") or "?"
            oy = int(p["original_year"]) if pd.notna(p["original_year"]) else None
            ly = int(p["latest_year"]) if pd.notna(p["latest_year"]) else None
            sl = int(p["years_slipped"]) if pd.notna(p["years_slipped"]) else None
            st_raw = p.get("status")
            st = "" if (st_raw is None or (isinstance(st_raw, float) and pd.isna(st_raw))) else str(st_raw)
            parts = [f"plan {ply}"]
            if oy and ly:
                parts.append(f"orig ISD '{oy % 100:02d} → now '{ly % 100:02d}")
            elif ly:
                parts.append(f"ISD '{ly % 100:02d}")
            if sl is not None and sl != 0:
                parts.append(f"slipped {sl} yr")
            if st:
                parts.append(st)
            return f'"{name}" ({", ".join(parts)})'

        active = active.sort_values("latest_year", na_position="last")
        summary = "; ".join(fmt(p) for _, p in active.iterrows())

        rows.append(dict(
            physical_line=line,
            n_tpp_projects=len(active),
            earliest_isd_active=int(earliest) if pd.notna(earliest) else None,
            latest_isd_active=int(latest_yr) if pd.notna(latest_yr) else None,
            oldest_plan_year=int(oldest_plan) if pd.notna(oldest_plan) else None,
            max_slip_years=int(max_slip) if pd.notna(max_slip) else None,
            projects_summary=summary,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV}  ({len(out)} physical lines, "
          f"{(out['n_tpp_projects'] > 0).sum()} with at least 1 TPP project)")

    # Quick sanity check
    print()
    print("=== sample matches for known LA Basin constraints ===")
    for needle in ["MIDWAY", "LUGO", "MIRA", "JULIAN", "DEVERS", "ALAMITOS"]:
        hits = out[out["physical_line"].str.contains(needle, case=False, na=False)
                    & (out["n_tpp_projects"] > 0)].head(2)
        if not hits.empty:
            for _, r in hits.iterrows():
                print(f"\n  {r['physical_line'][:65]}")
                print(f"    -> {r['n_tpp_projects']} project(s), earliest ISD {r['earliest_isd_active']}")
                print(f"    -> {r['projects_summary'][:200]}")


if __name__ == "__main__":
    main()
