"""Build a constraint -> TPP-projects crosswalk with multi-source provenance.

Inputs:
  data/tpp/tpp_projects.csv  (XLSX-derived: 230 projects across 9 PTOs)
  data/tpp/appendix_h.csv    (PDF narrative: per-project description + ISD)
  data/tpp/sources.json      (source URLs + labels for hyperlink attribution)
  data/wecc/wecc_paths_2026.csv  (path -> substation set for the WECC bridge)
  data/constraint_rating_crosswalk_*.csv  (per-month physical-line lookups)

Output:
  data/tpp_crosswalk.csv     (one row per physical_line, ready for the renderer)

Output columns:
  physical_line              — the constraint id (NOM:..., ITC:...)
  n_tpp_projects             — count of matching projects (after dedup)
  earliest_isd_active        — earliest expected in-service year
  latest_isd_active          — latest
  oldest_plan_year           — earliest plan year (proxy for how long it's been pending)
  max_slip_years             — biggest delta between original and current ISD
  projects_summary           — plain-text summary (legacy compat)
  projects_summary_html      — same content with <br>, <b>, <a href> hyperlinks
                               for the hover and details panel
  sources_checked_html       — always-populated list of (label, url) hyperlinks
                               so users can verify even when no match was found

Match rules:
  - Both-substation TPP projects: match if BOTH substations appear in the
    constraint's normalized substation pair (most precise).
  - One-substation TPP projects: match if that substation is one of the
    constraint's two endpoints.
  - WECC named-path bridge: if the constraint maps to a named path
    (e.g. Path 26 / "Northern -Southern California"), also match any TPP
    project whose substations appear in the path's definition.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
TPP_CSV = DATA / "tpp" / "tpp_projects.csv"
APPENDIX_H_CSV = DATA / "tpp" / "appendix_h.csv"
SOURCE_MANIFEST = DATA / "tpp" / "sources.json"
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


def normalize(s) -> str:
    if not isinstance(s, str):
        return ""
    t = re.sub(r"[^A-Za-z]+", "", s).upper()
    return ALIAS.get(t, t)


_VOWEL_RE = re.compile(r"[AEIOU]")


def devow(s: str) -> str:
    return _VOWEL_RE.sub("", s) if isinstance(s, str) else ""


def is_subseq(needle: str, hay: str) -> bool:
    """True if every char of needle appears in hay in the same order."""
    if not needle or not hay or len(needle) < 5:
        return False
    it = iter(hay)
    return all(c in it for c in needle)


def build_fuzzy_alias(tpp_subs: set[str]) -> dict[str, str]:
    """Map vowel-stripped form -> canonical TPP form. Bridges
    abbreviated OASIS spellings like CHCARITA → CHICARITA."""
    out: dict[str, str] = {}
    for s in tpp_subs:
        key = devow(s)
        # Skip ambiguous cases where two TPP names collapse to the same key
        if key in out and out[key] != s:
            out[key] = None  # mark ambiguous
        elif key not in out:
            out[key] = s
    # drop ambiguous keys
    return {k: v for k, v in out.items() if v is not None}


def resolve_constraint_sub(raw: str, tpp_subs: set[str],
                            fuzzy_alias: dict[str, str]) -> str | None:
    """Map a constraint-side substation token to its TPP equivalent.
    Tries: exact (after normalize) -> vowel-strip alias -> subsequence
    match (when needle consumes >=70% of haystack)."""
    if not raw:
        return None
    n = normalize(raw)
    if n in tpp_subs:
        return n
    dn = devow(n)
    if dn in fuzzy_alias:
        return fuzzy_alias[dn]
    # Subsequence fallback: find the TPP substation whose chars contain the
    # constraint's chars in order, requiring high coverage to avoid false
    # positives.
    if len(n) >= 5:
        best = None
        best_ratio = 0.0
        for ts in tpp_subs:
            if len(ts) < len(n):
                continue
            ratio = len(n) / len(ts)
            if ratio < 0.70:
                continue
            if is_subseq(n, ts) and ratio > best_ratio:
                best, best_ratio = ts, ratio
        if best:
            return best
    return None


def constraint_subs(physical_line: str) -> tuple[str | None, str | None]:
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


def link(label: str, url: str | None) -> str:
    """Render a hyperlink with descriptive anchor text. Always escape label."""
    safe = html.escape(label)
    if not url:
        return safe
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{safe}</a>'


# --- fuzzy join from XLSX -> Appendix H (so descriptions appear in hover) ----
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _norm_title(s: str) -> str:
    return _PUNCT_RE.sub(" ", str(s).lower()).strip()


def _fuzzy_match(xlsx_name: str, appendix_df: pd.DataFrame) -> pd.Series | None:
    if appendix_df.empty:
        return None
    target = _norm_title(xlsx_name)
    if not target:
        return None
    # Exact-on-normalized first
    keys = appendix_df["_norm"]
    exact = appendix_df[keys == target]
    if not exact.empty:
        return exact.iloc[0]
    # Substring containment in either direction
    contains = appendix_df[
        keys.apply(lambda k: target in k or k in target)
    ]
    if not contains.empty:
        return contains.iloc[0]
    return None


def build_path_subs() -> dict[str, set[str]]:
    """WECC named-path -> set of substations bridge."""
    path_subs: dict[str, set[str]] = {}
    if not WECC_CSV.exists():
        return path_subs
    wecc = pd.read_csv(WECC_CSV)
    for _, w in wecc.iterrows():
        name = w.get("path_name")
        defs = w.get("definitions")
        if not isinstance(name, str) or not isinstance(defs, str):
            continue
        tokens = re.split(r"[\s\-–/(),]+", defs)
        subs = set()
        for tok in tokens:
            t = normalize(tok)
            if t and len(t) >= 3 and not t.startswith("KV") and t != "DEFINITION":
                subs.add(t)
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
    return path_subs


def main():
    if not TPP_CSV.exists():
        raise SystemExit(f"missing {TPP_CSV}. Run tpp_parse.py first.")
    if not SOURCE_MANIFEST.exists():
        raise SystemExit(f"missing {SOURCE_MANIFEST}. Run download_tpp.py first.")
    sources = json.loads(SOURCE_MANIFEST.read_text())
    xlsx_meta = sources.get("xlsx") or {}
    appx_meta = sources.get("appendix_h") or {}
    pdf_meta = sources.get("pdf_sce") or {}

    tpp = pd.read_csv(TPP_CSV)
    print(f"loaded {len(tpp)} XLSX projects")

    appendix_df = pd.DataFrame()
    if APPENDIX_H_CSV.exists():
        appendix_df = pd.read_csv(APPENDIX_H_CSV)
        appendix_df["_norm"] = appendix_df["name"].astype(str).map(_norm_title)
        print(f"loaded {len(appendix_df)} Appendix H narrative entries")

    tpp["sub1_n"] = tpp["sub1"].map(normalize)
    tpp["sub2_n"] = tpp["sub2"].map(normalize)
    # Build a single set of TPP substations + a fuzzy lookup so we can
    # resolve abbreviated OASIS spellings (CHCARITA → CHICARITA) before
    # comparing.
    tpp_sub_set: set[str] = set()
    for s in pd.concat([tpp["sub1_n"], tpp["sub2_n"]]).dropna().tolist():
        if s:
            tpp_sub_set.add(s)
    fuzzy_alias = build_fuzzy_alias(tpp_sub_set)
    print(f"  {len(tpp_sub_set)} distinct TPP substations  ({len(fuzzy_alias)} fuzzy aliases)")

    # Pre-join Appendix H narrative onto each XLSX project (one-to-zero-or-one).
    appx_desc, appx_url, appx_label = [], [], []
    for _, p in tpp.iterrows():
        m = _fuzzy_match(p["project_name"], appendix_df) if not appendix_df.empty else None
        if m is not None and isinstance(m.get("brief_description"), str):
            appx_desc.append(m["brief_description"])
            appx_url.append(m.get("source_url") or appx_meta.get("url"))
            appx_label.append(m.get("source_label") or appx_meta.get("label"))
        else:
            appx_desc.append(None); appx_url.append(None); appx_label.append(None)
    tpp["appx_description"] = appx_desc
    tpp["appx_url"] = appx_url
    tpp["appx_label"] = appx_label
    print(f"  {sum(1 for d in appx_desc if d)} XLSX projects got Appendix H descriptions")

    path_subs = build_path_subs()
    print(f"loaded {len(path_subs)} WECC named-path substation sets")

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

    # Pre-build the "sources checked" hyperlink block — same on every row.
    src_lines = []
    if xlsx_meta.get("url"):
        src_lines.append(link(xlsx_meta["label"], xlsx_meta["url"]))
    if appx_meta.get("url"):
        src_lines.append(link(appx_meta["label"], appx_meta["url"]))
    if pdf_meta.get("url"):
        src_lines.append(link(pdf_meta["label"], pdf_meta["url"]))
    sources_checked_html = "<br>".join(
        f"&nbsp;&nbsp;• {l}" for l in src_lines
    )

    rows = []
    for line in sorted(phys_lines):
        s1_raw, s2_raw = constraint_subs(line)
        # Resolve OASIS abbreviations to TPP canonical spellings before
        # comparing — this is what unlocks SDG&E / PG&E coverage on
        # constraints like CHCARITA, KINGSBRG, LOSBANS, PENSQTOS, etc.
        s1 = resolve_constraint_sub(s1_raw, tpp_sub_set, fuzzy_alias) or s1_raw
        s2 = resolve_constraint_sub(s2_raw, tpp_sub_set, fuzzy_alias) or s2_raw
        wecc_path = phys_to_path.get(line)
        path_substations = path_subs.get(wecc_path, set()) if wecc_path else set()
        matches = []
        # Pass 1: both-substation precise matches
        if s1 and s2:
            both = tpp[tpp["sub2_n"].notna()]
            cset = {s1, s2}
            for _, p in both.iterrows():
                tset = {p["sub1_n"], p["sub2_n"]}
                if tset == cset:
                    matches.append(p)
        # Pass 2: single-substation matches
        if s1:
            singles = tpp[tpp["sub2_n"].isna() & tpp["sub1_n"].notna()]
            for _, p in singles.iterrows():
                if p["sub1_n"] == s1 or (s2 and p["sub1_n"] == s2):
                    matches.append(p)
        # Pass 3: WECC named-path bridge
        if path_substations:
            for _, p in tpp.iterrows():
                ts1, ts2 = p["sub1_n"], p["sub2_n"]
                if ts1 in path_substations or (
                    isinstance(ts2, str) and ts2 in path_substations
                ):
                    matches.append(p)

        if not matches:
            rows.append(dict(
                physical_line=line, n_tpp_projects=0,
                earliest_isd_active=None, latest_isd_active=None,
                oldest_plan_year=None, max_slip_years=None,
                projects_summary="",
                projects_summary_html="",
                sources_checked_html=sources_checked_html,
            ))
            continue

        m_df = pd.DataFrame(matches).drop_duplicates(subset=["project_name"])
        # Drop already-complete projects from the headline summary (they
        # don't change the forward-looking story)
        active = m_df[m_df["latest_isd"].astype(str) != "Complete"]
        if active.empty:
            active = m_df

        earliest = active["latest_year"].dropna().min()
        latest_yr = active["latest_year"].dropna().max()
        oldest_plan = active["original_year"].dropna().min()
        max_slip = active["years_slipped"].dropna().max()

        def fmt(p, as_html=False):
            name = p["project_name"]
            pto = p.get("pto") or "?"
            ply = p.get("plan_year_approved") or "?"
            oy = int(p["original_year"]) if pd.notna(p["original_year"]) else None
            ly = int(p["latest_year"]) if pd.notna(p["latest_year"]) else None
            sl = int(p["years_slipped"]) if pd.notna(p["years_slipped"]) else None
            st_raw = p.get("status")
            st = "" if (st_raw is None or (isinstance(st_raw, float) and pd.isna(st_raw))) else str(st_raw)
            parts = [f"{pto}", f"plan {ply}"]
            if oy and ly:
                parts.append(f"orig ISD '{oy % 100:02d} → now '{ly % 100:02d}")
            elif ly:
                parts.append(f"ISD '{ly % 100:02d}")
            if sl is not None and sl != 0:
                parts.append(f"slipped {sl} yr")
            if st:
                parts.append(st)
            meta = ", ".join(parts)
            if not as_html:
                return f'"{name}" ({meta})'
            # HTML form: project name in bold, then meta, then source links
            xlsx_label = p.get("source_label") or xlsx_meta.get("label") or "CAISO TPP XLSX"
            xlsx_url = p.get("source_url") or xlsx_meta.get("url")
            src_link = link(xlsx_label, xlsx_url)
            block = (
                f"<b>{html.escape(str(name))}</b> "
                f"<span style='color:#888'>({html.escape(meta)})</span>"
                f"<br>&nbsp;&nbsp;source: {src_link}"
            )
            appx_d = p.get("appx_description")
            if isinstance(appx_d, str) and appx_d.strip():
                appx_l = p.get("appx_label") or appx_meta.get("label") or "CAISO Appendix H"
                appx_u = p.get("appx_url") or appx_meta.get("url")
                desc_clean = html.escape(appx_d.replace("\n", " ").strip()[:240])
                block += (
                    f"<br>&nbsp;&nbsp;📋 <i>{desc_clean}</i>"
                    f"<br>&nbsp;&nbsp;source: {link(appx_l, appx_u)}"
                )
            return block

        active = active.sort_values("latest_year", na_position="last")
        summary = "; ".join(fmt(p, as_html=False) for _, p in active.iterrows())
        summary_html = "<br><br>".join(fmt(p, as_html=True) for _, p in active.iterrows())

        rows.append(dict(
            physical_line=line,
            n_tpp_projects=len(active),
            earliest_isd_active=int(earliest) if pd.notna(earliest) else None,
            latest_isd_active=int(latest_yr) if pd.notna(latest_yr) else None,
            oldest_plan_year=int(oldest_plan) if pd.notna(oldest_plan) else None,
            max_slip_years=int(max_slip) if pd.notna(max_slip) else None,
            projects_summary=summary,
            projects_summary_html=summary_html,
            sources_checked_html=sources_checked_html,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    matched = (out["n_tpp_projects"] > 0).sum()
    print(f"saved {OUT_CSV}  ({len(out)} physical lines, "
          f"{matched} with at least 1 TPP project — "
          f"{matched/len(out)*100:.1f}%)")

    print()
    print("=== sample matches for known constraints ===")
    for needle in ["MIDWAY", "PANOCHE", "LUGO", "MIRA", "JULIAN",
                    "DEVERS", "ALAMITOS", "LOSBANOS"]:
        hits = out[out["physical_line"].str.contains(needle, case=False, na=False)
                    & (out["n_tpp_projects"] > 0)].head(2)
        for _, r in hits.iterrows():
            print(f"\n  {r['physical_line'][:65]}")
            print(f"    -> {r['n_tpp_projects']} project(s), earliest ISD {r['earliest_isd_active']}")
            print(f"    -> {r['projects_summary'][:200]}")


if __name__ == "__main__":
    main()
