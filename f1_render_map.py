"""F1 — Plotly four-channel map render.

Channels (brief §2):
  hue          — archetype: WHITE / BLUE / RED / PURPLE
  intensity    — round-trip MCC spread (per-archetype quintile bucket)
  marker size  — sqrt(size_node) — proportional to constraint rent
  marker style — filled (persistent) vs hollow (spike-driven) via concentration

Per user request:
  - quintile-by-archetype color (not linear) so the long tail doesn't compress
  - top 1% spike-spread nodes listed in a side CSV so they don't get lost

Output: caiso_congestion_map_summer.html (standalone, MapLibre, no token).
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config


def ceil_int(x):
    """Round up to nearest whole dollar; NaN passes through unchanged."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    return int(math.ceil(x))

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"
# CLI: python3 f1_render_map.py [SEASON_TAG]
if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
    SEASON_TAG = sys.argv[1]

# Visual config
MIN_PIXEL = 4
MAX_PIXEL = 22
N_QUINTILES = 5
SPIKE_PERCENTILE = 0.99
BAR_TOP_N = 100
DURATIONS = [2, 4, 8]      # battery durations (hours) for the selector
DEFAULT_DURATION = 4       # initial view on page load

ARCHETYPE_CMAP = {
    "WHITE":  ("Greys",   0.15, 0.55),  # cmap, alpha-min, alpha-max
    "BLUE":   ("Blues",   0.30, 1.00),
    "RED":    ("Reds",    0.30, 1.00),
    "PURPLE": ("Purples", 0.30, 1.00),
}

# Human-readable descriptors shown in hovers / legend labels — preferred
# over the raw color names since the channel meanings are what matters.
ARCHETYPE_LABEL = {
    "WHITE":  "no local congestion",
    "BLUE":   "import pocket",
    "RED":    "export pocket",
    "PURPLE": "bidirectional",
}


MONTH_LABELS = {
    "2025-full": "Full year",
    "2025-01": "Jan 25", "2025-02": "Feb 25", "2025-03": "Mar 25",
    "2025-04": "Apr 25", "2025-05": "May 25", "2025-06": "Jun 25",
    "2025-07": "Jul 25", "2025-08": "Aug 25", "2025-09": "Sep 25",
    "2025-10": "Oct 25", "2025-11": "Nov 25", "2025-12": "Dec 25",
    "summer2025": "Aug 25",  # legacy alias for the original August pull
}
MONTH_LONG_LABELS = {
    "2025-full": "Full year 2025 (aggregated across all months)",
    "2025-01": "January 2025", "2025-02": "February 2025",
    "2025-03": "March 2025", "2025-04": "April 2025",
    "2025-05": "May 2025", "2025-06": "June 2025",
    "2025-07": "July 2025", "2025-08": "August 2025",
    "2025-09": "September 2025", "2025-10": "October 2025",
    "2025-11": "November 2025", "2025-12": "December 2025",
    "summer2025": "August 2025",
}
# The Full Year view is always the first nav entry and the default landing
# page (mirrored to index.html).
FULL_YEAR_TAG = "2025-full"


def discover_months() -> list[str]:
    """Find all completed monthly metric tags (have node_metrics_with_size_*.csv
    AND duration_sweep_*.csv AND node_coordinates_*.csv ready)."""
    import re
    pat = re.compile(r"node_metrics_with_size_(.+)\.csv$")
    tags = []
    for p in DATA_DIR.iterdir():
        m = pat.match(p.name)
        if not m:
            continue
        tag = m.group(1)
        # Require companion files to ensure the renderer can produce a full page
        if not (DATA_DIR / f"duration_sweep_{tag}.csv").exists():
            continue
        if not (DATA_DIR / f"node_coordinates_{tag}.csv").exists():
            continue
        tags.append(tag)
    # FULL_YEAR_TAG is always first; the rest sort by canonical YYYY-MM.
    def sort_key(t):
        if t == FULL_YEAR_TAG:
            return ""  # sorts before any "2025-MM"
        if t == "summer2025":
            return "2025-08"  # legacy alias
        return t
    return sorted(set(tags), key=sort_key)


def page_filename(tag: str) -> str:
    return f"caiso_congestion_map_{tag}.html"


def global_quintile_bounds(tags: list[str]) -> dict | None:
    """Return per-(archetype, duration) cutpoint arrays such that a node's
    spread value maps to the same quintile (and therefore the same saturation
    color) regardless of which month is being rendered.

    Cutpoints are the [0, 20, 40, 60, 80, 100] percentiles of spread_D{D}
    values restricted to that archetype, pooled across every month.
    Returns {archetype: {D: [c0..c5]}} or None if only a single month exists
    (per-month bounds remain fine in that case).
    """
    if len(tags) <= 1:
        return None
    import numpy as np

    # Gather (archetype, D, spread) triples across all months.
    pooled: dict[tuple[str, int], list] = {}
    for t in tags:
        m_path = DATA_DIR / f"node_metrics_with_size_{t}.csv"
        d_path = DATA_DIR / f"duration_sweep_{t}.csv"
        if not (m_path.exists() and d_path.exists()):
            continue
        m = pd.read_csv(m_path, usecols=["archetype"] + ["node"] if False else ["archetype"],
                         index_col=0)
        # Re-read with index for join
        m = pd.read_csv(m_path, index_col=0)[["archetype"]]
        d = pd.read_csv(d_path, index_col=0)[[f"spread_D{D}" for D in DURATIONS]]
        df = m.join(d, how="inner")
        for arch in df["archetype"].dropna().unique():
            sub = df[df["archetype"] == arch]
            for D in DURATIONS:
                vals = sub[f"spread_D{D}"].dropna().values
                pooled.setdefault((arch, D), []).extend(vals.tolist())

    out: dict = {}
    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for (arch, D), vals in pooled.items():
        if len(vals) < 5:
            continue
        cuts = list(np.quantile(vals, qs))
        out.setdefault(arch, {})[D] = cuts
    return out


def global_size_range(tags: list[str]) -> tuple[float, float] | None:
    """Scan every month's metric file and return the global (min, max) of
    sqrt(size_node), so per-month pixel sizes are comparable across months.
    Returns None if only one tag is available (fall back to per-month range).
    Caps the max at the 99th percentile to keep one freak outlier (e.g., a
    bad rating crosswalk match) from compressing the whole scale."""
    if len(tags) <= 1:
        return None
    import numpy as np
    sqrts = []
    for t in tags:
        p = DATA_DIR / f"node_metrics_with_size_{t}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, usecols=["size_node"])["size_node"].dropna()
        if not len(s):
            continue
        sqrts.append(np.sqrt(s.clip(lower=0).values))
    if not sqrts:
        return None
    all_sqrt = np.concatenate(sqrts)
    s_min = float(all_sqrt.min())
    s_max = float(np.quantile(all_sqrt, 0.99))  # cap at p99 to ignore freaks
    return s_min, s_max


def quintile_rank(s: pd.Series) -> pd.Series:
    """0..N_QUINTILES-1 quintile index per element; NaNs return 0."""
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(0, index=s.index, dtype=int)
    try:
        q = pd.qcut(valid, N_QUINTILES, labels=False, duplicates="drop")
    except ValueError:
        q = pd.Series(0, index=valid.index, dtype=int)
    out = pd.Series(0, index=s.index, dtype=int)
    out.loc[q.index] = q.astype(int)
    return out


def rgba_string(cmap_name: str, level: float, alpha: float) -> str:
    cmap = matplotlib.colormaps[cmap_name]
    r, g, b, _ = cmap(level)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"


def build_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Two views in one bar chart:
      (1) Top-N NODES — for each node, 3 grouped bars (D=2 / D=4 / D=8) so
          duration sensitivity is visible at a glance. Sorted by D=4 spread.
      (2) Top-N CONTROLLING CONSTRAINTS by total rent — one bar per physical
          line; size_node doesn't depend on D, so this view has no D variant.
    Includes unplaced nodes in view (1).
    """
    # ----- View 1: nodes by spread, grouped 3 bars per node -----
    by_spread = (metrics_df
                  .sort_values(f"spread_D{DEFAULT_DURATION}", ascending=False)
                  .head(BAR_TOP_N).copy())

    def make_node_hovertext(d: pd.DataFrame, D: int) -> list:
        out = []
        for node, r in d.iterrows():
            lat, lon = r.get("Latitude"), r.get("Longitude")
            loc = (f"{lat:.4f}, {lon:.4f}" if pd.notna(lat) and pd.notna(lon)
                   else "<i>(unplaced — no coordinates)</i>")
            plant = r["Plant Name"] if pd.notna(r.get("Plant Name")) else "(unmatched)"
            rating = r.get("kstar_rating_MW")
            rating_str = (f"{rating:.0f} MW" if pd.notna(rating) and rating > 0
                          else "(unknown)")
            out.append(
                f"<b>{node}</b> &nbsp;<span style='color:#666'>(D={D}h bar)</span><br>"
                f"archetype: {ARCHETYPE_LABEL.get(r['archetype'], r['archetype'])}<br>"
                f"spread @ D={D}h: <b>${ceil_int(r[f'spread_D{D}']):,d}/MWh</b><br>"
                f"spread @ D=2h: ${ceil_int(r['spread_D2']):,d} · "
                f"D=4h: ${ceil_int(r['spread_D4']):,d} · "
                f"D=8h: ${ceil_int(r['spread_D8']):,d}<br>"
                f"size: ${r['size_node']:,.0f}<br>"
                f"controlling line: {r.get('kstar_physical_line','(unknown)')}<br>"
                f"k* rating: {rating_str} ({r.get('kstar_rating_source','-')})<br>"
                f"conc: {r['conc']:.3f}  marker={r['marker']}<br>"
                f"EIA plant: {plant}<br>"
                f"lat,lon: {loc}"
            )
        return out

    def edge_color(d: pd.DataFrame) -> list:
        return ["#888" if pd.isna(lat) else "rgba(0,0,0,0.15)"
                for lat in d["Latitude"]]

    def edge_width(d: pd.DataFrame) -> list:
        return [1.5 if pd.isna(lat) else 0.3 for lat in d["Latitude"]]

    # Per-D colors for the bars. We re-shade the archetype color by D so the
    # three bars per node visually communicate which is which:
    #   D=2 → lighter (alpha 0.45 × archetype hue)
    #   D=4 → medium (alpha 0.75)
    #   D=8 → strongest (alpha 1.00)
    # The hue is still the archetype's quintile color at D=4 (consistent with
    # the map's default coloring), so the user can match across views.
    def bar_color_for_d(arch: str, q_d4: int, D: int) -> str:
        cmap_name, alpha_lo, alpha_hi = ARCHETYPE_CMAP[arch]
        level = 0.25 + 0.5 * (q_d4 / max(1, N_QUINTILES - 1))
        # Stronger alpha for longer-duration bar
        d_alpha = {2: 0.45, 4: 0.75, 8: 1.0}[D]
        return rgba_string(cmap_name, level, d_alpha)

    # ----- View 2: constraints aggregated by total rent -----
    has_constraint = metrics_df.dropna(subset=["kstar_physical_line", "size_node"])
    grouped = (has_constraint
        .groupby("kstar_physical_line")
        .agg(
            size=("size_node", "first"),             # identical within group
            rating=("kstar_rating_MW", "first"),
            rating_source=("kstar_rating_source", "first"),
            n_nodes=("size_node", "size"),
            avg_spread=("spread", "mean"),
            max_spread=("spread", "max"),
            archetype_mode=("archetype", lambda s: s.value_counts().idxmax()),
            example_nodes=("size_node", lambda s: ", ".join(s.index.astype(str)[:5])),
        )
        .sort_values("size", ascending=False)
        .head(BAR_TOP_N)
        .copy()
    )

    def short_label(line_key: str, maxlen: int = 55) -> str:
        # strip the NOM:/ITC: prefix for readability
        s = line_key.split(":", 1)[-1].strip()
        return s if len(s) <= maxlen else s[:maxlen-1] + "…"

    grouped["short_label"] = grouped.index.map(short_label)
    # Color per constraint = the most-saturated swatch of its dominant archetype
    grouped["color"] = grouped["archetype_mode"].map(
        lambda a: rgba_string(ARCHETYPE_CMAP[a][0], 0.7, ARCHETYPE_CMAP[a][2]))

    def make_constraint_hovertext(d: pd.DataFrame) -> list:
        out = []
        for line, r in d.iterrows():
            rating = r["rating"]
            rating_str = (f"{rating:.0f} MW" if pd.notna(rating) and rating > 0
                          else "(unknown)")
            out.append(
                f"<b>{line}</b><br>"
                f"<b>Size (rent):</b> ${r['size']:,.0f}<br>"
                f"<b>Rating:</b> {rating_str} ({r['rating_source']})<br>"
                f"<b>Nodes attributed:</b> {r['n_nodes']}<br>"
                f"<b>Dominant archetype:</b> {ARCHETYPE_LABEL.get(r['archetype_mode'], r['archetype_mode'])}<br>"
                f"<b>Node spread:</b> mean ${ceil_int(r['avg_spread']):,d}, max ${ceil_int(r['max_spread']):,d}<br>"
                f"<b>Example nodes:</b> {r['example_nodes']}"
            )
        return out

    # ----- Build figure -----
    # One bar per node, but the bar is a STACKED COMPOSITE of three segments:
    #   segment 1: spread at D=2h     (the "spike" value — captured by any battery)
    #   segment 2: spread_D4 − D=2    (incremental value of going from 2h → 4h)
    #   segment 3: spread_D8 − D=4    (incremental value of going from 4h → 8h)
    # Total bar length = spread_D8. Wide segment 1 = spike-driven; wide outer
    # segments = persistent (longer batteries unlock more value).
    seg_inc_d4 = (by_spread["spread_D4"] - by_spread["spread_D2"]).clip(lower=0)
    seg_inc_d8 = (by_spread["spread_D8"] - by_spread["spread_D4"]).clip(lower=0)

    def make_bar_hovertext(d: pd.DataFrame) -> list:
        """One hover entry per node — same text for every segment, so hovering
        anywhere on the composite bar shows the full picture."""
        out = []
        for node, r in d.iterrows():
            d2 = ceil_int(r["spread_D2"])
            d4 = ceil_int(r["spread_D4"])
            d8 = ceil_int(r["spread_D8"])
            lat, lon = r.get("Latitude"), r.get("Longitude")
            loc = (f"{lat:.4f}, {lon:.4f}" if pd.notna(lat) and pd.notna(lon)
                   else "<i>(unplaced — no coordinates)</i>")
            plant = r["Plant Name"] if pd.notna(r.get("Plant Name")) else "(unmatched)"
            rating = r.get("kstar_rating_MW")
            rating_str = (f"{rating:.0f} MW" if pd.notna(rating) and rating > 0
                          else "(unknown)")
            out.append(
                f"<b>{node}</b><br>"
                f"<b>Archetype:</b> {ARCHETYPE_LABEL.get(r['archetype'], r['archetype'])}<br>"
                f"<b>Spread (per MWh):</b> 2h ${d2:,d}  ·  4h ${d4:,d}  ·  8h ${d8:,d}<br>"
                f"<b>Size:</b> ${r['size_node']:,.0f}<br>"
                f"<b>Controlling line:</b> {r.get('kstar_physical_line','(unknown)')}<br>"
                f"<b>Rating:</b> {rating_str} ({r.get('kstar_rating_source','-')})<br>"
                f"<b>Concentration:</b> {r['conc']:.3f}  ·  <b>Marker:</b> {r['marker']}<br>"
                f"<b>EIA plant:</b> {plant}<br>"
                f"<b>Coordinates:</b> {loc}"
            )
        return out

    bar_hovertext = make_bar_hovertext(by_spread)

    # Use archetype quintile colors at the D=4 quintile (visually consistent
    # with the map's default), with progressively stronger alpha per segment.
    base_colors_d2  = [bar_color_for_d(a, q, 2) for a, q in zip(by_spread["archetype"], by_spread["q_rank_D4"])]
    base_colors_d4i = [bar_color_for_d(a, q, 4) for a, q in zip(by_spread["archetype"], by_spread["q_rank_D4"])]
    base_colors_d8i = [bar_color_for_d(a, q, 8) for a, q in zip(by_spread["archetype"], by_spread["q_rank_D4"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_spread["spread_D2"].values,
        y=by_spread.index.astype(str).tolist(),
        orientation="h",
        marker=dict(color=base_colors_d2,
                    line=dict(color=edge_color(by_spread), width=edge_width(by_spread))),
        text=bar_hovertext,
        hovertemplate="%{text}<extra></extra>",
        name="D = 2 h (base)",
        visible=True, legendgroup="durations", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=seg_inc_d4.values,
        y=by_spread.index.astype(str).tolist(),
        orientation="h",
        marker=dict(color=base_colors_d4i,
                    line=dict(color=edge_color(by_spread), width=edge_width(by_spread))),
        text=bar_hovertext,
        hovertemplate="%{text}<extra></extra>",
        name="+ 4 h increment",
        visible=True, legendgroup="durations", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=seg_inc_d8.values,
        y=by_spread.index.astype(str).tolist(),
        orientation="h",
        marker=dict(color=base_colors_d8i,
                    line=dict(color=edge_color(by_spread), width=edge_width(by_spread))),
        text=bar_hovertext,
        hovertemplate="%{text}<extra></extra>",
        name="+ 8 h increment",
        visible=True, legendgroup="durations", showlegend=True,
    ))
    # Constraint view (trace index 3) — hidden by default
    fig.add_trace(go.Bar(
        x=grouped["size"].values,
        y=grouped["short_label"].tolist(),
        orientation="h",
        marker=dict(color=grouped["color"].tolist(),
                    line=dict(color="rgba(0,0,0,0.2)", width=0.4)),
        text=make_constraint_hovertext(grouped),
        hovertemplate="%{text}<extra></extra>",
        name="constraint rent",
        visible=False, showlegend=False,
    ))

    fig.update_layout(
        barmode="stack",
        bargap=0.25,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.06,
                    xanchor="right", x=1.0,
                    title=dict(text="Composite bar segments:")),
        margin=dict(l=320, r=20, t=90, b=50),
        height=max(700, 18 * BAR_TOP_N + 120),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=10),
            categoryorder="array",
            categoryarray=by_spread.index.astype(str).tolist(),
        ),
        xaxis=dict(title="spread ($/MWh) — stacked: D=2 base + 4h increment + 8h increment",
                    gridcolor="#eee"),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.01, y=1.07, xanchor="left", yanchor="top",
            showactive=True,
            buttons=[
                dict(label="Top nodes by spread (composite: D=2 + 4h-inc + 8h-inc)",
                     method="update",
                     args=[
                         {"visible": [True, True, True, False]},
                         {"xaxis.title.text": "spread ($/MWh) — stacked: D=2 base + 4h increment + 8h increment",
                          "yaxis.categoryarray": by_spread.index.astype(str).tolist(),
                          "showlegend": True},
                     ]),
                dict(label="Top controlling constraints by rent ($)",
                     method="update",
                     args=[
                         {"visible": [False, False, False, True]},
                         {"xaxis.title.text": "constraint rent ($) = rating × Σ|μ|",
                          "yaxis.categoryarray": grouped["short_label"].tolist(),
                          "showlegend": False},
                     ]),
            ],
        )],
    )
    return fig


def build_month_nav(current_tag: str, all_tags: list[str]) -> str:
    """Render the month-link bar shown in the page header. If only one month
    is available, returns an empty string."""
    if len(all_tags) <= 1:
        return ""
    parts = ['<div class="month-nav"><span class="mlabel">View:</span>']
    for t in all_tags:
        classes = []
        if t == current_tag: classes.append("current")
        if t == FULL_YEAR_TAG: classes.append("full-year")
        cls = " ".join(classes)
        href = page_filename(t)
        label = MONTH_LABELS.get(t, t)
        title = MONTH_LONG_LABELS.get(t, t)
        parts.append(f'<a class="{cls}" href="{href}" title="{title}">{label}</a>')
    parts.append("</div>")
    return "".join(parts)


def build_page(map_html: str, bar_html: str,
               n_rendered: int, n_total: int, n_bar: int, n_metrics: int,
               quintile_bounds: dict, trace_colors_by_d: dict,
               month_nav_html: str = "", current_tag: str = "") -> str:
    """Wrap the Plotly figure in an HTML shell with a corner legend and a
    slide-in README panel."""
    # Color swatches for the corner legend — sample at the highest quintile
    # so the user sees the most-saturated example of each archetype.
    swatch = lambda arch: rgba_string(*[ARCHETYPE_CMAP[arch][0:1][0],
                                         0.75, ARCHETYPE_CMAP[arch][2]])
    color_w = rgba_string(ARCHETYPE_CMAP["WHITE"][0], 0.6, ARCHETYPE_CMAP["WHITE"][2])
    color_b = rgba_string(ARCHETYPE_CMAP["BLUE"][0], 0.75, ARCHETYPE_CMAP["BLUE"][2])
    color_r = rgba_string(ARCHETYPE_CMAP["RED"][0], 0.75, ARCHETYPE_CMAP["RED"][2])
    color_p = rgba_string(ARCHETYPE_CMAP["PURPLE"][0], 0.75, ARCHETYPE_CMAP["PURPLE"][2])

    def gradient_html(arch: str) -> str:
        bounds = quintile_bounds.get(arch)
        cmap_name, alpha_lo, alpha_hi = ARCHETYPE_CMAP[arch]
        label = ARCHETYPE_LABEL[arch]
        if not bounds:
            return f"<div class='qbar-row'><span class='qbar-label'>{label}</span> <i>(no nodes)</i></div>"
        cells = []
        for q in range(N_QUINTILES):
            level = 0.25 + 0.5 * (q / max(1, N_QUINTILES - 1))
            alpha = alpha_lo + (alpha_hi - alpha_lo) * (q / max(1, N_QUINTILES - 1))
            color = rgba_string(cmap_name, level, alpha)
            lo, hi = ceil_int(bounds[q]), ceil_int(bounds[q + 1])
            cells.append(
                f"<div class='qcell' style='background:{color}'>"
                f"<span class='qrange'>${lo:,}–${hi:,}</span>"
                f"</div>"
            )
        return (
            f"<div class='qbar-row'>"
            f"  <div class='qbar-label'>{label}</div>"
            f"  <div class='qbar'>{''.join(cells)}</div>"
            f"</div>"
        )

    gradient_bars = "\n".join(gradient_html(a) for a in ["BLUE", "RED", "PURPLE", "WHITE"])

    # Active-class strings for the duration picker buttons
    active_class = {D: (" active" if D == DEFAULT_DURATION else "") for D in DURATIONS}
    # Per-D color arrays embedded as JSON for client-side restyling
    duration_colors_json = json.dumps({str(D): trace_colors_by_d[D] for D in DURATIONS})

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CAISO congestion-relief node map — {SEASON_TAG}</title>
<style>
  html, body {{ margin:0; padding:0; height:100%; width:100%; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; overflow:hidden; }}
  /* Tab containers (mutually exclusive) */
  .view {{ position:absolute; top:42px; left:0; right:0; bottom:0; display:none; }}
  .view.active {{ display:block; }}
  .view#view-map {{ overflow:hidden; }}
  .view#view-bar {{ overflow-y:auto; background:#fafafa; }}
  .view#view-bar #bar-chart {{ background:#fff; min-height:100%; }}
  .view#view-bar .toolbar {{ padding:8px 14px; background:#fff; border-bottom:1px solid #e0e0e0;
                              font-size:12px; color:#555; }}
  #map {{ position:absolute; inset:0; }}
  /* Header bar */
  #header {{ position:absolute; top:0; left:0; right:0; height:42px; padding:0 14px;
             background:rgba(255,255,255,0.96); border-bottom:1px solid #ccc;
             display:flex; align-items:center; gap:14px; z-index:11; }}
  #header h1 {{ font-size:14px; font-weight:600; margin:0; color:#222; }}
  #header .meta {{ font-size:12px; color:#666; margin-left:auto; }}
  #header button {{ font-size:12px; background:#fff; border:1px solid #aaa;
                    border-radius:4px; padding:5px 10px; cursor:pointer; }}
  #header button:hover {{ background:#f0f0f0; }}
  /* Month navigation — inside header, between title and tab bar */
  .month-nav {{ display:flex; gap:0; align-items:center; }}
  .month-nav .mlabel {{ font-size:10.5px; color:#666; margin-right:6px;
                         text-transform:uppercase; letter-spacing:0.04em; }}
  .month-nav a {{ font-size:11px; padding:4px 7px; border:1px solid #aaa;
                   background:#fff; text-decoration:none; color:#222;
                   border-right:none; }}
  .month-nav a:first-of-type {{ border-radius:4px 0 0 4px; }}
  .month-nav a:last-of-type {{ border-radius:0 4px 4px 0; border-right:1px solid #aaa; }}
  .month-nav a:hover:not(.current) {{ background:#f0f0f0; }}
  .month-nav a.current {{ background:#2a6ab8; color:#fff; border-color:#2a6ab8; font-weight:600; }}
  .month-nav a.full-year {{ font-weight:600; background:#fafafa; }}
  .month-nav a.full-year + a {{ margin-left:6px; border-left:1px solid #aaa;
                                  border-radius:4px 0 0 4px; }}
  .month-nav a.full-year {{ border-radius:4px !important; border-right:1px solid #aaa; }}

  /* Tab bar — inside header */
  .tabs {{ display:flex; gap:0; }}
  .tab {{ font-size:12.5px; padding:6px 14px; border:1px solid #aaa; background:#fff;
          cursor:pointer; border-right:none; }}
  .tab:first-child {{ border-radius:4px 0 0 4px; }}
  .tab:last-child {{ border-radius:0 4px 4px 0; border-right:1px solid #aaa; }}
  .tab.active {{ background:#2a6ab8; color:#fff; border-color:#2a6ab8; }}
  .tab:hover:not(.active) {{ background:#f0f0f0; }}

  /* Slide-in README panel */
  #panel {{ position:absolute; top:42px; left:0; bottom:0; width:480px;
            background:#fff; border-right:1px solid #ccc; overflow-y:auto;
            transform:translateX(-100%); transition:transform 0.25s ease-out;
            z-index:9; box-shadow:0 0 12px rgba(0,0,0,0.15); }}
  #panel.open {{ transform:translateX(0); }}
  #panel .content {{ padding:18px 22px; font-size:13px; line-height:1.55; color:#222; }}
  #panel h2 {{ font-size:16px; margin:18px 0 6px; color:#111; border-bottom:1px solid #ddd; padding-bottom:4px; }}
  #panel h3 {{ font-size:13px; margin:12px 0 4px; color:#333; }}
  #panel p {{ margin:6px 0; }}
  #panel ul {{ padding-left:20px; margin:6px 0; }}
  #panel li {{ margin:3px 0; }}
  #panel code, #panel .eq {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:12px;
                              background:#f4f4f4; padding:1px 5px; border-radius:3px; }}
  #panel .eq {{ display:block; padding:6px 8px; margin:6px 0; font-size:12px; }}
  /* Quintile gradient bars (per archetype) */
  #panel .qbar-row {{ display:flex; align-items:center; gap:10px; margin:6px 0; }}
  #panel .qbar-label {{ flex:0 0 130px; font-size:11.5px; color:#333; }}
  #panel .qbar {{ display:flex; flex:1; height:30px; border:1px solid rgba(0,0,0,0.18);
                  border-radius:3px; overflow:hidden; }}
  #panel .qcell {{ flex:1; display:flex; align-items:center; justify-content:center;
                    border-right:1px solid rgba(0,0,0,0.12); position:relative; }}
  #panel .qcell:last-child {{ border-right:none; }}
  #panel .qrange {{ font-size:9.5px; color:rgba(0,0,0,0.78); font-weight:500;
                     text-shadow:0 0 3px rgba(255,255,255,0.6); white-space:nowrap; }}
  /* Variable-definition lists under each equation */
  #panel dl.vars {{ margin:4px 0 12px 8px; font-size:12px; }}
  #panel dl.vars dt {{ font-family:"SF Mono",Menlo,Consolas,monospace; display:inline-block;
                       background:#eef; padding:0 5px; border-radius:3px; font-weight:600; }}
  #panel dl.vars dd {{ display:inline; margin:0 0 0 6px; color:#444; }}
  #panel dl.vars dd::after {{ content:""; display:block; height:4px; }}

  /* Battery-duration picker (top-left, over the map) */
  #duration-picker {{ position:absolute; left:14px; top:55px; z-index:8;
                       background:rgba(255,255,255,0.95); border:1px solid #aaa;
                       border-radius:6px; padding:6px 10px; font-size:12px;
                       box-shadow:0 1px 4px rgba(0,0,0,0.12);
                       display:flex; align-items:center; gap:8px; }}
  #duration-picker .lbl {{ font-size:11.5px; color:#444; font-weight:500; }}
  #duration-picker .dbtn {{ font-size:12px; padding:4px 12px; border:1px solid #aaa;
                            background:#fff; cursor:pointer; border-radius:4px;
                            transition:background 0.1s, color 0.1s; }}
  #duration-picker .dbtn:hover:not(.active) {{ background:#f0f0f0; }}
  #duration-picker .dbtn.active {{ background:#2a6ab8; color:#fff;
                                    border-color:#2a6ab8; font-weight:600; }}
  #duration-picker .hint {{ font-size:10.5px; color:#777; margin-left:4px; }}

  /* Compact legend, lower-left */
  #legend {{ position:absolute; left:14px; bottom:14px; background:rgba(255,255,255,0.94);
             border:1px solid #ccc; border-radius:6px; padding:10px 12px;
             font-size:11.5px; line-height:1.5; max-width:280px; z-index:8;
             box-shadow:0 1px 4px rgba(0,0,0,0.1); }}
  #legend h4 {{ margin:0 0 6px; font-size:12px; }}
  #legend .row {{ display:flex; align-items:center; gap:8px; margin:3px 0; }}
  #legend .swatch {{ display:inline-block; width:14px; height:14px; border-radius:50%;
                     border:1px solid rgba(0,0,0,0.25); flex-shrink:0; }}
  #legend .size-scale {{ display:flex; align-items:center; gap:6px; margin-top:4px; }}
  #legend .size-dot {{ background:#888; border-radius:50%; display:inline-block;
                       border:1px solid rgba(0,0,0,0.25); }}
  #legend .hollow-dot {{ display:inline-block; background:#fff; border-radius:50%;
                          border:2px solid #6699cc; width:14px; height:14px; }}
  #legend .filled-dot {{ display:inline-block; background:#6699cc; border-radius:50%;
                          border:1px solid rgba(0,0,0,0.25); width:14px; height:14px; }}
</style>
</head>
<body>

<div id="header">
  <button id="toggle">📘 How to read this</button>
  <h1>CAISO congestion-relief node map</h1>
  {month_nav_html}
  <div class="tabs">
    <button class="tab active" data-view="map">🗺 Map ({n_rendered:,})</button>
    <button class="tab" data-view="bar">📊 Ranked (top {n_bar} of {n_metrics:,})</button>
  </div>
  <span class="meta">{MONTH_LONG_LABELS.get(current_tag, current_tag)} · {n_rendered:,} placed · {n_metrics-n_rendered:,} unplaced (in ranked tab only)</span>
</div>

<div id="panel">
  <div class="content">
    <h2>What is this?</h2>
    <p>An interactive screening map of CAISO pricing nodes designed to identify
       <b>where battery storage could relieve transmission congestion</b>.
       Each node is encoded by four channels (hue, saturation, size, marker style)
       derived from the <b>marginal congestion component (MCC) of locational price</b>,
       not the total price. This is a research instrument; deliverable relief
       requires a network-model run beyond scope.</p>

    <h2>The encoding (four channels)</h2>

    <h3>1. Hue → archetype (flow direction)</h3>
    <ul>
      <li><span style="color:#666">⬤</span> <b>WHITE</b> — no local congestion; price swing is purely energy-driven</li>
      <li><span style="color:#2a6ab8">⬤</span> <b>BLUE</b> — import pocket (evening MCC high; binding constraint feeds load)</li>
      <li><span style="color:#cc3333">⬤</span> <b>RED</b> — export pocket (midday MCC negative; trapped supply, e.g. solar oversupply)</li>
      <li><span style="color:#8855aa">⬤</span> <b>PURPLE</b> — bidirectional / double-duty (both import and export congestion). <b>Most interesting for siting.</b></li>
    </ul>

    <h3>2. Saturation → round-trip MCC spread (intensity)</h3>
    <p>Within each archetype, nodes are sorted into <b>spread quintiles</b> (q0–q4) —
       higher quintile = more saturated color. The quintile-based mapping
       prevents the long tail of extreme spreads (Kern County, etc.) from
       washing out the middle of the distribution. Boundaries are
       data-driven (each bucket has ~20% of that archetype's nodes); current
       month's values:</p>
    {gradient_bars}
    <p style="font-size:11px;color:#666;margin-top:6px">
      Numbers under each cell are the spread range (in $/MWh) covered by
      that quintile in this dataset. They update automatically each rerun.
    </p>

    <h3>3. Marker size → controlling-constraint rent</h3>
    <p>Radius ∝ <code>√(size_node)</code> where <code>size_node = rating × Σ|μ|</code>
       for the constraint that controls the node (its "k*"). Bigger circle =
       larger congestion prize on the wire this node sits on.</p>

    <h3>4. Marker style → bankability (concentration)</h3>
    <p>
      <span class="filled-dot"></span> <b>filled</b> circle — <i>persistent</i> congestion value (most days/hours)<br>
      <span class="hollow-dot"></span> <b>hollow</b> ring — <i>spike-driven</i> value (rare outage hours dominate). Less bankable.
    </p>
    <p>Threshold: hollow if top 1% of 5-min intervals contributes more than
       50% of total |MCC| value.</p>

    <h2>Battery duration (the 2 h / 4 h / 8 h buttons)</h2>
    <p>The picker at the top-left of the map switches the <b>battery duration
       <code>D</code></b> used to compute the round-trip spread (channel #2,
       color saturation). It changes how the map colors each node — same
       coordinates, same archetype, but a different intensity reflecting how
       much arbitrage a battery of that duration would capture at that node.</p>
    <p><b>Why duration matters:</b> a battery's daily arbitrage value depends
       on how many contiguous high/low hours it can ride. Different node
       behaviors favor different durations:</p>
    <ul>
      <li><b>Spike-driven nodes</b> have a handful of very-high-price hours per day
          — a short (2 h) battery can capture most of the value. Adding more
          duration unlocks little extra revenue.</li>
      <li><b>Persistent-spread nodes</b> have wider price differentials sustained
          over many hours — a long (8 h) battery unlocks meaningfully more
          revenue than a 2 h would. Path-26-fed LA Basin nodes like
          Alamitos sit here: the import constraint binds for most of the
          4–10 pm window, so an 8 h battery has more discharge hours to
          exploit.</li>
    </ul>
    <p><b>How to see this in the bar chart (📊 Ranked tab):</b> each top-100
       bar is a <b>stacked composite</b> showing where the value comes from:
    </p>
    <ul>
      <li>Innermost (faintest) segment = the <b>D=2 h spread</b> — the spike value
          a short battery captures.</li>
      <li>Middle segment = the <b>incremental value from doubling 2 h → 4 h</b>.</li>
      <li>Outermost (most saturated) segment = the <b>incremental value from
          doubling 4 h → 8 h</b>. Total bar length = D=8 h spread.</li>
    </ul>
    <p>Quick reads:</p>
    <ul>
      <li>If a bar is dominated by its innermost segment, the node is
          <b>spike-driven</b> — there's little benefit to building anything
          beyond a 2 h system there.</li>
      <li>If the outer two segments are visibly thick, the node is
          <b>duration-sensitive</b> — longer storage genuinely unlocks more
          value (often points to a constraint that binds for many contiguous
          hours).</li>
      <li>Hover any segment to see its specific contribution and the full
          2 h / 4 h / 8 h sweep.</li>
    </ul>

    <h2>Hover field reference</h2>
    <ul>
      <li><code>archetype</code> — flow-direction classification (WHITE / BLUE / RED / PURPLE)</li>
      <li><code>spread</code> — median daily round-trip arbitrage value of MCC, $/MWh. The bigger the spread, the more $$$ a battery sitting there could capture per day.</li>
      <li><code>(q0–q4)</code> — spread quintile within this archetype (saturation level)</li>
      <li><code>size</code> — <code>rating × Σ|μ|</code> over the month, in $. Total congestion-rent prize on the wire this node controls.</li>
      <li><code>controlling line</code> — physical transmission element with largest β·μ contribution from the regression. OASIS nomogram or intertie ID.</li>
      <li><code>k* rating</code> — MW capacity of that line. Sources: WECC named path catalog (gold), explicit intertie MVA, or voltage-class proxy (69→150, 115→250, 230→700, 500→2500).</li>
      <li><code>conc</code> — share of total |MCC| value contained in the top 1% of 5-min intervals. High conc → spike-driven (hollow).</li>
      <li><code>marker</code> — filled / hollow per <code>conc</code>.</li>
      <li><code>EIA plant</code> — fuzzy-matched generator name from EIA-860 used for coordinates. Some long-tail matches are approximate.</li>
      <li><code>lat,lon</code> — matched plant coordinates.</li>
    </ul>

    <h2>Core equations (brief §3)</h2>

    <h3>Price decomposition (CAISO RTM 5-min)</h3>
    <code class="eq">LMP(t) = MEC(t) + MCC<sub>node</sub>(t) + MLC<sub>node</sub>(t) + GHG(t)</code>
    <dl class="vars">
      <dt>t</dt><dd>time, in 5-min intervals (CAISO real-time market).</dd>
      <dt>LMP(t)</dt><dd>locational marginal price at the node, in $/MWh — the total price a generator gets paid (or a load pays) for energy at this location.</dd>
      <dt>MEC(t)</dt><dd>marginal energy component — system-wide reference price; identical at every node, so it carries no location information.</dd>
      <dt>MCC<sub>node</sub>(t)</dt><dd><b>marginal congestion component</b> — the only piece that varies by location, set by which transmission constraints are binding. <b>This is the signal we use throughout.</b></dd>
      <dt>MLC<sub>node</sub>(t)</dt><dd>marginal loss component — covers transmission line losses; small.</dd>
      <dt>GHG(t)</dt><dd>greenhouse-gas cost adder (cap-and-trade pass-through). Only nonzero hours when GHG-emitting units are on the margin.</dd>
    </dl>

    <h3>Daily spread (per node)</h3>
    <code class="eq">discharge(d) = sum of the D highest hourly MCC values on day d
charge(d)    = sum of the D lowest  hourly MCC values on day d
daily_spread(d) = discharge(d) − (1/η) × charge(d)
spread_node     = median over days of daily_spread(d)</code>
    <dl class="vars">
      <dt>D</dt><dd>battery duration in hours — how long the battery can keep discharging at rated power. Default <b>D = 4</b> (a 4-hour battery).</dd>
      <dt>η</dt><dd>round-trip efficiency — fraction of energy that survives one charge + discharge cycle. Default <b>η = 0.85</b> (15% losses). The factor <code>(1/η)</code> means you have to charge slightly more MWh than you'll later discharge.</dd>
      <dt>MCC (hourly)</dt><dd>marginal congestion component, averaged from 5-min to hourly. In $/MWh.</dd>
      <dt>d</dt><dd>calendar day index.</dd>
    </dl>
    <p>MCC stays <b>unclipped</b> — negative MCC during charge hours
       makes <code>charge</code> negative, so <code>−(1/η) × charge</code> turns into
       a positive contribution: the battery is effectively <i>paid to charge</i>
       and that's part of the prize.</p>

    <h3>Archetype classification</h3>
    <code class="eq">MCC_mid = mean MCC over 10:00–15:00 local time (midday window)
MCC_eve = mean MCC over 17:00–21:00 local time (evening window)
import_signal = MCC_eve
export_signal = −MCC_mid

if both signals < ε:        WHITE  (no local congestion)
if import ≥ ε, export < ε:  BLUE   (import pocket)
if export ≥ ε, import < ε:  RED    (export pocket)
if both signals ≥ ε:        PURPLE (bidirectional)</code>
    <dl class="vars">
      <dt>ε</dt><dd>flatness threshold in $/MWh — signals below ε are treated as "no congestion in that direction". Default <b>ε = $3/MWh</b>.</dd>
      <dt>import_signal</dt><dd>positive when the node sits behind a constraint that binds when load is high (evening peak).</dd>
      <dt>export_signal</dt><dd>positive when midday MCC is negative — happens when local generation can't get out because the wire to the rest of the grid is full (classic solar oversupply).</dd>
    </dl>

    <h3>Attribution &amp; sizing</h3>
    <code class="eq">MCC<sub>node</sub>(t) ≈ Σ<sub>k</sub> β<sub>node,k</sub> · μ<sub>k</sub>(t)        (Ridge regression)
k* = argmax<sub>k</sub>  mean<sub>t</sub> |β<sub>node,k</sub> · μ<sub>k</sub>(t)|     (the "controlling" physical line)
size_node = rating[k*]  ×  Σ<sub>t</sub> |μ<sub>k*</sub>(t)|     (the dollar prize on that line)</code>
    <dl class="vars">
      <dt>μ<sub>k</sub>(t)</dt><dd>shadow price of constraint <code>k</code> at time <code>t</code>, in $/MWh — CAISO publishes this. When a constraint binds, μ is the $/MWh the market would save if you could relax it by 1 MW. μ = 0 when the constraint isn't binding.</dd>
      <dt>k</dt><dd>index over binding transmission constraints (in our August 2025 panel: 287 distinct nomogram + intertie shadow-price series).</dd>
      <dt>β<sub>node,k</sub></dt><dd>regression coefficient — how much this node's MCC moves per $1 of shadow price on constraint <code>k</code>. Fitted by Ridge regression. Conceptually an empirical shift factor (PTDF); not exact PTDFs.</dd>
      <dt>k*</dt><dd>the constraint whose contribution <code>|β · μ|</code>, averaged over time, is largest for this node — i.e. the wire that <i>controls</i> this node's congestion. Aggregated to the physical-line level (scenarios combined).</dd>
      <dt>rating[k*]</dt><dd>nameplate MW capacity of the controlling line, from the WECC Path Rating Catalog or a voltage-class proxy.</dd>
      <dt><b>Σ<sub>t</sub> |μ<sub>k*</sub>(t)|</b></dt><dd>sum, over every 5-min interval in the month, of the absolute value of the controlling line's shadow price. Bigger means the line binds more often or harder — equivalently, total congestion-price "volume" on the line. Multiplied by the rating, this gives the total monthly <b>congestion rent</b> in dollars.</dd>
      <dt>Σ<sub>k</sub></dt><dd>sum over all constraints <code>k</code> in the panel.</dd>
      <dt>argmax<sub>k</sub></dt><dd>"the value of k that makes the following expression largest" — picks the dominant constraint.</dd>
    </dl>

    <h3>Bankability (concentration)</h3>
    <code class="eq">conc = (sum of |MCC| in the top 1% of 5-min intervals)
       / (total sum of |MCC| over all intervals)
hollow marker if  conc > 0.5</code>
    <dl class="vars">
      <dt>|MCC|</dt><dd>absolute value of the marginal congestion component (we use |MCC| because both positive — import — and negative — export — MCC are valuable to a battery).</dd>
      <dt>top 1%</dt><dd>over a 31-day month at 5-min resolution there are 8,928 intervals, so the top 1% is the ~89 most extreme intervals.</dd>
      <dt>conc threshold (0.5)</dt><dd>if more than half of all congestion value comes from those ~89 spike intervals, the node is "spike-driven" and rendered as a hollow ring. Otherwise it's persistent (filled circle).</dd>
    </dl>

    <h2>Data sources</h2>
    <ul>
      <li><b><a href="http://oasis.caiso.com" target="_blank" rel="noopener">CAISO OASIS</a></b>
          (accessed via the <a href="https://github.com/kmax12/gridstatus" target="_blank" rel="noopener"><code>gridstatus</code></a>
          Python library): 5-min RTM LMP with congestion component, plus
          nomogram + intertie shadow prices for August 2025.</li>
      <li><b><a href="https://www.wecc.org/wecc-document/26556" target="_blank" rel="noopener">WECC 2026 Path Rating Catalog (Public Version)</a></b>:
          named transmission-path MW ratings.</li>
      <li><b><a href="https://www.eia.gov/electricity/data/eia860/" target="_blank" rel="noopener">EIA-860</a></b>
          (2025 Early Release): generating-plant coordinates.</li>
    </ul>

    <h2>⚠ Important: plant matches are fuzzy</h2>
    <p>CAISO publishes prices by pricing-node <i>name</i> (e.g. <code>ALAMT1G_7_B1</code>),
       not coordinates. We attach a lat/lon by <b>fuzzy-matching the abbreviation
       to the EIA-860 plant catalog</b>:</p>
    <ul>
      <li>For ~10 well-known LA Basin plants (Alamitos, Harbor, Etiwanda, Huntington Beach, El Segundo, Ormond, Sentinel, …), we use a <b>hand-curated alias map</b> — these placements are reliable.</li>
      <li>For the rest, we use a <b>subsequence-density</b> matcher with three guards: (1) abbreviation must match the plant name with character density ≥ 0.5, (2) the EIA plant must be in the right Balancing Authority (CISO for SP15/NP15/ZP26; PACE/PACW for their hubs), with state fallback only if no BA match exists, and (3) when 3+ different CAISO abbreviations all match the same plant, all of them are demoted to "unplaced" because at least most of them are wrong.</li>
    </ul>
    <p><b>Even with those guards, a handful of placements are still wrong.</b>
       CAISO uses 5–8 char mnemonics that don't always map cleanly to EIA's full plant names, and many CAISO pnodes are for substations (Vincent, Lugo, Mira Loma) that don't appear in EIA-860 at all — those are intentionally left unplaced.</p>
    <p><b>Always hover-check the "EIA plant" field before acting on a placement.</b>
       If the plant name doesn't match what you'd expect for that node, treat
       the location as suspect. The bar-chart tab is the safer view if you only
       care about the rankings.</p>

    <h2>Why some dots stack with different colors</h2>
    <p>The map dot is a <b>geographic point</b>, but MCC is computed at a
       specific <b>electrical bus</b> in the network. One physical substation
       can have multiple distinct buses sharing the same lat/lon, and each
       bus generally has its own shift factors (PTDFs) to the binding
       constraints — so their MCCs can differ.</p>
    <p>Concrete sources of within-site divergence:</p>
    <ul>
      <li><b>Different voltage levels</b> (500 kV / 230 kV / 115 kV yards) at one substation. Each yard is a distinct bus and only "sees" the constraints electrically coupled to it.</li>
      <li><b>A transformer between two buses can itself bind</b> — when it does, the high-side and low-side prices diverge by the transformer's shadow price.</li>
      <li><b>Split-bus substations</b> with sectionalizing tie breakers; under contingencies the two sections can be electrically separated.</li>
      <li><b>Generator-tap buses</b> just upstream of the main station bus, through unit transformers — same address, different injection point in the model.</li>
    </ul>
    <p>So when you see a small cluster of dots in the same spot with different
       colors (e.g., one BLUE + one WHITE at Caribou hydro), it's usually the
       network model showing through — not a data bug. The pnode IDs and
       bus suffixes (<code>_B1</code> vs <code>_B2</code>) usually indicate
       which bus is which.</p>

    <h2>Other limitations</h2>
    <ul>
      <li>This is a <b>screening tool</b>. Map says "MCC pattern here suggests congestion." It does <i>not</i> confirm a battery sited there can actually relieve the constraint — that needs a network-model run with PTDFs.</li>
      <li>Empirical β are not true shift factors; they're regression-fit proxies.</li>
      <li>Aggregation nodes (DLAPs, sub-LAPs, trading hubs) excluded from this run.</li>
      <li>One season only (August 2025); seasonal variation not shown.</li>
    </ul>

  </div>
</div>

<div id="view-map" class="view active">
  <div id="duration-picker">
    <span class="lbl">Battery duration (D):</span>
    <button class="dbtn{active_class[2]}" data-d="2">2 h</button>
    <button class="dbtn{active_class[4]}" data-d="4">4 h</button>
    <button class="dbtn{active_class[8]}" data-d="8">8 h</button>
    <span class="hint">— color/saturation uses the selected D's spread quintile</span>
  </div>
  {map_html}
</div>

<div id="view-bar" class="view">
  <div class="toolbar">
    Two ranked views — switch with the buttons inside the chart:
    <b>(1) Top nodes by spread</b> — top {n_bar} of {n_metrics:,} metrics-ready nodes. Each bar is a <b>stacked composite</b>: innermost (faint) = D=2h base spread, middle = 4h increment, outer (most saturated) = 8h increment. Wide outer segments = duration-sensitive (longer batteries unlock more). <b>Dark gray outline</b> = unplaced node (not on map but metric is real). Hover any segment for its contribution + full sweep.
    <b>(2) Top controlling constraints by rent</b> — one bar per physical line; size_node doesn't depend on D so this view has no D variant. Hover shows # nodes attributed.
  </div>
  {bar_html}
</div>

<div id="legend">
  <h4>Legend</h4>
  <div class="row"><span class="swatch" style="background:{color_w}"></span>no local congestion</div>
  <div class="row"><span class="swatch" style="background:{color_b}"></span>import pocket (evening)</div>
  <div class="row"><span class="swatch" style="background:{color_r}"></span>export pocket (midday)</div>
  <div class="row"><span class="swatch" style="background:{color_p}"></span>bidirectional (double-duty)</div>
  <div class="row" style="margin-top:8px"><i>color saturation</i> &nbsp;→ spread quintile (within archetype)</div>
  <div class="row"><i>circle size</i> &nbsp;→ √(constraint rent on k*)</div>
  <div class="size-scale" style="margin-top:4px">
    <span class="size-dot" style="width:6px;height:6px"></span>
    <span class="size-dot" style="width:11px;height:11px"></span>
    <span class="size-dot" style="width:18px;height:18px"></span>
    <span style="font-size:10.5px;color:#555">small &nbsp;⟶&nbsp; large rent</span>
  </div>
  <div class="row" style="margin-top:8px"><span class="filled-dot"></span> filled = persistent value</div>
  <div class="row"><span class="hollow-dot"></span> hollow = spike-driven (rare hours)</div>
  <div class="row" style="margin-top:6px;font-size:10.5px;color:#555">
    Click <b>📘 How to read this map</b> for full reference.
  </div>
</div>

<script>
  // README slide-in panel
  const btn = document.getElementById("toggle");
  const panel = document.getElementById("panel");
  btn.addEventListener("click", () => {{
    panel.classList.toggle("open");
    btn.textContent = panel.classList.contains("open")
      ? "✕ Close panel"
      : "📘 How to read this";
  }});

  // Battery-duration picker (drives map marker.color via Plotly.restyle)
  const D_COLORS = {duration_colors_json};
  const dbtns = document.querySelectorAll("#duration-picker .dbtn");
  dbtns.forEach(btn => {{
    btn.addEventListener("click", () => {{
      const d = btn.dataset.d;
      dbtns.forEach(b => b.classList.toggle("active", b === btn));
      const mapDiv = document.getElementById("map");
      if (window.Plotly && mapDiv && D_COLORS[d]) {{
        Plotly.restyle(mapDiv, {{ "marker.color": D_COLORS[d] }});
      }}
    }});
  }});

  // Tab switching
  const tabs = document.querySelectorAll(".tab");
  const views = document.querySelectorAll(".view");
  const legendBox = document.getElementById("legend");
  function activate(viewName) {{
    tabs.forEach(t => t.classList.toggle("active", t.dataset.view === viewName));
    views.forEach(v => v.classList.toggle("active", v.id === ("view-" + viewName)));
    // Hide the floating legend on the bar tab (no spatial meaning there)
    legendBox.style.display = (viewName === "map") ? "block" : "none";
    // Force the newly-visible Plotly figure to re-fit its container
    const figId = (viewName === "map") ? "map" : "bar-chart";
    if (window.Plotly) {{
      const el = document.getElementById(figId);
      if (el) Plotly.Plots.resize(el);
    }}
  }}
  tabs.forEach(t => t.addEventListener("click", () => activate(t.dataset.view)));
</script>

</body>
</html>
"""


def main():
    # ----------------------------------------------------------------------
    # 1. Load + join
    # ----------------------------------------------------------------------
    metrics = pd.read_csv(DATA_DIR / f"node_metrics_with_size_{SEASON_TAG}.csv", index_col=0)
    coords  = pd.read_csv(DATA_DIR / f"node_coordinates_{SEASON_TAG}.csv")
    coords  = coords.rename(columns={"PNode ID": "node"}).set_index("node")

    df = metrics.join(coords[["Latitude", "Longitude", "placement",
                              "Plant Name", "abbrev", "match_score"]], how="left")
    total_nodes = len(df)
    print(f"total nodes: {total_nodes:,}")

    needed_metrics = ["archetype", "spread", "size_node"]
    df_metrics = df.dropna(subset=needed_metrics).copy()
    print(f"metrics-ready (for ranking): {len(df_metrics):,}")

    # Join in the duration-sweep spreads
    sweep_path = DATA_DIR / f"duration_sweep_{SEASON_TAG}.csv"
    sweep = pd.read_csv(sweep_path, index_col=0)
    df_metrics = df_metrics.join(sweep[[f"spread_D{D}" for D in DURATIONS]], how="left")
    # The base 'spread' column is the default-D (4h) — assert consistency
    if not df_metrics[f"spread_D{DEFAULT_DURATION}"].equals(df_metrics["spread"]):
        # small float diffs OK — but the base 'spread' from D4 should align
        max_diff = (df_metrics[f"spread_D{DEFAULT_DURATION}"] - df_metrics["spread"]).abs().max()
        print(f"NOTE: base spread vs D=4 spread max diff: {max_diff:.4f}")

    def node_color(arch: str, q: int) -> str:
        cmap_name, alpha_lo, alpha_hi = ARCHETYPE_CMAP[arch]
        level = 0.25 + 0.5 * (q / max(1, N_QUINTILES - 1))
        alpha = alpha_lo + (alpha_hi - alpha_lo) * (q / max(1, N_QUINTILES - 1))
        return rgba_string(cmap_name, level, alpha)

    # Per-archetype quintile rank + color, per duration.
    # If multi-month data exists, use GLOBAL cutpoints (pooled across months)
    # so a given $/MWh spread maps to the same color in every month.
    g_quintiles = global_quintile_bounds(discover_months())

    def quintile_with_cuts(values: pd.Series, cuts: list[float]) -> pd.Series:
        # cuts = [q0, q1, q2, q3, q4, q5] inclusive boundaries
        bins = cuts.copy()
        # Make outer edges open
        bins[0] = -float("inf")
        bins[-1] = float("inf")
        out = pd.cut(values, bins=bins, labels=False, include_lowest=True)
        return out.fillna(0).astype(int)

    for D in DURATIONS:
        col_q = f"q_rank_D{D}"
        col_c = f"color_D{D}"
        df_metrics[col_q] = 0
        for arch in df_metrics["archetype"].unique():
            mask = df_metrics["archetype"] == arch
            vals = df_metrics.loc[mask, f"spread_D{D}"]
            if g_quintiles and arch in g_quintiles and D in g_quintiles[arch]:
                cuts = g_quintiles[arch][D]
                df_metrics.loc[mask, col_q] = quintile_with_cuts(vals, cuts).values
            else:
                df_metrics.loc[mask, col_q] = quintile_rank(vals).values
        df_metrics[col_c] = [node_color(a, q)
                              for a, q in zip(df_metrics["archetype"], df_metrics[col_q])]

    # Back-compat aliases used elsewhere (spread / q_rank / color = default D)
    df_metrics["q_rank"] = df_metrics[f"q_rank_D{DEFAULT_DURATION}"]
    df_metrics["color"] = df_metrics[f"color_D{DEFAULT_DURATION}"]

    df_metrics["sqrt_size"] = np.sqrt(df_metrics["size_node"].clip(lower=0))
    # If multiple months exist, normalize against the GLOBAL sqrt-size range
    # so dollar-equivalent constraints render at the same pixel size across
    # months (otherwise a single outlier month can make all other months
    # look like tiny dots).
    all_months_for_scaling = discover_months()
    g = global_size_range(all_months_for_scaling)
    if g is not None:
        s_min, s_max = g
    else:
        s_min = float(df_metrics["sqrt_size"].min())
        s_max = float(df_metrics["sqrt_size"].max())
    if s_max > s_min:
        # Clip so freak-outlier nodes (above the global p99) cap at MAX_PIXEL
        # rather than blowing past it.
        df_metrics["pixel_size"] = (
            MIN_PIXEL +
            ((df_metrics["sqrt_size"] - s_min) / (s_max - s_min)).clip(lower=0, upper=1)
            * (MAX_PIXEL - MIN_PIXEL)
        )
    else:
        df_metrics["pixel_size"] = (MIN_PIXEL + MAX_PIXEL) / 2

    # 'placed' = the metrics-ready nodes that ALSO have coordinates
    placed = df_metrics.dropna(subset=["Latitude", "Longitude"]).copy()
    print(f"precise-placed (for map): {len(placed):,}")
    print(f"unplaced but on bar chart: {len(df_metrics) - len(placed):,}")

    # ----------------------------------------------------------------------
    # 5. Self-checks
    # ----------------------------------------------------------------------
    nan_cols = ["Latitude", "Longitude", "color", "pixel_size"]
    for c in nan_cols:
        assert placed[c].notna().all(), f"BLOCKER: NaN in {c}"

    print(f"about to render {len(placed):,} nodes")
    print(f"archetype counts (rendered):\n{placed['archetype'].value_counts().to_string()}")
    print(f"marker pixel size range: {placed['pixel_size'].min():.1f} .. {placed['pixel_size'].max():.1f}")

    # ----------------------------------------------------------------------
    # 6. Build Plotly figure — one trace per archetype × marker variant
    # ----------------------------------------------------------------------
    fig = go.Figure()

    # Plot order: WHITE first (background) → BLUE/RED → PURPLE (foreground)
    order = ["WHITE", "RED", "BLUE", "PURPLE"]

    # Hover template — spreads for all 3 durations always shown so the user
    # sees sensitivity regardless of which D is selected for coloring.
    hover_tpl = (
        "<b>%{customdata[0]}</b><br>"
        "<b>Archetype:</b> %{customdata[1]}<br>"
        "<b>Spread (per MWh):</b> 2h $%{customdata[11]:,d}  ·  4h $%{customdata[2]:,d}  ·  8h $%{customdata[12]:,d}<br>"
        "<b>Size:</b> $%{customdata[4]:,.0f}<br>"
        "<b>Controlling line:</b> %{customdata[5]}<br>"
        "<b>Rating:</b> %{customdata[6]} MW (%{customdata[7]})<br>"
        "<b>Concentration:</b> %{customdata[8]:.3f}  ·  <b>Marker:</b> %{customdata[9]}<br>"
        "<b>EIA plant:</b> %{customdata[10]}<br>"
        "<b>Coordinates:</b> %{lat:.4f}, %{lon:.4f}"
        "<extra></extra>"
    )

    # Track per-trace per-duration color arrays so the D selector can restyle.
    # Each entry is what marker.color should be for that trace at that D.
    trace_colors_by_d: dict[int, list] = {D: [] for D in DURATIONS}

    for arch in order:
        sub = placed[placed["archetype"] == arch]
        if sub.empty:
            continue
        # Customdata layout:
        # 0: node, 1: archetype-descriptor, 2: spread_D4 (default), 3: q_rank_D4,
        # 4: size, 5: controlling line, 6: rating, 7: rating source,
        # 8: conc, 9: marker, 10: EIA plant,
        # 11: spread_D2, 12: spread_D8
        cd = np.column_stack([
            sub.index.astype(str).values,
            sub["archetype"].map(ARCHETYPE_LABEL).values,
            np.ceil(sub["spread_D4"]).astype(int).values,
            sub["q_rank_D4"].values.astype(int),
            sub["size_node"].fillna(0).values,
            sub["kstar_physical_line"].fillna("(unknown)").values,
            sub["kstar_rating_MW"].fillna(-1).values,
            sub["kstar_rating_source"].fillna("(none)").values,
            sub["conc"].values,
            sub["marker"].values,
            sub["Plant Name"].fillna("(unmatched)").values,
            np.ceil(sub["spread_D2"]).astype(int).values,
            np.ceil(sub["spread_D8"]).astype(int).values,
        ])
        # Filled trace
        filled = sub[sub["marker"] == "filled"]
        hollow = sub[sub["marker"] == "hollow"]

        label = ARCHETYPE_LABEL[arch]
        if not filled.empty:
            mask = sub["marker"] == "filled"
            fig.add_trace(go.Scattermap(
                lat=filled["Latitude"],
                lon=filled["Longitude"],
                mode="markers",
                marker=dict(size=filled["pixel_size"],
                             color=filled[f"color_D{DEFAULT_DURATION}"].tolist()),
                customdata=cd[mask.values],
                hovertemplate=hover_tpl,
                name=f"{label} — filled ({len(filled)})",
                legendgroup=arch,
            ))
            for D in DURATIONS:
                trace_colors_by_d[D].append(filled[f"color_D{D}"].tolist())

        if not hollow.empty:
            mask = sub["marker"] == "hollow"
            fig.add_trace(go.Scattermap(
                lat=hollow["Latitude"],
                lon=hollow["Longitude"],
                mode="markers",
                marker=dict(size=hollow["pixel_size"],
                             color=hollow[f"color_D{DEFAULT_DURATION}"].tolist()),
                customdata=cd[mask.values],
                hovertemplate=hover_tpl,
                name=f"{label} — hollow ({len(hollow)})",
                legendgroup=arch,
            ))
            for D in DURATIONS:
                trace_colors_by_d[D].append(hollow[f"color_D{D}"].tolist())
            fig.add_trace(go.Scattermap(
                lat=hollow["Latitude"],
                lon=hollow["Longitude"],
                mode="markers",
                marker=dict(size=hollow["pixel_size"] * 0.55, color="white"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=arch,
            ))
            for D in DURATIONS:
                trace_colors_by_d[D].append("white")

    # Center the view on California.
    # NB: duration buttons live in HTML (over the map div) — see build_page.
    # The bottom-left #legend box is the authoritative key (consistent across
    # months and durations). Plotly's auto-legend would sample per-trace
    # marker colors/sizes that change with the underlying data — misleading
    # because color/size are per-node encodings, not per-trace classifications.
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=36.5, lon=-119.5),
            zoom=5.4,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    # Quintile bounds for the README gradient bars. Use global pooled
    # bounds (across all months) if multi-month, otherwise this month's.
    # We display the DEFAULT_DURATION (4 h) bucket in the README.
    quintile_bounds = {}
    if g_quintiles is not None:
        for arch in ARCHETYPE_CMAP:
            cuts = g_quintiles.get(arch, {}).get(DEFAULT_DURATION)
            quintile_bounds[arch] = cuts
    else:
        quintile_qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for arch in ARCHETYPE_CMAP:
            sub = df_metrics.loc[df_metrics["archetype"] == arch, "spread"]
            if len(sub) >= 5:
                quintile_bounds[arch] = sub.quantile(quintile_qs).tolist()
            else:
                quintile_bounds[arch] = None

    all_months = discover_months()
    month_nav = build_month_nav(SEASON_TAG, all_months)
    print(f"available months found: {len(all_months)} → month-nav " +
          ("rendered" if month_nav else "skipped (single month)"))

    out_html = Path(page_filename(SEASON_TAG))
    map_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="map")
    bar_fig = build_bar_chart(df_metrics)
    bar_html = bar_fig.to_html(include_plotlyjs=False, full_html=False, div_id="bar-chart")
    page = build_page(map_html, bar_html,
                       n_rendered=len(placed), n_total=total_nodes,
                       n_bar=min(BAR_TOP_N, len(df_metrics)),
                       n_metrics=len(df_metrics),
                       quintile_bounds=quintile_bounds,
                       trace_colors_by_d=trace_colors_by_d,
                       month_nav_html=month_nav,
                       current_tag=SEASON_TAG)
    out_html.write_text(page)
    print(f"saved {out_html.resolve()}")
    # index.html mirrors the FULL YEAR view (default landing page).
    # If the Full Year aggregate hasn't been built yet, fall back to the
    # latest available month so the site still has a root page.
    index_target = FULL_YEAR_TAG if FULL_YEAR_TAG in all_months else (
        all_months[-1] if all_months else None
    )
    if SEASON_TAG == index_target:
        Path("index.html").write_text(page)
        print(f"saved index.html (mirror of {index_target}, for GitHub Pages root)")

    # ----------------------------------------------------------------------
    # 7. Spike-spread side panel
    # ----------------------------------------------------------------------
    spike_threshold = placed["spread"].quantile(SPIKE_PERCENTILE)
    spikes = placed[placed["spread"] >= spike_threshold].sort_values("spread", ascending=False)
    spike_cols = ["archetype", "spread", "size_node", "kstar_physical_line",
                  "kstar_rating_MW", "Latitude", "Longitude", "Plant Name", "conc"]
    out_spike = DATA_DIR / f"spike_spread_nodes_{SEASON_TAG}.csv"
    spikes[spike_cols].to_csv(out_spike)
    print(f"saved {out_spike}  ({len(spikes)} nodes with spread >= ${spike_threshold:.0f})")

    # ----------------------------------------------------------------------
    # 8. PROBE_IMPORT sanity
    # ----------------------------------------------------------------------
    print()
    if config.PROBE_IMPORT in placed.index:
        r = placed.loc[config.PROBE_IMPORT]
        print(f"PROBE_IMPORT  {config.PROBE_IMPORT}:")
        print(f"  archetype:     {r['archetype']}")
        print(f"  q_rank:        {int(r['q_rank'])}")
        print(f"  spread:        ${r['spread']:.2f}")
        print(f"  size:          ${r['size_node']:,.0f}")
        print(f"  k*:            {r['kstar_physical_line']}")
        print(f"  k* rating:     {r['kstar_rating_MW']} MW ({r['kstar_rating_source']})")
        print(f"  marker:        {r['marker']}")
        print(f"  color:         {r['color']}")
        print(f"  pixel_size:    {r['pixel_size']:.1f}")
        print(f"  lat,lon:       {r['Latitude']:.4f}, {r['Longitude']:.4f}")
        print(f"  EIA plant:     {r['Plant Name']}")
    else:
        print(f"WARN: PROBE_IMPORT not in rendered set")

    print()
    print("F1 OK.")


if __name__ == "__main__":
    main()
