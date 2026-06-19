# Build Brief: CAISO Battery Congestion-Relief Node Map

## Objective

Build an interactive geographic map of CAISO pricing nodes that identifies **where battery storage could relieve transmission congestion**. Each node is encoded by four channels (hue, color intensity, circle size, marker style) derived from the **congestion component of the locational price**, not the full price. The map is a *screening tool*: it ranks candidate locations from public data. It does not confirm deliverable relief — that requires a network-model run, out of scope here.

Primary user: energy researcher evaluating storage siting (LA Basin import-pocket analog). Treat this as a research instrument, not a production app.

---

## 1. The single most important modeling decision

Use the **marginal congestion component (MCC)**, never the total LMP.

Locational marginal price decomposes additively:

```
LMP_node(t) = MEC(t) + MCC_node(t) + MLC_node(t)
```

- **MEC** (marginal energy component): one systemwide scalar, identical at every node. Carries the whole day/night energy swing. Mapping this would make nearly every node dark and tell you nothing about transmission.
- **MCC** (marginal congestion component): the only part that varies by location because a wire is constrained. **This is the signal.**
- **MLC** (marginal loss component): smaller, location-varying; ignore for v1.

Pull MCC as its own field. Do not subtract or reconstruct it from total LMP.

---

## 2. Final encoding spec (what to draw)

| Channel | Encodes | Rule |
|---|---|---|
| **Hue** | Flow direction / archetype | White = no local congestion; Blue = import pocket (evening MCC high); Red = export pocket (midday MCC negative); Purple = bidirectional (both) |
| **Color intensity** (lightness/alpha of the hue) | Round-trip MCC spread ($/MWh) | Higher spread = more saturated |
| **Circle size** (area) | Annual congestion rent of the node's controlling constraint ($/yr) | Bigger = larger prize. Area ∝ value (use sqrt radius scaling) |
| **Marker style** | Bankability (value concentration) | Filled = persistent/bankable; hollow ring = rare-spike-driven |

Note on marker style: literal dashed outlines are **not supported on map point markers** in Plotly. Use **filled circle vs. hollow ring** as the binary substitute for solid vs. dashed. This is faithful to the intent (flag the spike-driven outliers) — implement it, don't fake a dashed border.

(Two encodings from earlier design rounds are intentionally cut: contingency-vs-base-case differentiation, and measured-vs-proxy-rating fill style. Do not implement either.)

---

## 3. Core formulas

Define per node, over a chosen date range (run per season — see §5).

### 3a. Round-trip MCC spread (drives color intensity)

For battery duration `D` hours and round-trip efficiency `η` (default 0.85), for each day `d`:

```
discharge_value(d) = sum of the D highest hourly MCC values that day
charge_cost(d)     = sum of the D lowest  hourly MCC values that day
daily_spread(d)    = discharge_value(d) − (1/η) * charge_cost(d)
```

```
spread_node = median over days of daily_spread(d)
```

MCC can be **negative** (export congestion / trapped power). The formula handles it correctly: charging in negative-MCC hours makes `charge_cost` negative, so the `−(1/η)*charge_cost` term *adds* value. Do not clip MCC at zero.

### 3b. Archetype (drives hue)

Compute the average MCC by hour-of-day for the node, then two window means (windows are parameters):

```
MCC_mid = mean MCC over midday window  (default 10:00–15:00)
MCC_eve = mean MCC over evening window (default 17:00–21:00)

import_signal = MCC_eve            (positive: importing into a binding constraint at peak)
export_signal = −MCC_mid           (positive: negative midday MCC = export/oversupply congestion)
```

Classify with a flatness threshold `ε` (param, e.g. $3/MWh):

```
both signals < ε                       → WHITE  (no local congestion; swing is pure energy)
import_signal ≥ ε, export_signal < ε   → BLUE   (import-constrained load pocket)
export_signal ≥ ε, import_signal < ε   → RED    (export-constrained generation pocket)
both signals ≥ ε                       → PURPLE (bidirectional / double-duty — highest value)
```

### 3c. Controlling-constraint attribution (needed for size)

Goal: attach each node to the transmission constraint that drives its congestion, **without a network model**, using public price data.

Method — empirical shift-factor regression. The nodal MCC is, by construction, a sum over binding constraints of (shift factor × shadow price):

```
MCC_node(t) ≈ Σ_k  β_{node,k} * μ_k(t)
```

where `μ_k(t)` is the shadow price time series of binding constraint `k` (from OASIS — see §4). Regress each node's MCC on the panel of constraint shadow prices; the fitted coefficient `β_{node,k}` is an empirical estimate of the shift factor (PTDF). The node's **controlling constraint** `k*` is the one with the largest average contribution `|β_{node,k} * μ_k|`.

Fallback if shadow-price panel is unavailable or regression is too collinear: assign `k*` by **temporal coincidence** — the constraint most frequently binding during the node's own high-MCC hours.

Flag both methods as approximate. Do not present `β` as a true PTDF.

### 3d. Circle size (annual congestion rent of the prize)

Congestion rent on a constraint = shadow price × its flow limit (rating). Summed over the year:

```
constraint_rent_k = rating_k * Σ_t μ_k(t)          # rating × integrated shadow price
size_node         = constraint_rent_{k*}            # k* = node's controlling constraint
```

Nodes sharing a controlling constraint will get similar sizes — correct; they compete to relieve the same wire.

`rating_k`:
- If `k*` matches a named WECC path → use the public **WECC Path Rating Catalog** value (2026 public version; ratings in MW).
- Else → **voltage-class proxy**: map the local line's kV to a nominal MVA (e.g. 69 kV→150, 115 kV→250, 230 kV→700, 500 kV→2500 — tune these). Voltage class is public; exact branch ratings are CEII.

Render `marker_radius ∝ sqrt(size_node)` so area encodes value. Cap/clip outliers for legibility.

### 3e. Bankability (drives filled vs. hollow marker)

```
conc_node = (sum of MCC-congestion value in the top 1% of hours) / (total annual MCC value)
hollow (spike-driven) if conc_node > threshold (param, default 0.5); else filled
```

---

## 4. Data sources

| Need | Source | How | Confidence |
|---|---|---|---|
| Nodal LMP **with MCC component** | CAISO OASIS | `gridstatus` Python lib (CAISO LMP pull; request the congestion component) — reports `PRC_LMP` (day-ahead) / `PRC_INTVL_LMP` (real-time). **Verify field names and that MCC is exposed separately before building.** | moderate |
| Binding constraints + **shadow prices** (μ_k) | CAISO OASIS | `gridstatus` if it wraps the constraint report; else the OASIS system/nodal constraint report. **Verify availability.** | moderate |
| **Path ratings** | WECC Path Rating Catalog, public version (PDF, annual; 2026 current) | Parse to a `path_ratings.csv` lookup, or hand-enter the ~80 named paths. Constraint names in the CAISO market do **not** match WECC path names cleanly — name-matching is fuzzy; build a manual crosswalk for the top constraints. | high (catalog is public); moderate (name matching) |
| **Node coordinates** | No public node→lat/long table (CEII) | See §6 — this is the hard part | — |
| Plant coordinates (for generator nodes) | EIA-860 (via PUDL or `gridstatus`/EIA) | Join generator pnode → plant name → EIA-860 lat/long | high |

Do **not** invent OASIS report codes, field names, or node counts. Where uncertain, query a small sample first, print the schema, and adapt.

---

## 5. Parameters (expose as config / CLI)

- `D` — battery duration in hours (default 4; also support 2 and 8)
- `eta` — round-trip efficiency (default 0.85)
- `market` — `RTM` (5-minute) throughout. RTM captures realized congestion and outage spikes that DAM forecasts miss. Resample to hourly for spread/archetype, keep native for spike detection. Never use RUC (no component breakdown).
- `date_range` / `season` — **run per season**, not annual-averaged. California congestion is seasonal (summer evening import vs. spring midday solar glut); annual averaging blends them and hides the purple bidirectional nodes. Default: one representative month per season.
- `midday_window`, `evening_window` — hour ranges for archetype
- `epsilon` — flatness threshold for white classification
- `conc_threshold` — bankability cutoff for hollow markers

---

## 6. The hard part: geolocation (do not silently fake it)

CAISO publishes prices by pricing-node *name*, not coordinates. Plan, in priority order:

1. **Generator pnodes** — parse plant identifier from node name, fuzzy-match to EIA-860 → precise lat/long. Best coverage; do this first.
2. **Aggregation nodes** (DLAPs, sub-LAPs, trading hubs `TH_NP15`/`TH_SP15`/`TH_ZP26`) — place at known regional centroids. Approximate by design.
3. **Remaining internal buses** — likely unplaceable from public data. **Drop them or mark them explicitly as unplaced**; do not assign fake coordinates.

Report coverage in the output (e.g. "1,240 of N nodes placed; M generator nodes precise, K aggregation nodes regional"). Partial maps are acceptable and expected. State the count rather than trusting any node-count figure from memory.

---

## 7. Tech stack

- Python 3.11+, `pandas`, `numpy`
- `gridstatus` — CAISO OASIS pulls (LMP components, constraints)
- `geopandas`, EIA-860 via PUDL or `gridstatus` — geolocation
- **Rendering: Plotly** — `go.Scattermap` (MapLibre backend, **no API token needed** with an open base style). Output a **standalone self-contained HTML** file.
  - Precompute one RGBA hex per node in Python (hue from §3b, saturation/alpha from §3a) — Plotly markers take one color value, so combine hue+intensity yourself.
  - `marker.size` from `sqrt(size_node)`.
  - Filled vs. hollow: two traces, or `marker.symbol` / `marker.opacity` + outline, to get a clear filled-circle vs. open-ring binary.
  - Season selector: Plotly `updatemenus` buttons (one frame per season).
- Alternative if node count is large and Plotly lags: `kepler.gl` (export) or `pydeck`.

---

## 8. Outputs

1. `caiso_congestion_map_<season>.html` — the interactive map (primary deliverable).
2. `node_metrics.csv` — one row per node: name, lat, lon, archetype, spread, size, controlling_constraint, rating, rating_source (path/proxy), conc, placed_flag. Lets the user rank and audit independent of the visual.
3. `coverage_report.txt` — node counts, placement coverage, data date range, parameter values used.

---

## 9. Suggested build order (milestones)

1. **Pipeline on a small set.** Pull one month RTM LMP+MCC for trading hubs + DLAPs + ~50 generator nodes. Compute spread, archetype, bankability. Render. Validate the map renders and colors make sense before scaling.
2. **Attribution + size.** Add the shadow-price pull, the regression attribution (§3c), the path-rating crosswalk, and circle sizing.
3. **Geolocation scale-up.** Add the EIA-860 join for all generator nodes; place aggregation nodes; report coverage.
4. **Seasons + parameter sweep.** Add the season selector and the `D = 2/4/8` duration runs.
5. **Full nodal** only after 1–4 are stable.

---

## 10. Do-not-fabricate checklist

- Do not reconstruct MCC from total LMP — pull the component.
- Do not invent OASIS report codes/fields — sample and inspect first.
- Do not assign coordinates to nodes you cannot place — drop or flag them.
- Do not present regression `β` as a true PTDF, or `μ × rating` as deliverable relief (it is the constraint's total rent / prize size, not the MW one battery captures).
- Do not annual-average away the seasonal signal.
- If a data source is unavailable, stub the step with a clear `TODO` and a printed warning — never silently substitute fake values.
