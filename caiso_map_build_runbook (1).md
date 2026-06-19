# Build Runbook: CAISO Congestion-Relief Node Map (small, checkable steps)

This is the **execution order** for Claude Code. The companion file `caiso_congestion_map_build_brief.md` holds the full rationale and encoding spec; this runbook is what you actually run, in order.

**Working principle.** Build a *vertical slice* first — compute every metric for two nodes I already understand — and only scale after those two nodes come out right. Each step has:
- **Do** — the action (keep each step to one short script).
- **Self-check** — an assertion the agent prints; it must fail loudly, not warn quietly.
- **Inspect** — the artifact I will look at, and what "good" looks like.
- **Gate** (some steps) — a hard stop. Do not proceed until it passes.

**Do-not-fabricate rule (applies to every step):** if a data field, report, or match is missing, stop and print a `BLOCKER:` line describing exactly what's missing. Never substitute placeholder numbers, guessed coordinates, or invented field names to keep moving.

**Market source.** Use **real-time (RTM 5-minute, report `PRC_INTVL_LMP`)** throughout — it reflects realized congestion and the outage-driven spikes the bankability channel is meant to catch. Resample to hourly for the spread and archetype channels; keep native 5-minute for spike/concentration detection. Never use RUC prices — they carry no component breakdown.

---

## Step 0 — Pick ground-truth probe nodes

**Do.** Choose two pricing nodes I can reason about independently:
- `PROBE_IMPORT` — a node in the LA Basin / SP15 import pocket (expected archetype: **blue**, evening congestion).
- `PROBE_FLAT` — a trading hub or interior node expected to be **white** (little local congestion).

Record their exact CAISO node names in a `config.py`. These two nodes are the trust anchor used in Steps A2, B2, D2, F1.

**Inspect.** I confirm the two names are real CAISO nodes and match my expectation before anything is computed.

---

## Phase A — Data reconnaissance (verify the raw inputs are what we assume)

### A1. Environment + one-node pull
**Do.** Install `gridstatus`, `pandas`, `numpy`, `plotly`, `geopandas`. Pull **one day** of real-time (RTM 5-minute) LMP for `PROBE_FLAT`.
**Self-check.** Assert the dataframe is non-empty and has the expected ~288 five-minute rows. Print every column name.
**Inspect.** I read the column list and confirm a **congestion-component column exists and is separate from total LMP**.
**Gate.** If there is no separate congestion/MCC field, stop and print `BLOCKER: MCC component not exposed` — the whole design depends on it.

### A2. Verify the price decomposition holds
**Do.** Sum **all** posted price components and compute `LMP − sum(components)` per hour. CAISO posts MCE, MCC, MCL, and in some markets a GHG component — sum whatever fields are present; do not assume exactly three.
**Self-check.** Assert `max(abs(residual)) < $0.05`. Print the max residual and the names of the components summed.
**Inspect.** I see the identity reconstructs → we are using the real components, not a proxy.

### A3. Pull binding constraints + shadow prices
**Do.** Pull the binding-constraint / shadow-price report for the same day.
**Self-check.** Assert non-empty. Print distinct constraint names and the min/max shadow price.
**Inspect.** I see real constraint names and plausible `$/MWh` shadow prices.
**Gate.** If unavailable, print `BLOCKER: shadow-price panel missing` and note that attribution (Phase D) will fall back to temporal coincidence. Do not silently skip.

### A4. Inventory the scope
**Do.** List the count of distinct pricing nodes available and the queryable date range.
**Self-check.** Assert counts > 0.
**Inspect.** I read the **actual node count** (reported, not assumed) and confirm the date range covers the seasons I want.

---

## Phase B — Single-node vertical slice (validate the math on the two probes)

### B1. Pull one representative month for both probes
**Do.** Pull RTM MCC for `PROBE_IMPORT` and `PROBE_FLAT` for one summer month. Resample to hourly for the metrics; retain the native 5-minute series for concentration (Step B4).
**Self-check.** Assert ~720 hourly rows each after resampling; assert not all-NaN; print the NaN count.
**Inspect.** Save `probe_mcc_by_hour.png` — mean MCC vs. hour-of-day for both nodes. I eyeball the shape: the import node should rise in the evening; the flat node should stay near zero.

### B2. Archetype (hue) on the probes
**Do.** Compute `MCC_mid`, `MCC_eve`, `import_signal`, `export_signal`, and classify color per the brief (§3b).
**Self-check.** Assert `PROBE_IMPORT` classifies **BLUE**. Print all signals for both nodes.
**Inspect.** I confirm the node I understand is labeled correctly.
**Gate.** If the known import node is not blue, **stop and fix the classifier** — do not scale a miscalibrated rule.

### B3. Round-trip spread (intensity) on the probes
**Do.** Compute `daily_spread` and `spread_node` for both, with `D=4`, `η=0.85` (brief §3a). Keep negative MCC unclipped.
**Self-check.** Assert `spread(PROBE_IMPORT) > spread(PROBE_FLAT)`; assert spread ≥ 0. Print both.
**Inspect.** I check the magnitudes are believable in `$/MWh` (not absurdly large/small).

### B4. Bankability (marker style) on the probes
**Do.** Compute `conc_node` (top-1%-of-hours share) for both (brief §3e).
**Self-check.** Assert `conc` ∈ [0,1]. Print both.
**Inspect.** I cross-check against the B1 plot: a node whose congestion is one rare spike should show high `conc`.

---

## Phase C — Scale the per-node metrics

### C1. Run B1–B4 across all nodes (vectorized)
**Do.** Compute archetype, spread, and concentration for every available node. Write `node_metrics_raw.csv`.
**Self-check.** Assert output row count equals the Step A4 node count. **Count and print dropped/NaN nodes** — do not silently lose them.
**Inspect.** I review three artifacts:
- a histogram of `spread` across nodes (expect most near zero, a thin tail of congested nodes);
- a count of nodes by archetype (expect mostly white, a minority blue/red/purple);
- a table of the top 10 nodes by spread.
If "everything is dark blue," the metric is wrong — likely MCC contamination by energy price; stop.

---

## Phase D — Attribution + circle size

### D1. Build the shadow-price panel
**Do.** Assemble a time × constraint matrix `μ_k(t)` for the month. Write the top constraints by **integrated shadow price** `Σ_t μ_k(t)`.
**Self-check.** Assert panel has > 0 constraints and no all-zero columns. Print the top 15 constraints by integrated shadow price.
**Inspect.** I recognize major CAISO constraints / paths in that top list.

### D2. Empirical shift-factor regression on a probe
**Do.** Regress `MCC(PROBE_IMPORT, t)` on the panel `μ_k(t)`; the coefficients are empirical shift factors `β`. Assign the controlling constraint `k*` = largest mean `|β_k · μ_k|` (brief §3c).
**Self-check.** Print regression R² and the top 5 `β` constraints for the probe.
**Inspect.** I confirm the controlling constraint for the LA Basin probe is a **plausible local/import constraint**, not something electrically distant. This is the attribution trust gate.
**Gate.** If R² is near zero or the top constraint is implausible, switch to the temporal-coincidence fallback and re-inspect before continuing.

### D3. Constraint → rating crosswalk (the weakest link — audit by eye)
**Do.** For the top ~20 constraints by integrated shadow price, build a **manual crosswalk** to WECC Path Rating Catalog (2026 public) ratings in MW; for the rest, assign a voltage-class proxy MVA. Write `constraint_rating_crosswalk.csv` with columns: constraint, matched_path (or `PROXY`), kV, rating_MW, match_confidence.
**Self-check.** Print every row of the crosswalk.
**Inspect.** I **audit the name matches manually** — CAISO constraint names do not match WECC path names cleanly, so I verify each top match is real and flag bad ones. Do not auto-trust fuzzy matches.

### D4. Compute size and attach to nodes
**Do.** `constraint_rent_k = rating_k × Σ_t μ_k(t)`; `size_node = constraint_rent_{k*}` (brief §3d). Run D2's attribution for all nodes.
**Self-check.** Assert size ≥ 0 for all. Print the top 10 nodes by size.
**Inspect.** I confirm the **biggest-prize nodes sit on big paths**, and that a dramatic tiny radial node is *not* among the largest circles (that was the whole point of sizing by rating).

---

## Phase E — Geolocation (partial is fine; fake is not)

### E1. Place generator nodes via EIA-860
**Do.** Parse plant identifiers from generator node names, fuzzy-join to EIA-860, attach lat/long.
**Self-check.** Print the match rate. Assert all matched coordinates fall inside a CA/WECC bounding box.
**Inspect.** Save `placed_gen_nodes.png` — matched nodes on a plain basemap. I check they land on real California geography, not in the ocean.

### E2. Place aggregation nodes; flag the rest
**Do.** Put DLAPs / sub-LAPs / trading hubs at known regional centroids. Mark every remaining node `placed=False`.
**Self-check.** Print a coverage table: precise / regional / unplaced counts. Assert precise + regional + unplaced = total.
**Inspect.** I read `coverage_report.txt` and accept the coverage level (or decide to stop at generator nodes for v1).

---

## Phase F — Render

### F1. Render the four-channel map (one season)
**Do.** Build the Plotly `go.Scattermap` (MapLibre, no token): hue+intensity baked into per-node RGBA, `marker.size ∝ sqrt(size_node)`, filled circle vs. hollow ring for bankability. Output `caiso_congestion_map_summer.html`.
**Self-check.** Assert every placed node appears; assert no NaN in the color or size arrays.
**Inspect.** I open the HTML and find `PROBE_IMPORT`: it must be **blue, sized by its controlling path, filled if persistent**. If the probe looks right on the map, the full pipeline is trustworthy end to end.

### F2. Emit the audit outputs
**Do.** Write `node_metrics.csv` (name, lat, lon, archetype, spread, size, controlling_constraint, rating, rating_source, conc, placed) and finalize `coverage_report.txt` with the parameters used.
**Inspect.** I can sort and rank in the CSV independently of the visual.

---

## Phase G — Seasons + duration (only after F is trusted)

### G1. Per-season runs
**Do.** Re-run C–F for one representative month per season; add a Plotly season selector.
**Inspect.** I compare summer vs. spring: **purple/bidirectional nodes should appear and shift seasonally**. If seasons look identical, the per-season pull is not actually varying — stop.

### G2. Duration sweep
**Do.** Recompute spread and the map for `D = 2, 4, 8` hours.
**Inspect.** I check that longer duration changes which nodes light up (longer-binding constraints reward longer storage).

---

## Trust summary (the four gates that matter most)

1. **A1/A2** — the MCC component is real and separate. Without this, nothing downstream means anything.
2. **B2** — the node I already understand classifies correctly. Calibrates the whole color logic.
3. **D2/D3** — attribution lands on a plausible local constraint, and the rating crosswalk is human-audited (fuzzy name matching is the weakest link).
4. **F1** — the probe node renders correctly on the finished map.

Stop at any gate that fails. Each phase produces a printed check and an artifact I can open, so we never build on an unverified layer.
