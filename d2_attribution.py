"""D2 — empirical shift-factor regression for PROBE_IMPORT.

Per brief §3c:
  MCC_node(t) ≈ Σ_k β_{node,k} * μ_k(t)
Fit β. Controlling constraint k* = argmax_k mean_t(|β_k * μ_k(t)|).

We fit three models for comparison:
  1. OLS — exposes how degenerate collinear cancellations look
  2. Ridge (L2) — stabilizes collinear coefficients
  3. Lasso (L1) — drives degenerate coefficients to zero, sparse attribution

The trust gate is on the controlling constraint k* being a plausible LA Basin
element under the most stable model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

import config

DATA_DIR = Path("data")
SEASON_TAG = "summer2025"


def load_probe_mcc(node: str) -> pd.Series:
    cache = DATA_DIR / f"rtm_5min_{node}_{SEASON_TAG}.parquet"
    df = pd.read_parquet(cache)
    s = df[["Interval Start", "Congestion"]].copy()
    s["Interval Start"] = pd.to_datetime(s["Interval Start"], utc=True)
    return s.set_index("Interval Start")["Congestion"].sort_index()


def load_panel() -> pd.DataFrame:
    p = DATA_DIR / f"shadow_panel_wide_{SEASON_TAG}.parquet"
    df = pd.read_parquet(p)
    # Index should be UTC tz-aware
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


print("== D2: attribution regression ==")
print(f"probe: {config.PROBE_IMPORT}")

mcc = load_probe_mcc(config.PROBE_IMPORT)
panel = load_panel()

print(f"probe series:  {len(mcc):,} intervals, [{mcc.index.min()} .. {mcc.index.max()}]")
print(f"shadow panel:  {panel.shape[0]:,} intervals × {panel.shape[1]:,} constraints, "
      f"[{panel.index.min()} .. {panel.index.max()}]")

# Align on intersection
common = mcc.index.intersection(panel.index)
print(f"aligned intervals: {len(common):,}")

mcc = mcc.loc[common]
X_df = panel.loc[common]

# Drop columns that became all-zero on this subset (defensive)
nonzero_cols = X_df.columns[(X_df.abs().sum(axis=0) > 0)]
dropped = X_df.shape[1] - len(nonzero_cols)
if dropped:
    print(f"dropping {dropped} columns that are all-zero on aligned subset")
X_df = X_df[nonzero_cols]

K = X_df.shape[1]
T = len(mcc)
print(f"regression matrix: {T:,} rows × {K:,} columns")

X = X_df.values
y = mcc.values
mean_abs_mu = np.abs(X).mean(axis=0)
constraint_keys = list(X_df.columns)


def fit_and_attribute(name: str, model, X, y, keys, mean_abs_mu):
    model.fit(X, y)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / len(y)))

    contrib = np.abs(beta[None, :] * X).mean(axis=0)
    attrib = pd.DataFrame({
        "beta":              beta,
        "mean_abs_mu":       mean_abs_mu,
        "mean_abs_contrib":  contrib,
    }, index=keys)
    total = attrib["mean_abs_contrib"].sum()
    attrib["share_of_total"] = attrib["mean_abs_contrib"] / total if total > 0 else 0
    attrib["nonzero_beta"] = attrib["beta"].abs() > 1e-9
    attrib = attrib.sort_values("mean_abs_contrib", ascending=False)

    print()
    print(f"=========== {name} ===========")
    print(f"intercept: {intercept:+.4f} $/MWh   R²: {r2:.4f}   RMSE: {rmse:.3f} $/MWh")
    print(f"nonzero β: {int(attrib['nonzero_beta'].sum())} / {len(beta)}")
    print(f"top 10 by mean |β·μ|:")
    print(attrib.head(10).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"top 5 cumulative share: {attrib.head(5)['share_of_total'].sum():.1%}")
    return attrib, r2


print()
print(f"y stats — MCC mean: {float(np.mean(y)):+.3f}   std: {float(np.std(y)):.3f}")

# 1. OLS
ols_attrib, ols_r2 = fit_and_attribute("OLS", LinearRegression(), X, y, constraint_keys, mean_abs_mu)

# 2. Ridge (modest regularization)
ridge_attrib, ridge_r2 = fit_and_attribute("Ridge α=10", Ridge(alpha=10.0, fit_intercept=True),
                                            X, y, constraint_keys, mean_abs_mu)

# 3. Lasso (sparse) — need stronger alpha to drive cancellation coefs to zero
lasso_attrib, lasso_r2 = fit_and_attribute("Lasso α=0.05", Lasso(alpha=0.05, fit_intercept=True, max_iter=20000),
                                            X, y, constraint_keys, mean_abs_mu)

# Save the Lasso attribution (most interpretable)
out = DATA_DIR / f"attribution_{config.PROBE_IMPORT}_{SEASON_TAG}.csv"
lasso_attrib.to_csv(out)
print(f"\nsaved Lasso attribution -> {out}")

print()
print("== gate summary ==")
print(f"  OLS:   R²={ols_r2:.4f}   k* = {ols_attrib.index[0]}")
print(f"  Ridge: R²={ridge_r2:.4f}   k* = {ridge_attrib.index[0]}")
print(f"  Lasso: R²={lasso_r2:.4f}   k* = {lasso_attrib.index[0]}")


# --- Aggregate by physical line (drop the |contingency suffix) ---
def physical_line(key: str) -> str:
    # keys are "TYPE:line|contingency" — drop the contingency part
    if "|" not in key:
        return key
    head, _, _ = key.partition("|")
    return head


print()
print("== by-physical-line aggregation (Lasso, contingency scenarios summed) ==")
by_line = lasso_attrib.copy()
by_line["physical_line"] = [physical_line(k) for k in by_line.index]
agg = by_line.groupby("physical_line").agg(
    n_scenarios=("beta", "size"),
    sum_contrib=("mean_abs_contrib", "sum"),
).sort_values("sum_contrib", ascending=False)
agg["share"] = agg["sum_contrib"] / agg["sum_contrib"].sum()
print(agg.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

# Save by-line aggregation
out2 = DATA_DIR / f"attribution_byline_{config.PROBE_IMPORT}_{SEASON_TAG}.csv"
agg.to_csv(out2)
print(f"\nsaved {out2}")

print()
print(f"controlling physical line k* = {agg.index[0]}")
