"""
Step 10-13: Sub-Analysis 4 - DASS Range Restriction Correction
===============================================================
Tests if DASS null findings are artifacts of range restriction.
Applies Thorndike Case II correction with bootstrap CIs.

Data sources:
  - RQ 7.5.1 step03: theta_all (aggregated accuracy)
  - dfnonvr.csv: DASS subscale raw scores
  - Henry & Crawford (2005) community norms (hardcoded)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Path Setup ──────────────────────────────────────────────────────────────
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def make_log(step_name):
    log_file = RQ_DIR / "logs" / f"{step_name}.log"
    def log(msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{msg}\n")
            f.flush()
        print(msg, flush=True)
    return log


# ── Henry & Crawford (2005) Community Norms ──────────────────────────────────
POPULATION_NORMS = {
    'depression': {'mean': 5.55, 'sd': 7.48},
    'anxiety':    {'mean': 3.56, 'sd': 5.39},
    'stress':     {'mean': 9.27, 'sd': 8.04},
}


# ══════════════════════════════════════════════════════════════════════════════
# Extract DASS Observed Correlations and Descriptives
# ══════════════════════════════════════════════════════════════════════════════

log10 = make_log("step10_dass_observed_correlations")

log10("=" * 70)
log10("STEP 10: Extract DASS Observed Correlations and Descriptives")
log10("=" * 70)

# Load theta from RQ 7.5.1
theta_path = PROJECT_ROOT / "results" / "ch7" / "7.5.1" / "data" / "step02_theta_scores.csv"
assert theta_path.exists(), f"QUIT: {theta_path} not found"
df_theta = pd.read_csv(theta_path)
log10(f"Loaded 7.5.1 theta: {df_theta.shape}, columns: {df_theta.columns.tolist()}")

# Load DASS from dfnonvr.csv
dfnonvr = pd.read_csv(PROJECT_ROOT / "data" / "dfnonvr.csv")

# DASS column mapping
dass_cols = {
    'depression': 'total-dass-depression-items',
    'anxiety': 'total-dass-anxiety-items',
    'stress': 'total-dass-stress-items',
}

# Verify columns exist
for name, col in dass_cols.items():
    assert col in dfnonvr.columns, f"QUIT: {col} not found in dfnonvr.csv"

# Merge theta with DASS
df_dass = dfnonvr[['UID'] + list(dass_cols.values())].copy()
df_dass['UID'] = df_dass['UID'].astype(str)
df_theta['UID'] = df_theta['UID'].astype(str)

# theta column name
theta_col = 'theta_all' if 'theta_all' in df_theta.columns else df_theta.columns[1]
log10(f"Using theta column: {theta_col}")

df_merged = df_theta[['UID', theta_col]].merge(df_dass, on='UID', how='inner').dropna()
N = len(df_merged)
log10(f"Merged DASS + theta: N={N}")

# Compute correlations and descriptives
results = []
for subscale, col_name in dass_cols.items():
    dass_vals = df_merged[col_name].values
    theta_vals = df_merged[theta_col].values

    r, p = stats.pearsonr(dass_vals, theta_vals)
    results.append({
        'dass_subscale': subscale,
        'r_observed': r,
        'p': p,
        'N': N,
        'sample_mean': np.mean(dass_vals),
        'sample_sd': np.std(dass_vals, ddof=1),
    })
    log10(f"  {subscale}: r={r:.6f}, p={p:.6f}, M={np.mean(dass_vals):.2f}, SD={np.std(dass_vals, ddof=1):.2f}")

df_obs = pd.DataFrame(results)
df_obs.to_csv(RQ_DIR / "data" / "step10_dass_observed_stats.csv", index=False)
log10(f"DASS observed correlations computed: N=3")
log10(f"Saved: step10_dass_observed_stats.csv (3 rows)")

# Validation
assert len(df_obs) == 3
assert df_obs['r_observed'].between(-1, 1).all()
assert (df_obs['sample_sd'] > 0).all(), "QUIT: Zero variance in DASS subscale(s)"
log10("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Compute Restriction Ratios (u)
# ══════════════════════════════════════════════════════════════════════════════

log11 = make_log("step11_restriction_ratios")

log11("=" * 70)
log11("STEP 11: Compute Restriction Ratios (u)")
log11("=" * 70)

ratios = []
for _, row in df_obs.iterrows():
    subscale = row['dass_subscale']
    pop_sd = POPULATION_NORMS[subscale]['sd']
    u = row['sample_sd'] / pop_sd
    ratios.append({
        'dass_subscale': subscale,
        'sample_sd': row['sample_sd'],
        'population_sd': pop_sd,
        'restriction_ratio_u': u,
    })
    log11(f"  {subscale}: u = {row['sample_sd']:.3f} / {pop_sd:.3f} = {u:.4f}")

df_ratios = pd.DataFrame(ratios)
df_ratios.to_csv(RQ_DIR / "data" / "step11_restriction_ratios.csv", index=False)
log11(f"Restriction ratios computed: N=3")
mean_u = df_ratios['restriction_ratio_u'].mean()
log11(f"Mean u = {mean_u:.4f}")
log11(f"Saved: step11_restriction_ratios.csv (3 rows)")

# Validation
for _, row in df_ratios.iterrows():
    assert row['sample_sd'] > 0
    assert row['population_sd'] > 0
    assert 0 < row['restriction_ratio_u'] <= 2
    if row['restriction_ratio_u'] > 1.0:
        log11(f"WARNING: u > 1.0 for {row['dass_subscale']} (sample MORE variable than population) - unexpected")
    if row['restriction_ratio_u'] < 0.5:
        log11(f"Note: u < 0.5 indicates substantial range restriction for {row['dass_subscale']}")

log11("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Apply Thorndike Case II Correction
# ══════════════════════════════════════════════════════════════════════════════

log12 = make_log("step12_thorndike_correction")

log12("=" * 70)
log12("STEP 12: Apply Thorndike Case II Correction")
log12("=" * 70)

corrected = []
for i, row in df_obs.iterrows():
    subscale = row['dass_subscale']
    r_obs = row['r_observed']
    u = df_ratios.loc[df_ratios['dass_subscale'] == subscale, 'restriction_ratio_u'].values[0]

    # Thorndike Case II: r_corrected = r_observed / u
    r_corr = r_obs / u if abs(u) > 1e-10 else r_obs

    # Interpretation
    abs_r = abs(r_corr)
    if abs_r < 0.15:
        interp = 'negligible'
    elif abs_r < 0.20:
        interp = 'small'
    elif abs_r < 0.30:
        interp = 'moderate'
    else:
        interp = 'large'

    corrected.append({
        'dass_subscale': subscale,
        'r_observed': r_obs,
        'r_corrected': r_corr,
        'restriction_ratio_u': u,
        'interpretation': interp,
    })
    log12(f"  {subscale}: r_obs={r_obs:.6f} / u={u:.4f} = r_corr={r_corr:.6f} ({interp})")

df_corr = pd.DataFrame(corrected)
df_corr.to_csv(RQ_DIR / "data" / "step12_dass_corrected_correlations.csv", index=False)
log12(f"Thorndike Case II correction applied: N=3")
mean_abs_r = df_corr['r_corrected'].abs().mean()
log12(f"Mean |r_corrected| = {mean_abs_r:.4f}")
log12(f"Saved: step12_dass_corrected_correlations.csv (3 rows)")

# Validation
assert len(df_corr) == 3
for _, row in df_corr.iterrows():
    assert abs(row['r_corrected']) >= abs(row['r_observed']) - 1e-10, \
        f"|r_corrected| should >= |r_observed| for {row['dass_subscale']}"
    if abs(row['r_corrected']) > 2.0:
        log12(f"WARNING: Extreme corrected correlation for {row['dass_subscale']}: {row['r_corrected']:.4f}")

log12("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals for Corrected Correlations
# ══════════════════════════════════════════════════════════════════════════════

log13 = make_log("step13_bootstrap_corrected_ci")

log13("=" * 70)
log13("STEP 13: Bootstrap CIs for Corrected Correlations")
log13("=" * 70)

B = 5000
rng = np.random.RandomState(42)

# Prepare data arrays
theta_vals = df_merged[theta_col].values
dass_arrays = {}
for subscale, col_name in dass_cols.items():
    dass_arrays[subscale] = df_merged[col_name].values

bootstrap_iterations = []

log13(f"Running bootstrap with B={B} iterations...")

for b_iter in range(B):
    idx = rng.choice(N, size=N, replace=True)
    theta_b = theta_vals[idx]

    row = {'iteration': b_iter + 1}
    for subscale, dass_vals in dass_arrays.items():
        dass_b = dass_vals[idx]

        # Observed correlation in bootstrap sample
        r_obs_b = np.corrcoef(dass_b, theta_b)[0, 1]

        # Sample SD in bootstrap sample
        sd_b = np.std(dass_b, ddof=1)

        # Restriction ratio
        pop_sd = POPULATION_NORMS[subscale]['sd']
        u_b = sd_b / pop_sd if pop_sd > 0 else 1.0

        # Corrected correlation
        r_corr_b = r_obs_b / u_b if abs(u_b) > 1e-10 else r_obs_b

        row[f'r_corrected_{subscale}'] = r_corr_b

    bootstrap_iterations.append(row)

    if (b_iter + 1) % 1000 == 0:
        log13(f"  Progress: {b_iter+1}/{B} iterations")

log13(f"Bootstrap correction completed: {len(bootstrap_iterations)} iterations")

# Save bootstrap
df_boot = pd.DataFrame(bootstrap_iterations)
df_boot.to_csv(RQ_DIR / "data" / "step13_dass_bootstrap_corrected.csv", index=False)
log13(f"Saved: step13_dass_bootstrap_corrected.csv ({len(df_boot)} rows)")

# Compute CIs
ci_results = []
for subscale in ['depression', 'anxiety', 'stress']:
    col = f'r_corrected_{subscale}'
    vals = df_boot[col].dropna().values
    point = df_corr.loc[df_corr['dass_subscale'] == subscale, 'r_corrected'].values[0]
    ci_lower = np.percentile(vals, 2.5)
    ci_upper = np.percentile(vals, 97.5)

    ci_results.append({
        'dass_subscale': subscale,
        'r_corrected_point': point,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    })
    log13(f"  {subscale} r_corrected CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Check for extreme values
    n_extreme = np.sum(np.abs(vals) > 1.5)
    if n_extreme > 0.05 * B:
        log13(f"  WARNING: Wide CIs suggest correction uncertainty for {subscale}")

df_ci = pd.DataFrame(ci_results)
df_ci.to_csv(RQ_DIR / "data" / "step13_dass_corrected_ci.csv", index=False)
log13(f"Saved: step13_dass_corrected_ci.csv (3 rows)")

# Validation
assert len(df_boot) == B, f"Expected {B} rows, got {len(df_boot)}"
assert len(df_ci) == 3, f"Expected 3 rows in CI file, got {len(df_ci)}"
for _, row in df_ci.iterrows():
    assert row['ci_lower'] < row['r_corrected_point'] < row['ci_upper'], \
        f"Point estimate outside CI for {row['dass_subscale']}"

log13("Validation PASSED")

log13("=" * 70)
log13("SUB-ANALYSIS 4 COMPLETE")
log13("=" * 70)
