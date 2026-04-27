"""
Step 01-02: Sub-Analysis 1 - Formal Dissociation Test
=====================================================
Bootstrap sr² difference CI + Steiger's Z test

Tests if RPM sr² difference (accuracy 0.080 vs confidence 0.042) is significant.

Data sources:
  - RQ 7.1.1 step03: accuracy theta (theta_mean) + cognitive tests
  - RQ 7.3.1 step04: confidence theta (confidence_theta) + cognitive tests

Note: The 4_analysis.yaml incorrectly references 7.3.1/7.3.2. After dependency
verification:
  - Accuracy = theta_mean from RQ 7.1.1 (accuracy prediction RQ)
  - Confidence = confidence_theta from RQ 7.3.1 (confidence prediction RQ)
  - Both use RAVLT_T, BVMT_T, RPM_T as predictors (3-predictor model for fair comparison)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# ── Path Setup ──────────────────────────────────────────────────────────────
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.9.1
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # REMEMVR
LOG_FILE = RQ_DIR / "logs" / "step01_bootstrap_sr2_difference.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# ── Helper: Compute sr² for a predictor ─────────────────────────────────────
def compute_sr2(y, X_full, predictor_idx):
    """Compute semi-partial R² for predictor at predictor_idx using Type II SS."""
    X_with_const = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_with_const).fit()
    r2_full = model_full.rsquared

    # Drop predictor (column predictor_idx from X_full, then add constant)
    X_reduced = np.delete(X_full, predictor_idx, axis=1)
    X_reduced_const = sm.add_constant(X_reduced)
    model_reduced = sm.OLS(y, X_reduced_const).fit()
    r2_reduced = model_reduced.rsquared

    return r2_full - r2_reduced


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Bootstrap sr² Difference
# ══════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("STEP 01: Bootstrap sr² Difference CI")
log("=" * 70)

# ── Load Data ────────────────────────────────────────────────────────────────
# Accuracy theta from RQ 7.1.1 (predicting accuracy with cognitive tests)
accuracy_path = PROJECT_ROOT / "results" / "ch7" / "7.1.1" / "data" / "step03_merged_analysis.csv"
# Confidence theta from RQ 7.3.1 (predicting confidence with cognitive tests)
confidence_path = PROJECT_ROOT / "results" / "ch7" / "7.3.1" / "data" / "step04_analysis_dataset.csv"

assert accuracy_path.exists(), f"QUIT: Accuracy data not found: {accuracy_path}"
assert confidence_path.exists(), f"QUIT: Confidence data not found: {confidence_path}"

df_acc = pd.read_csv(accuracy_path)
df_conf = pd.read_csv(confidence_path)

log(f"Accuracy data: {df_acc.shape} from RQ 7.1.1")
log(f"  Columns: {df_acc.columns.tolist()}")
log(f"Confidence data: {df_conf.shape} from RQ 7.3.1")
log(f"  Columns: {df_conf.columns.tolist()}")

# ── Merge on UID ─────────────────────────────────────────────────────────────
# Select common predictors (3-predictor model for fair comparison)
predictors = ['RAVLT_T', 'BVMT_T', 'RPM_T']

df_acc_sel = df_acc[['UID', 'theta_mean'] + predictors].copy()
df_conf_sel = df_conf[['UID', 'confidence_theta'] + predictors].copy()

# Rename to avoid collision
df_conf_sel = df_conf_sel.rename(columns={p: f"{p}_conf" for p in predictors})

df_merged = df_acc_sel.merge(df_conf_sel, on='UID', how='inner')
df_merged = df_merged.dropna()
N = len(df_merged)

log(f"Merged dataset: N={N} participants (inner join)")
assert N >= 90, f"QUIT: Insufficient overlap N={N} < 90"

# Use accuracy predictors (both should be same T-scores)
X = df_merged[predictors].values
y_acc = df_merged['theta_mean'].values
y_conf = df_merged['confidence_theta'].values

# RPM is the 3rd predictor (index 2)
RPM_IDX = 2

# ── Point Estimates ──────────────────────────────────────────────────────────
sr2_acc_point = compute_sr2(y_acc, X, RPM_IDX)
sr2_conf_point = compute_sr2(y_conf, X, RPM_IDX)
delta_point = sr2_acc_point - sr2_conf_point

log(f"\nPoint Estimates:")
log(f"  RPM sr² (accuracy):   {sr2_acc_point:.6f}")
log(f"  RPM sr² (confidence): {sr2_conf_point:.6f}")
log(f"  Delta (acc - conf):   {delta_point:.6f}")

# ── Bootstrap ────────────────────────────────────────────────────────────────
B = 5000
rng = np.random.RandomState(42)

bootstrap_results = []
n_failed = 0

log(f"\nRunning bootstrap with B={B} iterations...")

for b in range(B):
    idx = rng.choice(N, size=N, replace=True)
    X_b = X[idx]
    y_acc_b = y_acc[idx]
    y_conf_b = y_conf[idx]

    try:
        sr2_acc_b = compute_sr2(y_acc_b, X_b, RPM_IDX)
        sr2_conf_b = compute_sr2(y_conf_b, X_b, RPM_IDX)
        delta_b = sr2_acc_b - sr2_conf_b
        bootstrap_results.append({
            'iteration': b + 1,
            'sr2_accuracy': sr2_acc_b,
            'sr2_confidence': sr2_conf_b,
            'delta': delta_b
        })
    except Exception as e:
        n_failed += 1
        if n_failed >= 10:
            log(f"QUIT: 10+ consecutive bootstrap failures. Last error: {e}")
            raise

    if (b + 1) % 1000 == 0:
        log(f"  Progress: {b+1}/{B} iterations completed")

log(f"Bootstrap completed: {len(bootstrap_results)} iterations ({n_failed} failed)")

# ── Save Bootstrap Deltas ────────────────────────────────────────────────────
df_boot = pd.DataFrame(bootstrap_results)
df_boot.to_csv(RQ_DIR / "data" / "step01_bootstrap_sr2_deltas.csv", index=False)
log(f"Saved: step01_bootstrap_sr2_deltas.csv ({len(df_boot)} rows)")

# ── Compute CIs ──────────────────────────────────────────────────────────────
deltas = df_boot['delta'].values

# Percentile CI
pct_lower = np.percentile(deltas, 2.5)
pct_upper = np.percentile(deltas, 97.5)

# BCa CI
# Bias correction
z0 = stats.norm.ppf(np.mean(deltas < delta_point))

# Acceleration (jackknife)
theta_jack = np.zeros(N)
for i in range(N):
    idx_jack = np.concatenate([np.arange(i), np.arange(i+1, N)])
    X_j = X[idx_jack]
    y_acc_j = y_acc[idx_jack]
    y_conf_j = y_conf[idx_jack]
    sr2_acc_j = compute_sr2(y_acc_j, X_j, RPM_IDX)
    sr2_conf_j = compute_sr2(y_conf_j, X_j, RPM_IDX)
    theta_jack[i] = sr2_acc_j - sr2_conf_j

theta_bar = theta_jack.mean()
a_hat = np.sum((theta_bar - theta_jack)**3) / (6 * np.sum((theta_bar - theta_jack)**2)**1.5 + 1e-10)

# BCa adjusted percentiles
alpha_levels = [0.025, 0.975]
bca_bounds = []
for alpha in alpha_levels:
    z_alpha = stats.norm.ppf(alpha)
    adjusted = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    adjusted = np.clip(adjusted, 0.001, 0.999)
    bca_bounds.append(np.percentile(deltas, adjusted * 100))

bca_lower, bca_upper = bca_bounds[0], bca_bounds[1]

# Proportion positive
prop_positive = np.mean(deltas > 0)

log(f"\nPercentile CI: [{pct_lower:.6f}, {pct_upper:.6f}]")
log(f"BCa CI: [{bca_lower:.6f}, {bca_upper:.6f}]")
log(f"Proportion positive: {prop_positive:.4f}")

# ── Save Summary ─────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'metric': ['point_estimate', 'percentile_ci_lower', 'percentile_ci_upper',
               'bca_ci_lower', 'bca_ci_upper', 'proportion_positive'],
    'value': [delta_point, pct_lower, pct_upper, bca_lower, bca_upper, prop_positive]
})
summary.to_csv(RQ_DIR / "data" / "step01_bootstrap_summary.csv", index=False)
log(f"Saved: step01_bootstrap_summary.csv (6 rows)")

# ── Interpretation ───────────────────────────────────────────────────────────
if pct_lower > 0:
    interp = "SIGNIFICANT: Dissociation confirmed (CI excludes zero)"
elif pct_upper < 0:
    interp = "SIGNIFICANT: Confidence sr² > accuracy sr² (unexpected direction)"
else:
    interp = "NON-SIGNIFICANT: CI includes zero; dissociation is numerical trend only"
log(f"\nInterpretation: {interp}")

# ── Validation ───────────────────────────────────────────────────────────────
assert len(df_boot) == B, f"Expected {B} rows, got {len(df_boot)}"
assert df_boot['sr2_accuracy'].between(0, 1).all(), "sr2_accuracy out of [0,1]"
assert df_boot['sr2_confidence'].between(0, 1).all(), "sr2_confidence out of [0,1]"
assert df_boot['delta'].between(-1, 1).all(), "delta out of [-1,1]"
assert not df_boot['delta'].isna().any(), "NaN in delta column"
assert pct_lower < delta_point < pct_upper, "Point estimate outside percentile CI"
log("Validation PASSED: All bootstrap checks OK")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Steiger's Z Test
# ══════════════════════════════════════════════════════════════════════════════

LOG_FILE_2 = RQ_DIR / "logs" / "step02_steiger_z_test.log"

def log2(msg):
    with open(LOG_FILE_2, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

log2("=" * 70)
log2("STEP 02: Steiger's Z Test for Dependent Correlations")
log2("=" * 70)

# ── Compute Correlations ─────────────────────────────────────────────────────
rpm_vals = df_merged['RPM_T'].values

r_ya = np.corrcoef(rpm_vals, y_acc)[0, 1]  # RPM-accuracy
r_yb = np.corrcoef(rpm_vals, y_conf)[0, 1]  # RPM-confidence
r_ab = np.corrcoef(y_acc, y_conf)[0, 1]     # accuracy-confidence

log2(f"Correlations:")
log2(f"  r_ya (RPM-accuracy):   {r_ya:.6f}")
log2(f"  r_yb (RPM-confidence): {r_yb:.6f}")
log2(f"  r_ab (acc-conf):       {r_ab:.6f}")

# ── Steiger (1980) Formula ───────────────────────────────────────────────────
# Z = (r_ya - r_yb) * sqrt(N-3) / sqrt(2 * (1 - r_ab) * det(R))
# det(R) = 1 - r_ya² - r_yb² - r_ab² + 2*r_ya*r_yb*r_ab
det_R = 1 - r_ya**2 - r_yb**2 - r_ab**2 + 2*r_ya*r_yb*r_ab

# Guard against negative det (shouldn't happen with valid correlations)
if det_R <= 0:
    log2(f"WARNING: det(R) = {det_R:.6f} <= 0, using absolute value")
    det_R = abs(det_R) + 1e-10

denominator = np.sqrt(2 * (1 - r_ab) * det_R)

if denominator < 1e-10:
    log2("WARNING: Denominator near zero, Z test unreliable")
    Z = 0.0
    p_uncorrected = 1.0
else:
    Z = (r_ya - r_yb) * np.sqrt(N - 3) / denominator
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(Z)))  # Two-tailed

p_bonferroni = min(p_uncorrected * 1, 1.0)  # Single test: no correction needed

log2(f"\nSteiger's Z = {Z:.6f}")
log2(f"p-value (uncorrected) = {p_uncorrected:.6f}")
log2(f"p-value (Bonferroni) = {p_bonferroni:.6f}")

# ── Save Results ─────────────────────────────────────────────────────────────
steiger_df = pd.DataFrame([{
    'r_ya': r_ya,
    'r_yb': r_yb,
    'r_ab': r_ab,
    'N': N,
    'Z': Z,
    'p_uncorrected': p_uncorrected,
    'p_bonferroni': p_bonferroni
}])
steiger_df.to_csv(RQ_DIR / "data" / "step02_steiger_z_test.csv", index=False)
log2(f"Saved: step02_steiger_z_test.csv (1 row)")

# ── Validation ───────────────────────────────────────────────────────────────
assert -1 <= r_ya <= 1, f"r_ya={r_ya} out of bounds"
assert -1 <= r_yb <= 1, f"r_yb={r_yb} out of bounds"
assert -1 <= r_ab <= 1, f"r_ab={r_ab} out of bounds"
assert 90 <= N <= 100, f"N={N} out of expected range"
assert np.isfinite(Z), f"Z={Z} is not finite"
assert 0 <= p_uncorrected <= 1, f"p={p_uncorrected} out of bounds"
log2("Validation PASSED: All Steiger's Z checks OK")

log("=" * 70)
log("SUB-ANALYSIS 1 COMPLETE")
log("=" * 70)
