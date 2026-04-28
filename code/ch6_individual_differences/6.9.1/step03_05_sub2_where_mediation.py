"""
Step 03-05: Sub-Analysis 2 - Where Domain Mediation
====================================================
Tests if age→Where memory is fully mediated by cognitive capacity,
similar to Overall (119.8%) and What (108.0%) domains.

Data sources:
  - RQ 7.2.2 step01: theta_where (aggregated across 4 sessions)
  - RQ 7.3.1 step02: cognitive tests (RAVLT_T, BVMT_T, RPM_T)
  - dfnonvr.csv: age
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── Path Setup ──────────────────────────────────────────────────────────────
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOG_FILE = RQ_DIR / "logs" / "step03_extract_where_data.log"

def log(msg, log_file=None):
    lf = log_file or LOG_FILE
    with open(lf, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Extract and Merge Where Domain Data
# ══════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("STEP 03: Extract and Merge Where Domain Data")
log("=" * 70)

# Load Where theta from RQ 7.2.2
theta_path = PROJECT_ROOT / "results" / "ch7" / "7.2.2" / "data" / "step01_merged_coefficients.csv"
assert theta_path.exists(), f"QUIT: {theta_path} not found"

df_theta = pd.read_csv(theta_path)
log(f"Loaded 7.2.2 data: {df_theta.shape}, columns: {df_theta.columns.tolist()}")
assert 'theta_where' in df_theta.columns, "QUIT: theta_where not found in RQ 7.2.2 output"

# Load cognitive tests from RQ 7.3.1
cog_path = PROJECT_ROOT / "results" / "ch7" / "7.3.1" / "data" / "step02_cognitive_tests.csv"
assert cog_path.exists(), f"QUIT: {cog_path} not found"

df_cog = pd.read_csv(cog_path)
log(f"Loaded 7.3.1 cognitive tests: {df_cog.shape}")

# Load age from dfnonvr.csv
dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
assert dfnonvr_path.exists(), f"QUIT: {dfnonvr_path} not found"

df_demo = pd.read_csv(dfnonvr_path)
log(f"Loaded dfnonvr: {df_demo.shape}")

# Merge
df_where = df_theta[['UID', 'theta_where']].copy()
df_age = df_demo[['UID', 'age']].copy()
df_cog_sel = df_cog[['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T']].copy()

df_merged = df_where.merge(df_age, on='UID', how='inner')
df_merged = df_merged.merge(df_cog_sel, on='UID', how='inner')
df_merged = df_merged.dropna()

N = len(df_merged)
log(f"Merged data: {N} participants with complete data")

if N < 90:
    log(f"WARNING: N < 90 ({N}), mediation may be underpowered")

# Validation
assert df_merged['age'].between(20, 70).all(), "age outside [20, 70]"
assert df_merged['theta_where'].between(-3, 3).all(), "theta_where outside [-3, 3]"
for col in ['RAVLT_T', 'BVMT_T', 'RPM_T']:
    assert df_merged[col].between(15, 85).all(), f"{col} outside [15, 85]"
assert df_merged['UID'].is_unique, "Duplicate UIDs found"
log("No missing values detected")

# Save
df_merged.to_csv(RQ_DIR / "data" / "step03_where_mediation_input.csv", index=False)
log(f"Saved: step03_where_mediation_input.csv ({N} rows)")
log("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Compute Mediation Path Coefficients
# ══════════════════════════════════════════════════════════════════════════════

LOG_FILE_4 = RQ_DIR / "logs" / "step04_compute_mediation_paths.log"

log("=" * 70, LOG_FILE_4)
log("STEP 04: Compute Mediation Path Coefficients", LOG_FILE_4)
log("=" * 70, LOG_FILE_4)

# Create cognitive composite (mean of T-scores)
df_merged['cognitive_composite'] = df_merged[['RAVLT_T', 'BVMT_T', 'RPM_T']].mean(axis=1)

age = df_merged['age'].values
theta_where = df_merged['theta_where'].values
cog_comp = df_merged['cognitive_composite'].values

# Path c (total effect): theta_where ~ age
X_c = sm.add_constant(age)
model_c = sm.OLS(theta_where, X_c).fit(cov_type='HC3')
c = model_c.params[1]
c_se = model_c.bse[1]
c_t = model_c.tvalues[1]
c_p = model_c.pvalues[1]

log(f"Path c (total effect): {c:.6f} (SE={c_se:.6f}, t={c_t:.4f}, p={c_p:.6f})", LOG_FILE_4)

# Path a: cognitive_composite ~ age
X_a = sm.add_constant(age)
model_a = sm.OLS(cog_comp, X_a).fit(cov_type='HC3')
a = model_a.params[1]
a_se = model_a.bse[1]
a_t = model_a.tvalues[1]
a_p = model_a.pvalues[1]

log(f"Path a (age->cognitive): {a:.6f} (SE={a_se:.6f}, t={a_t:.4f}, p={a_p:.6f})", LOG_FILE_4)

# Paths b + c': theta_where ~ age + cognitive_composite
X_bc = sm.add_constant(np.column_stack([age, cog_comp]))
model_bc = sm.OLS(theta_where, X_bc).fit(cov_type='HC3')
c_prime = model_bc.params[1]  # Direct effect of age
b = model_bc.params[2]         # Mediator effect
c_prime_se = model_bc.bse[1]
c_prime_t = model_bc.tvalues[1]
c_prime_p = model_bc.pvalues[1]
b_se = model_bc.bse[2]
b_t = model_bc.tvalues[2]
b_p = model_bc.pvalues[2]

log(f"Path b (cognitive->where): {b:.6f} (SE={b_se:.6f}, t={b_t:.4f}, p={b_p:.6f})", LOG_FILE_4)
log(f"Path c' (direct effect):   {c_prime:.6f} (SE={c_prime_se:.6f}, t={c_prime_t:.4f}, p={c_prime_p:.6f})", LOG_FILE_4)

# Indirect effect
indirect = a * b
# Sobel SE: sqrt(a²*se_b² + b²*se_a²)
sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
indirect_t = indirect / sobel_se if sobel_se > 0 else 0
indirect_p = 2 * (1 - stats.norm.cdf(abs(indirect_t)))

log(f"Indirect (a*b): {indirect:.6f} (Sobel SE={sobel_se:.6f}, Z={indirect_t:.4f}, p={indirect_p:.6f})", LOG_FILE_4)

# Proportion mediated
if abs(c) > 1e-10:
    prop_mediated = indirect / c
    prop_se = np.nan  # Complex, use bootstrap for CI
    prop_t = np.nan
    prop_p = np.nan
else:
    prop_mediated = np.nan
    prop_se = np.nan
    prop_t = np.nan
    prop_p = np.nan

log(f"Proportion mediated: {prop_mediated:.4f} ({prop_mediated*100:.1f}%)", LOG_FILE_4)
log("All mediation paths computed successfully", LOG_FILE_4)

# Save
paths_df = pd.DataFrame([
    {'path': 'c', 'coefficient': c, 'se': c_se, 't': c_t, 'p': c_p},
    {'path': 'a', 'coefficient': a, 'se': a_se, 't': a_t, 'p': a_p},
    {'path': 'b', 'coefficient': b, 'se': b_se, 't': b_t, 'p': b_p},
    {'path': 'c_prime', 'coefficient': c_prime, 'se': c_prime_se, 't': c_prime_t, 'p': c_prime_p},
    {'path': 'indirect', 'coefficient': indirect, 'se': sobel_se, 't': indirect_t, 'p': indirect_p},
    {'path': 'proportion_mediated', 'coefficient': prop_mediated, 'se': float('nan'), 't': float('nan'), 'p': float('nan')},
])
paths_df.to_csv(RQ_DIR / "data" / "step04_where_mediation_paths.csv", index=False)
log(f"Saved: step04_where_mediation_paths.csv (6 rows)", LOG_FILE_4)

# Validation
assert len(paths_df) == 6, f"Expected 6 paths, got {len(paths_df)}"
for _, row in paths_df.iterrows():
    if row['path'] != 'proportion_mediated':
        assert row['se'] > 0, f"SE for {row['path']} is not positive"
        assert 0 <= row['p'] <= 1, f"p for {row['path']} out of [0,1]"
log("Validation PASSED", LOG_FILE_4)


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap Mediation CI
# ══════════════════════════════════════════════════════════════════════════════

LOG_FILE_5 = RQ_DIR / "logs" / "step05_bootstrap_mediation.log"

log("=" * 70, LOG_FILE_5)
log("STEP 05: Bootstrap Mediation Confidence Intervals", LOG_FILE_5)
log("=" * 70, LOG_FILE_5)

B = 5000
rng = np.random.RandomState(42)

boot_results = []
n_extreme = 0

log(f"Running bootstrap with B={B} iterations...", LOG_FILE_5)

for b_iter in range(B):
    idx = rng.choice(N, size=N, replace=True)
    age_b = age[idx]
    theta_b = theta_where[idx]
    cog_b = cog_comp[idx]

    try:
        # Path a
        X_a_b = sm.add_constant(age_b)
        model_a_b = sm.OLS(cog_b, X_a_b).fit()
        a_b = model_a_b.params[1]

        # Path b + c'
        X_bc_b = sm.add_constant(np.column_stack([age_b, cog_b]))
        model_bc_b = sm.OLS(theta_b, X_bc_b).fit()
        b_b = model_bc_b.params[2]

        # Path c (total)
        X_c_b = sm.add_constant(age_b)
        model_c_b = sm.OLS(theta_b, X_c_b).fit()
        c_b = model_c_b.params[1]

        # Compute
        indirect_b = a_b * b_b
        if abs(c_b) > 1e-10:
            prop_b = indirect_b / c_b
        else:
            prop_b = np.nan
            n_extreme += 1

        if abs(prop_b) > 2.0:
            n_extreme += 1

        boot_results.append({
            'iteration': b_iter + 1,
            'indirect': indirect_b,
            'proportion_mediated': prop_b
        })
    except Exception:
        boot_results.append({
            'iteration': b_iter + 1,
            'indirect': np.nan,
            'proportion_mediated': np.nan
        })

    if (b_iter + 1) % 1000 == 0:
        log(f"  Progress: {b_iter+1}/{B} iterations", LOG_FILE_5)

log(f"Bootstrap mediation completed: {len(boot_results)} iterations", LOG_FILE_5)

if n_extreme > 0.05 * B:
    log(f"WARN: {n_extreme} extreme proportion_mediated values (>{0.05*B:.0f} threshold)", LOG_FILE_5)

# Save bootstrap
df_boot_med = pd.DataFrame(boot_results)
df_boot_med.to_csv(RQ_DIR / "data" / "step05_where_mediation_bootstrap.csv", index=False)
log(f"Saved: step05_where_mediation_bootstrap.csv ({len(df_boot_med)} rows)", LOG_FILE_5)

# Compute CIs
indirect_vals = df_boot_med['indirect'].dropna().values
prop_vals = df_boot_med['proportion_mediated'].dropna().values

indirect_ci_lower = np.percentile(indirect_vals, 2.5)
indirect_ci_upper = np.percentile(indirect_vals, 97.5)
prop_ci_lower = np.percentile(prop_vals, 2.5)
prop_ci_upper = np.percentile(prop_vals, 97.5)

# Bootstrap p-value (proportion where indirect <= 0)
p_boot = np.mean(indirect_vals <= 0)

log(f"\nIndirect effect CI: [{indirect_ci_lower:.6f}, {indirect_ci_upper:.6f}]", LOG_FILE_5)
log(f"Proportion mediated CI: [{prop_ci_lower:.4f}, {prop_ci_upper:.4f}]", LOG_FILE_5)
log(f"Bootstrap p-value: {p_boot:.6f}", LOG_FILE_5)

# Save summary
summary_med = pd.DataFrame({
    'metric': ['indirect_point', 'indirect_ci_lower', 'indirect_ci_upper',
               'proportion_point', 'proportion_ci_lower', 'proportion_ci_upper'],
    'value': [indirect, indirect_ci_lower, indirect_ci_upper,
              prop_mediated, prop_ci_lower, prop_ci_upper]
})
summary_med.to_csv(RQ_DIR / "data" / "step05_where_mediation_summary.csv", index=False)
log(f"Saved: step05_where_mediation_summary.csv (6 rows)", LOG_FILE_5)

# Interpretation
if indirect_ci_lower > 0:
    log("SIGNIFICANT mediation: CI excludes zero", LOG_FILE_5)
elif indirect_ci_upper < 0:
    log("SIGNIFICANT reverse mediation: CI excludes zero (negative)", LOG_FILE_5)
else:
    log("NON-SIGNIFICANT mediation: CI includes zero", LOG_FILE_5)

# Validation
assert len(df_boot_med) == B, f"Expected {B} rows, got {len(df_boot_med)}"
assert indirect_ci_lower < indirect < indirect_ci_upper, "Point estimate outside CI"
log("Validation PASSED", LOG_FILE_5)

log("=" * 70, LOG_FILE_5)
log("SUB-ANALYSIS 2 COMPLETE", LOG_FILE_5)
log("=" * 70, LOG_FILE_5)
