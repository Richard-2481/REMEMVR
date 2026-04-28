"""
Step 06-09: Sub-Analysis 3 - Day 0 Theta Replication
=====================================================
Tests if individual differences patterns replicate at Day 0 (encoding session),
uncontaminated by practice effects.

Data sources:
  - Ch5 5.1.1 step03: session-specific theta (test column = 1 for Day 0)
  - Ch5 5.2.1 step03: domain-specific session theta (composite_ID format)
  - RQ 7.3.1 step02: cognitive tests
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


def make_log(step_name):
    log_file = RQ_DIR / "logs" / f"{step_name}.log"
    def log(msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{msg}\n")
            f.flush()
        print(msg, flush=True)
    return log


# ══════════════════════════════════════════════════════════════════════════════
# Extract Day 0 Theta Scores
# ══════════════════════════════════════════════════════════════════════════════

log6 = make_log("step06_extract_day0_theta")

log6("=" * 70)
log6("STEP 06: Extract Day 0 Theta Scores")
log6("=" * 70)

# Overall theta from Ch5 5.1.1
theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
assert theta_path.exists(), f"QUIT: {theta_path} not found"

df_theta = pd.read_csv(theta_path)
log6(f"Loaded Ch5 5.1.1 theta: {df_theta.shape}, columns: {df_theta.columns.tolist()}")

assert 'test' in df_theta.columns, "QUIT: 'test' column not found"
assert 'Theta_All' in df_theta.columns, "QUIT: 'Theta_All' column not found"

# Filter to Day 0 (test == 1)
df_day0 = df_theta[df_theta['test'] == 1].copy()
df_day0 = df_day0.rename(columns={'Theta_All': 'theta_overall_day0'})
df_day0 = df_day0[['UID', 'theta_overall_day0']].copy()
df_day0['UID'] = df_day0['UID'].astype(str)

log6(f"Day 0 theta extracted: {len(df_day0)} participants")

# Domain-specific Day 0 theta from Ch5 5.2.1
domain_path = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
has_domain_day0 = False

if domain_path.exists():
    df_domain = pd.read_csv(domain_path)
    log6(f"Loaded Ch5 5.2.1 domain theta: {df_domain.shape}, columns: {df_domain.columns.tolist()}")

    # Composite_ID format: "UID_test"
    if 'composite_ID' in df_domain.columns:
        df_domain['UID'] = df_domain['composite_ID'].str.split('_').str[0]
        df_domain['test'] = df_domain['composite_ID'].str.split('_').str[1].astype(int)

        # Filter to Day 0 (test == 1)
        df_domain_day0 = df_domain[df_domain['test'] == 1].copy()

        domain_cols = []
        for col_name, rename in [('theta_what', 'theta_what_day0'),
                                  ('theta_where', 'theta_where_day0'),
                                  ('theta_when', 'theta_when_day0')]:
            if col_name in df_domain_day0.columns:
                df_domain_day0 = df_domain_day0.rename(columns={col_name: rename})
                domain_cols.append(rename)

        if domain_cols:
            df_day0 = df_day0.merge(
                df_domain_day0[['UID'] + domain_cols],
                on='UID', how='left'
            )
            has_domain_day0 = True
            log6(f"Domain-specific Day 0 theta added: {domain_cols}")
        else:
            log6("WARNING: No domain theta columns found in 5.2.1")
    else:
        log6("WARNING: composite_ID column not found in 5.2.1")
else:
    log6("WARN: Domain-specific Day 0 theta not available, proceeding with overall only")

# Validation
N_day0 = len(df_day0)
log6(f"Final Day 0 dataset: {N_day0} participants, columns: {df_day0.columns.tolist()}")

assert N_day0 >= 90, f"N={N_day0} below minimum of 90"
assert df_day0['theta_overall_day0'].between(-5, 5).all(), "theta_overall_day0 outside [-5, 5]"
assert not df_day0['theta_overall_day0'].isna().any(), "NaN in theta_overall_day0"
assert df_day0['UID'].is_unique, "Duplicate UIDs"

df_day0.to_csv(RQ_DIR / "data" / "step06_day0_theta.csv", index=False)
log6(f"Saved: step06_day0_theta.csv ({N_day0} rows)")
log6("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Day 0 Age-Theta Correlation
# ══════════════════════════════════════════════════════════════════════════════

log7 = make_log("step07_day0_age_correlation")

log7("=" * 70)
log7("STEP 07: Day 0 Age-Theta Correlation")
log7("=" * 70)

# Load age
dfnonvr = pd.read_csv(PROJECT_ROOT / "data" / "dfnonvr.csv")
df_age = dfnonvr[['UID', 'age']].copy()
df_age['UID'] = df_age['UID'].astype(str)

# Merge Day 0 theta with age
df_corr = df_day0[['UID', 'theta_overall_day0']].merge(df_age, on='UID', how='inner')
df_corr = df_corr.dropna()

# Day 0 correlation
r_day0, p_day0 = stats.pearsonr(df_corr['age'], df_corr['theta_overall_day0'])
N_corr = len(df_corr)

log7(f"Day 0 correlation computed: r = {r_day0:.6f}, p = {p_day0:.6f}, N = {N_corr}")

# Aggregated correlation (from RQ 7.2.2 data)
theta_agg_path = PROJECT_ROOT / "results" / "ch7" / "7.2.2" / "data" / "step01_merged_coefficients.csv"
df_agg = pd.read_csv(theta_agg_path)
df_agg_corr = df_agg[['UID', 'theta_all']].merge(df_age, on='UID', how='inner').dropna()
r_agg, p_agg = stats.pearsonr(df_agg_corr['age'], df_agg_corr['theta_all'])
N_agg = len(df_agg_corr)

log7(f"Aggregated correlation: r = {r_agg:.6f}, p = {p_agg:.6f}, N = {N_agg}")

# Save comparison
corr_df = pd.DataFrame([
    {'correlation_type': 'day0', 'r': r_day0, 'p': p_day0, 'N': N_corr},
    {'correlation_type': 'aggregated', 'r': r_agg, 'p': p_agg, 'N': N_agg},
])
corr_df.to_csv(RQ_DIR / "data" / "step07_day0_age_correlation.csv", index=False)
log7(f"Saved: step07_day0_age_correlation.csv (2 rows)")

# Validation
assert -1 <= r_day0 <= 1
assert -1 <= r_agg <= 1
assert 0 <= p_day0 <= 1
assert 0 <= p_agg <= 1
assert 90 <= N_corr <= 100
log7("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Day 0 RPM Dominance Test
# ══════════════════════════════════════════════════════════════════════════════

log8 = make_log("step08_day0_rpm_dominance")

log8("=" * 70)
log8("STEP 08: Day 0 RPM Dominance Test")
log8("=" * 70)

# Load cognitive tests
cog_path = PROJECT_ROOT / "results" / "ch7" / "7.3.1" / "data" / "step02_cognitive_tests.csv"
assert cog_path.exists(), f"QUIT: {cog_path} not found"

df_cog = pd.read_csv(cog_path)
df_cog['UID'] = df_cog['UID'].astype(str)

# Merge Day 0 theta with cognitive tests
df_rpm = df_day0[['UID', 'theta_overall_day0']].merge(
    df_cog[['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T']], on='UID', how='inner'
).dropna()

N_rpm = len(df_rpm)
log8(f"Merged Day 0 + cognitive tests: N={N_rpm}")

predictors = ['RAVLT_T', 'BVMT_T', 'RPM_T']

def compute_sr2_vals(y, X_df, predictors):
    """Compute sr² for each predictor."""
    X = X_df[predictors].values
    X_const = sm.add_constant(X)
    model_full = sm.OLS(y, X_const).fit(cov_type='HC3')
    r2_full = model_full.rsquared

    sr2_vals = {}
    for i, pred in enumerate(predictors):
        X_red = np.delete(X, i, axis=1)
        X_red_const = sm.add_constant(X_red)
        model_red = sm.OLS(y, X_red_const).fit()
        sr2_vals[pred] = r2_full - model_red.rsquared

    return sr2_vals, r2_full

# Day 0 regression
y_day0 = df_rpm['theta_overall_day0'].values
sr2_day0, r2_day0 = compute_sr2_vals(y_day0, df_rpm, predictors)

log8(f"Day 0 regression completed: R² = {r2_day0:.6f}")
for pred in predictors:
    log8(f"  {pred} sr² (day0) = {sr2_day0[pred]:.6f}")

# Aggregated regression (from RQ 7.1.1 known results)
# RPM sr² = 0.080, RAVLT sr² = 0.017, BVMT sr² = 0.011 (from 4-predictor model)
# For fair comparison, recompute with 3-predictor model
acc_path = PROJECT_ROOT / "results" / "ch7" / "7.1.1" / "data" / "step03_merged_analysis.csv"
df_acc = pd.read_csv(acc_path)
df_acc['UID'] = df_acc['UID'].astype(str)
df_acc_cog = df_acc.merge(df_cog[['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T']],
                           on='UID', how='inner', suffixes=('_orig', '')).dropna()

# Use theta_mean from 7.1.1
y_agg = df_acc_cog['theta_mean'].values
sr2_agg, r2_agg = compute_sr2_vals(y_agg, df_acc_cog, predictors)

log8(f"Aggregated regression (3-pred recomputed): R² = {r2_agg:.6f}")
for pred in predictors:
    log8(f"  {pred} sr² (aggregated) = {sr2_agg[pred]:.6f}")

# Build comparison table
rows = []
for analysis, sr2_dict in [('day0', sr2_day0), ('aggregated', sr2_agg)]:
    sorted_preds = sorted(sr2_dict.items(), key=lambda x: -x[1])
    for rank, (pred, sr2) in enumerate(sorted_preds, 1):
        rows.append({
            'analysis': analysis,
            'predictor': pred,
            'sr2': sr2,
            'rank': rank
        })

rpm_df = pd.DataFrame(rows)
rpm_df.to_csv(RQ_DIR / "data" / "step08_day0_rpm_dominance.csv", index=False)
log8(f"Saved: step08_day0_rpm_dominance.csv (6 rows)")

# Check RPM dominance
rpm_rank_day0 = rpm_df[(rpm_df['analysis'] == 'day0') & (rpm_df['predictor'] == 'RPM_T')]['rank'].values[0]
rpm_rank_agg = rpm_df[(rpm_df['analysis'] == 'aggregated') & (rpm_df['predictor'] == 'RPM_T')]['rank'].values[0]
log8(f"RPM rank: Day 0 = {rpm_rank_day0}, Aggregated = {rpm_rank_agg}")

# Validation
assert len(rpm_df) == 6
assert rpm_df['sr2'].between(0, 1).all()
assert rpm_df['rank'].between(1, 3).all()
log8("Validation PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Day 0 Domain Intercorrelation
# ══════════════════════════════════════════════════════════════════════════════

log9 = make_log("step09_day0_domain_correlation")

log9("=" * 70)
log9("STEP 09: Day 0 Domain Intercorrelation")
log9("=" * 70)

results_corr = []

if has_domain_day0 and 'theta_what_day0' in df_day0.columns and 'theta_where_day0' in df_day0.columns:
    df_domain_corr = df_day0[['UID', 'theta_what_day0', 'theta_where_day0']].dropna()
    N_domain = len(df_domain_corr)

    if N_domain >= 50:
        r_domain_day0, p_domain_day0 = stats.pearsonr(
            df_domain_corr['theta_what_day0'],
            df_domain_corr['theta_where_day0']
        )
        log9(f"Day 0 domain correlation computed: r = {r_domain_day0:.6f}, p = {p_domain_day0:.6f}, N = {N_domain}")
        results_corr.append({'analysis': 'day0', 'r': r_domain_day0, 'p': p_domain_day0, 'N': N_domain})
    else:
        log9(f"WARNING: N={N_domain} too small for domain correlation, skipping Day 0")
else:
    log9("Domain-specific Day 0 theta not available, skipping Day 0 comparison")

# Aggregated domain correlation from Ch5 5.2.1
df_agg_domain = pd.read_csv(PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv")
df_agg_domain['UID'] = df_agg_domain['composite_ID'].str.split('_').str[0]
df_agg_domain_mean = df_agg_domain.groupby('UID')[['theta_what', 'theta_where']].mean().reset_index()
df_agg_domain_mean = df_agg_domain_mean.dropna()

r_agg_domain, p_agg_domain = stats.pearsonr(df_agg_domain_mean['theta_what'], df_agg_domain_mean['theta_where'])
N_agg_domain = len(df_agg_domain_mean)
log9(f"Aggregated domain correlation: r = {r_agg_domain:.6f}, p = {p_agg_domain:.6f}, N = {N_agg_domain}")
results_corr.append({'analysis': 'aggregated', 'r': r_agg_domain, 'p': p_agg_domain, 'N': N_agg_domain})

domain_corr_df = pd.DataFrame(results_corr)
domain_corr_df.to_csv(RQ_DIR / "data" / "step09_day0_domain_intercorrelation.csv", index=False)
log9(f"Saved: step09_day0_domain_intercorrelation.csv ({len(domain_corr_df)} rows)")

# Validation
for _, row in domain_corr_df.iterrows():
    assert -1 <= row['r'] <= 1
    assert 0 <= row['p'] <= 1
    assert 50 <= row['N'] <= 100
log9("Validation PASSED")

log9("=" * 70)
log9("SUB-ANALYSIS 3 COMPLETE")
log9("=" * 70)
