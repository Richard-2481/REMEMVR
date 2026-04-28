#!/usr/bin/env python3
"""
RQ 6.6.2: Individual Difference Predictors of High-Confidence Errors
=====================================================================

Pipeline: Multiple regression predicting HCE_rate_mean from:
- Baseline accuracy (Ch5 5.1.1 Day 0 theta)
- Baseline confidence (RQ 6.1.1 Day 0 theta)
- Age (from dfData.csv)
- Confidence bias (z_conf - z_acc at Day 0)

Hypotheses:
- Dunning-Kruger: Low baseline accuracy -> high HCE (negative beta)
- Confidence bias: High overconfidence -> high HCE (positive beta)
- Metacognitive skill: Low baseline confidence -> high HCE (negative beta)
- Age NULL: No age effect (consistent with Ch5/Ch6 pattern)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"
RESULTS_DIR = RQ_DIR / "results"

# Input files (dependencies)
HCE_RATES_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.6.1/data/step01_hce_rates.csv")
ACCURACY_THETA_FILE = Path("/home/etai/projects/REMEMVR/results/ch5/5.1.1/data/step03_theta_scores.csv")
CONFIDENCE_THETA_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.1.1/data/step03_theta_confidence.csv")
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")

# Create directories
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Log message to file and stdout with flush."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Clear log file
with open(LOG_FILE, 'w') as f:
    f.write("RQ 6.6.2: Individual Difference Predictors of HCE\n")
    f.write("=" * 60 + "\n\n")

# Merge Predictor Data Sources

log("STEP 00: Merge Predictor Data Sources")
log("-" * 40)

# --- Load HCE rates ---
log(f"Loading HCE rates from: {HCE_RATES_FILE}")
df_hce = pd.read_csv(HCE_RATES_FILE)
log(f"  HCE rates: {len(df_hce)} rows, columns: {list(df_hce.columns)}")

# Aggregate to person-level (mean across 4 tests)
df_hce['UID'] = df_hce['UID'].astype(str)
hce_agg = df_hce.groupby('UID')['HCE_rate'].mean().reset_index()
hce_agg.columns = ['UID', 'HCE_rate_mean']
log(f"  Aggregated to {len(hce_agg)} participants (mean across 4 tests)")
log(f"  HCE_rate_mean range: [{hce_agg['HCE_rate_mean'].min():.4f}, {hce_agg['HCE_rate_mean'].max():.4f}]")

# --- Load baseline accuracy (Ch5 5.1.1 Day 0) ---
log(f"\nLoading accuracy theta from: {ACCURACY_THETA_FILE}")
df_acc = pd.read_csv(ACCURACY_THETA_FILE)
log(f"  Accuracy theta: {len(df_acc)} rows, columns: {list(df_acc.columns)}")

# Filter to T1 (Day 0) - column is lowercase 'test'
df_acc['UID'] = df_acc['UID'].astype(str)
df_acc_t1 = df_acc[df_acc['test'] == 1][['UID', 'Theta_All']].copy()
df_acc_t1.columns = ['UID', 'baseline_accuracy']
log(f"  Filtered to T1 (Day 0): {len(df_acc_t1)} rows")
log(f"  baseline_accuracy range: [{df_acc_t1['baseline_accuracy'].min():.4f}, {df_acc_t1['baseline_accuracy'].max():.4f}]")

# --- Load baseline confidence (RQ 6.1.1 Day 0) ---
log(f"\nLoading confidence theta from: {CONFIDENCE_THETA_FILE}")
df_conf = pd.read_csv(CONFIDENCE_THETA_FILE)
log(f"  Confidence theta: {len(df_conf)} rows, columns: {list(df_conf.columns)}")

# Parse composite_ID to extract UID and filter to T1
# Format: A010_T1, A010_T2, etc.
df_conf['UID'] = df_conf['composite_ID'].str.split('_').str[0]
df_conf['test_num'] = df_conf['composite_ID'].str.split('_').str[1].str.replace('T', '').astype(int)
df_conf_t1 = df_conf[df_conf['test_num'] == 1][['UID', 'theta_All']].copy()
df_conf_t1.columns = ['UID', 'baseline_confidence']
log(f"  Filtered to T1 (Day 0): {len(df_conf_t1)} rows")
log(f"  baseline_confidence range: [{df_conf_t1['baseline_confidence'].min():.4f}, {df_conf_t1['baseline_confidence'].max():.4f}]")

# --- Load Age ---
log(f"\nLoading Age from: {DFDATA_FILE}")
df_data = pd.read_csv(DFDATA_FILE)
log(f"  dfData: {len(df_data)} rows")

# Get unique UID-Age pairs
df_data['UID'] = df_data['UID'].astype(str)
df_age = df_data[['UID', 'age']].drop_duplicates()
df_age.columns = ['UID', 'Age']  # Rename to match spec
log(f"  Unique UID-Age pairs: {len(df_age)}")
log(f"  Age range: [{df_age['Age'].min()}, {df_age['Age'].max()}]")

# --- Merge all predictors ---
log("\nMerging all predictors...")

# Start with HCE rates
merged = hce_agg.merge(df_acc_t1, on='UID', how='left')
log(f"  After merging accuracy: {len(merged)} rows, NaN in baseline_accuracy: {merged['baseline_accuracy'].isna().sum()}")

merged = merged.merge(df_conf_t1, on='UID', how='left')
log(f"  After merging confidence: {len(merged)} rows, NaN in baseline_confidence: {merged['baseline_confidence'].isna().sum()}")

merged = merged.merge(df_age, on='UID', how='left')
log(f"  After merging age: {len(merged)} rows, NaN in Age: {merged['Age'].isna().sum()}")

# --- Compute confidence bias ---
log("\nComputing confidence bias...")
# Z-standardize both baselines first
z_acc = (merged['baseline_accuracy'] - merged['baseline_accuracy'].mean()) / merged['baseline_accuracy'].std()
z_conf = (merged['baseline_confidence'] - merged['baseline_confidence'].mean()) / merged['baseline_confidence'].std()
merged['confidence_bias'] = z_conf - z_acc
log(f"  confidence_bias range: [{merged['confidence_bias'].min():.4f}, {merged['confidence_bias'].max():.4f}]")

# --- Validation ---
log("\nValidating Step 00 output...")
assert len(merged) == 100, f"Expected 100 participants, got {len(merged)}"
assert merged.isna().sum().sum() == 0, f"Found NaN values: {merged.isna().sum()}"
assert merged['HCE_rate_mean'].min() >= 0 and merged['HCE_rate_mean'].max() <= 1, "HCE_rate_mean outside [0,1]"
assert merged['UID'].nunique() == 100, "Duplicate UIDs found"

# Save Step 00 output
merged.to_csv(DATA_DIR / "step00_predictor_data.csv", index=False)
log(f"\nStep 00 OUTPUT: {DATA_DIR / 'step00_predictor_data.csv'}")
log(f"  Shape: {merged.shape}")
log(f"  Columns: {list(merged.columns)}")
log("  Data merge complete: 100 participants with complete predictors")
log("  All source files merged successfully")

# Standardize Predictors

log("\n" + "=" * 60)
log("STEP 01: Standardize Predictors")
log("-" * 40)

df = merged.copy()

# Standardize all 4 predictors (leave outcome unstandardized for interpretability)
predictors = ['baseline_accuracy', 'baseline_confidence', 'Age', 'confidence_bias']
for pred in predictors:
    z_col = f"z_{pred}"
    df[z_col] = (df[pred] - df[pred].mean()) / df[pred].std()
    log(f"  {z_col}: mean={df[z_col].mean():.6f}, SD={df[z_col].std():.6f}")

# Validation
log("\nValidating z-scores...")
for pred in predictors:
    z_col = f"z_{pred}"
    mean_check = abs(df[z_col].mean()) < 0.01
    sd_check = abs(df[z_col].std() - 1.0) < 0.01
    assert mean_check, f"{z_col} mean not near 0: {df[z_col].mean()}"
    assert sd_check, f"{z_col} SD not near 1: {df[z_col].std()}"
    log(f"  {z_col}: PASS (mean={df[z_col].mean():.6f}, SD={df[z_col].std():.6f})")

log("Standardization complete: 4 predictors z-scored")
log("Z-score validation PASS: mean near 0, SD near 1")

# Save standardized data
standardized_cols = ['UID', 'HCE_rate_mean', 'z_baseline_accuracy', 'z_baseline_confidence', 'z_Age', 'z_confidence_bias']
df_std = df[standardized_cols].copy()
df_std.to_csv(DATA_DIR / "step01_standardized_predictors.csv", index=False)
log(f"\nStep 01 OUTPUT: {DATA_DIR / 'step01_standardized_predictors.csv'}")

# Fit Multiple Regression Model

log("\n" + "=" * 60)
log("STEP 02: Fit Multiple Regression Model")
log("-" * 40)

# Fit OLS regression
formula = "HCE_rate_mean ~ z_baseline_accuracy + z_baseline_confidence + z_Age + z_confidence_bias"
log(f"Formula: {formula}")

model = smf.ols(formula, data=df_std).fit()

log(f"\nModel Summary:")
log(f"  R-squared: {model.rsquared:.4f}")
log(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
log(f"  F-statistic: {model.fvalue:.4f}")
log(f"  F p-value: {model.f_pvalue:.6f}")
log(f"  N observations: {model.nobs}")

# Check model diagnostics
residuals = model.resid
shapiro_stat, shapiro_p = stats.shapiro(residuals)
log(f"\nResidual diagnostics:")
log(f"  Mean residual: {residuals.mean():.6f}")
log(f"  SD residual: {residuals.std():.6f}")
log(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p < 0.05:
    log("  WARNING: Residuals slightly non-normal (regression robust to minor violations)")
else:
    log("  Residuals approximately normal")

# Save model summary
with open(DATA_DIR / "step02_regression_model_summary.txt", 'w') as f:
    f.write(str(model.summary()))
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("Residual Diagnostics\n")
    f.write(f"Mean: {residuals.mean():.6f}\n")
    f.write(f"SD: {residuals.std():.6f}\n")
    f.write(f"Shapiro-Wilk W: {shapiro_stat:.4f}, p={shapiro_p:.4f}\n")

log(f"\nStep 02 OUTPUT: {DATA_DIR / 'step02_regression_model_summary.txt'}")
log("Regression model fitted successfully")
log(f"R-squared: {model.rsquared:.4f}")

# Extract Coefficients with Dual P-Values (Decision D068)

log("\n" + "=" * 60)
log("STEP 03: Extract Coefficients with Dual P-Values")
log("-" * 40)

# Extract coefficient table
coef_data = []
n_predictors = 4  # Excluding intercept for Bonferroni

for term in model.params.index:
    coef = model.params[term]
    se = model.bse[term]
    t_val = model.tvalues[term]
    p_uncorr = model.pvalues[term]

    # Bonferroni correction (only for predictors, not intercept)
    if term == 'Intercept':
        p_bonf = p_uncorr  # No correction for intercept
    else:
        p_bonf = min(p_uncorr * n_predictors, 1.0)

    sig_uncorr = p_uncorr < 0.05
    sig_bonf = p_bonf < 0.05

    coef_data.append({
        'predictor': term,
        'coefficient': coef,
        'SE': se,
        't_value': t_val,
        'p_uncorrected': p_uncorr,
        'p_bonferroni': p_bonf,
        'sig_uncorrected': sig_uncorr,
        'sig_bonferroni': sig_bonf
    })

df_coef = pd.DataFrame(coef_data)

log("\nCoefficient Table (Decision D068 Dual P-Values):")
log("-" * 80)
for _, row in df_coef.iterrows():
    sig_marker = "***" if row['sig_bonferroni'] else ("*" if row['sig_uncorrected'] else "")
    log(f"  {row['predictor']:25s} β={row['coefficient']:8.5f} SE={row['SE']:.5f} t={row['t_value']:7.3f} "
        f"p={row['p_uncorrected']:.4f} p_bonf={row['p_bonferroni']:.4f} {sig_marker}")

# Validation
assert len(df_coef) == 5, f"Expected 5 rows (Intercept + 4 predictors), got {len(df_coef)}"
assert 'p_uncorrected' in df_coef.columns and 'p_bonferroni' in df_coef.columns, "Missing dual p-value columns"
assert all(df_coef['p_bonferroni'] >= df_coef['p_uncorrected']), "p_bonferroni should be >= p_uncorrected"
assert all(df_coef['SE'] > 0), "All SE must be positive"

log("\nCoefficients extracted with dual p-values (Decision D068)")
log("Bonferroni correction applied: 4 predictors")

df_coef.to_csv(DATA_DIR / "step03_regression_coefficients.csv", index=False)
log(f"\nStep 03 OUTPUT: {DATA_DIR / 'step03_regression_coefficients.csv'}")

# Compute Effect Sizes

log("\n" + "=" * 60)
log("STEP 04: Compute Effect Sizes")
log("-" * 40)

effect_sizes = []

# Overall R-squared
effect_sizes.append({'metric': 'R_squared', 'value': model.rsquared})
effect_sizes.append({'metric': 'Adjusted_R_squared', 'value': model.rsquared_adj})

log(f"  R_squared: {model.rsquared:.4f}")
log(f"  Adjusted_R_squared: {model.rsquared_adj:.4f}")

# Partial R-squared per predictor
# Method: Compare full model vs reduced model (excluding one predictor)
predictors_list = ['z_baseline_accuracy', 'z_baseline_confidence', 'z_Age', 'z_confidence_bias']

log("\nPartial R-squared (unique variance explained):")
for pred in predictors_list:
    # Reduced model formula (exclude target predictor)
    other_preds = [p for p in predictors_list if p != pred]
    formula_reduced = f"HCE_rate_mean ~ {' + '.join(other_preds)}"

    model_reduced = smf.ols(formula_reduced, data=df_std).fit()
    partial_r2 = model.rsquared - model_reduced.rsquared

    # Handle negative partial R-squared (can happen with correlated predictors)
    partial_r2 = max(partial_r2, 0.0)

    effect_sizes.append({'metric': f'partial_R2_{pred.replace("z_", "")}', 'value': partial_r2})
    log(f"  {pred}: partial R² = {partial_r2:.4f}")

df_effect = pd.DataFrame(effect_sizes)

# Validation
assert all(df_effect['value'] >= 0) and all(df_effect['value'] <= 1), "R-squared values outside [0,1]"
assert df_effect[df_effect['metric'] == 'Adjusted_R_squared']['value'].values[0] <= \
       df_effect[df_effect['metric'] == 'R_squared']['value'].values[0], "Adjusted R² > R²"

log("\nEffect sizes computed: R-squared and partial R-squared")
log(f"Overall R-squared: {model.rsquared:.4f}")

df_effect.to_csv(DATA_DIR / "step04_effect_sizes.csv", index=False)
log(f"\nStep 04 OUTPUT: {DATA_DIR / 'step04_effect_sizes.csv'}")

# SUMMARY

log("\n" + "=" * 60)
log("ANALYSIS COMPLETE: RQ 6.6.2")
log("=" * 60)

# Interpret results
log("\nHYPOTHESIS TEST RESULTS:")
log("-" * 40)

# Get coefficients for interpretation
coef_dict = dict(zip(df_coef['predictor'], df_coef['coefficient']))
pval_dict = dict(zip(df_coef['predictor'], df_coef['p_bonferroni']))
sig_dict = dict(zip(df_coef['predictor'], df_coef['sig_bonferroni']))

# 1. Dunning-Kruger: Low accuracy -> high HCE (expect negative beta)
acc_beta = coef_dict['z_baseline_accuracy']
acc_p = pval_dict['z_baseline_accuracy']
acc_sig = sig_dict['z_baseline_accuracy']
if acc_beta < 0 and acc_sig:
    log(f"1. Dunning-Kruger: SUPPORTED (β={acc_beta:.4f}, p_bonf={acc_p:.4f})")
    log("   Low baseline accuracy predicts more HCEs")
elif acc_beta < 0:
    log(f"1. Dunning-Kruger: Direction correct but NOT significant (β={acc_beta:.4f}, p_bonf={acc_p:.4f})")
else:
    log(f"1. Dunning-Kruger: NOT SUPPORTED (β={acc_beta:.4f}, p_bonf={acc_p:.4f})")

# 2. Confidence bias: High overconfidence -> high HCE (expect positive beta)
bias_beta = coef_dict['z_confidence_bias']
bias_p = pval_dict['z_confidence_bias']
bias_sig = sig_dict['z_confidence_bias']
if bias_beta > 0 and bias_sig:
    log(f"2. Confidence Bias: SUPPORTED (β={bias_beta:.4f}, p_bonf={bias_p:.4f})")
    log("   High overconfidence predicts more HCEs")
elif bias_beta > 0:
    log(f"2. Confidence Bias: Direction correct but NOT significant (β={bias_beta:.4f}, p_bonf={bias_p:.4f})")
else:
    log(f"2. Confidence Bias: NOT SUPPORTED (β={bias_beta:.4f}, p_bonf={bias_p:.4f})")

# 3. Metacognitive skill: Low confidence -> high HCE (expect negative beta)
conf_beta = coef_dict['z_baseline_confidence']
conf_p = pval_dict['z_baseline_confidence']
conf_sig = sig_dict['z_baseline_confidence']
if conf_beta < 0 and conf_sig:
    log(f"3. Metacognitive Skill: SUPPORTED (β={conf_beta:.4f}, p_bonf={conf_p:.4f})")
    log("   Low baseline confidence predicts more HCEs")
elif conf_beta < 0:
    log(f"3. Metacognitive Skill: Direction correct but NOT significant (β={conf_beta:.4f}, p_bonf={conf_p:.4f})")
else:
    log(f"3. Metacognitive Skill: NOT SUPPORTED (β={conf_beta:.4f}, p_bonf={conf_p:.4f})")

# 4. Age NULL hypothesis (expect p > 0.05)
age_beta = coef_dict['z_Age']
age_p = pval_dict['z_Age']
age_sig = sig_dict['z_Age']
if not age_sig:
    log(f"4. Age NULL: CONFIRMED (β={age_beta:.4f}, p_bonf={age_p:.4f})")
    log("   Age does NOT predict HCE rates (consistent with Ch5/Ch6 pattern)")
else:
    log(f"4. Age NULL: REJECTED - Age DOES predict HCE (β={age_beta:.4f}, p_bonf={age_p:.4f})")

log(f"\nOVERALL MODEL FIT:")
log(f"  R² = {model.rsquared:.4f} ({model.rsquared*100:.1f}% variance explained)")
log(f"  F({model.df_model:.0f},{model.df_resid:.0f}) = {model.fvalue:.2f}, p = {model.f_pvalue:.4f}")

# Count significant predictors
n_sig_bonf = sum(1 for p in [acc_sig, conf_sig, age_sig, bias_sig] if p)
log(f"\n  Significant predictors (Bonferroni): {n_sig_bonf}/4")

log("\n" + "=" * 60)
log("All steps completed successfully")
log("=" * 60)
