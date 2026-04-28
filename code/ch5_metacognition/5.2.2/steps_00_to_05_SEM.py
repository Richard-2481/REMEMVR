#!/usr/bin/env python3
"""
RQ 6.2.2: Over-Underconfidence Trajectory (SEM VERSION)
=======================================================

Tests whether overconfidence (confidence > accuracy) INCREASES from Day 0 to Day 6
as accuracy declines faster than confidence adjusts.

**PHASE 2 SEM PROTOTYPE:** This version uses SEM-based latent calibration scores
instead of simple difference scores to properly account for measurement error.

Input: SEM calibration scores from RQ 6.2.1 (step02_calibration_scores_SEM.csv)
Output: Classified observations, proportion overconfident by timepoint, trend test, mean calibration

**Key Difference from Original:**
- Original: calibration = z_theta_confidence - z_theta_accuracy (r_diff=-0.25, CATASTROPHIC)
- SEM Version: calibration from latent variable model (r=0.70, MARGINAL)
- Expected: Stronger effect detection due to reduced measurement error
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# SETUP

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.2
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"
DATA_DIR = RQ_DIR / "data"
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # REMEMVR root

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """Log message to both file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


# Clear log file
LOG_FILE.write_text("")
log("=" * 70)
log("RQ 6.2.2: Over-Underconfidence Trajectory Analysis (SEM VERSION)")
log("=" * 70)
log("Phase 2 Prototype: Using SEM latent calibration (r=0.70 vs r_diff=-0.25)")
log("=" * 70)

# Load Calibration Data from RQ 6.2.1

log("\n" + "=" * 70)
log("STEP 00: Load SEM Calibration Data from RQ 6.2.1")
log("=" * 70)

# Load SEM-based calibration scores from RQ 6.2.1 (Phase 2 prototype)
source_file = PROJECT_ROOT / "results" / "ch6" / "6.2.1" / "data" / "step02_calibration_scores_SEM.csv"
log(f"Source file (SEM version): {source_file}")
log("Using SEM latent calibration (r=0.70) instead of simple difference (r_diff=-0.25)")

if not source_file.exists():
    log(f"ERROR: Source file not found: {source_file}")
    log("FAIL: RQ 6.2.1 must complete before this RQ")
    sys.exit(1)

df_raw = pd.read_csv(source_file)
log(f"Loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
log(f"Columns: {list(df_raw.columns)}")

# Check actual column names (may differ from 4_analysis.yaml specification)
# Actual from 6.2.1: UID, test, composite_ID, TSVR_hours, z_theta_accuracy, z_theta_confidence, calibration
expected_cols = ['UID', 'test', 'calibration']  # Minimum required
for col in expected_cols:
    if col not in df_raw.columns:
        log(f"ERROR: Missing required column: {col}")
        sys.exit(1)

# Rename calibration column for consistency (lowercase -> Calibration)
df = df_raw.copy()
if 'calibration' in df.columns and 'Calibration' not in df.columns:
    df.rename(columns={'calibration': 'Calibration'}, inplace=True)
    log("Renamed 'calibration' -> 'Calibration' for consistency")

# Validate row count
if len(df) != 400:
    log(f"WARNING: Expected 400 rows, found {len(df)}")

# Check for missing calibration values
n_missing = df['Calibration'].isna().sum()
if n_missing > 0:
    log(f"ERROR: {n_missing} missing values in Calibration column")
    sys.exit(1)
log("No missing values in Calibration column")

# Save step 00 output
output_file = DATA_DIR / "step00_calibration_loaded.csv"
df.to_csv(output_file, index=False)
log(f"Saved: {output_file} ({len(df)} rows)")

# Validation summary
log(f"\nLoaded calibration data: {len(df)} rows, {len(df.columns)} columns")
log("STEP 00 COMPLETE")

# Classify Observations by Calibration Sign

log("\n" + "=" * 70)
log("STEP 01: Classify Observations by Calibration Sign")
log("=" * 70)

# Classification parameters
EPSILON = 0.1  # Threshold for "calibrated" zone (within +/- 0.1 SD)

def classify_calibration(cal_value, epsilon=0.1):
    """Classify calibration value as Overconfident, Underconfident, or Calibrated."""
    if cal_value > epsilon:
        return "Overconfident"
    elif cal_value < -epsilon:
        return "Underconfident"
    else:
        return "Calibrated"

# Apply classification
df['Classification'] = df['Calibration'].apply(classify_calibration, epsilon=EPSILON)

# Count by category
classification_counts = df['Classification'].value_counts()
log(f"\nClassification with epsilon={EPSILON}:")
for cat in ['Overconfident', 'Underconfident', 'Calibrated']:
    if cat in classification_counts:
        n = classification_counts[cat]
        pct = 100 * n / len(df)
        log(f"  {cat}: {n} ({pct:.1f}%)")

# Verify all observations classified
if df['Classification'].isna().sum() > 0:
    log("ERROR: Some observations not classified")
    sys.exit(1)

# Save step 01 output
output_file = DATA_DIR / "step01_calibration_classified.csv"
df.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df)} rows)")
log(f"Classification complete: {len(df)} observations classified")
log("STEP 01 COMPLETE")

# Compute Proportion Overconfident Per Timepoint

log("\n" + "=" * 70)
log("STEP 02: Compute Proportion Overconfident Per Timepoint")
log("=" * 70)

def wilson_score_ci(n_success, n_total, confidence=0.95):
    """Compute Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denominator

    return max(0, center - margin), min(1, center + margin)

# Group by test and compute proportions
proportion_results = []
for test in ['T1', 'T2', 'T3', 'T4']:
    test_data = df[df['test'] == test]
    n_total = len(test_data)
    n_overconfident = (test_data['Classification'] == 'Overconfident').sum()
    proportion = n_overconfident / n_total if n_total > 0 else np.nan
    ci_lower, ci_upper = wilson_score_ci(n_overconfident, n_total)

    proportion_results.append({
        'test': test,
        'N_total': n_total,
        'N_overconfident': n_overconfident,
        'proportion_overconfident': proportion,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper
    })
    log(f"{test}: {n_overconfident}/{n_total} = {proportion:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

df_proportions = pd.DataFrame(proportion_results)

# Validate
if len(df_proportions) != 4:
    log(f"ERROR: Expected 4 rows, found {len(df_proportions)}")
    sys.exit(1)

# Check N_total = 100 for all
for _, row in df_proportions.iterrows():
    if row['N_total'] != 100:
        log(f"WARNING: N_total = {row['N_total']} for {row['test']}, expected 100")

# Save step 02 output
output_file = DATA_DIR / "step02_proportion_overconfident.csv"
df_proportions.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_proportions)} rows)")
log(f"Proportion overconfident computed: {len(df_proportions)} timepoints")
log("STEP 02 COMPLETE")

# Trend Test (Logistic Regression)

log("\n" + "=" * 70)
log("STEP 03: Trend Test (Logistic Regression)")
log("=" * 70)

# Create binary outcome: 1 if Overconfident, 0 otherwise
df['overconfident_binary'] = (df['Classification'] == 'Overconfident').astype(int)

# Create time predictor: nominal days (T1=0, T2=1, T3=3, T4=6)
time_map = {'T1': 0, 'T2': 1, 'T3': 3, 'T4': 6}
df['time_ordinal'] = df['test'].map(time_map)

# Log data summary
log(f"\nBinary outcome:")
log(f"  Overconfident (1): {df['overconfident_binary'].sum()}")
log(f"  Not Overconfident (0): {(df['overconfident_binary'] == 0).sum()}")

# Fit logistic regression
X = sm.add_constant(df['time_ordinal'])
y = df['overconfident_binary']

try:
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(method='newton', disp=False, maxiter=100)
    converged = logit_result.converged
    log(f"\nLogistic regression converged: {converged}")
except Exception as e:
    log(f"ERROR: Logistic regression failed: {e}")
    sys.exit(1)

if not converged:
    log("WARNING: Model did not converge, results may be unreliable")

# Extract results
params = logit_result.params
bse = logit_result.bse
zvalues = logit_result.tvalues
pvalues = logit_result.pvalues

# Compute odds ratios
odds_ratios = np.exp(params)
or_ci = np.exp(logit_result.conf_int())

# Create results DataFrame
trend_results = []
for i, term in enumerate(['Intercept', 'time_ordinal']):
    trend_results.append({
        'term': term,
        'estimate': params.iloc[i],
        'SE': bse.iloc[i],
        'z': zvalues.iloc[i],
        'p_value': pvalues.iloc[i],
        'OR': odds_ratios.iloc[i],
        'OR_CI_lower': or_ci.iloc[i, 0],
        'OR_CI_upper': or_ci.iloc[i, 1]
    })

df_trend = pd.DataFrame(trend_results)

# Log key results
log("\nTrend test results:")
for _, row in df_trend.iterrows():
    log(f"  {row['term']}: β={row['estimate']:.4f}, SE={row['SE']:.4f}, z={row['z']:.3f}, p={row['p_value']:.4f}")

time_row = df_trend[df_trend['term'] == 'time_ordinal'].iloc[0]
log(f"\nOdds ratio per day: {time_row['OR']:.4f} [{time_row['OR_CI_lower']:.4f}, {time_row['OR_CI_upper']:.4f}]")
log(f"Trend test complete: slope = {time_row['estimate']:.4f}, p = {time_row['p_value']:.4f}")

# Interpret
if time_row['p_value'] < 0.05:
    if time_row['estimate'] > 0:
        log("\nINTERPRETATION: SIGNIFICANT INCREASE in overconfidence over time (p < 0.05)")
    else:
        log("\nINTERPRETATION: SIGNIFICANT DECREASE in overconfidence over time (p < 0.05)")
else:
    log("\nINTERPRETATION: No significant trend in overconfidence over time (p >= 0.05)")

# Save step 03 output
output_file = DATA_DIR / "step03_trend_test.csv"
df_trend.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_trend)} rows)")
log("STEP 03 COMPLETE")

# Compute Mean Calibration Per Timepoint

log("\n" + "=" * 70)
log("STEP 04: Compute Mean Calibration Per Timepoint")
log("=" * 70)

# Load original data with calibration values
df_cal = pd.read_csv(DATA_DIR / "step00_calibration_loaded.csv")

# Group by test and compute summary statistics
mean_results = []
for test in ['T1', 'T2', 'T3', 'T4']:
    test_data = df_cal[df_cal['test'] == test]['Calibration']
    n = len(test_data)
    mean_cal = test_data.mean()
    sd_cal = test_data.std(ddof=1)  # Sample SD
    se_cal = sd_cal / np.sqrt(n)
    ci_lower = mean_cal - 1.96 * se_cal
    ci_upper = mean_cal + 1.96 * se_cal

    mean_results.append({
        'test': test,
        'N': n,
        'mean_calibration': mean_cal,
        'SD_calibration': sd_cal,
        'SE_calibration': se_cal,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper
    })
    log(f"{test}: mean = {mean_cal:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], SD = {sd_cal:.4f}, N = {n}")

df_mean_cal = pd.DataFrame(mean_results)

# Validate
if len(df_mean_cal) != 4:
    log(f"ERROR: Expected 4 rows, found {len(df_mean_cal)}")
    sys.exit(1)

# Save step 04 output
output_file = DATA_DIR / "step04_mean_calibration.csv"
df_mean_cal.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_mean_cal)} rows)")
log("Mean calibration computed: 4 timepoints")
log("STEP 04 COMPLETE")

# Prepare Plot Data

log("\n" + "=" * 70)
log("STEP 05: Prepare Overconfidence Trajectory Plot Data")
log("=" * 70)

# Load proportion and mean data
df_prop = pd.read_csv(DATA_DIR / "step02_proportion_overconfident.csv")
df_mean = pd.read_csv(DATA_DIR / "step04_mean_calibration.csv")

# Merge on test
df_plot = df_prop.merge(df_mean, on='test', suffixes=('_prop', '_mean'))

# Rename columns for clarity
df_plot = df_plot.rename(columns={
    'CI_lower_prop': 'prop_CI_lower',
    'CI_upper_prop': 'prop_CI_upper',
    'CI_lower_mean': 'mean_CI_lower',
    'CI_upper_mean': 'mean_CI_upper'
})

# Add time_numeric column
time_map = {'T1': 0, 'T2': 1, 'T3': 3, 'T4': 6}
df_plot['time_numeric'] = df_plot['test'].map(time_map)

# Sort by time
df_plot = df_plot.sort_values('time_numeric')

# Select final columns
final_cols = [
    'test', 'time_numeric',
    'proportion_overconfident', 'prop_CI_lower', 'prop_CI_upper',
    'mean_calibration', 'mean_CI_lower', 'mean_CI_upper'
]
df_plot_final = df_plot[final_cols]

log(f"\nPlot data merged: {len(df_plot_final)} rows")
log("\nPlot data summary:")
for _, row in df_plot_final.iterrows():
    log(f"  {row['test']} (Day {row['time_numeric']}): prop_over={row['proportion_overconfident']:.3f}, mean_cal={row['mean_calibration']:.4f}")

# Validate merge
if len(df_plot_final) != 4:
    log(f"ERROR: Expected 4 rows after merge, found {len(df_plot_final)}")
    sys.exit(1)
log("Merge successful: 4 tests matched across proportion and mean files")

# Save step 05 output
output_file = DATA_DIR / "step05_overconfidence_trajectory_data.csv"
df_plot_final.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_plot_final)} rows)")
log("Plot data preparation complete: 4 rows created")
log("STEP 05 COMPLETE")

# SUMMARY

log("\n" + "=" * 70)
log("ANALYSIS COMPLETE - SUMMARY")
log("=" * 70)

# Reload key results for summary
df_prop = pd.read_csv(DATA_DIR / "step02_proportion_overconfident.csv")
df_trend = pd.read_csv(DATA_DIR / "step03_trend_test.csv")
df_mean = pd.read_csv(DATA_DIR / "step04_mean_calibration.csv")

# Summary statistics
log("\n1. PROPORTION OVERCONFIDENT TRAJECTORY:")
for _, row in df_prop.iterrows():
    log(f"   {row['test']}: {row['proportion_overconfident']:.1%} [{row['CI_lower']:.1%}, {row['CI_upper']:.1%}]")

# Change from T1 to T4
prop_t1 = df_prop[df_prop['test'] == 'T1']['proportion_overconfident'].iloc[0]
prop_t4 = df_prop[df_prop['test'] == 'T4']['proportion_overconfident'].iloc[0]
delta_prop = prop_t4 - prop_t1
log(f"\n   Change T1→T4: {delta_prop:+.1%}")

log("\n2. TREND TEST (Logistic Regression):")
time_row = df_trend[df_trend['term'] == 'time_ordinal'].iloc[0]
log(f"   Slope (log-odds/day): β = {time_row['estimate']:.4f} (SE = {time_row['SE']:.4f})")
log(f"   z = {time_row['z']:.3f}, p = {time_row['p_value']:.4f}")
log(f"   Odds ratio: {time_row['OR']:.4f} [{time_row['OR_CI_lower']:.4f}, {time_row['OR_CI_upper']:.4f}]")

if time_row['p_value'] < 0.05:
    significance = "SIGNIFICANT"
else:
    significance = "NON-SIGNIFICANT"
log(f"   Result: {significance} (α = 0.05)")

log("\n3. MEAN CALIBRATION TRAJECTORY:")
for _, row in df_mean.iterrows():
    log(f"   {row['test']}: {row['mean_calibration']:+.4f} [{row['CI_lower']:+.4f}, {row['CI_upper']:+.4f}]")

# Change from T1 to T4
mean_t1 = df_mean[df_mean['test'] == 'T1']['mean_calibration'].iloc[0]
mean_t4 = df_mean[df_mean['test'] == 'T4']['mean_calibration'].iloc[0]
delta_mean = mean_t4 - mean_t1
log(f"\n   Change T1→T4: {delta_mean:+.4f}")

# Interpretation
log("\n4. INTERPRETATION:")
if time_row['p_value'] < 0.05:
    if time_row['estimate'] > 0:
        log("   OVERCONFIDENCE INCREASES SIGNIFICANTLY OVER TIME")
        log("   Confidence lags behind accuracy decline (dual-process hypothesis supported)")
    else:
        log("   OVERCONFIDENCE DECREASES SIGNIFICANTLY OVER TIME")
        log("   Confidence declines faster than accuracy (unexpected pattern)")
else:
    log("   NO SIGNIFICANT TREND IN OVERCONFIDENCE")
    log("   Confidence and accuracy may decline in parallel (coupled system)")

# Cross-check with RQ 6.2.1
log("\n5. CROSS-REFERENCE WITH RQ 6.2.1:")
log(f"   RQ 6.2.1 found calibration worsens significantly (p_LRT=0.004)")
log(f"   This RQ tests directionality: Does proportion overconfident INCREASE?")
if mean_t1 < 0 and mean_t4 > 0:
    log("   CONFIRMED: Shift from underconfidence (T1) to overconfidence (T4)")
elif mean_t1 < 0 and mean_t4 < 0:
    log("   Pattern: Underconfident throughout, but trending toward zero")
else:
    log(f"   Pattern: T1={mean_t1:+.3f}, T4={mean_t4:+.3f}")

log("\n" + "=" * 70)
log("ALL STEPS COMPLETE")
log("=" * 70)
