#!/usr/bin/env python3
"""
Simplified Step 06: Individual Predictors Analysis for RQ 7.3.1
Extract individual coefficients with dual p-values
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_individual_predictors.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Load data
log("Step 06: Individual Predictors (Simplified)")
df = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")
log(f"Analysis dataset: {len(df)} rows")

# Setup regression
X = df[['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']]
y = df['confidence_theta']
X_with_const = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X_with_const).fit()
log(f"R² = {model.rsquared:.4f}, F = {model.fvalue:.3f}, p = {model.f_pvalue:.6f}")

# Calculate VIF
vif_values = []
for i in range(1, X_with_const.shape[1]):
    vif = variance_inflation_factor(X_with_const.values, i)
    vif_values.append(vif)

# Calculate semi-partial correlations
sr2_values = []
full_r2 = model.rsquared
for predictor in X.columns:
    X_reduced = X.drop(columns=[predictor])
    X_reduced_const = sm.add_constant(X_reduced)
    reduced_model = sm.OLS(y, X_reduced_const).fit()
    sr2 = full_r2 - reduced_model.rsquared
    sr2_values.append(sr2)

# Apply corrections
p_uncorrected = model.pvalues[1:].values  # Skip intercept
_, p_fdr, _, _ = multipletests(p_uncorrected, method='fdr_bh')

# Bonferroni for cognitive tests only
p_bonferroni = []
for i, predictor in enumerate(X.columns):
    if predictor in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
        p_bonferroni.append(min(p_uncorrected[i] * 5, 1.0))
    else:
        p_bonferroni.append(p_uncorrected[i])

# Create results DataFrame
results = []
conf_int = model.conf_int()
for i, predictor in enumerate(X.columns):
    results.append({
        'predictor': predictor,
        'beta': model.params[i+1],
        'se': model.bse[i+1],
        'ci_lower': conf_int.iloc[i+1, 0],
        'ci_upper': conf_int.iloc[i+1, 1],
        'sr2': sr2_values[i],
        'p_uncorrected': p_uncorrected[i],
        'p_bonferroni': p_bonferroni[i],
        'p_fdr': p_fdr[i],
        'vif': vif_values[i]
    })
    
    # Report significance
    sig_bonf = p_bonferroni[i] < (0.000358 if predictor in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T'] else 0.05)
    log(f"{predictor}: β={results[-1]['beta']:.4f}, p={p_uncorrected[i]:.4f}, sig_bonf={sig_bonf}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(RQ_DIR / "data" / "step06_individual_predictors.csv", index=False)
log(f"Individual predictors results: {len(results_df)} predictors")

# Save diagnostics
with open(RQ_DIR / "data" / "step06_assumption_diagnostics.txt", 'w') as f:
    f.write("Regression Assumption Diagnostics\n")
    f.write("="*50 + "\n\n")
    f.write(f"Sample size: {len(df)}\n")
    f.write(f"R-squared: {model.rsquared:.4f}\n")
    f.write(f"F-statistic: {model.fvalue:.3f} (p={model.f_pvalue:.6f})\n\n")
    f.write("VIF Values:\n")
    for i, predictor in enumerate(X.columns):
        f.write(f"  {predictor}: {vif_values[i]:.3f}\n")
    f.write("\nSignificance Summary:\n")
    for res in results:
        sig = "***" if res['p_bonferroni'] < 0.001 else ("**" if res['p_bonferroni'] < 0.01 else ("*" if res['p_bonferroni'] < 0.05 else ""))
        f.write(f"  {res['predictor']}: β={res['beta']:.4f}, p={res['p_uncorrected']:.4f} {sig}\n")

log("Step 06 complete")