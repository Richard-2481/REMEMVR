#!/usr/bin/env python3
"""partial_correlation: Compute partial correlations between RAVLT predictors (forgetting index AND"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.bootstrap import bootstrap_correlation_ci

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2
LOG_FILE = RQ_DIR / "logs" / "step05_partial_correlation.log"

BONFERRONI_FACTOR = 28
PREDICTORS = ['RAVLT_Forgetting', 'RAVLT_Pct_Ret']

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: partial_correlation (Forgetting + Pct Retention)")
        # Load Input Data

        log("Loading merged dataset...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path} ({len(df)} rows, {len(df.columns)} cols)")

        all_results = []

        for predictor in PREDICTORS:
            log(f"\n=== Partial correlation: {predictor} ~ REMEMVR_Slope ===")

            # Handle potential NaN
            mask = df[predictor].notna()
            df_pred = df[mask].copy()

            # Extract variables
            pred_values = df_pred[predictor].values
            rememvr_slope = df_pred['REMEMVR_Slope'].values

            # Control variables
            ravlt_trial5 = df_pred['ravlt-trial-5-score'].values
            rememvr_intercept = df_pred['REMEMVR_Intercept'].values
            X_controls = np.column_stack([ravlt_trial5, rememvr_intercept])

            log(f"{predictor} - N={len(pred_values)}, Mean={np.mean(pred_values):.3f}, SD={np.std(pred_values):.3f}")
            log(f"REMEMVR_Slope - N={len(rememvr_slope)}, Mean={np.mean(rememvr_slope):.4f}, SD={np.std(rememvr_slope):.4f}")
            log(f"Control 1: ravlt-trial-5-score - Mean={np.mean(ravlt_trial5):.3f}, SD={np.std(ravlt_trial5):.3f}")
            log(f"Control 2: REMEMVR_Intercept - Mean={np.mean(rememvr_intercept):.3f}, SD={np.std(rememvr_intercept):.3f}")
            # Residualize Variables

            log(f"Residualizing {predictor} on control variables...")
            reg_pred = LinearRegression()
            reg_pred.fit(X_controls, pred_values)
            pred_residuals = pred_values - reg_pred.predict(X_controls)
            log(f"{predictor} residuals - Mean={np.mean(pred_residuals):.6f}, SD={np.std(pred_residuals):.3f}")
            log(f"R^2 for {predictor} ~ Controls: {reg_pred.score(X_controls, pred_values):.4f}")

            log(f"Residualizing REMEMVR_Slope on control variables...")
            reg_slope = LinearRegression()
            reg_slope.fit(X_controls, rememvr_slope)
            slope_residuals = rememvr_slope - reg_slope.predict(X_controls)
            log(f"Slope residuals - Mean={np.mean(slope_residuals):.6f}, SD={np.std(slope_residuals):.4f}")
            log(f"R^2 for Slope ~ Controls: {reg_slope.score(X_controls, rememvr_slope):.4f}")
            # Compute Partial Correlation

            log(f"Computing partial correlation (correlation of residuals)...")
            partial_r, p_uncorrected = pearsonr(pred_residuals, slope_residuals)
            log(f"Partial r = {partial_r:.4f}, p_uncorrected = {p_uncorrected:.6f}")

            # Bootstrap CI
            log(f"Computing bootstrap confidence intervals...")
            boot_result = bootstrap_correlation_ci(
                x=pred_residuals,
                y=slope_residuals,
                n_bootstrap=1000,
                confidence=0.95,
                method='pearson',
                seed=42
            )
            ci_lower = boot_result['ci_lower']
            ci_upper = boot_result['ci_upper']
            log(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

            # Bonferroni correction (D068)
            p_bonferroni = min(1.0, p_uncorrected * BONFERRONI_FACTOR)
            log(f"[D068] p_bonferroni = {p_bonferroni:.6f} (factor = {BONFERRONI_FACTOR})")

            all_results.append({
                'predictor': predictor,
                'partial_r': partial_r,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'N': len(pred_residuals)
            })
        # Save Results

        log("\nSaving partial correlation results...")

        results_df = pd.DataFrame(all_results)
        output_path = RQ_DIR / "data" / "step05_partial_correlation.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(results_df)} rows, {len(results_df.columns)} cols)")
        # Run Validation Tool

        log("Running validate_correlation_test_d068...")

        for _, row in results_df.iterrows():
            row_df = pd.DataFrame([{
                'correlation': row['partial_r'],
                'CI_lower': row['CI_lower'],
                'CI_upper': row['CI_upper'],
                'p_uncorrected': row['p_uncorrected'],
                'p_bonferroni': row['p_bonferroni'],
                'N': row['N']
            }])
            validation_result = validate_correlation_test_d068(
                correlation_df=row_df
            )
            if isinstance(validation_result, dict):
                if validation_result.get('valid', False):
                    log(f"PASS - {row['predictor']} partial correlation meets D068 requirements")
                else:
                    log(f"FAIL - {row['predictor']} validation errors")

        # Summary
        log("\nPartial Correlation Results:")
        log("=" * 70)
        for _, row in results_df.iterrows():
            log(f"  {row['predictor']}:")
            log(f"    partial r = {row['partial_r']:.4f}, 95% CI [{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]")
            log(f"    p_uncorrected = {row['p_uncorrected']:.6f}, p_bonferroni = {row['p_bonferroni']:.6f}")
        log("=" * 70)

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
