#!/usr/bin/env python3
"""bivariate_correlation: Compute bivariate Pearson correlations between RAVLT predictors (forgetting index"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.bootstrap import bootstrap_correlation_ci

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2
LOG_FILE = RQ_DIR / "logs" / "step04_bivariate_correlation.log"

BONFERRONI_FACTOR = 28  # Decision D068: total Ch7 hypotheses

# Predictors to correlate with REMEMVR_Slope
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
        log("Step 04: bivariate_correlation (Forgetting + Pct Retention)")
        # Load Input Data

        log("Loading merged dataset...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path} ({len(df)} rows, {len(df.columns)} cols)")

        y = df['REMEMVR_Slope'].values
        log(f"REMEMVR_Slope - N={len(y)}, Mean={np.mean(y):.4f}, SD={np.std(y):.4f}")
        # Compute Correlations for Each Predictor

        all_results = []

        for predictor in PREDICTORS:
            log(f"\n=== Correlating {predictor} with REMEMVR_Slope ===")

            # Handle potential NaN in RAVLT_Pct_Ret
            mask = df[predictor].notna()
            x = df.loc[mask, predictor].values
            y_pred = df.loc[mask, 'REMEMVR_Slope'].values

            log(f"{predictor} - N={len(x)}, Mean={np.mean(x):.3f}, SD={np.std(x):.3f}")

            # Pearson correlation with p-value
            r, p_uncorrected = pearsonr(x, y_pred)
            log(f"Pearson r = {r:.4f}, p_uncorrected = {p_uncorrected:.6f}")

            # Bootstrap confidence interval
            log(f"Computing bootstrap confidence intervals...")
            boot_result = bootstrap_correlation_ci(
                x=x,
                y=y_pred,
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
                'correlation': r,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'N': len(x)
            })
        # Save Results

        log("\nSaving correlation results...")

        results_df = pd.DataFrame(all_results)

        output_path = RQ_DIR / "data" / "step04_bivariate_correlation.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(results_df)} rows, {len(results_df.columns)} cols)")
        # Run Validation Tool

        log("Running validate_correlation_test_d068...")

        # Validate each row
        for _, row in results_df.iterrows():
            row_df = pd.DataFrame([{
                'correlation': row['correlation'],
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
                    log(f"PASS - {row['predictor']} meets D068 requirements")
                else:
                    log(f"FAIL - {row['predictor']} validation errors")
                    if 'failed_checks' in validation_result:
                        for check in validation_result['failed_checks']:
                            log(f"Failed: {check}")

        # Summary
        log("\nBivariate Correlation Results:")
        log("=" * 70)
        for _, row in results_df.iterrows():
            log(f"  {row['predictor']}:")
            log(f"    r = {row['correlation']:.4f}, 95% CI [{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]")
            log(f"    p_uncorrected = {row['p_uncorrected']:.6f}, p_bonferroni = {row['p_bonferroni']:.6f}")
            log(f"    N = {int(row['N'])}")
        log("=" * 70)

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
