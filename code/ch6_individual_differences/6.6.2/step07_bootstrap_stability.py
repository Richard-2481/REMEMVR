#!/usr/bin/env python3
"""bootstrap_stability: Assess correlation stability through bootstrap resampling, outlier-robust"""

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

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2
LOG_FILE = RQ_DIR / "logs" / "step07_bootstrap_stability.log"

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
        log("Step 07: bootstrap_stability (Forgetting + Pct Retention)")
        # Load Input Data

        log("Loading merged dataset...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path} ({len(df)} rows, {len(df.columns)} cols)")

        all_stability_results = []

        for predictor in PREDICTORS:
            log(f"\n=== Bootstrap Stability for {predictor} ===")

            # Handle NaN
            mask = df[predictor].notna()
            x = df.loc[mask, predictor].values
            y = df.loc[mask, 'REMEMVR_Slope'].values

            log(f"{predictor} - N={len(x)}, Mean={np.mean(x):.3f}, SD={np.std(x):.3f}")

            # -----------------------------------------------------------------
            # Full Sample Bootstrap
            # -----------------------------------------------------------------
            log(f"Full sample bootstrap (n=1000)...")
            full_boot = bootstrap_correlation_ci(
                x=x, y=y, n_bootstrap=1000, confidence=0.95, method='pearson', seed=42
            )
            log(f"Full sample r={full_boot['r']:.4f}, CI=[{full_boot['ci_lower']:.4f}, {full_boot['ci_upper']:.4f}]")

            all_stability_results.append({
                'predictor': predictor,
                'metric': 'full_sample_correlation',
                'mean_estimate': full_boot['r'],
                'CI_lower': full_boot['ci_lower'],
                'CI_upper': full_boot['ci_upper'],
                'stability_index': 1.0
            })

            # -----------------------------------------------------------------
            # Outlier-Robust Analysis (Remove |z| > 2.5)
            # -----------------------------------------------------------------
            log(f"Outlier-robust analysis (remove |z| > 2.5)...")
            z_x = np.abs((x - np.mean(x)) / np.std(x))
            z_y = np.abs((y - np.mean(y)) / np.std(y))
            clean_idx = (z_x <= 2.5) & (z_y <= 2.5)
            x_clean = x[clean_idx]
            y_clean = y[clean_idx]

            log(f"Removed {len(x) - len(x_clean)} outliers ({100 * (1 - len(x_clean)/len(x)):.1f}%)")
            log(f"Clean sample N={len(x_clean)}")

            if len(x_clean) >= 20:  # Ensure adequate sample size
                clean_boot = bootstrap_correlation_ci(
                    x=x_clean, y=y_clean, n_bootstrap=1000, confidence=0.95, method='pearson', seed=42
                )
                stability = 1.0 - abs(full_boot['r'] - clean_boot['r'])
                log(f"Outlier-robust r={clean_boot['r']:.4f}, CI=[{clean_boot['ci_lower']:.4f}, {clean_boot['ci_upper']:.4f}]")
                log(f"Stability index={stability:.4f}")

                all_stability_results.append({
                    'predictor': predictor,
                    'metric': 'outlier_robust_correlation',
                    'mean_estimate': clean_boot['r'],
                    'CI_lower': clean_boot['ci_lower'],
                    'CI_upper': clean_boot['ci_upper'],
                    'stability_index': stability
                })
            else:
                log(f"Clean sample too small (N={len(x_clean)}), skipping outlier-robust analysis")

            # -----------------------------------------------------------------
            # Subsample Stability (Random 80%)
            # -----------------------------------------------------------------
            log(f"Subsample stability (random 80%)...")
            np.random.seed(42)
            n_subsample = int(0.8 * len(x))
            subsample_idx = np.random.choice(len(x), n_subsample, replace=False)
            x_sub = x[subsample_idx]
            y_sub = y[subsample_idx]

            log(f"Subsample N={len(x_sub)} (80% of original)")

            sub_boot = bootstrap_correlation_ci(
                x=x_sub, y=y_sub, n_bootstrap=1000, confidence=0.95, method='pearson', seed=42
            )
            subsample_stability = 1.0 - abs(full_boot['r'] - sub_boot['r'])
            log(f"Subsample r={sub_boot['r']:.4f}, CI=[{sub_boot['ci_lower']:.4f}, {sub_boot['ci_upper']:.4f}]")
            log(f"Stability index={subsample_stability:.4f}")

            all_stability_results.append({
                'predictor': predictor,
                'metric': 'subsample_correlation',
                'mean_estimate': sub_boot['r'],
                'CI_lower': sub_boot['ci_lower'],
                'CI_upper': sub_boot['ci_upper'],
                'stability_index': subsample_stability
            })
        # Save Stability Results

        log("\nSaving bootstrap stability results...")
        stability_df = pd.DataFrame(all_stability_results)
        output_path = RQ_DIR / "data" / "step07_bootstrap_results.csv"
        stability_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(stability_df)} rows, {len(stability_df.columns)} cols)")
        # Run Validation Tool

        log("Running validate_numeric_range on stability_index...")
        validation_result = validate_numeric_range(
            data=stability_df['stability_index'],
            min_val=0.0,
            max_val=1.0,
            column_name='stability_index'
        )
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            if validation_result.get('valid', False):
                log("PASS - All stability indices in valid range [0, 1]")
            else:
                log("FAIL - Some stability indices out of range")
        else:
            log(f"{validation_result}")

        # Summary
        log("\nBootstrap Stability Results:")
        log("=" * 70)
        for _, row in stability_df.iterrows():
            log(f"  [{row['predictor']}] {row['metric']}: r={row['mean_estimate']:.4f}, stability={row['stability_index']:.4f}")
        log("=" * 70)

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
