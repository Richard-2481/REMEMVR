#!/usr/bin/env python3
"""diagnostic_tests: Test correlation assumptions including normality (Kolmogorov-Smirnov),"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import kstest, normaltest, pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2
LOG_FILE = RQ_DIR / "logs" / "step06_diagnostics.log"

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
        log("Step 06: diagnostic_tests (Forgetting + Pct Retention)")
        # Load Input Data

        log("Loading merged dataset...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path} ({len(df)} rows, {len(df.columns)} cols)")

        diagnostics = []

        for predictor in PREDICTORS:
            log(f"\n=== Diagnostics for {predictor} vs REMEMVR_Slope ===")

            # Handle NaN
            mask = df[predictor].notna()
            x = df.loc[mask, predictor].values
            y = df.loc[mask, 'REMEMVR_Slope'].values

            log(f"{predictor} - N={len(x)}, Mean={np.mean(x):.3f}, SD={np.std(x):.3f}")
            log(f"REMEMVR_Slope - N={len(y)}, Mean={np.mean(y):.4f}, SD={np.std(y):.4f}")

            # -----------------------------------------------------------------
            # Test 1: Normality of predictor (K-S)
            # -----------------------------------------------------------------
            log(f"Testing normality of {predictor} (Kolmogorov-Smirnov)...")
            x_std = (x - np.mean(x)) / np.std(x)
            ks_x, p_ks_x = kstest(x_std, 'norm')
            conclusion_x = 'PASS' if p_ks_x > 0.05 else 'FAIL'
            log(f"K-S test {predictor}: statistic={ks_x:.4f}, p={p_ks_x:.4f}, conclusion={conclusion_x}")
            diagnostics.append({
                'predictor': predictor,
                'test_type': f'normality_{predictor}',
                'statistic': ks_x,
                'p_value': p_ks_x,
                'conclusion': conclusion_x
            })

            # -----------------------------------------------------------------
            # Test 2: Normality of REMEMVR_Slope (K-S)
            # -----------------------------------------------------------------
            log(f"Testing normality of REMEMVR_Slope (Kolmogorov-Smirnov)...")
            y_std = (y - np.mean(y)) / np.std(y)
            ks_y, p_ks_y = kstest(y_std, 'norm')
            conclusion_y = 'PASS' if p_ks_y > 0.05 else 'FAIL'
            log(f"K-S test REMEMVR: statistic={ks_y:.4f}, p={p_ks_y:.4f}, conclusion={conclusion_y}")
            diagnostics.append({
                'predictor': predictor,
                'test_type': 'normality_REMEMVR',
                'statistic': ks_y,
                'p_value': p_ks_y,
                'conclusion': conclusion_y
            })

            # -----------------------------------------------------------------
            # Test 3: Linearity (Pearson vs Spearman)
            # -----------------------------------------------------------------
            log(f"Testing linearity (Pearson vs Spearman comparison)...")
            r_pearson, _ = pearsonr(x, y)
            r_spearman, _ = spearmanr(x, y)
            linearity_diff = abs(r_pearson - r_spearman)
            conclusion_linear = 'PASS' if linearity_diff < 0.1 else 'FAIL'
            log(f"Pearson r={r_pearson:.4f}, Spearman r={r_spearman:.4f}, diff={linearity_diff:.4f}")
            log(f"Linearity conclusion: {conclusion_linear} (threshold=0.1)")
            diagnostics.append({
                'predictor': predictor,
                'test_type': 'linearity',
                'statistic': linearity_diff,
                'p_value': np.nan,
                'conclusion': conclusion_linear
            })

            # -----------------------------------------------------------------
            # Test 4: Outlier Detection (|z| > 3)
            # -----------------------------------------------------------------
            log(f"Detecting outliers (|z-score| > 3)...")
            z_x = np.abs((x - np.mean(x)) / np.std(x))
            z_y = np.abs((y - np.mean(y)) / np.std(y))
            outlier_mask = (z_x > 3) | (z_y > 3)
            n_outliers = np.sum(outlier_mask)
            outlier_pct = (n_outliers / len(x)) * 100
            log(f"Outliers detected: {n_outliers} ({outlier_pct:.2f}%)")
            if n_outliers > 0:
                outlier_indices = np.where(outlier_mask)[0].tolist()
                log(f"Outlier indices: {outlier_indices}")
            diagnostics.append({
                'predictor': predictor,
                'test_type': 'outliers',
                'statistic': n_outliers,
                'p_value': np.nan,
                'conclusion': f"{n_outliers} outliers detected ({outlier_pct:.2f}%)"
            })
        # Save Diagnostic Results

        log("\nSaving diagnostic results...")
        diagnostics_df = pd.DataFrame(diagnostics)
        output_path = RQ_DIR / "data" / "step06_diagnostics.csv"
        diagnostics_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(diagnostics_df)} rows, {len(diagnostics_df.columns)} cols)")
        # Run Validation Tool

        log("Running validate_data_format...")
        expected_columns = ['predictor', 'test_type', 'statistic', 'p_value', 'conclusion']
        validation_result = validate_data_format(
            df=diagnostics_df,
            required_cols=expected_columns
        )
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            if validation_result.get('valid', False):
                log("PASS - Diagnostic results format valid")
            else:
                log("FAIL - Format validation errors detected")
        else:
            log(f"{validation_result}")

        # Summary
        log("\nDiagnostic Test Results:")
        log("=" * 60)
        for _, row in diagnostics_df.iterrows():
            log(f"  [{row['predictor']}] {row['test_type']}: {row['conclusion']}")
        log("=" * 60)

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
