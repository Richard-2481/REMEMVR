#!/usr/bin/env python3
"""Extract Accuracy Variance: Extract variance components from Ch5 5.1.4 model-averaged PowerLaw accuracy"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Configuration

SCRIPT_PATH = Path(__file__).resolve()
RQ_DIR = SCRIPT_PATH.parents[1]  # results/ch6/6.9.2
PROJECT_ROOT = SCRIPT_PATH.parents[4]  # REMEMVR/

LOG_FILE = RQ_DIR / "logs" / "step01_extract_accuracy_variance.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract Accuracy Variance Components")
        # Load Accuracy Random Effects from Ch5 5.1.4

        log("\nLoading accuracy random effects from Ch5 5.1.4...")
        re_file = PROJECT_ROOT / "results/ch5/5.1.4/data/step06_averaged_random_effects.csv"

        if not re_file.exists():
            raise FileNotFoundError(f"EXPECTATIONS ERROR: Expected file missing: {re_file}")

        df_re = pd.read_csv(re_file)
        log(f"{re_file.name} ({len(df_re)} rows, {len(df_re.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'intercept_avg', 'slope_avg']
        missing_cols = set(required_cols) - set(df_re.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in random effects: {missing_cols}")

        log(f"  Columns: {list(df_re.columns)}")
        log(f"  N participants: {len(df_re)}")
        # Compute Variance Components from Random Effects
        # Variance components: var_intercept, var_slope, cov_int_slope
        # Using ddof=1 for sample variance (unbiased estimator)

        log("\nComputing variance components...")

        var_intercept = np.var(df_re['intercept_avg'], ddof=1)
        var_slope = np.var(df_re['slope_avg'], ddof=1)

        # Covariance between intercept and slope
        cov_matrix = np.cov(df_re['intercept_avg'], df_re['slope_avg'])
        cov_int_slope = cov_matrix[0, 1]

        log(f"  var_intercept: {var_intercept:.6f}")
        log(f"  var_slope: {var_slope:.6f}")
        log(f"  cov_int_slope: {cov_int_slope:.6f}")

        # Validate non-negative variance
        if var_intercept < 0 or var_slope < 0:
            raise ValueError(f"Negative variance detected: var_int={var_intercept}, var_slope={var_slope}")
        # Extract Residual Variance from Ch5 5.1.4
        # Try variance components file first, fallback to searching other files

        log("\nExtracting residual variance from Ch5 5.1.4...")

        var_residual = None
        var_file = PROJECT_ROOT / "results/ch5/5.1.4/data/step06_averaged_variance_components.csv"

        if var_file.exists():
            try:
                df_var = pd.read_csv(var_file)
                log(f"{var_file.name}")

                # Check if file has var_residual column directly
                if 'var_residual' in df_var.columns:
                    var_residual = df_var['var_residual'].values[0]
                    log(f"  var_residual: {var_residual:.6f}")
            except Exception as e:
                log(f"  WARNING: Could not extract from {var_file.name}: {e}")

        # If not found, look for alternative file patterns
        if var_residual is None:
            log("  Searching for alternative variance files...")
            search_dir = PROJECT_ROOT / "results/ch5/5.1.4/data"
            variance_files = list(search_dir.glob("*variance*.csv"))

            for vf in variance_files:
                try:
                    df_var = pd.read_csv(vf)

                    # Try direct column access
                    if 'var_residual' in df_var.columns:
                        var_residual = df_var['var_residual'].values[0]
                        log(f"  Found in {vf.name}: var_residual = {var_residual:.6f}")
                        break

                    # Try component/value format
                    if 'component' in df_var.columns and 'value' in df_var.columns:
                        residual_row = df_var[df_var['component'].str.contains('residual', case=False, na=False)]
                        if not residual_row.empty:
                            var_residual = residual_row['value'].values[0]
                            log(f"  Found in {vf.name}: var_residual = {var_residual:.6f}")
                            break
                except:
                    continue

        if var_residual is None:
            raise ValueError("EXPECTATIONS ERROR: Could not extract var_residual from Ch5 5.1.4")

        # Validate non-negative residual variance
        if var_residual < 0:
            raise ValueError(f"Negative residual variance: {var_residual}")
        # Compute ICC Values
        # ICC_slope: Proportion of slope variance relative to total variance
        # ICC_conditional: Proportion accounting for correlation with intercepts

        log("\nComputing ICC values...")

        # ICC_slope = var_slope / (var_slope + var_residual)
        ICC_slope = var_slope / (var_slope + var_residual)

        # ICC_conditional = var_slope / (var_intercept + var_slope + var_residual)
        ICC_conditional = var_slope / (var_intercept + var_slope + var_residual)

        log(f"  ICC_slope: {ICC_slope:.6f}")
        log(f"  ICC_conditional: {ICC_conditional:.6f}")

        # Validate ICC bounds [0, 1]
        if not (0 <= ICC_slope <= 1):
            raise ValueError(f"ICC_slope out of bounds: {ICC_slope}")
        if not (0 <= ICC_conditional <= 1):
            raise ValueError(f"ICC_conditional out of bounds: {ICC_conditional}")
        # Create Output DataFrames
        # Variance components table with metadata

        log("\nCreating output files...")

        variance_components = pd.DataFrame({
            'component': [
                'var_intercept',
                'var_slope',
                'cov_int_slope',
                'var_residual',
                'ICC_slope',
                'ICC_conditional',
                'N_participants',
                'source_RQ',
                'functional_form',
                'N_models'
            ],
            'value': [
                var_intercept,
                var_slope,
                cov_int_slope,
                var_residual,
                ICC_slope,
                ICC_conditional,
                len(df_re),
                'ch5/5.1.4',
                'PowerLaw_lambda_0.41',
                9.84  # Model-averaged from Ch5 5.1.4
            ]
        })

        # Save variance components
        output_var = RQ_DIR / "data" / "step01_accuracy_variance_components.csv"
        variance_components.to_csv(output_var, index=False, encoding='utf-8')
        log(f"{output_var.name} ({len(variance_components)} rows)")

        # Save copy of random effects for reference
        output_re = RQ_DIR / "data" / "step01_accuracy_random_effects.csv"
        df_re.to_csv(output_re, index=False, encoding='utf-8')
        log(f"{output_re.name} ({len(df_re)} rows)")
        # Validation Summary
        # Report key statistics for validation

        log("\nSummary:")
        log(f"  All variance components >= 0: PASS")
        log(f"  ICC_slope in [0, 1]: PASS ({ICC_slope:.6f})")
        log(f"  ICC_conditional in [0, 1]: PASS ({ICC_conditional:.6f})")
        log(f"  N_participants = {len(df_re)}: {'PASS' if len(df_re) == 100 else 'CHECK'}")

        log("\nStep 01 complete - Accuracy variance components extracted")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
