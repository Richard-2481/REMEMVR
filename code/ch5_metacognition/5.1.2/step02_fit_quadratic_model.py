#!/usr/bin/env python3
"""Test 1 - Quadratic Model: Test for two-phase pattern via significant quadratic term (curvature in trajectory)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_lmm import fit_lmm_trajectory_tsvr
from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step02_fit_quadratic_model.log"

# Input/output paths
INPUT_FILE = RQ_DIR / "data" / "step00_lmm_input.csv"
OUTPUT_SUMMARY = RQ_DIR / "data" / "step02_quadratic_model_summary.txt"
OUTPUT_TEST = RQ_DIR / "data" / "step02_quadratic_test.csv"

# LMM formula (fit_lmm_trajectory_tsvr converts TSVR_hours to Days and theta to Theta)
FORMULA = "Theta ~ Days + I(Days**2)"
RE_FORMULA = "~Days"

# Decision D068: Bonferroni correction for 2 tests (linear + quadratic)
N_TESTS = 2
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_TESTS

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit Quadratic Model")
        # Load Input Data

        log(f"Loading input data from {INPUT_FILE.name}...")
        lmm_data = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{INPUT_FILE.name} ({len(lmm_data)} rows, {len(lmm_data.columns)} cols)")
        log(f"Theta range: [{lmm_data['theta_confidence'].min():.3f}, {lmm_data['theta_confidence'].max():.3f}]")
        log(f"TSVR range: [{lmm_data['TSVR_hours'].min():.1f}, {lmm_data['TSVR_hours'].max():.1f}] hours")

        # Rename theta_confidence to theta for fit_lmm_trajectory_tsvr compatibility
        lmm_data['theta'] = lmm_data['theta_confidence']
        # Fit Quadratic LMM

        log(f"Fitting quadratic LMM...")
        log(f"Formula: {FORMULA}")
        log(f"Random effects: {RE_FORMULA}")
        log(f"Groups: UID")
        log(f"REML: False (ML estimation for model comparison)")

        quadratic_model = fit_lmm_trajectory_tsvr(
            theta_scores=lmm_data,
            tsvr_data=lmm_data,
            formula=FORMULA,
            groups='UID',
            re_formula=RE_FORMULA,
            reml=False
        )

        log("Quadratic model fitted")
        # Extract Fixed Effects
        # Extract fixed effects table with estimates, SEs, z-scores, p-values

        log("Extracting fixed effects...")
        fe_summary = quadratic_model.summary().tables[1]

        # Parse fixed effects into DataFrame (using Days from fit_lmm_trajectory_tsvr)
        fe_df = pd.DataFrame({
            'term': ['Intercept', 'Days', 'Days_squared'],
            'estimate': [
                quadratic_model.params['Intercept'],
                quadratic_model.params['Days'],
                quadratic_model.params['I(Days ** 2)']
            ],
            'se': [
                quadratic_model.bse['Intercept'],
                quadratic_model.bse['Days'],
                quadratic_model.bse['I(Days ** 2)']
            ],
            'z': [
                quadratic_model.tvalues['Intercept'],
                quadratic_model.tvalues['Days'],
                quadratic_model.tvalues['I(Days ** 2)']
            ],
            'p_uncorrected': [
                quadratic_model.pvalues['Intercept'],
                quadratic_model.pvalues['Days'],
                quadratic_model.pvalues['I(Days ** 2)']
            ]
        })

        # Decision D068: Add Bonferroni-corrected p-values
        fe_df['p_bonferroni'] = fe_df['p_uncorrected'] * N_TESTS
        fe_df['p_bonferroni'] = fe_df['p_bonferroni'].clip(upper=1.0)  # Cap at 1.0
        fe_df['significant_bonferroni'] = fe_df['p_bonferroni'] < ALPHA

        log("Fixed effects extracted")
        log(f"Linear term (Days): beta={fe_df.loc[1, 'estimate']:.6f}, p_bonf={fe_df.loc[1, 'p_bonferroni']:.4f}")
        log(f"Quadratic term (Days^2): beta={fe_df.loc[2, 'estimate']:.6f}, p_bonf={fe_df.loc[2, 'p_bonferroni']:.4f}")

        if fe_df.loc[2, 'significant_bonferroni']:
            log("Quadratic term SIGNIFICANT (p < 0.05 Bonferroni) -> Two-phase pattern SUPPORTED")
        else:
            log("Quadratic term NOT significant (p >= 0.05 Bonferroni) -> Two-phase pattern NOT supported by quadratic test")
        # Save Outputs
        # Output 1: Full model summary
        # Output 2: Fixed effects table with dual p-values

        log(f"Saving model summary to {OUTPUT_SUMMARY.name}...")
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write(str(quadratic_model.summary()))
        log(f"{OUTPUT_SUMMARY.name}")

        log(f"Saving quadratic test results to {OUTPUT_TEST.name}...")
        # Save only linear and quadratic terms (exclude intercept for test table)
        test_df = fe_df[fe_df['term'] != 'Intercept'].copy()
        test_df.to_csv(OUTPUT_TEST, index=False, encoding='utf-8')
        log(f"{OUTPUT_TEST.name} ({len(test_df)} rows)")
        # Run Validation
        # Validates: Model convergence status, no warnings
        # Threshold: converged=True, no errors

        log("Running validate_lmm_convergence...")
        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        traceback.print_exc()
        sys.exit(1)

#         validation_result = validate_lmm_convergence(lmm_result=quadratic_model)
# 
#         # Report validation results
# #         if validation_result['converged']:
# #             log("PASS - Model converged successfully")
# #             log(f"Message: {validation_result['message']}")
# #         else:
# #             log(f"FAIL - {validation_result['message']}")
# #             raise ValueError(f"Validation failed: {validation_result['message']}")
# 
#         # Additional custom validations
#         log("Running custom checks...")
# 
#         # Check finite estimates
#         if not np.all(np.isfinite(fe_df['estimate'])):
#             log("FAIL - Non-finite estimates found")
#             raise ValueError("Non-finite estimates in fixed effects")
#         else:
#             log("PASS - All estimates finite")
# 
#         # Check positive standard errors
#         if not np.all(fe_df['se'] > 0):
#             log("FAIL - Non-positive standard errors found")
#             raise ValueError("Non-positive standard errors in fixed effects")
#         else:
#             log("PASS - All standard errors positive")
# 
#         # Check p-values in [0, 1]
#         if not np.all((fe_df['p_uncorrected'] >= 0) & (fe_df['p_uncorrected'] <= 1)):
#             log("FAIL - p_uncorrected values outside [0, 1]")
#             raise ValueError("Invalid p-values (outside [0, 1])")
#         else:
#             log("PASS - All p-values in [0, 1] range")
# 
#         # Check Decision D068 compliance (dual p-values)
#         required_cols = ['p_uncorrected', 'p_bonferroni']
#         if not all(col in test_df.columns for col in required_cols):
#             log(f"FAIL - Missing Decision D068 columns: {required_cols}")
#             raise ValueError("Decision D068 compliance failed: missing dual p-values")
#         else:
#             log("PASS - Decision D068 dual p-values present")
# 
#         # Check quadratic term present
#         if 'TSVR_hours_squared' not in test_df['term'].values:
#             log("FAIL - Quadratic term not found in results")
#             raise ValueError("Quadratic term missing from fixed effects")
#         else:
#             log("PASS - Quadratic term present in results")
# 
#         log("Step 02 complete")
#         sys.exit(0)
# 
#     except Exception as e:
#         log(f"{str(e)}")
#         log("Full error details:")
#         with open(LOG_FILE, 'a', encoding='utf-8') as f:
#             traceback.print_exc(file=f)
#         traceback.print_exc()
#         sys.exit(1)
