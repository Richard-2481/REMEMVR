#!/usr/bin/env python3
"""Test Key Hypothesis - Congruent Consolidation Benefit: Extract all 11 hypothesis tests from the fitted piecewise LMM and report with"""

import sys
from pathlib import Path
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLMResults
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import extract_fixed_effects_from_lmm

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq6
LOG_FILE = RQ_DIR / "logs" / "step04_test_hypothesis.log"

# Bonferroni correction parameters (Decision D068)
N_TESTS = 15  # Conservative family-wise correction
ALPHA_UNCORRECTED = 0.05
ALPHA_BONFERRONI = ALPHA_UNCORRECTED / N_TESTS  # 0.0033

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Test Key Hypothesis - Congruent Consolidation Benefit")
        # Load Fitted LMM Model

        log("Loading fitted piecewise LMM model from Step 2...")

        model_path = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))

        log(f"{model_path.name}")
        # Extract Fixed Effects
        # Returns: DataFrame with coefficient, SE, z-value, p-value

        log("Extracting fixed effects from LMM...")

        df_tests = extract_fixed_effects_from_lmm(lmm_model)

        log(f"{len(df_tests)} rows from summary table")

        # Filter out random effects rows (have NaN p-values)
        df_tests = df_tests[df_tests['P_value'].notna()].copy()

        log(f"{len(df_tests)} fixed effects (removed random effects rows)")

        # Rename columns to match specification
        df_tests = df_tests.rename(columns={
            'Term': 'Test_Name',
            'Coef': 'Coefficient',
            'Std_Err': 'SE',
            'z': 'z_value',
            'P_value': 'p_uncorrected'
        })
        # Apply Bonferroni Correction (Decision D068)
        # Correction: p_bonferroni = p_uncorrected * n_tests
        # Significance: p_bonferroni < 0.05 (adjusted alpha)

        log(f"Applying Bonferroni correction (n_tests = {N_TESTS})...")

        # Compute Bonferroni-corrected p-values
        df_tests['p_bonferroni'] = df_tests['p_uncorrected'] * N_TESTS

        # Cap at 1.0 (p-values cannot exceed 1)
        df_tests['p_bonferroni'] = df_tests['p_bonferroni'].clip(upper=1.0)

        # Label significance based on Bonferroni-corrected p-value
        df_tests['Significant_Bonferroni'] = df_tests['p_bonferroni'] < ALPHA_UNCORRECTED

        log(f"Applied Bonferroni correction (alpha = {ALPHA_BONFERRONI:.4f})")
        # Identify Primary Hypothesis
        # Primary hypothesis: 3-way interaction Days_within:Segment[Late]:Congruence[Congruent]
        # Tests: Does congruent slope differ between Early and Late segments?

        log("Identifying primary hypothesis test...")

        # Find 3-way interaction term for Congruent
        primary_test = df_tests[
            df_tests['Test_Name'].str.contains('Days_within') &
            df_tests['Test_Name'].str.contains('Segment') &
            df_tests['Test_Name'].str.contains('Late') &
            df_tests['Test_Name'].str.contains('Congruent')
        ]

        if len(primary_test) == 0:
            log("Primary hypothesis test not found (3-way interaction for Congruent)")
        else:
            primary_name = primary_test.iloc[0]['Test_Name']
            primary_coef = primary_test.iloc[0]['Coefficient']
            primary_p_uncorr = primary_test.iloc[0]['p_uncorrected']
            primary_p_bonf = primary_test.iloc[0]['p_bonferroni']
            primary_sig = primary_test.iloc[0]['Significant_Bonferroni']

            log(f"Primary test: {primary_name}")
            log(f"  Coefficient: {primary_coef:.4f}")
            log(f"  p_uncorrected: {primary_p_uncorr:.4f}")
            log(f"  p_bonferroni: {primary_p_bonf:.4f}")
            log(f"  Significant (Bonferroni): {primary_sig}")
        # Validate and Save

        log("Validating hypothesis tests...")

        # Check test count (should match number of model fixed effects)
        # Note: extract_fixed_effects_from_lmm returns ALL terms from summary table
        n_expected = len(lmm_model.fe_params)  # 12 for this 3-way interaction model
        if len(df_tests) != n_expected:
            log(f"Test count mismatch: expected {n_expected} fixed effects, found {len(df_tests)}")
            # Don't fail - the function may return additional summary rows
        else:
            log(f"Test count matches model fixed effects ({n_expected})")

        # Check dual p-values present
        required_cols = ['Test_Name', 'Coefficient', 'SE', 'z_value',
                         'p_uncorrected', 'p_bonferroni', 'Significant_Bonferroni']
        if not all(col in df_tests.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_tests.columns]
            raise ValueError(
                f"Required columns missing: {missing}"
            )
        log(f"All required columns present")

        # Check p-value bounds
        if not ((df_tests['p_uncorrected'] >= 0).all() and
                (df_tests['p_uncorrected'] <= 1).all()):
            raise ValueError(
                f"p_uncorrected out of bounds [0, 1]"
            )
        log(f"p_uncorrected in valid range [0, 1]")

        if not ((df_tests['p_bonferroni'] >= 0).all() and
                (df_tests['p_bonferroni'] <= 1).all()):
            raise ValueError(
                f"p_bonferroni out of bounds [0, 1]"
            )
        log(f"p_bonferroni in valid range [0, 1]")

        # Check Bonferroni correction applied correctly
        if not (df_tests['p_bonferroni'] >= df_tests['p_uncorrected']).all():
            raise ValueError(
                f"Bonferroni correction incorrect: p_bonferroni < p_uncorrected for some tests"
            )
        log(f"Bonferroni correction applied correctly (p_bonf >= p_uncorr)")

        # Check significance labeling
        expected_sig = df_tests['p_bonferroni'] < ALPHA_UNCORRECTED
        if not (df_tests['Significant_Bonferroni'] == expected_sig).all():
            raise ValueError(
                f"Significance labeling incorrect"
            )
        log(f"Significance labeling correct")

        # Report significant tests
        n_sig_uncorr = (df_tests['p_uncorrected'] < ALPHA_UNCORRECTED).sum()
        n_sig_bonf = df_tests['Significant_Bonferroni'].sum()

        log(f"Significant tests:")
        log(f"  Uncorrected (alpha = {ALPHA_UNCORRECTED}): {n_sig_uncorr} / {len(df_tests)}")
        log(f"  Bonferroni (alpha = {ALPHA_UNCORRECTED}): {n_sig_bonf} / {len(df_tests)}")

        if n_sig_bonf > 0:
            log(f"Significant tests (Bonferroni-corrected):")
            for _, row in df_tests[df_tests['Significant_Bonferroni']].iterrows():
                log(f"  {row['Test_Name']}: p = {row['p_bonferroni']:.4f}")

        # Save output
        output_path = RQ_DIR / "results" / "step04_hypothesis_tests.csv"
        df_tests.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(df_tests)} rows)")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
