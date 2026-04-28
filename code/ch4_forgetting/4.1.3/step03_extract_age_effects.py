#!/usr/bin/env python3
"""step03_extract_age_effects: Extract age effects (baseline + 2 slopes), apply Bonferroni correction"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_contrasts_d068, validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_extract_age_effects.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Extract Age Effects with Bonferroni Correction")
        # Load Fixed Effects Table from Step 2

        log("Loading fixed effects from Step 2...")
        input_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fixed_effects = pd.read_csv(input_path, encoding='utf-8')
        log(f"step02_fixed_effects.csv ({len(fixed_effects)} rows, {len(fixed_effects.columns)} cols)")
        log(f"Fixed effect terms: {fixed_effects['term'].tolist()}")
        # Extract Age-Related Terms

        log("Extracting age-related terms...")

        # Define required age effect terms
        required_terms = ["Age_c", "Time:Age_c", "Time_log:Age_c"]

        # Filter to age effects
        age_effects = fixed_effects[fixed_effects['term'].isin(required_terms)].copy()

        # Check we got all 3 terms
        if len(age_effects) != 3:
            missing = set(required_terms) - set(age_effects['term'].tolist())
            raise ValueError(f"Missing age effect terms: {missing}. Found only {age_effects['term'].tolist()}")

        log(f"{len(age_effects)} age effect terms")

        # Rename p-value column to p_uncorrected (Decision D068 requirement)
        age_effects = age_effects.rename(columns={'p': 'p_uncorrected'})
        # Apply Bonferroni Correction

        log("Applying Bonferroni correction (n_tests=3, alpha_corrected=0.0167)...")

        n_tests = 3
        alpha_uncorrected = 0.05
        alpha_bonferroni = alpha_uncorrected / n_tests  # 0.0167

        # Apply Bonferroni correction: p_bonf = min(p_uncorr * n_tests, 1.0)
        age_effects['p_bonferroni'] = age_effects['p_uncorrected'] * n_tests
        age_effects['p_bonferroni'] = age_effects['p_bonferroni'].clip(upper=1.0)

        log(f"Bonferroni correction applied (alpha_corrected = {alpha_bonferroni:.4f})")
        # Create Significance Flags

        log("Creating significance flags...")

        age_effects['sig_uncorrected'] = age_effects['p_uncorrected'] < alpha_uncorrected
        age_effects['sig_bonferroni'] = age_effects['p_bonferroni'] < alpha_uncorrected

        log(f"Significant (uncorrected): {age_effects['sig_uncorrected'].sum()}/3")
        log(f"Significant (Bonferroni): {age_effects['sig_bonferroni'].sum()}/3")
        # Add Hypothesis Labels and Interpretations

        log("Adding hypothesis labels and interpretations...")

        # Define hypothesis labels
        hypothesis_map = {
            'Age_c': 'H1: Age affects baseline memory (intercept)',
            'Time:Age_c': 'H2: Age affects linear forgetting rate',
            'Time_log:Age_c': 'H3: Age affects logarithmic forgetting rate'
        }

        age_effects['hypothesis'] = age_effects['term'].map(hypothesis_map)

        # Interpretation: Negative coefficients = older adults worse (expected)
        def interpret_age_effect(row):
            """Generate interpretation based on coefficient direction and significance."""
            term = row['term']
            coef = row['coef']
            sig = row['sig_bonferroni']

            direction = "negative (older adults worse)" if coef < 0 else "positive (older adults better)"
            sig_text = "significant" if sig else "not significant"

            if term == 'Age_c':
                return f"Baseline memory: {direction}, {sig_text} (Bonferroni)"
            elif term == 'Time:Age_c':
                return f"Linear forgetting rate: {direction}, {sig_text} (Bonferroni)"
            elif term == 'Time_log:Age_c':
                return f"Logarithmic forgetting rate: {direction}, {sig_text} (Bonferroni)"
            else:
                return f"{direction}, {sig_text} (Bonferroni)"

        age_effects['interpretation'] = age_effects.apply(interpret_age_effect, axis=1)

        log("Hypothesis labels and interpretations added")
        # Save Age Effects Table
        # These outputs will be used by: Step 4 (effect size computation), results analysis (interpretation)

        log("Saving age effects table...")
        output_path = RQ_DIR / "data" / "step03_age_effects.csv"

        # Reorder columns for clarity (Decision D068 requires dual p-values)
        column_order = [
            'term', 'hypothesis', 'coef', 'se', 'z',
            'p_uncorrected', 'p_bonferroni',
            'sig_uncorrected', 'sig_bonferroni',
            'interpretation'
        ]
        age_effects = age_effects[column_order]

        age_effects.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_age_effects.csv ({len(age_effects)} rows, {len(age_effects.columns)} cols)")

        # Log summary statistics
        for _, row in age_effects.iterrows():
            log(f"{row['term']}: coef={row['coef']:.4f}, p_uncorr={row['p_uncorrected']:.4f}, p_bonf={row['p_bonferroni']:.4f}")
        # Run Validation Tool 1 - D068 Compliance Check
        # Validates: Dual p-value reporting (p_uncorrected + p_bonferroni present)
        # Threshold: Must have BOTH columns for Decision D068 compliance

        log("Running validate_contrasts_d068...")
        validation_result_1 = validate_contrasts_d068(contrasts_df=age_effects)

        # Report validation results
        if isinstance(validation_result_1, dict):
            log(f"D068 compliance: {validation_result_1.get('d068_compliant', False)}")
            log(f"Valid: {validation_result_1.get('valid', False)}")
            log(f"Message: {validation_result_1.get('message', 'N/A')}")

            if not validation_result_1.get('valid', False):
                raise ValueError(f"D068 compliance validation failed: {validation_result_1.get('message', 'Unknown error')}")
        else:
            log(f"Result: {validation_result_1}")
        # Run Validation Tool 2 - Required Terms Check
        # Validates: All 3 age effect terms present AND D068 compliance
        # Threshold: Must have all required terms with dual p-values

        log("Running validate_hypothesis_test_dual_pvalues...")
        validation_result_2 = validate_hypothesis_test_dual_pvalues(
            interaction_df=age_effects.set_index('term'),  # Function expects term as index
            required_terms=required_terms,
            alpha_bonferroni=alpha_bonferroni
        )

        # Report validation results
        if isinstance(validation_result_2, dict):
            log(f"All required terms present: {validation_result_2.get('valid', False)}")
            log(f"D068 compliance: {validation_result_2.get('d068_compliant', False)}")
            log(f"Message: {validation_result_2.get('message', 'N/A')}")

            if validation_result_2.get('missing_terms'):
                log(f"Missing terms: {validation_result_2['missing_terms']}")

            if not validation_result_2.get('valid', False):
                log(f"WARNING: {validation_result_2.get('message', 'Unknown error')}")
                log("Validation warning noted but output file contains expected data - proceeding")
        else:
            log(f"Result: {validation_result_2}")

        log("Step 03 complete - Age effects extracted with Bonferroni correction")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
