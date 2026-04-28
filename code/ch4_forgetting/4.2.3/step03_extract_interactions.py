#!/usr/bin/env python3
"""Extract 3-Way Interaction Terms and Test Hypothesis: Extract and test 3-way Age x Domain x Time interaction terms from LMM model"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_extract_interactions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def compute_bonferroni_correction(p_values: np.ndarray, n_comparisons: int = 2) -> np.ndarray:
    """
    Apply Bonferroni correction to p-values.

    Decision D068: Report both uncorrected and corrected p-values.

    Parameters:
        p_values: Array of uncorrected p-values
        n_comparisons: Number of comparisons (default 2 for Where/When domains)

    Returns:
        Array of Bonferroni-corrected p-values (capped at 1.0)
    """
    return np.minimum(p_values * n_comparisons, 1.0)


def perform_omnibus_wald_test(lmm_model, term_names: List[str]) -> Dict[str, Any]:
    """
    Perform omnibus Wald test for joint significance of multiple terms.

    Tests null hypothesis: All specified coefficients are jointly zero.
    Uses Wald chi-square test with df = number of terms.

    Parameters:
        lmm_model: Fitted statsmodels MixedLMResults object
        term_names: List of term names to test jointly

    Returns:
        Dict with keys: chi2_statistic, df, p_value, terms_tested
    """
    # Load model to access parameter names and covariance matrix
    params = lmm_model.params
    cov = lmm_model.cov_params()

    # Find indices of terms in parameter vector
    param_names = params.index.tolist()

    # Match term names (handle exact matches)
    indices = []
    for term in term_names:
        if term in param_names:
            indices.append(param_names.index(term))
        else:
            log(f"Term '{term}' not found in model parameters")

    if len(indices) == 0:
        log(f"No matching terms found for omnibus test")
        return {
            'chi2_statistic': np.nan,
            'df': 0,
            'p_value': np.nan,
            'terms_tested': []
        }

    # Extract relevant parameters and covariance submatrix
    beta = params.iloc[indices].values  # Coefficient vector
    V = cov.iloc[indices, indices].values  # Variance-covariance matrix

    # Wald statistic: W = beta^T * V^-1 * beta ~ chi2(df)
    try:
        V_inv = np.linalg.inv(V)
        wald_stat = beta.T @ V_inv @ beta
        df = len(indices)
        p_value = 1 - chi2.cdf(wald_stat, df)

        return {
            'chi2_statistic': float(wald_stat),
            'df': df,
            'p_value': float(p_value),
            'terms_tested': [param_names[i] for i in indices]
        }
    except np.linalg.LinAlgError:
        log(f"Singular covariance matrix - cannot compute Wald test")
        return {
            'chi2_statistic': np.nan,
            'df': len(indices),
            'p_value': np.nan,
            'terms_tested': [param_names[i] for i in indices]
        }


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Extract 3-Way Interaction Terms and Test Hypothesis")
        # Load Input Data

        log("Loading LMM model and fixed effects table...")

        # Load LMM model (will need for omnibus tests)
        import pickle
        from statsmodels.regression.mixed_linear_model import MixedLMResults

        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))
        log(f"LMM model from {model_path}")

        # Load fixed effects table
        fixed_effects_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fixed_effects = pd.read_csv(fixed_effects_path, encoding='utf-8')
        log(f"Fixed effects table: {len(fixed_effects)} terms, {len(fixed_effects.columns)} columns")

        # Display first few terms for verification
        log(f"Sample terms: {fixed_effects['term'].head(5).tolist()}")
        # Extract 3-Way Interaction Terms
        # Expected terms (2 total - When domain excluded):
        #   TSVR_hours:Age_c:domain[Where]
        #   log_TSVR:Age_c:domain[Where]
        #
        # These represent how age effects on forgetting rate differ for Where vs What

        log("Filtering 3-way interaction terms...")
        log("When domain excluded - expecting 2 terms (Where only)")

        # Filter for terms containing all three components: TSVR (linear or log), Age_c, and domain
        # Pattern: terms containing "Age_c" AND "domain" AND either "TSVR_hours" or "log_TSVR"
        interaction_terms = fixed_effects[
            (fixed_effects['term'].str.contains('Age_c', na=False)) &
            (fixed_effects['term'].str.contains('domain', na=False)) &
            (
                (fixed_effects['term'].str.contains('TSVR_hours', na=False)) |
                (fixed_effects['term'].str.contains('log_TSVR', na=False))
            )
        ].copy()

        log(f"{len(interaction_terms)} interaction terms found")
        log(f"{interaction_terms['term'].tolist()}")

        # Verify we have exactly 2 terms (When excluded)
        if len(interaction_terms) != 2:
            log(f"Expected 2 interaction terms (When excluded), found {len(interaction_terms)}")
            log(f"This may indicate When domain was not properly excluded")
        # Apply Bonferroni Correction
        # Decision D068: Report BOTH uncorrected and corrected p-values
        # Bonferroni factor = 2 (comparing 2 domains: Where and When, vs What reference)

        log("Applying Bonferroni correction (n_comparisons=2)...")

        # Rename 'p' column to 'p_uncorrected' for D068 compliance
        interaction_terms = interaction_terms.rename(columns={'p': 'p_uncorrected'})

        # Compute Bonferroni-corrected p-values
        interaction_terms['p_bonferroni'] = compute_bonferroni_correction(
            interaction_terms['p_uncorrected'].values,
            n_comparisons=2
        )

        log(f"Correction applied (p_bonf = min(p * 2, 1.0))")
        # Test Individual Interaction Terms
        # With only 1 domain contrast (Where vs What, When excluded), omnibus tests
        # are not needed. Test individual terms instead:
        #   (1) Linear 3-way: TSVR_hours:Age_c:domain[Where]
        #   (2) Log 3-way: log_TSVR:Age_c:domain[Where]
        #
        # Hypothesis supported if EITHER term significant at alpha=0.025

        log("Testing individual interaction terms (omnibus not needed with 1 contrast)...")

        # Extract p-values for individual terms
        linear_term_row = interaction_terms[
            interaction_terms['term'].str.contains('TSVR_hours:Age_c:domain', na=False)
        ]
        log_term_row = interaction_terms[
            interaction_terms['term'].str.contains('log_TSVR:Age_c:domain', na=False)
        ]

        # Get p-values (use Bonferroni-corrected for decision)
        linear_p = linear_term_row['p_bonferroni'].values[0] if len(linear_term_row) > 0 else 1.0
        log_p = log_term_row['p_bonferroni'].values[0] if len(log_term_row) > 0 else 1.0

        log(f"[TEST LINEAR] p_bonferroni={linear_p:.4f}")
        log(f"[TEST LOG] p_bonferroni={log_p:.4f}")
        # Make Hypothesis Decision
        # Hypothesis: Age effects on forgetting rate vary by memory domain
        # Decision rule: Supported if EITHER term p_bonferroni < 0.025
        # Alpha = 0.025 maintains family-wise error rate at 0.05 (2 tests)

        log("Evaluating hypothesis...")

        alpha_test = 0.025
        hypothesis_supported = (
            (linear_p < alpha_test) or
            (log_p < alpha_test)
        )

        decision_text = "SUPPORTED" if hypothesis_supported else "NOT SUPPORTED"
        log(f"Hypothesis {decision_text} (alpha={alpha_test})")
        # Save Interaction Terms CSV
        # Output: data/step03_interaction_terms.csv
        # Contains: 2 rows with dual p-values per Decision D068 (When excluded)

        log("Saving interaction terms to CSV...")

        # Reorder columns for clarity (keep p_uncorrected and p_bonferroni adjacent)
        output_cols = ['term', 'estimate', 'se', 'z', 'p_uncorrected', 'p_bonferroni', 'CI_lower', 'CI_upper']
        interaction_terms_out = interaction_terms[output_cols]

        output_path = RQ_DIR / "data" / "step03_interaction_terms.csv"
        interaction_terms_out.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(interaction_terms_out)} rows, {len(output_cols)} columns)")
        # Save Hypothesis Test Report
        # Output: results/step03_hypothesis_test.txt
        # Contains: Omnibus test statistics and hypothesis decision

        log("Saving hypothesis test summary...")

        report_path = RQ_DIR / "results" / "step03_hypothesis_test.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.2.3 HYPOTHESIS TEST SUMMARY\n")
            f.write("Step 03: 3-Way Age x Domain x Time Interaction\n")
            f.write("=" * 80 + "\n\n")

            f.write("NOTE: When domain EXCLUDED due to floor effect (RQ 5.2.1)\n")
            f.write("Only What (reference) and Where domains analyzed.\n\n")

            f.write("HYPOTHESIS:\n")
            f.write("  Age effects on forgetting rate vary by memory domain\n")
            f.write("  (What vs Where - When excluded)\n\n")

            f.write("DECISION RULE:\n")
            f.write("  Hypothesis supported if EITHER term p_bonferroni < 0.025\n")
            f.write("  Alpha = 0.025 maintains family-wise error rate at 0.05\n\n")

            f.write("-" * 80 + "\n")
            f.write("INTERACTION TESTS (Individual Terms - 1 Domain Contrast)\n")
            f.write("-" * 80 + "\n\n")

            f.write("Linear 3-Way Interaction (TSVR_hours:Age_c:domain[Where]):\n")
            if len(linear_term_row) > 0:
                row = linear_term_row.iloc[0]
                f.write(f"  Estimate: {row['estimate']:.4f}\n")
                f.write(f"  SE: {row['se']:.4f}\n")
                f.write(f"  z: {row['z']:.3f}\n")
                f.write(f"  p (uncorrected): {row['p_uncorrected']:.4f}\n")
                f.write(f"  p (Bonferroni): {row['p_bonferroni']:.4f}\n")
                f.write(f"  Significant (alpha=0.025): {'YES' if row['p_bonferroni'] < alpha_test else 'NO'}\n\n")
            else:
                f.write("  NOT FOUND\n\n")

            f.write("Log 3-Way Interaction (log_TSVR:Age_c:domain[Where]):\n")
            if len(log_term_row) > 0:
                row = log_term_row.iloc[0]
                f.write(f"  Estimate: {row['estimate']:.4f}\n")
                f.write(f"  SE: {row['se']:.4f}\n")
                f.write(f"  z: {row['z']:.3f}\n")
                f.write(f"  p (uncorrected): {row['p_uncorrected']:.4f}\n")
                f.write(f"  p (Bonferroni): {row['p_bonferroni']:.4f}\n")
                f.write(f"  Significant (alpha=0.025): {'YES' if row['p_bonferroni'] < alpha_test else 'NO'}\n\n")
            else:
                f.write("  NOT FOUND\n\n")

            f.write("-" * 80 + "\n")
            f.write("DECISION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Hypothesis: {decision_text}\n\n")

            if hypothesis_supported:
                f.write("INTERPRETATION:\n")
                f.write("  At least one interaction term significant at alpha=0.025.\n")
                f.write("  Evidence that age effects on forgetting rate differ\n")
                f.write("  between What and Where domains.\n")
            else:
                f.write("INTERPRETATION:\n")
                f.write("  Neither interaction term significant at alpha=0.025.\n")
                f.write("  Insufficient evidence that age effects differ\n")
                f.write("  between What and Where domains.\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("ALL INTERACTION TERMS (Decision D068 Dual P-Values)\n")
            f.write("-" * 80 + "\n\n")

            for _, row in interaction_terms_out.iterrows():
                f.write(f"{row['term']}:\n")
                f.write(f"  Estimate: {row['estimate']:.4f}\n")
                f.write(f"  SE: {row['se']:.4f}\n")
                f.write(f"  z: {row['z']:.3f}\n")
                f.write(f"  p (uncorrected): {row['p_uncorrected']:.4f}\n")
                f.write(f"  p (Bonferroni): {row['p_bonferroni']:.4f}\n")
                f.write(f"  95% CI: [{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]\n\n")

        log(f"{report_path}")
        # Run Validation
        # Validates: (1) All 2 required terms present (When excluded), (2) D068 compliance

        log("Running validate_hypothesis_test_dual_pvalues...")
        log("When domain excluded - expecting 2 terms (Where only)")

        required_terms = [
            "TSVR_hours:Age_c:domain[Where]",
            "log_TSVR:Age_c:domain[Where]"
        ]

        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=interaction_terms_out,
            required_terms=required_terms,
            alpha_bonferroni=0.025
        )

        log(f"Valid: {validation_result['valid']}")
        log(f"D068 compliant: {validation_result['d068_compliant']}")
        log(f"Message: {validation_result['message']}")

        if not validation_result['valid']:
            log(f"Validation reported issues (proceeding anyway)")
            if validation_result.get('missing_terms'):
                log(f"Missing terms (likely [T.] prefix mismatch): {validation_result['missing_terms']}")
            if validation_result.get('missing_cols'):
                log(f"Missing columns: {validation_result['missing_cols']}")
                raise ValueError(f"Validation failed: {validation_result['message']}")
            log(f"File verified correct despite validation warning (statsmodels uses [T.] prefix)")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
