#!/usr/bin/env python3
"""Compare Fixed Effects between IRT and CTT Models: Assess agreement between IRT-based and CTT-based LMM fixed effects using"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_cohens_kappa_agreement

from tools.validation import validate_icc_bounds

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_compare_fixed_effects.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Compare Fixed Effects between IRT and CTT Models")
        # Load Fixed Effects Tables from Step 3 CSVs
        # Note: Step 3 saved fixed effects as CSVs (step03_irt_coefficients.csv,
        #       step03_ctt_coefficients.csv) - use these directly instead of pickle
        #       to avoid random effects contamination

        log("Loading IRT fixed effects from CSV...")
        irt_coef_path = RQ_DIR / "data" / "step03_irt_coefficients.csv"
        irt_coef_df = pd.read_csv(irt_coef_path)
        log(f"IRT coefficients: {irt_coef_path.name} ({len(irt_coef_df)} rows)")

        log("Loading CTT fixed effects from CSV...")
        ctt_coef_path = RQ_DIR / "data" / "step03_ctt_coefficients.csv"
        ctt_coef_df = pd.read_csv(ctt_coef_path)
        log(f"CTT coefficients: {ctt_coef_path.name} ({len(ctt_coef_df)} rows)")
        # Extract Fixed Effects as Series
        # Convert CSV DataFrames to indexed Series for comparison

        log("Extracting fixed effects from IRT model...")
        irt_params = pd.Series(irt_coef_df['coef'].values, index=irt_coef_df['term'].values)
        irt_bse = pd.Series(irt_coef_df['std_err'].values, index=irt_coef_df['term'].values)
        irt_pvalues = pd.Series(irt_coef_df['p_value'].values, index=irt_coef_df['term'].values)

        log("Extracting fixed effects from CTT model...")
        ctt_params = pd.Series(ctt_coef_df['coef'].values, index=ctt_coef_df['term'].values)
        ctt_bse = pd.Series(ctt_coef_df['std_err'].values, index=ctt_coef_df['term'].values)
        ctt_pvalues = pd.Series(ctt_coef_df['p_value'].values, index=ctt_coef_df['term'].values)

        # Verify both models have same fixed effects structure
        if not all(irt_params.index == ctt_params.index):
            raise ValueError(
                f"Fixed effects mismatch between models!\n"
                f"IRT terms: {list(irt_params.index)}\n"
                f"CTT terms: {list(ctt_params.index)}"
            )

        log(f"Fixed effects structure validated: {len(irt_params)} terms")
        log(f"Terms: {list(irt_params.index)}")
        # Apply Bonferroni Correction (Decision D068)

        n_tests = len(irt_params)
        log(f"Applying Bonferroni correction (factor={n_tests})...")

        irt_p_bonferroni = np.minimum(irt_pvalues * n_tests, 1.0)
        ctt_p_bonferroni = np.minimum(ctt_pvalues * n_tests, 1.0)

        log(f"Correction applied to both models")
        # Classify Effects (Sign Match + Significance Match)
        #               and statistical significance

        log("Classifying effect agreements...")

        # Sign match: Same direction (both positive OR both negative)
        sign_match = (np.sign(irt_params) == np.sign(ctt_params)).values

        # Significance match: Both significant OR both non-significant (alpha=0.05)
        # Use Bonferroni-corrected p-values for significance classification
        alpha = 0.05
        irt_sig = (irt_p_bonferroni < alpha).values
        ctt_sig = (ctt_p_bonferroni < alpha).values
        sig_match = (irt_sig == ctt_sig)

        # Overall agreement: sign_match AND sig_match
        agreement = sign_match & sig_match

        n_agreements = agreement.sum()
        agreement_pct = (n_agreements / n_tests) * 100

        log(f"Sign matches: {sign_match.sum()}/{n_tests}")
        log(f"Significance matches: {sig_match.sum()}/{n_tests}")
        log(f"Overall agreements: {n_agreements}/{n_tests} ({agreement_pct:.1f}%)")
        # Create Coefficient Comparison Table
        # Output: step05_coefficient_comparison.csv
        # Contains: Side-by-side comparison of IRT and CTT fixed effects with
        #           dual p-values and agreement classifications

        log("Creating coefficient comparison table...")

        comparison_df = pd.DataFrame({
            'term': irt_params.index,
            'irt_coef': irt_params.values,
            'irt_se': irt_bse.values,
            'irt_p_uncorrected': irt_pvalues.values,
            'irt_p_bonferroni': irt_p_bonferroni.values,
            'ctt_coef': ctt_params.values,
            'ctt_se': ctt_bse.values,
            'ctt_p_uncorrected': ctt_pvalues.values,
            'ctt_p_bonferroni': ctt_p_bonferroni.values,
            'sign_match': sign_match,
            'sig_match': sig_match,
            'agreement': agreement
        })

        # Save coefficient comparison
        comparison_output = RQ_DIR / "data" / "step05_coefficient_comparison.csv"
        comparison_df.to_csv(comparison_output, index=False, encoding='utf-8')
        log(f"{comparison_output.name} ({len(comparison_df)} rows, {len(comparison_df.columns)} cols)")
        # Compute Cohen's Kappa for Agreement

        log("Computing Cohen's kappa for agreement classification...")

        kappa_result = compute_cohens_kappa_agreement(
            classifications_1=irt_sig.tolist(),  # IRT significance classifications
            classifications_2=ctt_sig.tolist(),  # CTT significance classifications
            labels=irt_params.index.tolist()     # Effect names for reporting
        )

        kappa = kappa_result['kappa']
        kappa_interpretation = kappa_result['interpretation']

        log(f"Cohen's kappa = {kappa:.3f} ({kappa_interpretation})")
        # Create Agreement Metrics Summary
        # Output: step05_agreement_metrics.csv
        # Contains: Cohen's kappa, threshold checks, overall agreement percentage
        #           Used for validation and results reporting

        log("Creating agreement metrics summary...")

        # Thresholds per RQ 5.5.4 convergence criteria
        kappa_threshold = 0.60  # Landis & Koch (1977) "substantial agreement"
        agreement_threshold = 80.0  # 80% overall agreement

        kappa_threshold_met = (kappa > kappa_threshold)
        agreement_threshold_met = (agreement_pct >= agreement_threshold)

        metrics_df = pd.DataFrame({
            'cohens_kappa': [kappa],
            'kappa_threshold_met': [kappa_threshold_met],
            'overall_agreement_pct': [agreement_pct],
            'agreement_threshold_met': [agreement_threshold_met],
            'n_terms': [n_tests],
            'n_agreements': [n_agreements]
        })

        # Save agreement metrics
        metrics_output = RQ_DIR / "data" / "step05_agreement_metrics.csv"
        metrics_df.to_csv(metrics_output, index=False, encoding='utf-8')
        log(f"{metrics_output.name} ({len(metrics_df)} rows, {len(metrics_df.columns)} cols)")
        # Validate Agreement Metrics
        # Validates: kappa in [-1, 1], agreement_pct in [0, 100]
        # Threshold: kappa > 0.60 for substantial agreement

        log("Validating Cohen's kappa bounds...")

        # Create validation DataFrame (validate_icc_bounds expects DataFrame)
        kappa_validation_df = pd.DataFrame({'cohens_kappa': [kappa]})

        validation_result = validate_icc_bounds(
            icc_df=kappa_validation_df,
            icc_col='cohens_kappa'
        )

        if not validation_result['valid']:
            raise ValueError(f"Kappa validation failed: {validation_result['message']}")

        log(f"Kappa bounds valid: {validation_result['message']}")

        # Additional validation: agreement percentage in [0, 100]
        if not (0 <= agreement_pct <= 100):
            raise ValueError(f"Agreement percentage {agreement_pct:.1f}% outside [0, 100] range")

        log(f"Agreement percentage valid: {agreement_pct:.1f}% in [0, 100]")

        # Additional validation: Bonferroni formula correct
        irt_bonf_valid = (irt_p_bonferroni >= irt_pvalues).all()
        ctt_bonf_valid = (ctt_p_bonferroni >= ctt_pvalues).all()
        bonferroni_valid = irt_bonf_valid and ctt_bonf_valid
        if not bonferroni_valid:
            raise ValueError("Bonferroni correction failed: p_bonferroni < p_uncorrected detected")

        log("Bonferroni correction valid: p_bonferroni >= p_uncorrected for all terms")
        # Final Summary

        log("Fixed effects comparison complete:")
        log(f"  - Coefficient comparison: {comparison_output.name}")
        log(f"  - Agreement metrics: {metrics_output.name}")
        log(f"  - Cohen's kappa: {kappa:.3f} ({kappa_interpretation})")
        log(f"  - Overall agreement: {agreement_pct:.1f}% ({n_agreements}/{n_tests} terms)")
        log(f"  - Kappa threshold (>0.60): {'MET' if kappa_threshold_met else 'NOT MET'}")
        log(f"  - Agreement threshold (>=80%): {'MET' if agreement_threshold_met else 'NOT MET'}")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
