#!/usr/bin/env python3
"""compare_coefficients: Extract fixed effects from both IRT and CTT LMM models, compare statistical"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_compare_coefficients.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def compute_cohens_kappa(agreement_vector: np.ndarray, n_categories: int = 2) -> float:
    """
    Compute Cohen's kappa for agreement between two raters.

    Formula: kappa = (p_o - p_e) / (1 - p_e)
    where:
      p_o = observed agreement proportion
      p_e = expected agreement by chance

    For binary classification (significant vs non-significant):
      p_e = p(both sig) + p(both nonsig)
          = (n_sig_irt/n * n_sig_ctt/n) + (n_nonsig_irt/n * n_nonsig_ctt/n)

    Parameters:
    -----------
    agreement_vector : np.ndarray
        Boolean array where True = agreement, False = disagreement
    n_categories : int
        Number of categories (default 2 for binary sig/nonsig)

    Returns:
    --------
    float
        Cohen's kappa coefficient in [-1, 1]
        kappa > 0.60 = substantial agreement (Landis & Koch 1977)
        kappa > 0.80 = almost perfect agreement
    """
    n = len(agreement_vector)
    p_o = agreement_vector.mean()  # Observed agreement proportion

    # For chance agreement, we need the marginal probabilities
    # This is computed from the actual data in the calling function
    # Here we just use a simplified formula for binary classification
    # which assumes equal marginal distributions (conservative estimate)
    p_e = 1 / n_categories  # Chance agreement for balanced binary case

    # Adjusted formula using actual marginals (computed in main)
    # p_e will be passed in from actual sig/nonsig rates

    if p_e >= 1.0:
        return 0.0  # Perfect disagreement case

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_cohens_kappa_from_marginals(
    n_agree: int,
    n_total: int,
    n_sig_irt: int,
    n_sig_ctt: int
) -> float:
    """
    Compute Cohen's kappa using actual marginal frequencies.

    This is the proper formula accounting for actual sig/nonsig rates.

    Parameters:
    -----------
    n_agree : int
        Number of coefficients with agreement
    n_total : int
        Total number of coefficients
    n_sig_irt : int
        Number of significant coefficients in IRT model
    n_sig_ctt : int
        Number of significant coefficients in CTT model

    Returns:
    --------
    float
        Cohen's kappa with marginal correction
    """
    # Observed agreement
    p_o = n_agree / n_total

    # Expected agreement by chance
    # p(both sig) = p(sig_irt) * p(sig_ctt)
    # p(both nonsig) = p(nonsig_irt) * p(nonsig_ctt)
    p_sig_irt = n_sig_irt / n_total
    p_sig_ctt = n_sig_ctt / n_total
    p_nonsig_irt = 1 - p_sig_irt
    p_nonsig_ctt = 1 - p_sig_ctt

    p_e = (p_sig_irt * p_sig_ctt) + (p_nonsig_irt * p_nonsig_ctt)

    if p_e >= 1.0:
        return 0.0  # Perfect chance agreement

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Compare Fixed Effects Between IRT and CTT Models")
        # Load Fixed Effects Tables

        log("Loading IRT fixed effects...")
        irt_fixed = pd.read_csv(RQ_DIR / "results" / "step03_irt_lmm_fixed_effects.csv")
        log(f"IRT fixed effects: {len(irt_fixed)} rows, {len(irt_fixed.columns)} columns")
        log(f"IRT columns: {irt_fixed.columns.tolist()}")

        log("Loading CTT fixed effects...")
        ctt_fixed = pd.read_csv(RQ_DIR / "results" / "step03_ctt_lmm_fixed_effects.csv")
        log(f"CTT fixed effects: {len(ctt_fixed)} rows, {len(ctt_fixed.columns)} columns")
        log(f"CTT columns: {ctt_fixed.columns.tolist()}")
        # STEP 1.5: Standardize Term Names (Case Sensitivity Fix)
        # Fix case mismatch: IRT has C(domain)[T.When] but CTT has C(domain)[T.when]
        # Standardize to Title Case for consistent merging

        log("Fixing domain case mismatch in term names...")

        def standardize_domain_case(term):
            """Standardize domain names to Title Case in coefficient terms."""
            # Replace lowercase domain names with Title Case
            term = term.replace('[T.what]', '[T.What]')
            term = term.replace('[T.where]', '[T.Where]')
            term = term.replace('[T.when]', '[T.When]')
            return term

        irt_fixed['term'] = irt_fixed['term'].apply(standardize_domain_case)
        ctt_fixed['term'] = ctt_fixed['term'].apply(standardize_domain_case)

        log(f"IRT terms: {irt_fixed['term'].tolist()}")
        log(f"CTT terms: {ctt_fixed['term'].tolist()}")
        # Merge Fixed Effects on Term (Coefficient Name)
        # Merge IRT and CTT fixed effects tables on 'term' column
        # This creates side-by-side comparison of same coefficients across models

        log("Merging IRT and CTT fixed effects on term...")
        coef_comparison = irt_fixed.merge(
            ctt_fixed,
            on='term',
            how='inner',  # Only keep coefficients present in BOTH models
            suffixes=('_irt', '_ctt')
        )
        log(f"{len(coef_comparison)} coefficients present in both models")

        if len(coef_comparison) == 0:
            raise ValueError("No matching coefficients found between IRT and CTT models - check term names")
        # Classify Significance (p < 0.05)
        # For each coefficient, classify as significant (TRUE) or non-significant (FALSE)
        # in each model separately

        log("Classifying significance (alpha = 0.05)...")
        alpha = 0.05

        coef_comparison['sig_irt'] = coef_comparison['p_uncorrected_irt'] < alpha
        coef_comparison['sig_ctt'] = coef_comparison['p_uncorrected_ctt'] < alpha

        n_sig_irt = coef_comparison['sig_irt'].sum()
        n_sig_ctt = coef_comparison['sig_ctt'].sum()

        log(f"IRT significant: {n_sig_irt}/{len(coef_comparison)} coefficients")
        log(f"CTT significant: {n_sig_ctt}/{len(coef_comparison)} coefficients")
        # Compute Agreement (Both Sig OR Both Nonsig)
        # Agreement = TRUE when:
        #   - Both models find coefficient significant, OR
        #   - Both models find coefficient non-significant
        # Disagreement = One model significant, other non-significant

        log("Computing agreement classification...")
        coef_comparison['agreement'] = (
            (coef_comparison['sig_irt'] & coef_comparison['sig_ctt']) |  # Both sig
            (~coef_comparison['sig_irt'] & ~coef_comparison['sig_ctt'])  # Both nonsig
        )

        n_agree = coef_comparison['agreement'].sum()
        n_total = len(coef_comparison)
        raw_agreement_pct = (n_agree / n_total) * 100

        log(f"Raw agreement: {n_agree}/{n_total} = {raw_agreement_pct:.1f}%")
        # Compute Cohen's Kappa (Accounts for Chance Agreement)
        # Cohen's kappa adjusts for the agreement expected by chance alone
        # kappa = (p_o - p_e) / (1 - p_e)
        # Interpretation (Landis & Koch 1977):
        #   kappa < 0.00: Poor agreement
        #   kappa 0.00-0.20: Slight agreement
        #   kappa 0.21-0.40: Fair agreement
        #   kappa 0.41-0.60: Moderate agreement
        #   kappa 0.61-0.80: Substantial agreement ← Target threshold
        #   kappa 0.81-1.00: Almost perfect agreement

        log("Computing Cohen's kappa (all coefficients)...")
        kappa_all = compute_cohens_kappa_from_marginals(
            n_agree=n_agree,
            n_total=n_total,
            n_sig_irt=n_sig_irt,
            n_sig_ctt=n_sig_ctt
        )
        log(f"All coefficients: kappa = {kappa_all:.3f}")

        # Interpretation
        if kappa_all >= 0.81:
            kappa_interp = "Almost perfect agreement"
        elif kappa_all >= 0.61:
            kappa_interp = "Substantial agreement"
        elif kappa_all >= 0.41:
            kappa_interp = "Moderate agreement"
        elif kappa_all >= 0.21:
            kappa_interp = "Fair agreement"
        elif kappa_all >= 0.00:
            kappa_interp = "Slight agreement"
        else:
            kappa_interp = "Poor agreement"

        log(f"Interpretation: {kappa_interp}")
        # Compute Kappa for Interaction Terms Only
        # Focus on interaction terms (TSVR_hours:domain, log(TSVR_hours+1):domain)
        # These are the key terms for testing domain-specific forgetting patterns

        log("Computing Cohen's kappa (interaction terms only)...")

        # Identify interaction terms (contain ':' character)
        interaction_mask = coef_comparison['term'].str.contains(':', regex=False)
        interaction_terms = coef_comparison[interaction_mask]

        if len(interaction_terms) > 0:
            n_agree_int = interaction_terms['agreement'].sum()
            n_total_int = len(interaction_terms)
            n_sig_irt_int = interaction_terms['sig_irt'].sum()
            n_sig_ctt_int = interaction_terms['sig_ctt'].sum()

            kappa_interactions = compute_cohens_kappa_from_marginals(
                n_agree=n_agree_int,
                n_total=n_total_int,
                n_sig_irt=n_sig_irt_int,
                n_sig_ctt=n_sig_ctt_int
            )
            log(f"Interaction terms: kappa = {kappa_interactions:.3f} (n={n_total_int})")
        else:
            kappa_interactions = np.nan
            log("No interaction terms found")
        # Compute Beta Ratio (CTT / IRT)
        # Beta ratio shows the scaling factor between CTT and IRT coefficients
        # If beta_ratio ≈ constant across terms, models differ only by scaling
        # If beta_ratio varies widely, models show different effect patterns

        log("Computing beta ratios (CTT / IRT)...")

        # Avoid division by zero - set ratio to NaN when IRT estimate = 0
        coef_comparison['beta_ratio'] = np.where(
            coef_comparison['estimate_irt'] != 0,
            coef_comparison['estimate_ctt'] / coef_comparison['estimate_irt'],
            np.nan
        )

        # Log summary statistics
        valid_ratios = coef_comparison['beta_ratio'].dropna()
        if len(valid_ratios) > 0:
            log(f"Mean ratio: {valid_ratios.mean():.3f}")
            log(f"Median ratio: {valid_ratios.median():.3f}")
            log(f"Range: [{valid_ratios.min():.3f}, {valid_ratios.max():.3f}]")
        else:
            log("No valid ratios (all IRT estimates = 0)")
        # Flag Large Discrepancies
        # Flag coefficients where |beta_irt - beta_ctt| > 2*SE
        # This identifies cases where coefficient estimates differ beyond measurement error

        log("Flagging large coefficient discrepancies...")
        discrepancy_multiplier = 2.0

        # Compute difference
        beta_diff = np.abs(coef_comparison['estimate_irt'] - coef_comparison['estimate_ctt'])

        # Compute pooled SE (conservative: larger of the two SEs)
        pooled_se = np.maximum(coef_comparison['SE_irt'], coef_comparison['SE_ctt'])

        # Flag if difference exceeds threshold
        coef_comparison['discrepancy_flag'] = beta_diff > (discrepancy_multiplier * pooled_se)

        n_discrepancies = coef_comparison['discrepancy_flag'].sum()
        log(f"{n_discrepancies}/{n_total} coefficients flagged (|diff| > {discrepancy_multiplier}*SE)")

        if n_discrepancies > 0:
            flagged_terms = coef_comparison.loc[coef_comparison['discrepancy_flag'], 'term'].tolist()
            log(f"Flagged terms: {flagged_terms}")
        # Rename Columns for Output
        # Clean up column names for final output table

        log("Preparing output columns...")
        coef_comparison = coef_comparison.rename(columns={
            'estimate_irt': 'estimate_irt',
            'SE_irt': 'SE_irt',
            'p_uncorrected_irt': 'p_irt',
            'estimate_ctt': 'estimate_ctt',
            'SE_ctt': 'SE_ctt',
            'p_uncorrected_ctt': 'p_ctt'
        })

        # Select and order output columns
        output_cols = [
            'term',
            'estimate_irt', 'SE_irt', 'p_irt', 'sig_irt',
            'estimate_ctt', 'SE_ctt', 'p_ctt', 'sig_ctt',
            'agreement', 'beta_ratio', 'discrepancy_flag'
        ]
        coef_comparison = coef_comparison[output_cols]
        # Save Coefficient Comparison Table
        # Save side-by-side comparison with all computed metrics

        log("Saving coefficient comparison table...")
        comparison_path = RQ_DIR / "results" / "step05_coefficient_comparison.csv"
        coef_comparison.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"{comparison_path} ({len(coef_comparison)} rows, {len(coef_comparison.columns)} columns)")
        # Create Agreement Metrics Summary
        # Summarize key agreement metrics in a single table

        log("Creating agreement metrics summary...")

        agreement_metrics = pd.DataFrame([
            {
                'metric': 'raw_agreement_pct',
                'value': raw_agreement_pct,
                'threshold': 80.0,  # 80% agreement threshold
                'pass': raw_agreement_pct >= 80.0
            },
            {
                'metric': 'kappa_all_coefficients',
                'value': kappa_all,
                'threshold': 0.60,  # Substantial agreement threshold (Landis & Koch)
                'pass': kappa_all >= 0.60
            },
            {
                'metric': 'kappa_interactions_only',
                'value': kappa_interactions if not np.isnan(kappa_interactions) else None,
                'threshold': 0.60,
                'pass': kappa_interactions >= 0.60 if not np.isnan(kappa_interactions) else None
            }
        ])

        # Log summary
        log("Agreement metrics:")
        for _, row in agreement_metrics.iterrows():
            status = "" if row['pass'] else ""
            log(f"  {status} {row['metric']}: {row['value']:.3f} (threshold: {row['threshold']:.2f})")
        # Save Agreement Metrics

        log("Saving agreement metrics...")
        metrics_path = RQ_DIR / "results" / "step05_agreement_metrics.csv"
        agreement_metrics.to_csv(metrics_path, index=False, encoding='utf-8')
        log(f"{metrics_path} ({len(agreement_metrics)} rows, {len(agreement_metrics.columns)} columns)")
        # Run Validation
        # Validate output structure using tools.validation.validate_dataframe_structure

        log("Validating coefficient comparison structure...")

        # Validate coefficient comparison
        # Note: Row count depends on model structure (domain-specific models have fewer shared terms)
        coef_validation = validate_dataframe_structure(
            df=coef_comparison,
            expected_rows=(3, 12),  # Flexible range (domain models may have only 3-4 shared terms)
            expected_columns=output_cols
        )

        if not coef_validation['valid']:
            raise ValueError(f"Coefficient comparison validation failed: {coef_validation['message']}")

        log(f"Coefficient comparison: {coef_validation['message']}")

        # Validate metrics table
        log("Validating agreement metrics structure...")
        metrics_validation = validate_dataframe_structure(
            df=agreement_metrics,
            expected_rows=3,
            expected_columns=['metric', 'value', 'threshold', 'pass']
        )

        if not metrics_validation['valid']:
            raise ValueError(f"Agreement metrics validation failed: {metrics_validation['message']}")

        log(f"Agreement metrics: {metrics_validation['message']}")

        # Additional validation: Check for NaN in critical columns
        log("Checking for NaN values...")

        nan_check_cols = ['p_irt', 'p_ctt', 'estimate_irt', 'estimate_ctt']
        for col in nan_check_cols:
            n_nan = coef_comparison[col].isna().sum()
            if n_nan > 0:
                raise ValueError(f"Found {n_nan} NaN values in {col} column")

        log("No NaN values in p-values or estimates")

        # Validate kappa range
        log("Checking Cohen's kappa range...")
        if not (-1 <= kappa_all <= 1):
            raise ValueError(f"Cohen's kappa out of range [-1, 1]: {kappa_all}")

        log("Cohen's kappa in valid range [-1, 1]")

        log("Step 5 complete")
        log(f"Key findings:")
        log(f"  - Compared {n_total} coefficients across IRT and CTT models")
        log(f"  - Raw agreement: {raw_agreement_pct:.1f}%")
        log(f"  - Cohen's kappa: {kappa_all:.3f} ({kappa_interp})")
        log(f"  - Flagged {n_discrepancies} large discrepancies")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
