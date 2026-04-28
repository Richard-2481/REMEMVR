#!/usr/bin/env python3
"""Correlation Analysis with Steiger's z-test: Test primary hypothesis that purified CTT correlates more strongly with IRT"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compare_correlations_dependent

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_correlation_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Correlation Analysis with Steiger's z-test")
        # Load Input Data

        log("Loading input data...")

        # Load IRT theta scores from RQ 5.1
        df_theta = pd.read_csv(RQ_DIR / "data" / "step00_theta_scores.csv")
        log(f"step00_theta_scores.csv ({len(df_theta)} rows, {len(df_theta.columns)} cols)")

        # Load Full CTT scores from Step 2
        df_ctt_full = pd.read_csv(RQ_DIR / "data" / "step02_ctt_full_scores.csv")
        log(f"step02_ctt_full_scores.csv ({len(df_ctt_full)} rows, {len(df_ctt_full.columns)} cols)")

        # Load Purified CTT scores from Step 3
        df_ctt_purified = pd.read_csv(RQ_DIR / "data" / "step03_ctt_purified_scores.csv")
        log(f"step03_ctt_purified_scores.csv ({len(df_ctt_purified)} rows, {len(df_ctt_purified.columns)} cols)")
        # Merge Data Sources on composite_ID
        # Merge all three measurement approaches into single DataFrame

        log("Merging theta, full CTT, and purified CTT on composite_ID...")

        # Merge theta + full CTT (What/Where only - no When)
        df_merged = df_theta.merge(
            df_ctt_full[['composite_ID', 'CTT_full_what', 'CTT_full_where']],
            on='composite_ID',
            how='inner'
        )

        # Merge + purified CTT (What/Where only - no When)
        df_merged = df_merged.merge(
            df_ctt_purified[['composite_ID', 'CTT_purified_what', 'CTT_purified_where']],
            on='composite_ID',
            how='inner'
        )

        log(f"Combined dataset: {len(df_merged)} rows, {len(df_merged.columns)} cols")

        # Check for missing data
        n_missing = df_merged.isnull().sum().sum()
        if n_missing > 0:
            log(f"{n_missing} missing values detected in merged data")
            log("Missing data by column:")
            for col in df_merged.columns:
                n_col_missing = df_merged[col].isnull().sum()
                if n_col_missing > 0:
                    log(f"  {col}: {n_col_missing} missing")
        # Run Correlation Analysis with Steiger's z-test

        log("Running Steiger's z-test for each domain...")
        log("When domain EXCLUDED per RQ 5.2.1 floor effect")

        # Define domains to analyze (What/Where only - When excluded)
        domains = ['what', 'where']

        # Initialize results list
        results = []

        # Sample size for correlation tests
        n = len(df_merged)
        log(f"Sample size for correlations: n = {n}")

        # Loop through domains
        for domain in domains:
            log(f"Processing domain: {domain}")

            # Extract domain-specific columns
            # Variable 1: Full CTT
            full_ctt = df_merged[f'CTT_full_{domain}'].values

            # Variable 2: IRT theta
            irt_theta = df_merged[f'theta_{domain}'].values

            # Variable 3: Purified CTT
            purified_ctt = df_merged[f'CTT_purified_{domain}'].values

            # Remove rows with any NaN in this domain's trio
            mask = ~(np.isnan(full_ctt) | np.isnan(irt_theta) | np.isnan(purified_ctt))
            full_ctt_clean = full_ctt[mask]
            irt_theta_clean = irt_theta[mask]
            purified_ctt_clean = purified_ctt[mask]

            n_valid = len(full_ctt_clean)
            log(f"  Valid observations after removing NaN: {n_valid}")

            # Compute 3 pairwise correlations
            # r12: Full CTT <-> IRT theta
            r12 = np.corrcoef(full_ctt_clean, irt_theta_clean)[0, 1]

            # r13: Full CTT <-> Purified CTT
            r13 = np.corrcoef(full_ctt_clean, purified_ctt_clean)[0, 1]

            # r23: Purified CTT <-> IRT theta
            r23 = np.corrcoef(purified_ctt_clean, irt_theta_clean)[0, 1]

            log(f"  Correlations:")
            log(f"    r(Full CTT, IRT) = {r12:.3f}")
            log(f"    r(Full CTT, Purified CTT) = {r13:.3f}")
            log(f"    r(Purified CTT, IRT) = {r23:.3f}")

            # Apply Steiger's z-test
            # Tests H0: r12 = r23 (Full CTT-IRT vs Purified CTT-IRT)
            steiger_result = compare_correlations_dependent(
                r12=r12,  # Full CTT <-> IRT theta
                r13=r13,  # Full CTT <-> Purified CTT
                r23=r23,  # Purified CTT <-> IRT theta
                n=n_valid
            )

            log(f"  Steiger's z = {steiger_result['z_statistic']:.3f}")
            log(f"  p (uncorrected) = {steiger_result['p_value']:.4f}")

            # Compute delta_r = r23 - r12 (improvement from purification)
            delta_r = r23 - r12
            log(f"  Delta r (Purified - Full) = {delta_r:.3f}")

            # Bonferroni correction (2 domains tested - What/Where only)
            p_bonferroni = min(steiger_result['p_value'] * 2, 1.0)
            log(f"  p (Bonferroni, k=2) = {p_bonferroni:.4f}")

            # Interpretation
            if p_bonferroni < 0.05:
                if delta_r > 0:
                    interpretation = "Purified CTT significantly higher correlation with IRT (Bonferroni p < 0.05)"
                else:
                    interpretation = "Full CTT significantly higher correlation with IRT (Bonferroni p < 0.05)"
            else:
                interpretation = "No significant difference between Full and Purified CTT correlations with IRT"

            log(f"  Interpretation: {interpretation}")

            # Append result
            results.append({
                'domain': domain,
                'r_full_irt': r12,
                'r_purified_irt': r23,
                'r_full_purified': r13,
                'delta_r': delta_r,
                'steiger_z': steiger_result['z_statistic'],
                'p_uncorrected': steiger_result['p_value'],
                'p_bonferroni': p_bonferroni,
                'interpretation': interpretation
            })

        log("Steiger's z-test complete for all domains")
        # Save Analysis Outputs
        # Output: CSV with Steiger's z-test results per domain
        # Contains: Correlations, z-statistic, dual p-values, interpretation

        log("Saving correlation analysis results...")

        # Create DataFrame from results
        df_correlations = pd.DataFrame(results)

        # Save to CSV
        output_path = RQ_DIR / "data" / "step05_correlation_analysis.csv"
        df_correlations.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step05_correlation_analysis.csv ({len(df_correlations)} rows, {len(df_correlations.columns)} cols)")

        # Log summary statistics
        log("Correlation analysis results:")
        log(f"  Domains analyzed: {len(df_correlations)}")
        log(f"  Mean r(Full CTT, IRT): {df_correlations['r_full_irt'].mean():.3f}")
        log(f"  Mean r(Purified CTT, IRT): {df_correlations['r_purified_irt'].mean():.3f}")
        log(f"  Mean delta_r: {df_correlations['delta_r'].mean():.3f}")
        log(f"  Significant differences (Bonferroni p < 0.05): {(df_correlations['p_bonferroni'] < 0.05).sum()}")
        # Run Validation Tool
        # Validates: Decision D068 dual p-value reporting (uncorrected + Bonferroni)
        # Checks: Required columns present, p-values in [0,1], correlations in [-1,1]

        log("Running validate_correlation_test_d068...")

        # Required columns per Decision D068
        required_cols = [
            'domain',
            'r_full_irt',
            'r_purified_irt',
            'steiger_z',
            'p_uncorrected',
            'p_bonferroni'
        ]

        validation_result = validate_correlation_test_d068(
            correlation_df=df_correlations,
            required_cols=required_cols
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASSED - {validation_result['message']}")
        else:
            log(f"FAILED - {validation_result['message']}")
            if validation_result.get('missing_cols'):
                log(f"  Missing columns: {validation_result['missing_cols']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional range checks
        log("Checking value ranges...")

        # Check correlations in [-1, 1]
        for col in ['r_full_irt', 'r_purified_irt', 'r_full_purified']:
            if not df_correlations[col].between(-1, 1).all():
                raise ValueError(f"Column {col} has values outside [-1, 1] range")
        log("  All correlations in [-1, 1] range")

        # Check p-values in [0, 1]
        for col in ['p_uncorrected', 'p_bonferroni']:
            if not df_correlations[col].between(0, 1).all():
                raise ValueError(f"Column {col} has values outside [0, 1] range")
        log("  All p-values in [0, 1] range")

        # Check D068 compliance
        if validation_result.get('d068_compliant'):
            log("  Decision D068 compliant (dual p-value reporting)")
        else:
            raise ValueError("Decision D068 violation - missing dual p-values")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
