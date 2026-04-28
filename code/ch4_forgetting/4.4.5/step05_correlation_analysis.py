#!/usr/bin/env python3
"""
Step 05: Correlation Analysis with Steiger's Z-Test

Tests hypothesis that Purified CTT correlates more strongly with IRT theta than Full CTT
using Steiger's z-test for dependent correlations with Bonferroni correction (Decision D068).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_correlation_analysis.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def steiger_z_test(r12, r13, r23, n):
    """
    Steiger's z-test for comparing two dependent correlations.

    Tests if r12 significantly differs from r13, where both share variable 1.

    Args:
        r12: Correlation between variable 1 and 2 (e.g., theta vs Full CTT)
        r13: Correlation between variable 1 and 3 (e.g., theta vs Purified CTT)
        r23: Correlation between variable 2 and 3 (e.g., Full CTT vs Purified CTT)
        n: Sample size

    Returns:
        tuple: (z_statistic, p_value)

    Reference:
        Steiger, J. H. (1980). Tests for comparing elements of a correlation matrix.
        Psychological Bulletin, 87(2), 245-251.
    """
    # Average correlation
    r_avg = (r12 + r13) / 2.0

    # Denominator term
    denom = np.sqrt(2 * (1 - r23) + r_avg**2 * (1 - r23)**2)

    # Z-statistic
    z = (r12 - r13) * np.sqrt(n - 3) / denom

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value

if __name__ == "__main__":
    try:
        log("Step 05: Correlation Analysis with Steiger's Z-Test")

        # Load theta scores from RQ 5.4.1
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step03_theta_scores.csv"
        log(f"Reading {theta_path}")
        theta_scores = pd.read_csv(theta_path, encoding='utf-8')
        log(f"{len(theta_scores)} rows")

        # Load Full CTT scores
        ctt_full_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        log(f"Reading {ctt_full_path}")
        ctt_full_scores = pd.read_csv(ctt_full_path, encoding='utf-8')
        log(f"{len(ctt_full_scores)} rows")

        # Load Purified CTT scores
        ctt_purified_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        log(f"Reading {ctt_purified_path}")
        ctt_purified_scores = pd.read_csv(ctt_purified_path, encoding='utf-8')
        log(f"{len(ctt_purified_scores)} rows")

        # Merge all three datasets
        log("Merging theta, Full CTT, and Purified CTT on composite_ID")
        merged = theta_scores.merge(ctt_full_scores, on='composite_ID', how='inner')
        merged = merged.merge(ctt_purified_scores, on='composite_ID', how='inner')
        log(f"{len(merged)} rows retained after merge")

        # Perform Steiger's z-test for each dimension
        results = []
        family_alpha = 0.05
        n_tests = 3
        bonferroni_alpha = family_alpha / n_tests

        log(f"Performing Steiger's z-test (Bonferroni alpha = {bonferroni_alpha:.4f})")

        for dimension in ['common', 'congruent', 'incongruent']:
            log(f"\n{dimension.capitalize()}")

            # Column names
            theta_col = f'theta_{dimension}'
            ctt_full_col = f'ctt_full_{dimension}'
            ctt_purified_col = f'ctt_purified_{dimension}'

            # Correlations
            r_full = merged[theta_col].corr(merged[ctt_full_col])  # r12: theta vs Full
            r_purified = merged[theta_col].corr(merged[ctt_purified_col])  # r13: theta vs Purified
            r23 = merged[ctt_full_col].corr(merged[ctt_purified_col])  # r23: Full vs Purified

            log(f"  r(theta, Full CTT) = {r_full:.3f}")
            log(f"  r(theta, Purified CTT) = {r_purified:.3f}")
            log(f"  r(Full CTT, Purified CTT) = {r23:.3f}")

            # Delta r
            delta_r = r_purified - r_full
            log(f"  Delta r (Purified - Full) = {delta_r:+.3f}")

            # Steiger's z-test
            z_stat, p_uncorrected = steiger_z_test(r_full, r_purified, r23, n=len(merged))
            p_bonferroni = min(p_uncorrected * n_tests, 1.0)

            log(f"  Steiger's z = {z_stat:.3f}")
            log(f"  p (uncorrected) = {p_uncorrected:.4f}")
            log(f"  p (Bonferroni) = {p_bonferroni:.4f}")

            # Normality check (Shapiro-Wilk for each variable)
            normality_checks = []
            for col, name in [(theta_col, 'theta'), (ctt_full_col, 'Full'), (ctt_purified_col, 'Purified')]:
                _, p_shapiro = stats.shapiro(merged[col])
                normality_checks.append(p_shapiro > 0.05)  # True if normal

            if all(normality_checks):
                normality_check = "PASS"
            else:
                normality_check = "FAIL (non-normal distribution detected)"
                log(f"  Normality assumption violated")

            results.append({
                'dimension': dimension.capitalize(),
                'r_full': r_full,
                'r_purified': r_purified,
                'delta_r': delta_r,
                'steiger_z': z_stat,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'normality_check': normality_check,
                'N': len(merged)
            })

        # Create DataFrame
        correlation_analysis = pd.DataFrame(results)

        # Validation: Check correlations in [-1, 1]
        log("\nChecking correlations in [-1, 1]")
        corr_cols = ['r_full', 'r_purified']
        for col in corr_cols:
            min_val = correlation_analysis[col].min()
            max_val = correlation_analysis[col].max()
            if min_val < -1.0 or max_val > 1.0:
                raise ValueError(f"{col} out of range: [{min_val:.3f}, {max_val:.3f}]")
        log("All correlations in valid range")

        # Validation: Check Bonferroni correction
        log("Checking p_bonferroni = min(p_uncorrected * 3, 1.0)")
        for idx, row in correlation_analysis.iterrows():
            expected_p_bonf = min(row['p_uncorrected'] * n_tests, 1.0)
            if not np.isclose(row['p_bonferroni'], expected_p_bonf):
                raise ValueError(f"Bonferroni correction error for {row['dimension']}: {row['p_bonferroni']} != {expected_p_bonf}")
        log("Bonferroni correction verified")

        # Save results
        output_path = RQ_DIR / "data" / "step05_correlation_analysis.csv"
        log(f"\nWriting {output_path}")
        correlation_analysis.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{len(correlation_analysis)} rows, {len(correlation_analysis.columns)} columns")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
