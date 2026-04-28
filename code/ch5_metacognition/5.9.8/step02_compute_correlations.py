#!/usr/bin/env python3
"""Compute Within-Schema Correlations at Key Timepoints: Compute Pearson correlations between accuracy and confidence within each schema condition"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.8
LOG_FILE = RQ_DIR / "logs" / "step02_compute_correlations.log"
INPUT_FILE = RQ_DIR / "data" / "step01_reshaped_long.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step02_within_schema_correlations.csv"

# Key timepoints for analysis
KEY_TIMEPOINTS = ['T1', 'T2', 'T4']

# Bonferroni correction parameters
N_TESTS = 9  # 3 schema conditions x 3 timepoints
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_TESTS  # 0.0056

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute Within-Schema Correlations")
        log(f"Key timepoints: {KEY_TIMEPOINTS}")
        log(f"Bonferroni correction: alpha = {ALPHA}/{N_TESTS} = {ALPHA_BONFERRONI:.4f}")
        # Load and Filter Data
        log("Loading long-format data...")
        df = pd.read_csv(INPUT_FILE)
        log(f"{len(df)} rows, {len(df.columns)} columns")

        log("Filtering to key timepoints...")
        df_key = df[df['test'].isin(KEY_TIMEPOINTS)].copy()
        log(f"{len(df_key)} rows retained")

        # Verify expected row count
        expected_rows = 100 * 3 * 3  # 100 UIDs x 3 timepoints x 3 schema conditions
        if len(df_key) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(df_key)}")
        # Compute Correlations for Each Schema x Timepoint
        log("Computing correlations...")

        results = []

        for schema in ['Common', 'Congruent', 'Incongruent']:
            for timepoint in KEY_TIMEPOINTS:
                # Filter to specific schema and timepoint
                subset = df_key[
                    (df_key['schema_condition'] == schema) &
                    (df_key['test'] == timepoint)
                ].copy()

                n = len(subset)

                if n < 3:
                    log(f"{schema} x {timepoint}: only {n} observations, skipping")
                    continue

                # Compute Pearson correlation
                r, p_uncorrected = pearsonr(
                    subset['theta_accuracy'],
                    subset['theta_confidence']
                )

                # Fisher Z transformation
                fisher_z = np.arctanh(r)
                SE_z = 1 / np.sqrt(n - 3)

                # 95% CI using Fisher Z
                z_critical = 1.96
                CI_z_lower = fisher_z - z_critical * SE_z
                CI_z_upper = fisher_z + z_critical * SE_z

                # Transform back to r scale
                CI_r_lower = np.tanh(CI_z_lower)
                CI_r_upper = np.tanh(CI_z_upper)

                # Bonferroni correction
                p_bonferroni = min(p_uncorrected * N_TESTS, 1.0)  # Cap at 1.0
                sig_bonferroni = p_bonferroni < ALPHA

                log(f"{schema:12s} x {timepoint}: r = {r:.3f} [CI: {CI_r_lower:.3f}, {CI_r_upper:.3f}], p = {p_uncorrected:.4f}, p_bonf = {p_bonferroni:.4f}")

                results.append({
                    'schema_condition': schema,
                    'timepoint': timepoint,
                    'n': n,
                    'r': r,
                    'fisher_z': fisher_z,
                    'SE_z': SE_z,
                    'CI_r_lower': CI_r_lower,
                    'CI_r_upper': CI_r_upper,
                    'p_uncorrected': p_uncorrected,
                    'p_bonferroni': p_bonferroni,
                    'sig_bonferroni': sig_bonferroni
                })

        df_corr = pd.DataFrame(results)
        log(f"Computed {len(df_corr)} correlations")
        # Validate Results
        log("Checking correlation results...")

        errors = []

        # Check row count
        if len(df_corr) != N_TESTS:
            errors.append(f"Expected {N_TESTS} rows, got {len(df_corr)}")
        else:
            log("Row count: 9")

        # Check correlation bounds
        if (df_corr['r'] < -1).any() or (df_corr['r'] > 1).any():
            errors.append("Correlation(s) out of bounds [-1, 1]")
        else:
            log("All r in [-1, 1]")

        # Check positive correlations (expected)
        if (df_corr['r'] < 0).any():
            log("Some correlations negative (unexpected)")

        if (df_corr['r'] < 0.50).any():
            log("Some correlations < 0.50 (weaker than expected)")

        # Check SE_z positive
        if (df_corr['SE_z'] <= 0).any():
            errors.append("Found non-positive SE_z values")
        else:
            log("All SE_z > 0")

        # Check SE_z approximately 0.10 (for n=100)
        se_mean = df_corr['SE_z'].mean()
        log(f"Mean SE_z = {se_mean:.4f} (expected ~0.101 for n=100)")

        # Check p-value ranges
        if (df_corr['p_uncorrected'] < 0).any() or (df_corr['p_uncorrected'] > 1).any():
            errors.append("p_uncorrected out of range [0, 1]")
        if (df_corr['p_bonferroni'] < 0).any() or (df_corr['p_bonferroni'] > 1).any():
            errors.append("p_bonferroni out of range [0, 1]")
        else:
            log("All p-values in [0, 1]")

        # Check CI ordering
        ci_violations = (df_corr['CI_r_lower'] >= df_corr['r']) | (df_corr['r'] >= df_corr['CI_r_upper'])
        if ci_violations.any():
            errors.append("CI bounds violate CI_lower < r < CI_upper")
        else:
            log("All CIs properly ordered")

        # Check n = 100 for all rows
        if not (df_corr['n'] == 100).all():
            errors.append(f"Not all n = 100. Range: [{df_corr['n'].min()}, {df_corr['n'].max()}]")
        else:
            log("All n = 100")

        # Report descriptive statistics
        log(f"Correlation range: [{df_corr['r'].min():.3f}, {df_corr['r'].max():.3f}]")
        log(f"Mean correlation: {df_corr['r'].mean():.3f} (SD = {df_corr['r'].std():.3f})")
        log(f"Bonferroni-significant: {df_corr['sig_bonferroni'].sum()}/{len(df_corr)}")
        # Save Output
        if errors:
            log("FAIL - Errors detected:")
            for error in errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed with {len(errors)} error(s)")

        log("PASS - All checks passed")
        log("Fisher Z transformation applied for CIs")
        log(f"Bonferroni correction: alpha = {ALPHA_BONFERRONI:.4f} (Decision D068)")
        log("Dual p-values reported (uncorrected and Bonferroni)")

        df_corr.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"Output: {OUTPUT_FILE}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
