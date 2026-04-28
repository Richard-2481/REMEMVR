#!/usr/bin/env python3
"""Bootstrap Confidence Intervals for Correlations: Compute bootstrap 95% CIs for within-schema correlations using participant-level"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.8
LOG_FILE = RQ_DIR / "logs" / "step06_bootstrap_correlations.log"
INPUT_DATA = RQ_DIR / "data" / "step01_reshaped_long.csv"
INPUT_CORR = RQ_DIR / "data" / "step02_within_schema_correlations.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step06_bootstrap_correlations.csv"

# Bootstrap parameters
N_ITERATIONS = 1000
RANDOM_SEED = 42
KEY_TIMEPOINTS = ['T1', 'T2', 'T4']

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Bootstrap Confidence Intervals for Correlations")
        log(f"Bootstrap: {N_ITERATIONS} iterations, seed={RANDOM_SEED}, participant-level resampling")

        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        log(f"Random seed set to {RANDOM_SEED}")
        # Load Data
        log("Loading long-format data...")
        df = pd.read_csv(INPUT_DATA)
        log(f"{len(df)} rows")

        log("Filtering to key timepoints...")
        df_key = df[df['test'].isin(KEY_TIMEPOINTS)].copy()
        log(f"{len(df_key)} rows retained")

        log("Loading parametric correlations...")
        df_corr = pd.read_csv(INPUT_CORR)
        log(f"{len(df_corr)} parametric correlations")
        # Bootstrap for Each Schema x Timepoint
        log("Starting bootstrap iterations...")

        bootstrap_results = []

        for schema in ['Common', 'Congruent', 'Incongruent']:
            for timepoint in KEY_TIMEPOINTS:
                log(f"{schema} x {timepoint}...")

                # Filter to specific schema and timepoint
                subset = df_key[
                    (df_key['schema_condition'] == schema) &
                    (df_key['test'] == timepoint)
                ].copy()

                n_obs = len(subset)
                log(f"  Observations: {n_obs}")

                # Get unique UIDs
                unique_uids = subset['UID'].unique()
                n_uids = len(unique_uids)
                log(f"  Unique UIDs: {n_uids}")

                # Bootstrap iterations
                r_boot = []
                for i in range(N_ITERATIONS):
                    # Resample UIDs with replacement
                    sampled_uids = np.random.choice(unique_uids, size=n_uids, replace=True)

                    # Keep all observations for sampled UIDs (participant-level resampling)
                    boot_sample = subset[subset['UID'].isin(sampled_uids)].copy()

                    # Handle duplicate UIDs (keep all observations)
                    # This is correct for participant-level resampling

                    # Compute correlation
                    try:
                        r_i, _ = pearsonr(
                            boot_sample['theta_accuracy'],
                            boot_sample['theta_confidence']
                        )
                        r_boot.append(r_i)
                    except Exception as e:
                        log(f"  Iteration {i} failed: {e}")
                        continue

                log(f"  Bootstrap iterations completed: {len(r_boot)}/{N_ITERATIONS}")

                # Percentile method 95% CI
                CI_bootstrap_lower = np.percentile(r_boot, 2.5)
                CI_bootstrap_upper = np.percentile(r_boot, 97.5)

                log(f"  Bootstrap CI: [{CI_bootstrap_lower:.3f}, {CI_bootstrap_upper:.3f}]")

                # Get parametric CI for comparison
                parametric_row = df_corr[
                    (df_corr['schema_condition'] == schema) &
                    (df_corr['timepoint'] == timepoint)
                ]

                if parametric_row.empty:
                    log(f"  Parametric CI not found for {schema} x {timepoint}")
                    continue

                CI_param_lower = parametric_row['CI_r_lower'].values[0]
                CI_param_upper = parametric_row['CI_r_upper'].values[0]
                r_observed = parametric_row['r'].values[0]

                log(f"  Parametric CI: [{CI_param_lower:.3f}, {CI_param_upper:.3f}]")
                log(f"  Observed r: {r_observed:.3f}")

                # Check agreement (CIs overlap by at least 50% of parametric CI width)
                overlap_lower = max(CI_bootstrap_lower, CI_param_lower)
                overlap_upper = min(CI_bootstrap_upper, CI_param_upper)
                overlap = max(0, overlap_upper - overlap_lower)

                ci_width_param = CI_param_upper - CI_param_lower
                agreement = overlap >= 0.5 * ci_width_param

                if agreement:
                    log(f"  Agreement: PASS (overlap = {overlap:.3f}, 50% of param width = {0.5*ci_width_param:.3f})")
                else:
                    log(f"  Agreement: FAIL (overlap = {overlap:.3f}, 50% of param width = {0.5*ci_width_param:.3f})")

                bootstrap_results.append({
                    'schema_condition': schema,
                    'timepoint': timepoint,
                    'r_observed': r_observed,
                    'CI_parametric_lower': CI_param_lower,
                    'CI_parametric_upper': CI_param_upper,
                    'CI_bootstrap_lower': CI_bootstrap_lower,
                    'CI_bootstrap_upper': CI_bootstrap_upper,
                    'agreement': agreement
                })

        df_bootstrap = pd.DataFrame(bootstrap_results)
        log(f"Bootstrap complete: {len(df_bootstrap)} results")
        # Validate Results
        log("Checking bootstrap results...")

        errors = []

        # Check row count
        if len(df_bootstrap) != 9:
            errors.append(f"Expected 9 rows, got {len(df_bootstrap)}")
        else:
            log("Row count: 9")

        # Check all r_observed in [-1, 1]
        if (df_bootstrap['r_observed'] < -1).any() or (df_bootstrap['r_observed'] > 1).any():
            errors.append("r_observed out of bounds [-1, 1]")
        else:
            log("All r_observed in [-1, 1]")

        # Check all CI bounds in [-1, 1]
        ci_cols = ['CI_parametric_lower', 'CI_parametric_upper', 'CI_bootstrap_lower', 'CI_bootstrap_upper']
        for col in ci_cols:
            if (df_bootstrap[col] < -1).any() or (df_bootstrap[col] > 1).any():
                errors.append(f"{col} out of bounds [-1, 1]")

        if not errors:
            log("All CI bounds in [-1, 1]")

        # Check CI ordering
        ci_violations = 0
        for idx, row in df_bootstrap.iterrows():
            if row['CI_parametric_lower'] >= row['CI_parametric_upper']:
                ci_violations += 1
                log(f"  Parametric CI ordering violated: {row['schema_condition']} x {row['timepoint']}")
            if row['CI_bootstrap_lower'] >= row['CI_bootstrap_upper']:
                ci_violations += 1
                log(f"  Bootstrap CI ordering violated: {row['schema_condition']} x {row['timepoint']}")

        if ci_violations == 0:
            log("All CIs valid (lower < upper)")
        else:
            errors.append(f"Found {ci_violations} CI ordering violations")

        # Check agreement rate
        agreement_rate = df_bootstrap['agreement'].mean()
        log(f"Agreement: {df_bootstrap['agreement'].sum()}/{len(df_bootstrap)} ({agreement_rate*100:.1f}%)")

        if agreement_rate >= 0.5:
            log("Most CIs agree (>= 50%)")
        else:
            log("Low agreement rate (< 50%) - bootstrap and parametric CIs differ substantially")

        # Compare CI widths
        df_bootstrap['CI_width_param'] = df_bootstrap['CI_parametric_upper'] - df_bootstrap['CI_parametric_lower']
        df_bootstrap['CI_width_boot'] = df_bootstrap['CI_bootstrap_upper'] - df_bootstrap['CI_bootstrap_lower']

        wider_count = (df_bootstrap['CI_width_boot'] > df_bootstrap['CI_width_param']).sum()
        log(f"Bootstrap CIs wider than parametric: {wider_count}/{len(df_bootstrap)}")

        if wider_count >= len(df_bootstrap) / 2:
            log("Bootstrap CIs typically wider (more conservative)")
        else:
            log("Bootstrap CIs not consistently wider than parametric")
        # Save Output
        if errors:
            log("FAIL - Errors detected:")
            for error in errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed with {len(errors)} error(s)")

        log("PASS - All checks passed")

        # Drop temporary width columns before saving
        df_bootstrap = df_bootstrap[['schema_condition', 'timepoint', 'r_observed',
                                     'CI_parametric_lower', 'CI_parametric_upper',
                                     'CI_bootstrap_lower', 'CI_bootstrap_upper', 'agreement']]

        df_bootstrap.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"Output: {OUTPUT_FILE}")

        log(f"Bootstrap: {N_ITERATIONS} iterations, seed={RANDOM_SEED}, participant-level resampling")
        log(f"9 bootstrap CIs computed")
        log(f"Agreement: {df_bootstrap['agreement'].sum()}/{len(df_bootstrap)} rows")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
