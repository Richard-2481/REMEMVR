#!/usr/bin/env python3
"""Bootstrap 95% CI for Delta_f2: Compute bootstrap 95% confidence interval for Delta_f2 (difference in effect"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_bootstrap.log"

# Bootstrap parameters
N_ITERATIONS = 10000
SEED = 42
np.random.seed(SEED)

# TSVR mapping (normalized format: 1/2/3/4)
TSVR_MAP = {'1': 1.0, '2': 26.0, '3': 74.0, '4': 148.0}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def compute_marginal_r2(lmm_result) -> float:
    """Compute marginal R-squared (fixed effects variance / total variance)."""
    fitted_fixed = lmm_result.fittedvalues
    var_fixed = np.var(fitted_fixed)
    var_resid = lmm_result.scale
    var_random = lmm_result.cov_re.values[0, 0] if hasattr(lmm_result, 'cov_re') else 0
    r2 = var_fixed / (var_fixed + var_random + var_resid)
    return r2

def bootstrap_iteration(df_acc_wide, df_conf_wide, uids: List[str], iteration: int) -> float:
    """
    Single bootstrap iteration: resample participants, fit models, compute Delta_f2.

    Returns: Delta_f2 (f2_accuracy - f2_confidence) for this iteration
    """
    # Resample participants WITH replacement
    resampled_uids = np.random.choice(uids, size=len(uids), replace=True)

    # Get all observations for resampled participants
    df_acc_boot = df_acc_wide[df_acc_wide['UID'].isin(resampled_uids)].copy()
    df_conf_boot = df_conf_wide[df_conf_wide['UID'].isin(resampled_uids)].copy()

    # Reshape to LONG (source/destination)
    # Accuracy
    df_acc_source = df_acc_boot[['UID', 'test', 'TSVR_hours', 'theta_source']].copy()
    df_acc_source.columns = ['UID', 'test', 'TSVR_hours', 'theta']
    df_acc_source['location'] = 'source'

    df_acc_dest = df_acc_boot[['UID', 'test', 'TSVR_hours', 'theta_destination']].copy()
    df_acc_dest.columns = ['UID', 'test', 'TSVR_hours', 'theta']
    df_acc_dest['location'] = 'destination'

    df_acc_long = pd.concat([df_acc_source, df_acc_dest], ignore_index=True)

    # Confidence
    df_conf_source = df_conf_boot[['UID', 'test', 'TSVR_hours', 'theta_Source']].copy()
    df_conf_source.columns = ['UID', 'test', 'TSVR_hours', 'theta']
    df_conf_source['location'] = 'source'

    df_conf_dest = df_conf_boot[['UID', 'test', 'TSVR_hours', 'theta_Destination']].copy()
    df_conf_dest.columns = ['UID', 'test', 'TSVR_hours', 'theta']
    df_conf_dest['location'] = 'destination'

    df_conf_long = pd.concat([df_conf_source, df_conf_dest], ignore_index=True)

    # Fit 4 models
    try:
        # Accuracy full
        acc_full = smf.mixedlm(
            formula='theta ~ location * TSVR_hours',
            data=df_acc_long,
            groups=df_acc_long['UID'],
            re_formula='~1'
        ).fit(reml=False)
        r2_acc_full = compute_marginal_r2(acc_full)

        # Accuracy reduced
        acc_reduced = smf.mixedlm(
            formula='theta ~ location + TSVR_hours',
            data=df_acc_long,
            groups=df_acc_long['UID'],
            re_formula='~1'
        ).fit(reml=False)
        r2_acc_reduced = compute_marginal_r2(acc_reduced)

        # Confidence full
        conf_full = smf.mixedlm(
            formula='theta ~ location * TSVR_hours',
            data=df_conf_long,
            groups=df_conf_long['UID'],
            re_formula='~1'
        ).fit(reml=False)
        r2_conf_full = compute_marginal_r2(conf_full)

        # Confidence reduced
        conf_reduced = smf.mixedlm(
            formula='theta ~ location + TSVR_hours',
            data=df_conf_long,
            groups=df_conf_long['UID'],
            re_formula='~1'
        ).fit(reml=False)
        r2_conf_reduced = compute_marginal_r2(conf_reduced)

        # Compute f-squared
        f2_acc = (r2_acc_full - r2_acc_reduced) / (1 - r2_acc_full)
        f2_conf = (r2_conf_full - r2_conf_reduced) / (1 - r2_conf_full)
        delta_f2 = f2_acc - f2_conf

        return delta_f2

    except Exception as e:
        # Model fitting failed (convergence issue)
        log(f"Iteration {iteration}: {str(e)[:50]}")
        return np.nan

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Bootstrap 95% CI for Delta_f2")
        log(f"n_iterations={N_ITERATIONS}, seed={SEED}")
        # STEP 5.1: LOAD DATA
        log("\n[STEP 5.1] Load accuracy and confidence data")

        acc_path = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step03_theta_scores.csv"
        df_acc_wide = pd.read_csv(acc_path)

        # Parse composite_ID (format: A010_1, A010_2, etc.)
        df_acc_wide['UID'] = df_acc_wide['composite_ID'].str.split('_').str[0]
        df_acc_wide['test'] = df_acc_wide['composite_ID'].str.split('_').str[1]
        df_acc_wide['TSVR_hours'] = df_acc_wide['test'].map(TSVR_MAP)
        log(f"Accuracy: {len(df_acc_wide)} rows")

        conf_path = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step03_theta_confidence.csv"
        df_conf_wide = pd.read_csv(conf_path)

        # Parse composite_ID (format: A010_T1, A010_T2, etc.)
        df_conf_wide['UID'] = df_conf_wide['composite_ID'].str.split('_').str[0]
        df_conf_wide['test_raw'] = df_conf_wide['composite_ID'].str.split('_').str[1]

        # Normalize test format: T1->1, T2->2, T3->3, T4->4
        df_conf_wide['test'] = df_conf_wide['test_raw'].str.replace('T', '')
        df_conf_wide['TSVR_hours'] = df_conf_wide['test'].map(TSVR_MAP)
        log(f"Confidence: {len(df_conf_wide)} rows")
        log(f"Confidence test format: T1/T2/T3/T4 -> 1/2/3/4")

        # Get unique UIDs
        uids_acc = df_acc_wide['UID'].unique()
        uids_conf = df_conf_wide['UID'].unique()
        uids = np.intersect1d(uids_acc, uids_conf)
        log(f"{len(uids)} participants in both datasets")
        # STEP 5.2: BOOTSTRAP LOOP
        log(f"\n[STEP 5.2] Bootstrap loop: {N_ITERATIONS} iterations")
        log(f"This will take ~15 minutes (40,000 LMM fits)")

        bootstrap_results = []
        converged_count = 0

        for i in range(N_ITERATIONS):
            if (i + 1) % 1000 == 0:
                log(f"Iteration {i+1}/{N_ITERATIONS} ({converged_count} converged)")

            delta_f2 = bootstrap_iteration(df_acc_wide, df_conf_wide, uids, i)

            if not np.isnan(delta_f2):
                converged_count += 1

            bootstrap_results.append({
                'iteration': i,
                'Delta_f2': delta_f2
            })

        df_bootstrap = pd.DataFrame(bootstrap_results)
        log(f"\nBootstrap complete: {converged_count}/{N_ITERATIONS} converged")

        # Remove NaN values
        df_bootstrap_clean = df_bootstrap.dropna(subset=['Delta_f2'])
        log(f"{len(df_bootstrap_clean)} valid iterations (>{len(df_bootstrap)*0.95:.0f} required)")

        if len(df_bootstrap_clean) < N_ITERATIONS * 0.95:
            raise ValueError(f"Too many convergence failures: {N_ITERATIONS - len(df_bootstrap_clean)}/{N_ITERATIONS}")
        # STEP 5.3: COMPUTE SUMMARY STATISTICS
        log("\n[STEP 5.3] Compute summary statistics")

        mean_delta_f2 = df_bootstrap_clean['Delta_f2'].mean()
        se_delta_f2 = df_bootstrap_clean['Delta_f2'].std()
        ci_lower = df_bootstrap_clean['Delta_f2'].quantile(0.025)
        ci_upper = df_bootstrap_clean['Delta_f2'].quantile(0.975)
        excludes_zero = (ci_lower > 0) or (ci_upper < 0)

        log(f"Mean Delta_f2: {mean_delta_f2:.4f}")
        log(f"SE Delta_f2: {se_delta_f2:.4f}")
        log(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        log(f"Excludes zero: {excludes_zero}")

        df_summary = pd.DataFrame([{
            'mean_Delta_f2': mean_delta_f2,
            'se_Delta_f2': se_delta_f2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'excludes_zero': excludes_zero
        }])
        # STEP 5.4: COMPUTE PERCENTILES
        log("\n[STEP 5.4] Compute distribution percentiles")

        percentiles = [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]
        percentile_values = df_bootstrap_clean['Delta_f2'].quantile(np.array(percentiles) / 100)

        df_percentiles = pd.DataFrame({
            'percentile': percentiles,
            'value': percentile_values.values
        })

        for p, v in zip(percentiles, percentile_values):
            log(f"[P{p:>5.1f}] {v:.4f}")
        # STEP 5.5: VALIDATE RESULTS
        log("\n[STEP 5.5] Validate bootstrap results")

        result = validate_numeric_range(
            df_bootstrap_clean['Delta_f2'].values, -1, 1, 'Delta_f2'
        )
        if result['valid']:
            log(f"Delta_f2 values in reasonable range")
        else:
            log(f"{result['message']}")
        # STEP 5.6: SAVE OUTPUTS
        log("\n[STEP 5.6] Save outputs")

        # Save all bootstrap results
        bootstrap_path = RQ_DIR / "data" / "step05_bootstrap_results.csv"
        df_bootstrap.to_csv(bootstrap_path, index=False, encoding='utf-8')
        log(f"{bootstrap_path.name} ({len(df_bootstrap)} rows)")

        # Save summary
        summary_path = RQ_DIR / "data" / "step05_bootstrap_summary.csv"
        df_summary.to_csv(summary_path, index=False, encoding='utf-8')
        log(f"{summary_path.name}")

        # Save percentiles
        percentiles_path = RQ_DIR / "data" / "step05_bootstrap_distribution.csv"
        df_percentiles.to_csv(percentiles_path, index=False, encoding='utf-8')
        log(f"{percentiles_path.name}")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
