#!/usr/bin/env python3
"""Bootstrap Confidence Intervals: Compute robust bootstrap confidence intervals for all regression coefficients"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step05_bootstrap_ci.log"

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Bootstrap Confidence Intervals")
        # Load Input Data

        log("Loading step03 merged and standardized data...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'slope', 'age_std', 'sex', 'education_std',
                        'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                        'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"All required columns present")
        log(f"Sample size: {len(df)} participants")
        # Prepare Data for Bootstrap
        # Response variable: slope
        # Predictors: age_std, sex, education_std, RAVLT_T_std, BVMT_T_std, RPM_T_std

        log("Preparing bootstrap regression data...")

        y = df['slope'].values
        X_cols = ['age_std', 'sex', 'education_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                  'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        X = df[X_cols].values
        n_samples = len(df)

        log(f"Response variable (slope): mean={y.mean():.6f}, std={y.std():.6f}")
        log(f"Predictors: {X_cols}")
        log(f"Sample size: {n_samples} participants")
        # Run Bootstrap Resampling

        log(f"Starting bootstrap with {N_BOOTSTRAP} iterations (seed={RANDOM_STATE})...")

        # Set random seed for reproducibility
        np.random.seed(RANDOM_STATE)

        # Storage for bootstrap coefficients
        # Shape: (n_bootstrap, n_predictors + 1)  [+1 for intercept]
        bootstrap_coefs = np.zeros((N_BOOTSTRAP, len(X_cols) + 1))

        # Bootstrap loop
        for i in range(N_BOOTSTRAP):
            # Progress logging every 100 iterations
            if (i + 1) % 100 == 0:
                log(f"Completed {i+1}/{N_BOOTSTRAP} iterations...")

            # Participant-level resampling WITH replacement
            resample_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Resample X and y
            X_resample = X[resample_indices]
            y_resample = y[resample_indices]

            # Add constant and fit OLS
            X_resample_const = sm.add_constant(X_resample)
            model = sm.OLS(y_resample, X_resample_const).fit()

            # Store coefficients (intercept + predictors)
            bootstrap_coefs[i, :] = model.params

        log(f"Completed all {N_BOOTSTRAP} iterations")
        # Compute Bootstrap Statistics and CIs
        # Percentile method: 2.5th and 97.5th percentiles for 95% CI

        log("Computing bootstrap confidence intervals...")

        # Compute percentiles
        alpha = 1 - CONFIDENCE_LEVEL
        ci_lower_pct = alpha / 2 * 100  # 2.5th percentile
        ci_upper_pct = (1 - alpha / 2) * 100  # 97.5th percentile

        log(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
        log(f"Percentiles: {ci_lower_pct}th and {ci_upper_pct}th")

        # Parameter names (with intercept)
        param_names = ['const'] + X_cols

        # Prepare results list
        results_list = []

        for i, param_name in enumerate(param_names):
            # Bootstrap distribution for this coefficient
            boot_dist = bootstrap_coefs[:, i]

            # Compute statistics
            mean_boot = np.mean(boot_dist)
            std_boot = np.std(boot_dist, ddof=1)  # Sample std with Bessel's correction
            ci_lower = np.percentile(boot_dist, ci_lower_pct)
            ci_upper = np.percentile(boot_dist, ci_upper_pct)

            results_list.append({
                'coefficient': param_name,
                'mean': mean_boot,
                'std': std_boot,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

            log(f"{param_name}: mean={mean_boot:.6f}, std={std_boot:.6f}, CI=[{ci_lower:.6f}, {ci_upper:.6f}]")
        # Save Bootstrap Results
        # Output: step05_bootstrap_ci.csv with all coefficients

        log("Saving bootstrap confidence intervals...")
        output_path = RQ_DIR / "data" / "step05_bootstrap_ci.csv"
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(results_df)} rows, {len(results_df.columns)} cols)")
        # Validate Bootstrap Results
        # Check CI ordering: ci_lower < mean < ci_upper for all coefficients

        log("Validating bootstrap confidence intervals...")

        valid_ordering = []
        for idx, row in results_df.iterrows():
            coef_name = row['coefficient']
            ci_lower = row['ci_lower']
            mean_val = row['mean']
            ci_upper = row['ci_upper']

            # Check ordering
            if ci_lower < mean_val < ci_upper:
                valid_ordering.append(True)
                log(f"{coef_name}: {ci_lower:.6f} < {mean_val:.6f} < {ci_upper:.6f}")
            else:
                valid_ordering.append(False)
                log(f"{coef_name}: Invalid CI ordering")

        if all(valid_ordering):
            log(f"All {len(valid_ordering)} coefficients have valid CI ordering")
        else:
            failed_count = sum(not v for v in valid_ordering)
            log(f"{failed_count}/{len(valid_ordering)} coefficients have invalid CI ordering")

        # Check bootstrap iterations
        log(f"Completed {N_BOOTSTRAP} bootstrap iterations")

        # Check confidence level
        log(f"Confidence level: {CONFIDENCE_LEVEL*100}%")

        log("Step 05 complete - Bootstrap confidence intervals computed")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
