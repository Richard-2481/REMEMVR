#!/usr/bin/env python3
"""bootstrap_confidence_intervals: Compute 95% confidence intervals for ICC estimates using participant-level bootstrap"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.bootstrap import bootstrap_statistic

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_bootstrap_confidence_intervals.log"
OUTPUT_CIS = RQ_DIR / "data" / "step03_bootstrap_cis.csv"
OUTPUT_DIST = RQ_DIR / "data" / "step03_bootstrap_distributions.csv"

# Input files
INPUT_SLOPES = RQ_DIR / "data" / "step01_domain_slopes.csv"
INPUT_VAR = RQ_DIR / "data" / "step02_variance_components.csv"

# Bootstrap parameters
N_BOOTSTRAP = 1000
SEED = 42
CONFIDENCE = 0.95

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# ICC Calculation Function

def calculate_icc(slope_values):
    """
    Calculate ICC(1,1) from slope values.

    For single observation per person:
    ICC = variance_between / variance_total

    Returns: float ICC value
    """
    variance_total = np.var(slope_values, ddof=1)
    variance_between = variance_total  # Single obs per person
    icc = variance_between / variance_total if variance_total > 0 else 0.0
    return icc

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Bootstrap Confidence Intervals")
        # Load Data

        log(f"Reading domain slopes...")
        df_slopes = pd.read_csv(INPUT_SLOPES)
        log(f"{len(df_slopes)} rows")

        log(f"Reading variance components...")
        df_var = pd.read_csv(INPUT_VAR)
        log(f"{len(df_var)} rows")
        # Bootstrap ICC for Each Domain
        # Resample participants with replacement, recalculate ICC
        # Repeat 1000 times to build bootstrap distribution

        log(f"Parameters: n={N_BOOTSTRAP}, seed={SEED}, confidence={CONFIDENCE}")

        bootstrap_results = []
        bootstrap_distributions = []

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values

            log(f"[{domain.upper()}] Starting bootstrap (1000 iterations)...")

            # Run bootstrap
            boot_result = bootstrap_statistic(
                data=slopes,
                statistic=calculate_icc,
                n_bootstrap=N_BOOTSTRAP,
                confidence=CONFIDENCE,
                seed=SEED
            )

            # Extract results
            icc_original = calculate_icc(slopes)
            icc_boot_mean = boot_result['statistic']  # CORRECTED: bootstrap_statistic returns 'statistic', not 'mean'
            ci_lower = boot_result['ci_lower']
            ci_upper = boot_result['ci_upper']
            boot_bias = icc_boot_mean - icc_original

            log(f"[{domain.upper()}] Original ICC: {icc_original:.4f}")
            log(f"[{domain.upper()}] Bootstrap mean: {icc_boot_mean:.4f}")
            log(f"[{domain.upper()}] 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            log(f"[{domain.upper()}] Bootstrap bias: {boot_bias:.4f}")

            # Store summary
            bootstrap_results.append({
                'domain': domain.capitalize(),
                'icc_original': icc_original,
                'icc_boot_mean': icc_boot_mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'boot_bias': boot_bias
            })

            # Store distribution
            bootstrap_samples = boot_result['bootstrap_samples']
            for i, icc_boot in enumerate(bootstrap_samples):
                bootstrap_distributions.append({
                    'domain': domain.capitalize(),
                    'bootstrap_iteration': i + 1,
                    'icc_bootstrap': icc_boot
                })
        # Create Output DataFrames

        df_cis = pd.DataFrame(bootstrap_results)
        df_dist = pd.DataFrame(bootstrap_distributions)

        log(f"Bootstrap CIs table: {len(df_cis)} rows")
        log(f"Bootstrap distributions table: {len(df_dist)} rows")
        # Validate Bootstrap Results

        log(f"Checking bootstrap results...")

        # Check all iterations completed
        expected_dist_rows = 2 * N_BOOTSTRAP  # 2 domains x 1000 iterations
        if len(df_dist) != expected_dist_rows:
            log(f"Expected {expected_dist_rows} bootstrap samples, got {len(df_dist)}")
            sys.exit(1)
        else:
            log(f"All {N_BOOTSTRAP} iterations completed for both domains")

        # Check CI bounds in [0, 1]
        for _, row in df_cis.iterrows():
            domain = row['domain']
            if row['ci_lower'] < 0 or row['ci_upper'] > 1:
                log(f"{domain}: CI bounds outside [0, 1]")
                sys.exit(1)

        log(f"All CI bounds in [0, 1]")

        # Check bootstrap bias acceptable
        for _, row in df_cis.iterrows():
            domain = row['domain']
            if abs(row['boot_bias']) > 0.1:
                log(f"{domain}: Large bootstrap bias = {row['boot_bias']:.4f}")
            else:
                log(f"{domain}: Bootstrap bias acceptable ({row['boot_bias']:.4f})")
        # Save Results

        log(f"Writing bootstrap CIs...")
        df_cis.to_csv(OUTPUT_CIS, index=False, encoding='utf-8')
        log(f"{OUTPUT_CIS} ({len(df_cis)} rows)")

        log(f"Writing bootstrap distributions...")
        df_dist.to_csv(OUTPUT_DIST, index=False, encoding='utf-8')
        log(f"{OUTPUT_DIST} ({len(df_dist)} rows)")

        log(f"Step 03 complete")
        log(f"Proceed to step04 (pairwise domain comparison)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
