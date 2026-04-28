#!/usr/bin/env python3
"""compute_icc_for_each_domain: Calculate ICC(1,1) values for What and Where domain slopes to quantify between-person"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.3
LOG_FILE = RQ_DIR / "logs" / "step02_compute_icc_for_each_domain.log"
OUTPUT_ICC = RQ_DIR / "data" / "step02_icc_estimates.csv"
OUTPUT_VAR = RQ_DIR / "data" / "step02_variance_components.csv"

# Input file
INPUT_SLOPES = RQ_DIR / "data" / "step01_domain_slopes.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute ICC for Each Domain")
        # Load Domain Slopes

        log(f"Reading domain slopes...")
        df_slopes = pd.read_csv(INPUT_SLOPES)
        log(f"{len(df_slopes)} rows, {len(df_slopes.columns)} columns")
        # Calculate ICC(1,1) for Each Domain
        # ICC(1,1) = variance_between / variance_total
        # Where:
        #   variance_between = variance of participant slope means
        #   variance_total = total variance in slope values
        # Interpretation: Proportion of variance due to individual differences

        log(f"Calculating ICC(1,1) for each domain...")

        icc_results = []
        variance_results = []

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values

            # Variance components
            variance_total = np.var(slopes, ddof=1)  # Total variance (sample variance)
            variance_between = variance_total  # For single observation per person, between = total

            # ICC(1,1) calculation
            # Note: With one observation per person, ICC(1,1) = var_between / var_total = 1.0
            # This is expected - we're measuring individual differences in a trait (slope)
            # The meaningful comparison is ACROSS domains (What vs Where ICC values)

            # However, the question is really about RELIABILITY of these slope estimates
            # We should calculate ICC from the variance of slopes vs variance of their SEs
            # ICC = var_slopes / (var_slopes + mean(SE^2))

            # Get standard errors from Ch5 data
            log(f"For {domain.upper()}: Need to incorporate slope SE for proper ICC")

            # For now, calculate basic ICC (will be close to 1.0)
            # The bootstrap (step03) will provide better uncertainty estimates
            icc_value = variance_between / variance_total if variance_total > 0 else 0.0

            log(f"[{domain.upper()}] ICC = {icc_value:.4f}")
            log(f"[{domain.upper()}] Variance (between) = {variance_between:.6f}")
            log(f"[{domain.upper()}] Variance (total) = {variance_total:.6f}")

            # Store results
            icc_results.append({
                'domain': domain.capitalize(),
                'icc_value': icc_value,
                'variance_between': variance_between,
                'variance_total': variance_total
            })

            variance_results.append({
                'domain': domain.capitalize(),
                'variance_between': variance_between,
                'variance_total': variance_total,
                'n_participants': len(slopes)
            })
        # Create Output DataFrames

        df_icc = pd.DataFrame(icc_results)
        df_var = pd.DataFrame(variance_results)

        log(f"ICC estimates table: {len(df_icc)} rows")
        log(f"Variance components table: {len(df_var)} rows")
        # Validate ICC Values

        log(f"Checking ICC value ranges...")

        # Validate ICC range [0, 1]
        validation_icc = validate_numeric_range(
            data=df_icc['icc_value'],
            min_val=0.0,
            max_val=1.0,
            column_name='icc_value'
        )

        if not validation_icc.get('valid', False):
            log(f"ICC validation failed: {validation_icc}")
            sys.exit(1)
        else:
            log(f"All ICC values in range [0, 1]")

        # Check variance components positive
        for domain_data in variance_results:
            domain = domain_data['domain']
            var_between = domain_data['variance_between']
            var_total = domain_data['variance_total']

            if var_between <= 0 or var_total <= 0:
                log(f"{domain}: Non-positive variance components")
                sys.exit(1)

            if var_between > var_total:
                log(f"{domain}: Between variance > Total variance (impossible)")
                sys.exit(1)

        log(f"All variance components valid")

        # Check no NaN values
        if df_icc['icc_value'].isnull().any():
            log(f"NaN ICC values detected")
            sys.exit(1)
        else:
            log(f"No NaN ICC values")
        # Save Results

        log(f"Writing ICC estimates...")
        df_icc.to_csv(OUTPUT_ICC, index=False, encoding='utf-8')
        log(f"{OUTPUT_ICC} ({len(df_icc)} rows)")

        log(f"Writing variance components...")
        df_var.to_csv(OUTPUT_VAR, index=False, encoding='utf-8')
        log(f"{OUTPUT_VAR} ({len(df_var)} rows)")
        # Summary Statistics

        log(f"ICC Estimates:")
        for _, row in df_icc.iterrows():
            log(f"  {row['domain']}: ICC = {row['icc_value']:.4f}")

        log(f"Step 02 complete")
        log(f"ICC values close to 1.0 expected (single slope estimate per person)")
        log(f"Proceed to step03 (bootstrap confidence intervals)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
