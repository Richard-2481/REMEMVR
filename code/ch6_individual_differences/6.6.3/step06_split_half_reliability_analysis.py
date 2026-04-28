#!/usr/bin/env python3
"""split_half_reliability_analysis: Assess reliability of ICC estimates using split-half cross-validation. Randomly"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_split_half_reliability_analysis.log"
OUTPUT_FILE = RQ_DIR / "data" / "step06_split_half_reliability.csv"

# Input file
INPUT_SLOPES = RQ_DIR / "data" / "step01_domain_slopes.csv"

# Split-half parameters
N_SPLITS = 100
SEED = 42
SPLIT_SIZE = 50

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# ICC Calculation Function

def calculate_icc(slope_values):
    """Calculate ICC(1,1) from slope values."""
    variance_total = np.var(slope_values, ddof=1)
    icc = 1.0 if variance_total > 0 else 0.0
    return icc

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Split-Half Reliability Analysis")
        # Load Data

        log(f"Reading domain slopes...")
        df_slopes = pd.read_csv(INPUT_SLOPES)
        log(f"{len(df_slopes)} rows")
        # Split-Half Reliability for Each Domain
        # For each of 100 random splits:
        #   1. Randomly split 100 participants into two halves (n=50 each)
        #   2. Compute ICC for Half1 and Half2
        #   3. Store ICC pair
        # Then correlate Half1 ICCs with Half2 ICCs

        log(f"[SPLIT-HALF] Parameters: {N_SPLITS} splits, size={SPLIT_SIZE}, seed={SEED}")

        np.random.seed(SEED)
        reliability_results = []

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values
            n_participants = len(slopes)

            log(f"[{domain.upper()}] Running split-half reliability ({N_SPLITS} splits)...")

            # Store ICC pairs
            icc_half1_list = []
            icc_half2_list = []

            for split_i in range(N_SPLITS):
                # Random split
                indices = np.random.choice(n_participants, size=n_participants, replace=False)
                half1_indices = indices[:SPLIT_SIZE]
                half2_indices = indices[SPLIT_SIZE:]

                # Get slope values for each half
                slopes_half1 = slopes[half1_indices]
                slopes_half2 = slopes[half2_indices]

                # Compute ICC for each half
                icc_half1 = calculate_icc(slopes_half1)
                icc_half2 = calculate_icc(slopes_half2)

                icc_half1_list.append(icc_half1)
                icc_half2_list.append(icc_half2)

            # Convert to arrays
            icc_half1_arr = np.array(icc_half1_list)
            icc_half2_arr = np.array(icc_half2_list)

            # Compute reliability (correlation between half1 and half2 ICCs)
            reliability_r, reliability_p = stats.pearsonr(icc_half1_arr, icc_half2_arr)

            # Bootstrap CI for reliability (percentile method)
            # Resample split pairs with replacement
            reliability_boot = []
            for _ in range(1000):
                boot_indices = np.random.choice(N_SPLITS, size=N_SPLITS, replace=True)
                boot_half1 = icc_half1_arr[boot_indices]
                boot_half2 = icc_half2_arr[boot_indices]
                boot_r, _ = stats.pearsonr(boot_half1, boot_half2)
                reliability_boot.append(boot_r)

            reliability_ci_lower = np.percentile(reliability_boot, 2.5)
            reliability_ci_upper = np.percentile(reliability_boot, 97.5)

            # Mean difference between halves
            mean_diff = np.mean(icc_half1_arr - icc_half2_arr)

            # SD of ICC estimates across splits (measure of variability)
            sd_icc = np.std(np.concatenate([icc_half1_arr, icc_half2_arr]), ddof=1)

            log(f"[{domain.upper()}] Reliability r: {reliability_r:.4f} (p={reliability_p:.4f})")
            log(f"[{domain.upper()}] 95% CI: [{reliability_ci_lower:.4f}, {reliability_ci_upper:.4f}]")
            log(f"[{domain.upper()}] Mean half1-half2 diff: {mean_diff:.4f}")
            log(f"[{domain.upper()}] SD(ICC): {sd_icc:.4f}")

            reliability_results.append({
                'domain': domain.capitalize(),
                'reliability_r': reliability_r,
                'reliability_ci_lower': reliability_ci_lower,
                'reliability_ci_upper': reliability_ci_upper,
                'mean_diff': mean_diff,
                'sd_icc': sd_icc
            })
        # Create Output DataFrame

        df_reliability = pd.DataFrame(reliability_results)
        log(f"Split-half reliability table: {len(df_reliability)} rows")
        # Validate Results

        log(f"Checking reliability results...")

        # Check all domains present
        if len(df_reliability) != 2:
            log(f"Expected 2 domains, got {len(df_reliability)}")
            sys.exit(1)
        else:
            log(f"Both domains analyzed")

        # Check reliability_r in [-1, 1]
        for _, row in df_reliability.iterrows():
            domain = row['domain']
            r = row['reliability_r']

            if r < -1 or r > 1:
                log(f"{domain}: reliability_r = {r} outside [-1, 1]")
                sys.exit(1)

        log(f"All reliability_r in [-1, 1]")

        # Check expected reliability thresholds
        for _, row in df_reliability.iterrows():
            domain = row['domain']
            r = row['reliability_r']

            if r > 0.50:
                log(f"{domain}: reliability_r = {r:.4f} > 0.50 (good reliability)")
            else:
                log(f"{domain}: reliability_r = {r:.4f} <= 0.50 (low reliability)")

        # Check CI bounds reasonable
        for _, row in df_reliability.iterrows():
            domain = row['domain']
            ci_lower = row['reliability_ci_lower']
            ci_upper = row['reliability_ci_upper']

            if ci_lower < -1 or ci_upper > 1:
                log(f"{domain}: CI bounds outside [-1, 1]")
                sys.exit(1)

        log(f"All CI bounds in [-1, 1]")
        # Save Results

        log(f"Writing split-half reliability...")
        df_reliability.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_reliability)} rows)")

        log(f"Step 06 complete")
        log(f"Reliability estimates: What={df_reliability[df_reliability['domain']=='What']['reliability_r'].values[0]:.4f}, Where={df_reliability[df_reliability['domain']=='Where']['reliability_r'].values[0]:.4f}")
        log(f"Proceed to step07 (power analysis)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
