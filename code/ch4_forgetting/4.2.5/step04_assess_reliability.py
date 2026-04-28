#!/usr/bin/env python3
"""Assess Reliability: Compute Cronbach's alpha internal consistency for both full and purified CTT item sets per domain with bootstrap 95% confidence intervals. Tests whether IRT purification maintains CTT reliability."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_cronbachs_alpha

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_assess_reliability.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Assess Reliability")
        # Load Input Data
        #           Item mapping with retention status from step01

        log("Loading input data...")

        # Load raw item responses (all TQ_* columns)
        df_raw = pd.read_csv(RQ_DIR / "data" / "step00_raw_scores.csv", encoding='utf-8')
        log(f"step00_raw_scores.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Load item mapping with retention status
        df_mapping = pd.read_csv(RQ_DIR / "data" / "step01_item_mapping.csv", encoding='utf-8')
        log(f"step01_item_mapping.csv ({len(df_mapping)} items)")

        # Verify TQ_* columns exist in raw data
        tq_cols = [col for col in df_raw.columns if col.startswith('TQ_')]
        log(f"Found {len(tq_cols)} TQ_* item columns in raw data")
        # Run Analysis Tool (Cronbach's Alpha for Full and Purified Sets)

        log("Computing Cronbach's alpha for full vs purified item sets...")
        log("When domain EXCLUDED per RQ 5.2.1 floor effect")

        # Define domains (What/Where only - When excluded)
        domains = ['what', 'where']

        # Initialize results list
        results_list = []

        # Loop through domains
        for domain in domains:
            log(f"Processing domain: {domain}")

            # -----------------------------------------------------------------------
            # Full Item Set (ALL items for this domain)
            # -----------------------------------------------------------------------
            # Get all items for this domain (retained=True OR False)
            full_items = df_mapping[df_mapping['domain'] == domain]['item_name'].tolist()
            log(f"  [FULL SET] {len(full_items)} items for domain {domain}")

            # Select these columns from raw data
            full_data = df_raw[full_items].copy()

            # Compute Cronbach's alpha with bootstrap
            log(f"  [FULL SET] Running bootstrap (n=1000)...")
            alpha_full_result = compute_cronbachs_alpha(data=full_data, n_bootstrap=1000)

            log(f"  [FULL SET] alpha = {alpha_full_result['alpha']:.3f} "
                f"[{alpha_full_result['ci_lower']:.3f}, {alpha_full_result['ci_upper']:.3f}]")

            # -----------------------------------------------------------------------
            # Purified Item Set (ONLY retained items)
            # -----------------------------------------------------------------------
            # Get ONLY retained items for this domain (retained=True)
            purified_items = df_mapping[
                (df_mapping['domain'] == domain) &
                (df_mapping['retained'] == True)
            ]['item_name'].tolist()
            log(f"  [PURIFIED SET] {len(purified_items)} items for domain {domain}")

            # Select these columns from raw data
            purified_data = df_raw[purified_items].copy()

            # Compute Cronbach's alpha with bootstrap
            log(f"  [PURIFIED SET] Running bootstrap (n=1000)...")
            alpha_purified_result = compute_cronbachs_alpha(data=purified_data, n_bootstrap=1000)

            log(f"  [PURIFIED SET] alpha = {alpha_purified_result['alpha']:.3f} "
                f"[{alpha_purified_result['ci_lower']:.3f}, {alpha_purified_result['ci_upper']:.3f}]")

            # -----------------------------------------------------------------------
            # Compute delta_alpha (change due to purification)
            # -----------------------------------------------------------------------
            delta_alpha = alpha_purified_result['alpha'] - alpha_full_result['alpha']
            log(f"  delta_alpha = {delta_alpha:+.3f} "
                f"({'improvement' if delta_alpha > 0 else 'reduction'})")

            # -----------------------------------------------------------------------
            # Store results
            # -----------------------------------------------------------------------
            results_list.append({
                'domain': domain,
                'alpha_full': alpha_full_result['alpha'],
                'CI_lower_full': alpha_full_result['ci_lower'],
                'CI_upper_full': alpha_full_result['ci_upper'],
                'n_items_full': alpha_full_result['n_items'],
                'alpha_purified': alpha_purified_result['alpha'],
                'CI_lower_purified': alpha_purified_result['ci_lower'],
                'CI_upper_purified': alpha_purified_result['ci_upper'],
                'n_items_purified': alpha_purified_result['n_items'],
                'delta_alpha': delta_alpha
            })

        log("Cronbach's alpha computation complete for all domains")
        # Save Analysis Outputs
        # These outputs will be used by: Step 5 (correlation analysis validation)
        #                                 Step 8 (results reporting)

        log("Saving reliability assessment results...")

        # Create DataFrame from results
        df_reliability = pd.DataFrame(results_list)

        # Save to CSV
        output_path = RQ_DIR / "data" / "step04_reliability_assessment.csv"
        df_reliability.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step04_reliability_assessment.csv ({len(df_reliability)} rows, {len(df_reliability.columns)} cols)")

        # Print summary to log
        log("Reliability Assessment Results:")
        for _, row in df_reliability.iterrows():
            log(f"  Domain: {row['domain']}")
            log(f"    Full Set:     alpha = {row['alpha_full']:.3f} "
                f"[{row['CI_lower_full']:.3f}, {row['CI_upper_full']:.3f}] "
                f"({row['n_items_full']} items)")
            log(f"    Purified Set: alpha = {row['alpha_purified']:.3f} "
                f"[{row['CI_lower_purified']:.3f}, {row['CI_upper_purified']:.3f}] "
                f"({row['n_items_purified']} items)")
            log(f"    Delta:        {row['delta_alpha']:+.3f}")
        # Run Validation Tool
        # Validates: All alpha values in [0, 1] range
        # Threshold: min_val=0.0, max_val=1.0

        log("Running validate_numeric_range...")

        # Validate alpha_full values
        for col in ['alpha_full', 'CI_lower_full', 'CI_upper_full',
                    'alpha_purified', 'CI_lower_purified', 'CI_upper_purified']:
            validation_result = validate_numeric_range(
                data=df_reliability[col],
                min_val=0.0,
                max_val=1.0,
                column_name=col
            )

            if validation_result['valid']:
                log(f"{col}: PASS (all values in [0, 1])")
            else:
                log(f"{col}: FAIL - {validation_result['message']}")
                raise ValueError(f"Validation failed for {col}: {validation_result['message']}")

        # Additional validation: CI_lower < alpha < CI_upper
        log("Checking CI bounds...")
        for _, row in df_reliability.iterrows():
            domain = row['domain']

            # Check full set CIs
            if not (row['CI_lower_full'] <= row['alpha_full'] <= row['CI_upper_full']):
                raise ValueError(
                    f"Domain {domain} full set: alpha {row['alpha_full']:.3f} "
                    f"not in CI [{row['CI_lower_full']:.3f}, {row['CI_upper_full']:.3f}]"
                )

            # Check purified set CIs
            if not (row['CI_lower_purified'] <= row['alpha_purified'] <= row['CI_upper_purified']):
                raise ValueError(
                    f"Domain {domain} purified set: alpha {row['alpha_purified']:.3f} "
                    f"not in CI [{row['CI_lower_purified']:.3f}, {row['CI_upper_purified']:.3f}]"
                )

        log("CI bounds check: PASS (alpha within CIs for all domains)")

        # Check typical alpha range [0.70, 0.95]
        log("Checking typical alpha range...")
        for _, row in df_reliability.iterrows():
            domain = row['domain']
            alpha_full = row['alpha_full']
            alpha_purified = row['alpha_purified']

            if alpha_full < 0.70:
                log(f"Domain {domain} full set alpha = {alpha_full:.3f} < 0.70 (below acceptable)")
            elif alpha_full > 0.95:
                log(f"Domain {domain} full set alpha = {alpha_full:.3f} > 0.95 (unusually high)")
            else:
                log(f"Domain {domain} full set alpha = {alpha_full:.3f} in typical range [0.70, 0.95]")

            if alpha_purified < 0.70:
                log(f"Domain {domain} purified set alpha = {alpha_purified:.3f} < 0.70 (below acceptable)")
            elif alpha_purified > 0.95:
                log(f"Domain {domain} purified set alpha = {alpha_purified:.3f} > 0.95 (unusually high)")
            else:
                log(f"Domain {domain} purified set alpha = {alpha_purified:.3f} in typical range [0.70, 0.95]")

        log("Step 04 complete - Reliability assessment validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
