#!/usr/bin/env python3
"""Correlation Analysis with Steiger's Z-Test: Test whether Purified CTT shows higher correlation with IRT theta compared to"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_correlation_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Correlation Analysis with Steiger's Z-Test")
        # Load Input Data
        #           Purified CTT scores from Step 3

        log("Loading theta scores from RQ 5.5.1...")
        theta_file = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step03_theta_scores.csv"
        df_theta = pd.read_csv(theta_file, encoding='utf-8')
        log(f"Theta scores: {len(df_theta)} rows, {len(df_theta.columns)} cols")
        log(f"Theta columns: {df_theta.columns.tolist()}")

        # Parse composite_ID into UID and test
        log("Parsing composite_ID into UID and test...")
        df_theta[['UID', 'test_num']] = df_theta['composite_ID'].str.split('_', expand=True)
        df_theta['test'] = 'T' + df_theta['test_num']
        df_theta = df_theta.drop(columns=['test_num', 'composite_ID'])
        log(f"Parsed UID and test from composite_ID")

        log("Loading Full CTT scores from Step 2...")
        ctt_full_file = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        df_ctt_full = pd.read_csv(ctt_full_file, encoding='utf-8')
        log(f"Full CTT scores: {len(df_ctt_full)} rows, {len(df_ctt_full.columns)} cols")

        log("Loading Purified CTT scores from Step 3...")
        ctt_purified_file = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        df_ctt_purified = pd.read_csv(ctt_purified_file, encoding='utf-8')
        log(f"Purified CTT scores: {len(df_ctt_purified)} rows, {len(df_ctt_purified.columns)} cols")
        # Merge Data by Location Type

        log("Merging theta with CTT scores...")

        # Reshape theta from wide to long format (source vs destination)
        df_theta_long = pd.concat([
            df_theta[['UID', 'test', 'theta_source']].assign(location_type='source').rename(columns={'theta_source': 'theta'}),
            df_theta[['UID', 'test', 'theta_destination']].assign(location_type='destination').rename(columns={'theta_destination': 'theta'})
        ], ignore_index=True)
        log(f"Theta reshaped to long format: {len(df_theta_long)} rows")

        # Merge theta with Full CTT
        df_merged = df_theta_long.merge(
            df_ctt_full,
            on=['UID', 'test', 'location_type'],
            how='inner'
        )
        log(f"Theta + Full CTT: {len(df_merged)} rows")

        # Merge with Purified CTT
        df_merged = df_merged.merge(
            df_ctt_purified,
            on=['UID', 'test', 'location_type'],
            how='inner'
        )
        log(f"All three measurements: {len(df_merged)} rows")
        log(f"Merged columns: {df_merged.columns.tolist()}")
        # Compute Correlations and Run Steiger's Z-Test

        log("Running Steiger's z-test per location type...")

        results = []
        bonferroni_factor = 2  # 2 location types (Decision D068)

        for location_type in ['source', 'destination']:
            log(f"Processing {location_type} memory...")

            # Filter to current location_type
            df_loc = df_merged[df_merged['location_type'] == location_type].copy()
            n = len(df_loc)
            log(f"{location_type}: n={n} observations")

            # Compute correlations
            r_full = df_loc['theta'].corr(df_loc['ctt_full_score'])
            r_purified = df_loc['theta'].corr(df_loc['ctt_purified_score'])
            r_full_purified = df_loc['ctt_full_score'].corr(df_loc['ctt_purified_score'])

            log(f"{location_type} r(theta, Full_CTT) = {r_full:.4f}")
            log(f"{location_type} r(theta, Purified_CTT) = {r_purified:.4f}")
            log(f"{location_type} r(Full_CTT, Purified_CTT) = {r_full_purified:.4f}")

            # Steiger's z-test
            # r12 = r(theta, Full_CTT)
            # r13 = r(theta, Purified_CTT)
            # r23 = r(Full_CTT, Purified_CTT)
            steiger_result = compare_correlations_dependent(
                r12=r_full,
                r13=r_purified,
                r23=r_full_purified,
                n=n
            )

            # Extract results
            steiger_z = steiger_result['z_statistic']
            p_uncorrected = steiger_result['p_value']
            delta_r = r_purified - r_full

            # Bonferroni correction (capped at 1.0)
            p_bonferroni = min(p_uncorrected * bonferroni_factor, 1.0)

            log(f"{location_type} z={steiger_z:.4f}, p_uncorrected={p_uncorrected:.4f}, p_bonferroni={p_bonferroni:.4f}")
            log(f"{location_type} delta_r={delta_r:.4f} (Purified - Full)")

            # Store results
            results.append({
                'location_type': location_type,
                'r_full': r_full,
                'r_purified': r_purified,
                'delta_r': delta_r,
                'r_full_purified': r_full_purified,
                'steiger_z': steiger_z,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'n': n
            })

        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        log("Steiger's z-test complete for both location types")
        # Save Analysis Outputs
        # Output: data/step05_correlation_analysis.csv
        # Contains: Steiger's z-test results with dual p-values (Decision D068)
        # Columns: location_type, r_full, r_purified, delta_r, r_full_purified,
        #          steiger_z, p_uncorrected, p_bonferroni, n

        output_file = RQ_DIR / "data" / "step05_correlation_analysis.csv"
        log(f"Saving results to {output_file}...")
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        log(f"{output_file.name} ({len(df_results)} rows, {len(df_results.columns)} cols)")
        # Run Validation Tool
        # Validates: Decision D068 compliance (dual p-value reporting)
        # Threshold: p_bonferroni >= p_uncorrected

        log("Running validate_correlation_test_d068...")
        validation_result = validate_correlation_test_d068(
            correlation_df=df_results,
            required_cols=['p_uncorrected', 'p_bonferroni']
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - Decision D068 compliant")
            log(f"d068_compliant: {validation_result['d068_compliant']}")
            log(f"All required columns present")
        else:
            log(f"FAIL - {validation_result['message']}")
            if validation_result.get('missing_cols'):
                log(f"Missing columns: {validation_result['missing_cols']}")
            sys.exit(1)

        # Additional validation checks
        log("Running additional checks...")

        # Check correlation ranges
        all_corrs = df_results[['r_full', 'r_purified', 'r_full_purified']].values.flatten()
        if np.any(all_corrs < 0.50) or np.any(all_corrs > 1.00):
            log(f"WARNING - Some correlations outside expected range [0.50, 1.00]")
            log(f"Range: [{all_corrs.min():.4f}, {all_corrs.max():.4f}]")
        else:
            log(f"PASS - All correlations in reasonable range [0.50, 1.00]")

        # Check p_bonferroni >= p_uncorrected
        if np.all(df_results['p_bonferroni'] >= df_results['p_uncorrected']):
            log(f"PASS - p_bonferroni >= p_uncorrected (as expected)")
        else:
            log(f"FAIL - p_bonferroni < p_uncorrected (correction error)")
            sys.exit(1)

        # Check all correlations positive
        if np.all(all_corrs > 0):
            log(f"PASS - All correlations positive (CTT and IRT agree)")
        else:
            log(f"WARNING - Some correlations negative or zero")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
