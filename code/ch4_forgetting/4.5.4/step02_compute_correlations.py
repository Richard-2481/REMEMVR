#!/usr/bin/env python3
"""step02_compute_correlations: Compute Pearson correlations between IRT theta scores and CTT mean scores,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_pearson_correlations_with_correction

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_compute_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute Pearson Correlations between IRT and CTT Scores")
        # Load Input Data
        #           CTT mean scores computed in Step 1 (800 rows)

        log("Loading IRT theta scores from RQ 5.5.1...")
        theta_long = pd.read_csv(RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv", encoding='utf-8')
        log(f"step00_irt_theta_from_rq551.csv ({len(theta_long)} rows, {len(theta_long.columns)} cols)")

        log("Loading CTT mean scores from Step 1...")
        ctt_scores = pd.read_csv(RQ_DIR / "data" / "step01_ctt_scores.csv", encoding='utf-8')
        log(f"step01_ctt_scores.csv ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        # Merge IRT and CTT Scores

        log("Merging IRT theta and CTT scores on composite_ID + location_type...")

        # Select required columns from each dataset
        theta_subset = theta_long[['composite_ID', 'location_type', 'irt_theta']].copy()
        ctt_subset = ctt_scores[['composite_ID', 'location_type', 'ctt_mean_score']].copy()

        # Merge on composite_ID + location_type
        merged_data = pd.merge(
            theta_subset,
            ctt_subset,
            on=['composite_ID', 'location_type'],
            how='inner',
            validate='one_to_one'  # Ensure no duplicates
        )

        log(f"Combined dataset: {len(merged_data)} rows")

        # Verify merge completeness
        if len(merged_data) != 800:
            log(f"Expected 800 rows after merge, got {len(merged_data)}")

        # Check for missing values
        missing_irt = merged_data['irt_theta'].isna().sum()
        missing_ctt = merged_data['ctt_mean_score'].isna().sum()
        if missing_irt > 0 or missing_ctt > 0:
            log(f"Missing values detected: IRT={missing_irt}, CTT={missing_ctt}")
        # Run Analysis Tool - Compute Pearson Correlations
        #               plus overall (all pooled), with Holm-Bonferroni correction

        log("Running compute_pearson_correlations_with_correction...")
        log("Parameters:")
        log("  irt_col: 'irt_theta'")
        log("  ctt_col: 'ctt_mean_score'")
        log("  factor_col: 'location_type'")
        log("  thresholds: [0.70, 0.90]")

        correlations = compute_pearson_correlations_with_correction(
            df=merged_data,
            irt_col='irt_theta',
            ctt_col='ctt_mean_score',
            factor_col='location_type',
            thresholds=[0.70, 0.90]
        )

        log("Correlation analysis complete")
        log(f"Generated {len(correlations)} correlation results")
        log(f"Output columns: {list(correlations.columns)}")

        # Rename 'factor' to 'location_type' for consistency with spec (tool outputs 'factor')
        if 'factor' in correlations.columns and 'location_type' not in correlations.columns:
            correlations = correlations.rename(columns={'factor': 'location_type'})
            log("Renamed 'factor' -> 'location_type'")
        # Save Analysis Outputs
        # Output: step02_correlations.csv
        # Contains: Pearson correlations with dual p-values (uncorrected + Holm)
        # Columns: location_type, r, CI_lower, CI_upper, p_uncorrected, p_holm, n, threshold_0.70, threshold_0.90
        # These outputs will be used by: Step 5 (fixed effects comparison), results analysis (final report)

        output_path = RQ_DIR / "data" / "step02_correlations.csv"
        log(f"Saving {output_path.name}...")
        correlations.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_correlations.csv ({len(correlations)} rows, {len(correlations.columns)} cols)")

        # Log correlation results for transparency
        log("Results summary:")
        for idx, row in correlations.iterrows():
            loc_type = row['location_type']
            r_val = row['r']
            ci_lower = row['CI_lower']
            ci_upper = row['CI_upper']
            p_uncorr = row['p_uncorrected']
            p_holm = row['p_holm']
            n_obs = row['n']
            log(f"  {loc_type}: r={r_val:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}], p_uncorr={p_uncorr:.4f}, p_holm={p_holm:.4f}, n={n_obs}")
        # Run Validation Tool - Validate D068 Compliance
        # Validates: Decision D068 dual p-value reporting compliance
        # Criteria: p_uncorrected + p_holm columns present, p_holm >= p_uncorrected,
        #           r in [-1, 1], exactly 3 rows, all location types present

        log("Running validate_correlation_test_d068...")
        validation_result = validate_correlation_test_d068(
            correlation_df=correlations,
            required_cols=None  # Use default D068 columns
        )

        # Report validation results
        if validation_result['valid']:
            log("PASS - All D068 compliance checks passed")
            log(f"Message: {validation_result['message']}")
        else:
            log("FAIL - D068 compliance issues detected")
            log(f"Message: {validation_result['message']}")
            if validation_result.get('missing_cols'):
                log(f"Missing columns: {validation_result['missing_cols']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation checks
        log("Additional checks...")

        # Check exactly 3 rows
        if len(correlations) != 3:
            raise ValueError(f"Expected 3 rows (source, destination, overall), got {len(correlations)}")
        log("Row count: 3 rows (PASS)")

        # Check r in [-1, 1]
        r_out_of_bounds = correlations[(correlations['r'] < -1) | (correlations['r'] > 1)]
        if len(r_out_of_bounds) > 0:
            raise ValueError(f"Correlation coefficient out of bounds: {r_out_of_bounds['r'].tolist()}")
        log("Correlation bounds: all r in [-1, 1] (PASS)")

        # Check p_holm >= p_uncorrected (correction cannot reduce p-value)
        violations = correlations[correlations['p_holm'] < correlations['p_uncorrected']]
        if len(violations) > 0:
            raise ValueError(f"Holm correction violation: p_holm < p_uncorrected for {violations['location_type'].tolist()}")
        log("Holm correction monotonicity: all p_holm >= p_uncorrected (PASS)")

        # Check all location types present
        expected_types = {'source', 'destination', 'Overall'}  # Note: Overall capitalized
        actual_types = set(correlations['location_type'].unique())
        missing_types = expected_types - actual_types
        if missing_types:
            raise ValueError(f"Missing location types: {missing_types}")
        log(f"Location types: {actual_types} (PASS)")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
