#!/usr/bin/env python3
"""Test consolidation benefit per location type: Test whether each location type (Source, Destination) shows consolidation benefit"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Import statsmodels for loading LMM model
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.2
LOG_FILE = RQ_DIR / "logs" / "step05_test_consolidation_benefit.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Test consolidation benefit per location type")
        # Load Input Data

        log("Loading segment-location slopes from Step 4...")
        slopes_file = RQ_DIR / "data/step04_segment_location_slopes.csv"
        df_slopes = pd.read_csv(slopes_file)
        log(f"{slopes_file.name} ({len(df_slopes)} rows, {len(df_slopes.columns)} cols)")

        # Load fitted LMM model for variance-covariance matrix
        log("Loading fitted LMM model for variance-covariance matrix...")
        model_file = RQ_DIR / "data/step03_piecewise_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_file))
        log(f"{model_file.name} (MixedLMResults object)")
        # Extract Early/Late Slopes Per Location Type
        # Extract the 4 slopes from Step 4 output:
        #   - Source_Early, Source_Late
        #   - Destination_Early, Destination_Late

        log("Extracting Early/Late slopes per location type...")

        # Source slopes
        source_early = df_slopes[(df_slopes['Segment'] == 'Early') &
                                 (df_slopes['LocationType'] == 'Source')].iloc[0]
        source_late = df_slopes[(df_slopes['Segment'] == 'Late') &
                                (df_slopes['LocationType'] == 'Source')].iloc[0]

        # Destination slopes
        dest_early = df_slopes[(df_slopes['Segment'] == 'Early') &
                               (df_slopes['LocationType'] == 'Destination')].iloc[0]
        dest_late = df_slopes[(df_slopes['Segment'] == 'Late') &
                              (df_slopes['LocationType'] == 'Destination')].iloc[0]

        log(f"Source Early slope: {source_early['slope']:.4f} +/- {source_early['SE']:.4f}")
        log(f"Source Late slope: {source_late['slope']:.4f} +/- {source_late['SE']:.4f}")
        log(f"Destination Early slope: {dest_early['slope']:.4f} +/- {dest_early['SE']:.4f}")
        log(f"Destination Late slope: {dest_late['slope']:.4f} +/- {dest_late['SE']:.4f}")
        # Compute Consolidation Benefit with Delta Method SE
        # Consolidation benefit = Early_slope - Late_slope
        #
        # INTERPRETATION:
        # - If Early slope is more negative than Late slope, difference will be NEGATIVE
        # - Negative difference = Early forgetting is steeper (more negative slope)
        # - This indicates consolidation benefit (stabilization over time)
        # - Significant consolidation benefit = 95% CI excludes 0
        #
        # DELTA METHOD:
        # SE(difference) = sqrt(Var(Early) + Var(Late) - 2*Cov(Early, Late))
        # This accounts for covariance between Early and Late slopes (both from same model)

        log("Computing consolidation benefit per location type with delta method SE...")

        # Get variance-covariance matrix from fitted model
        vcov_matrix = lmm_model.cov_params()

        # Get coefficient names to map slopes to vcov indices
        coef_names = lmm_model.params.index.tolist()
        log(f"Model coefficients: {coef_names}")

        # Map coefficient names to indices
        # Piecewise LMM formula: theta ~ Days_within * Segment * LocationType
        # Early Source slope: Days_within (reference level)
        # Late Source slope: Days_within + Days_within:Segment[T.Late]
        # Early Destination slope: Days_within + Days_within:LocationType[T.Destination]
        # Late Destination slope: Days_within + Days_within:Segment[T.Late] + Days_within:LocationType[T.Destination] + Days_within:Segment[T.Late]:LocationType[T.Destination]

        # Find coefficient indices
        idx_days_within = coef_names.index('Days_within')
        idx_days_segment = coef_names.index('Days_within:Segment[T.Late]')
        idx_days_location = coef_names.index('Days_within:LocationType[T.Destination]')
        idx_3way = coef_names.index('Days_within:Segment[T.Late]:LocationType[T.Destination]')

        # Function to compute delta method SE for slope difference
        def compute_difference_se(early_indices, late_indices, vcov):
            """
            Compute SE of (Early - Late) using delta method.

            early_indices: list of coefficient indices contributing to Early slope
            late_indices: list of coefficient indices contributing to Late slope
            vcov: variance-covariance matrix

            Returns: SE of difference
            """
            # Create gradient vectors (partial derivatives)
            # For Early - Late, gradient is [+1 for Early terms, -1 for Late terms]
            n_params = len(vcov)
            gradient = np.zeros(n_params)

            # Early slope: +1 for all contributing terms
            for idx in early_indices:
                gradient[idx] += 1.0

            # Late slope: -1 for all contributing terms
            for idx in late_indices:
                gradient[idx] -= 1.0

            # Delta method: Var(f(θ)) = ∇f^T * Σ * ∇f
            variance = gradient @ vcov @ gradient
            se = np.sqrt(variance)

            return se

        # -------------------------------------------------------------------------
        # SOURCE CONSOLIDATION BENEFIT
        # -------------------------------------------------------------------------
        # Early Source slope: β_Days_within
        # Late Source slope: β_Days_within + β_Days_within:Segment[T.Late]
        # Difference: β_Days_within - (β_Days_within + β_Days_within:Segment[T.Late]) = -β_Days_within:Segment[T.Late]

        source_diff = source_early['slope'] - source_late['slope']
        source_early_indices = [idx_days_within]
        source_late_indices = [idx_days_within, idx_days_segment]
        source_diff_se = compute_difference_se(source_early_indices, source_late_indices, vcov_matrix)

        source_ci_lower = source_diff - 1.96 * source_diff_se
        source_ci_upper = source_diff + 1.96 * source_diff_se
        source_significant = (source_ci_lower > 0) or (source_ci_upper < 0)

        log(f"Source consolidation benefit: {source_diff:.4f} +/- {source_diff_se:.4f}")
        log(f"95% CI: [{source_ci_lower:.4f}, {source_ci_upper:.4f}]")
        log(f"Significant: {source_significant}")

        # -------------------------------------------------------------------------
        # DESTINATION CONSOLIDATION BENEFIT
        # -------------------------------------------------------------------------
        # Early Destination slope: β_Days_within + β_Days_within:LocationType[T.Destination]
        # Late Destination slope: β_Days_within + β_Days_within:Segment[T.Late] + β_Days_within:LocationType[T.Destination] + β_3way
        # Difference: -β_Days_within:Segment[T.Late] - β_3way

        dest_diff = dest_early['slope'] - dest_late['slope']
        dest_early_indices = [idx_days_within, idx_days_location]
        dest_late_indices = [idx_days_within, idx_days_segment, idx_days_location, idx_3way]
        dest_diff_se = compute_difference_se(dest_early_indices, dest_late_indices, vcov_matrix)

        dest_ci_lower = dest_diff - 1.96 * dest_diff_se
        dest_ci_upper = dest_diff + 1.96 * dest_diff_se
        dest_significant = (dest_ci_lower > 0) or (dest_ci_upper < 0)

        log(f"Destination consolidation benefit: {dest_diff:.4f} +/- {dest_diff_se:.4f}")
        log(f"95% CI: [{dest_ci_lower:.4f}, {dest_ci_upper:.4f}]")
        log(f"Significant: {dest_significant}")
        # Save Consolidation Benefit Results
        # Output: 2 rows (Source, Destination) with Early/Late slopes and difference

        log("Saving consolidation benefit results...")

        df_consolidation = pd.DataFrame([
            {
                'LocationType': 'Source',
                'Early_slope': source_early['slope'],
                'Late_slope': source_late['slope'],
                'Difference': source_diff,
                'SE': source_diff_se,
                'CI_lower': source_ci_lower,
                'CI_upper': source_ci_upper,
                'Significant': source_significant
            },
            {
                'LocationType': 'Destination',
                'Early_slope': dest_early['slope'],
                'Late_slope': dest_late['slope'],
                'Difference': dest_diff,
                'SE': dest_diff_se,
                'CI_lower': dest_ci_lower,
                'CI_upper': dest_ci_upper,
                'Significant': dest_significant
            }
        ])

        output_file = RQ_DIR / "data/step05_consolidation_benefit.csv"
        df_consolidation.to_csv(output_file, index=False, encoding='utf-8')
        log(f"{output_file.name} ({len(df_consolidation)} rows, {len(df_consolidation.columns)} cols)")
        # Run Validation Tool
        # Validates: Expected row count (2), column count (8), no NaN values
        # Threshold: All validation checks must pass

        log("Running validate_dataframe_structure...")
        validation_result = validate_dataframe_structure(
            df=df_consolidation,
            expected_rows=2,
            expected_columns=['LocationType', 'Early_slope', 'Late_slope', 'Difference',
                            'SE', 'CI_lower', 'CI_upper', 'Significant']
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
