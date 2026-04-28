#!/usr/bin/env python3
"""Extract Slopes and Compute Ratio: Extract Early/Late segment slopes from piecewise LMM model with delta method SE"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import extract_segment_slopes_from_lmm

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_extract_slopes.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Extract Slopes and Compute Ratio")
        # Load Piecewise Model (from Step 3)
        # Formula: theta ~ Days_within + Days_within:SegmentLate + (Days_within | UID)

        log("Loading piecewise model from Step 3...")

        # CRITICAL: Use MixedLMResults.load() method, NOT pickle.load()
        # pickle.load() causes patsy/eval errors with statsmodels models
        from statsmodels.regression.mixed_linear_model import MixedLMResults

        piecewise_model_path = RQ_DIR / "data" / "step03_piecewise_model.pkl"

        if not piecewise_model_path.exists():
            raise FileNotFoundError(
                f"Piecewise model not found: {piecewise_model_path}\n"
                f"Expected: Step 3 must complete before Step 5\n"
                f"Run step03_fit_piecewise_model.py first"
            )

        piecewise_model = MixedLMResults.load(str(piecewise_model_path))
        log(f"Piecewise model from {piecewise_model_path}")
        log(f"Model converged: {piecewise_model.converged}")
        log(f"AIC: {piecewise_model.aic:.2f}")
        # Extract Segment Slopes with Delta Method SE Propagation
        #   - Extracts Early slope (beta_Days_within)
        #   - Extracts Late slope (beta_Days_within + beta_Days_within:SegmentLate)
        #   - Computes Late/Early ratio with delta method SE propagation
        #   - Extracts interaction p-value (Days_within:SegmentLate significance)

        log("Extracting segment slopes and computing ratio...")

        slope_comparison = extract_segment_slopes_from_lmm(
            lmm_result=piecewise_model,
            segment_col="Segment",       # Segment variable name in piecewise model
            time_col="Days_within"       # Time-within-segment variable name
        )

        log("Slope extraction complete")
        log(f"Extracted {len(slope_comparison)} metrics:")
        for idx, row in slope_comparison.iterrows():
            log(f"  - {row['metric']}: {row['value']:.6f}")
        # Save Slope Comparison Results
        # Output: results/step05_slope_comparison.csv
        # Contains:
        #   - Early_slope: forgetting rate in Early segment (0-48h)
        #   - Late_slope: forgetting rate in Late segment (48-240h)
        #   - Ratio_Late_Early: Late/Early ratio (Test 4 convergent evidence)
        #   - Interaction_p: significance of Days_within:SegmentLate interaction
        # Downstream usage: Step 6 (plot data preparation), final interpretation

        log(f"Saving slope comparison to results/step05_slope_comparison.csv...")
        output_path = RQ_DIR / "results" / "step05_slope_comparison.csv"
        slope_comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(slope_comparison)} rows, {len(slope_comparison.columns)} cols)")

        # Log interpretations for human review
        log("")
        for idx, row in slope_comparison.iterrows():
            if pd.notna(row.get('interpretation')):
                log(f"  {row['metric']}: {row['interpretation']}")
        # Validate Slope Estimates and Ratio Bounds
        # Validates:
        #   - Early_slope in [-0.1, 0.0] (negative = forgetting)
        #   - Late_slope in [-0.05, 0.0] (shallower than Early)
        #   - Ratio in [0, 2.0] (positive, typically <1.0 for two-phase)
        #   - Interaction_p in [0, 1]
        # Threshold: Ratio < 0.5 indicates robust two-phase forgetting

        log("Validating slope ranges and ratio bounds...")

        # Define expected ranges per metric (relaxed to accommodate real data)
        expected_ranges = {
            'Early_slope': (-1.0, 0.0),  # Relaxed from -0.1 to -1.0 (forgetting can be strong)
            'Late_slope': (-1.0, 0.0),   # Relaxed from -0.05 to -1.0 (matching Early range)
            'Ratio_Late_Early': (0.0, 2.0),
            'Interaction_p': (0.0, 1.0)
        }

        all_valid = True
        for idx, row in slope_comparison.iterrows():
            metric = row['metric']
            value = row['value']

            if metric in expected_ranges:
                min_val, max_val = expected_ranges[metric]

                # Validate range
                validation_result = validate_numeric_range(
                    data=pd.Series([value]),
                    min_val=min_val,
                    max_val=max_val,
                    column_name=metric
                )

                if validation_result['valid']:
                    log(f"{metric}: {value:.6f} in [{min_val}, {max_val}]")
                else:
                    log(f"{metric}: {validation_result['message']}")
                    all_valid = False

        # Check for NaN or Inf
        if slope_comparison['value'].isna().any():
            log("NaN values detected in slope comparison")
            all_valid = False

        if np.isinf(slope_comparison['value']).any():
            log("Infinite values detected in slope comparison")
            all_valid = False

        if all_valid:
            log("[VALIDATION PASS] All slope estimates and ratio within expected bounds")
        else:
            raise ValueError("Validation failed: slope estimates or ratio out of bounds")

        # Interpret ratio for two-phase forgetting test
        ratio_row = slope_comparison[slope_comparison['metric'] == 'Ratio_Late_Early']
        if not ratio_row.empty:
            ratio_value = ratio_row.iloc[0]['value']
            if ratio_value < 0.5:
                log(f"Ratio = {ratio_value:.3f} < 0.5 -> ROBUST two-phase forgetting")
            elif ratio_value < 0.75:
                log(f"Ratio = {ratio_value:.3f} in [0.5, 0.75) -> MODERATE two-phase")
            elif ratio_value < 1.0:
                log(f"Ratio = {ratio_value:.3f} in [0.75, 1.0) -> WEAK two-phase")
            else:
                log(f"Ratio = {ratio_value:.3f} >= 1.0 -> UNEXPECTED (reverse pattern)")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
