#!/usr/bin/env python3
"""Test 3 - Slope Ratio: Test for two-phase pattern by computing Late/Early slope ratio. If ratio < 0.5,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
# Note: Using fit_lmm_trajectory (not _tsvr) because we need custom piecewise variables
from tools.analysis_lmm import fit_lmm_trajectory
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step04_compute_slope_ratio.log"

# Input/output paths
INPUT_PIECEWISE = RQ_DIR / "data" / "step01_piecewise_input.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step04_slope_ratio.csv"

# Piecewise model formula (using Days for numerical stability)
# Note: Using fit_lmm_trajectory directly with Days conversion
PIECEWISE_FORMULA = "theta_confidence ~ Time_Early_Days + Time_Late_Days"
RE_FORMULA = "~1"  # Random intercept only for piecewise (avoid convergence issues)

# Slope ratio threshold
RATIO_THRESHOLD = 0.5

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute Slope Ratio")
        # Load Input Data and Re-fit Piecewise Model

        log(f"Loading piecewise data from {INPUT_PIECEWISE.name}...")
        piecewise_data = pd.read_csv(INPUT_PIECEWISE, encoding='utf-8')
        log(f"{INPUT_PIECEWISE.name} ({len(piecewise_data)} rows)")

        # Convert hours to days for numerical stability (per Decision D070)
        piecewise_data['Time_Early_Days'] = piecewise_data['Time_Early'] / 24.0
        piecewise_data['Time_Late_Days'] = piecewise_data['Time_Late'] / 24.0
        log(f"Time_Early_Days range: [{piecewise_data['Time_Early_Days'].min():.2f}, {piecewise_data['Time_Early_Days'].max():.2f}]")
        log(f"Time_Late_Days range: [{piecewise_data['Time_Late_Days'].min():.2f}, {piecewise_data['Time_Late_Days'].max():.2f}]")

        # Re-fit piecewise model to get slope estimates
        log(f"Re-fitting piecewise model for slope extraction...")
        log(f"Formula: {PIECEWISE_FORMULA}")

        piecewise_model = fit_lmm_trajectory(
            data=piecewise_data,
            formula=PIECEWISE_FORMULA,
            groups='UID',
            re_formula=RE_FORMULA,
            reml=False
        )

        log("Piecewise model re-fitted")
        # Extract Early and Late Slopes
        # expects specific piecewise formula structure that we don't have)

        log("Extracting Early and Late slopes from model coefficients...")

        # Extract slopes directly from model parameters (using Days column names)
        beta_early = piecewise_model.params['Time_Early_Days']
        se_early = piecewise_model.bse['Time_Early_Days']

        beta_late = piecewise_model.params['Time_Late_Days']
        se_late = piecewise_model.bse['Time_Late_Days']

        log(f"Early slope: beta={beta_early:.6f}, SE={se_early:.6f}")
        log(f"Late slope: beta={beta_late:.6f}, SE={se_late:.6f}")
        # Compute Slope Ratio
        # Ratio = |beta_Late| / |beta_Early|
        # Ratio < 0.5 indicates two-phase pattern (late decline < half of early)

        log("Computing slope ratio...")

        # Use absolute values for ratio (both should be negative for decline)
        abs_beta_early = abs(beta_early)
        abs_beta_late = abs(beta_late)

        if abs_beta_early == 0:
            log("Early slope is zero, cannot compute ratio")
            raise ValueError("Early slope is zero, ratio undefined")

        ratio_value = abs_beta_late / abs_beta_early
        two_phase_evidence = ratio_value < RATIO_THRESHOLD

        log(f"Slope ratio: {ratio_value:.4f} (|Late| / |Early|)")
        log(f"Threshold: {RATIO_THRESHOLD}")

        if two_phase_evidence:
            log(f"Ratio < {RATIO_THRESHOLD} -> Two-phase pattern SUPPORTED by slope ratio test")
        else:
            log(f"Ratio >= {RATIO_THRESHOLD} -> Two-phase pattern NOT supported by slope ratio test")
        # Create Output DataFrame
        # Format: segment, slope, se, ratio_value, two_phase_evidence
        # Rows: Early, Late, Ratio summary

        slope_ratio_df = pd.DataFrame({
            'segment': ['Early', 'Late', 'Ratio'],
            'slope': [beta_early, beta_late, np.nan],
            'se': [se_early, se_late, np.nan],
            'ratio_value': [np.nan, np.nan, ratio_value],
            'two_phase_evidence': [np.nan, np.nan, two_phase_evidence]
        })

        log("Slope ratio DataFrame created")
        # Save Output
        # Output: data/step04_slope_ratio.csv
        # Contains: Early/Late slopes with SEs, ratio value, evidence conclusion

        log(f"Saving slope ratio results to {OUTPUT_FILE.name}...")
        slope_ratio_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(slope_ratio_df)} rows)")
        # Run Validation
        # Validates: Slopes in reasonable range, positive SEs, valid ratio

        log("Running custom slope ratio checks...")

        # Extract only Early/Late slopes (exclude Ratio row)
        slopes_only = slope_ratio_df[slope_ratio_df['segment'] != 'Ratio']['slope'].values

        # Check both slopes are negative (decline expected)
        if beta_early >= 0:
            log(f"WARNING - Early slope is non-negative ({beta_early:.6f}), expected decline")
        else:
            log("PASS - Early slope is negative (decline)")

        if beta_late >= 0:
            log(f"WARNING - Late slope is non-negative ({beta_late:.6f}), expected decline")
        else:
            log("PASS - Late slope is negative (decline)")

        # Check standard errors are positive
        if se_early <= 0 or se_late <= 0:
            log("FAIL - Non-positive standard errors found")
            raise ValueError("Standard errors must be positive")
        else:
            log("PASS - All standard errors positive")

        # Check ratio is non-negative (absolute values used)
        if ratio_value < 0:
            log(f"FAIL - Negative ratio value ({ratio_value:.4f})")
            raise ValueError("Ratio should be non-negative (uses absolute values)")
        else:
            log("PASS - Ratio is non-negative")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
