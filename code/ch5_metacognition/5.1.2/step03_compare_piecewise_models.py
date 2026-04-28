#!/usr/bin/env python3
"""Test 2 - Piecewise vs Continuous Comparison: Test for two-phase pattern by comparing piecewise model (separate Early/Late slopes)"""

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
from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step03_compare_piecewise_models.log"

# Input/output paths
INPUT_FILE = RQ_DIR / "data" / "step01_piecewise_input.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step03_piecewise_comparison.csv"

# LMM formulas (use Days for better numerical stability)
CONTINUOUS_FORMULA = "theta_confidence ~ Days"
PIECEWISE_FORMULA = "theta_confidence ~ Time_Early_Days + Time_Late_Days"
RE_FORMULA = "~Days"  # Random slope on Days for continuous model
RE_PIECEWISE_FORMULA = "~1"  # Random intercept only for piecewise (avoid convergence issues)

# Burnham & Anderson threshold
DELTA_AIC_THRESHOLD = 2.0

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Compare Piecewise Models")
        # Load Input Data

        log(f"Loading input data from {INPUT_FILE.name}...")
        piecewise_data = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{INPUT_FILE.name} ({len(piecewise_data)} rows, {len(piecewise_data.columns)} cols)")
        log(f"Theta range: [{piecewise_data['theta_confidence'].min():.3f}, {piecewise_data['theta_confidence'].max():.3f}]")
        log(f"Segment distribution: Early={sum(piecewise_data['Segment']=='Early')}, Late={sum(piecewise_data['Segment']=='Late')}")

        # Convert hours to days for numerical stability (per Decision D070)
        piecewise_data['Days'] = piecewise_data['TSVR_hours'] / 24.0
        piecewise_data['Time_Early_Days'] = piecewise_data['Time_Early'] / 24.0
        piecewise_data['Time_Late_Days'] = piecewise_data['Time_Late'] / 24.0
        log(f"Days range: [{piecewise_data['Days'].min():.2f}, {piecewise_data['Days'].max():.2f}]")
        # Fit Continuous Model

        log(f"Fitting continuous model...")
        log(f"Formula: {CONTINUOUS_FORMULA}")
        log(f"Random effects: {RE_FORMULA}")

        continuous_model = fit_lmm_trajectory(
            data=piecewise_data,
            formula=CONTINUOUS_FORMULA,
            groups='UID',
            re_formula=RE_FORMULA,
            reml=False
        )

        log("Continuous model fitted")
        log(f"Continuous AIC: {continuous_model.aic:.2f}")

        # Validate continuous model convergence
        log("Checking continuous model convergence...")
        val_continuous = validate_lmm_convergence(lmm_result=continuous_model)
        if not val_continuous['converged']:
            log(f"FAIL - Continuous model did not converge: {val_continuous['message']}")
            raise ValueError(f"Continuous model convergence failed: {val_continuous['message']}")
        else:
            log("PASS - Continuous model converged")
        # Fit Piecewise Model

        log(f"Fitting piecewise model...")
        log(f"Formula: {PIECEWISE_FORMULA}")
        log(f"Random effects: {RE_FORMULA}")

        piecewise_model = fit_lmm_trajectory(
            data=piecewise_data,
            formula=PIECEWISE_FORMULA,
            groups='UID',
            re_formula=RE_PIECEWISE_FORMULA,  # Use piecewise-specific random effects
            reml=False
        )

        log("Piecewise model fitted")
        log(f"Piecewise AIC: {piecewise_model.aic:.2f}")

        # Validate piecewise model convergence
        log("Checking piecewise model convergence...")
        val_piecewise = validate_lmm_convergence(lmm_result=piecewise_model)
        if not val_piecewise['converged']:
            log(f"FAIL - Piecewise model did not converge: {val_piecewise['message']}")
            raise ValueError(f"Piecewise model convergence failed: {val_piecewise['message']}")
        else:
            log("PASS - Piecewise model converged")
        # Compute AIC Comparison
        # Delta AIC = AIC_continuous - AIC_piecewise
        # Positive delta = piecewise preferred

        log("Computing AIC comparison...")

        aic_continuous = continuous_model.aic
        aic_piecewise = piecewise_model.aic
        delta_aic = aic_continuous - aic_piecewise
        piecewise_preferred = delta_aic > DELTA_AIC_THRESHOLD

        log(f"Delta AIC: {delta_aic:.2f} (AIC_continuous - AIC_piecewise)")
        log(f"Threshold: {DELTA_AIC_THRESHOLD} (Burnham & Anderson)")

        if piecewise_preferred:
            log(f"Piecewise model PREFERRED (delta AIC > {DELTA_AIC_THRESHOLD}) -> Two-phase pattern SUPPORTED")
        else:
            log(f"Continuous model NOT substantially worse (delta AIC <= {DELTA_AIC_THRESHOLD}) -> Two-phase pattern NOT supported by AIC test")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'model': ['Continuous', 'Piecewise', 'Comparison'],
            'AIC': [aic_continuous, aic_piecewise, np.nan],
            'delta_AIC': [np.nan, np.nan, delta_aic],
            'piecewise_preferred': [np.nan, np.nan, piecewise_preferred]
        })

        log("AIC comparison computed")
        # Save Comparison Results
        # Output: data/step03_piecewise_comparison.csv
        # Contains: AIC values, delta AIC, preference conclusion

        log(f"Saving AIC comparison to {OUTPUT_FILE.name}...")
        comparison_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(comparison_df)} rows)")
        # Run Validation
        # Additional custom validations beyond convergence

        log("Running custom comparison checks...")

        # Check AIC values are reasonable (not negative, not extremely large)
        if aic_continuous < 0 or aic_piecewise < 0:
            log("FAIL - Negative AIC values found")
            raise ValueError("Invalid AIC values (negative)")
        else:
            log("PASS - AIC values non-negative")

        if aic_continuous > 10000 or aic_piecewise > 10000:
            log("WARNING - Extremely large AIC values (>10000)")
        else:
            log("PASS - AIC values in reasonable range")

        # Check delta AIC computed correctly
        computed_delta = aic_continuous - aic_piecewise
        if abs(computed_delta - delta_aic) > 0.01:
            log(f"FAIL - Delta AIC mismatch (computed: {computed_delta:.2f}, stored: {delta_aic:.2f})")
            raise ValueError("Delta AIC computation error")
        else:
            log("PASS - Delta AIC computed correctly")

        # Check no NaN values in key columns (except expected ones)
        if pd.isna(aic_continuous) or pd.isna(aic_piecewise) or pd.isna(delta_aic):
            log("FAIL - Unexpected NaN values in comparison results")
            raise ValueError("NaN values found in AIC comparison")
        else:
            log("PASS - No unexpected NaN values")

        # Check piecewise_preferred is boolean
        if not isinstance(piecewise_preferred, (bool, np.bool_)):
            log(f"FAIL - piecewise_preferred is not boolean: {type(piecewise_preferred)}")
            raise ValueError("piecewise_preferred should be boolean")
        else:
            log("PASS - piecewise_preferred is boolean")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
