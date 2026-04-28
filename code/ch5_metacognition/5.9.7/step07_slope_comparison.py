#!/usr/bin/env python3
"""
Step 07: Recognition vs Free Recall Slope Comparison
RQ 6.9.7 - Paradigm-Specific Calibration Trajectory

PURPOSE: Test Recognition advantage hypothesis by comparing calibration trajectory slopes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_slope_comparison.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 7: slope_comparison")

        # Load LMM fixed effects
        fixed_path = RQ_DIR / "data" / "step03_lmm_fixed_effects.csv"
        fixed_effects = pd.read_csv(fixed_path, encoding='utf-8')
        log(f"{fixed_path.name}")

        # Validate format
        validation_result = validate_data_format(
            fixed_effects,
            required_cols=['term', 'coef', 'se']
        )

        if not validation_result.get('valid', False):
            log(f"Data format validation failed")
            sys.exit(1)

        # Extract slopes
        # Free Recall slope = TSVR_hours coefficient (reference level)
        # Recognition slope = TSVR_hours + paradigm[T.recognition]:TSVR_hours interaction

        log("Extracting paradigm-specific slopes...")

        # Find TSVR_hours main effect (Free Recall slope)
        tsvr_row = fixed_effects[fixed_effects['term'] == 'TSVR_hours']
        if len(tsvr_row) == 0:
            log("TSVR_hours term not found in fixed effects")
            sys.exit(1)

        slope_free = tsvr_row['coef'].values[0]
        se_free = tsvr_row['se'].values[0]

        # Find Recognition×TSVR interaction
        recog_int_row = fixed_effects[fixed_effects['term'].str.contains('recognition.*TSVR', case=False, na=False)]
        if len(recog_int_row) == 0:
            log("Recognition×TSVR interaction term not found")
            slope_recognition = slope_free
            se_recognition = se_free
        else:
            interaction_coef = recog_int_row['coef'].values[0]
            se_interaction = recog_int_row['se'].values[0]
            slope_recognition = slope_free + interaction_coef
            # SE of sum (assuming independence for simplicity)
            se_recognition = np.sqrt(se_free**2 + se_interaction**2)

        log(f"  Free Recall slope: {slope_free:.6f} (SE={se_free:.6f})")
        log(f"  Recognition slope: {slope_recognition:.6f} (SE={se_recognition:.6f})")

        # Test difference
        difference = slope_recognition - slope_free
        se_difference = np.sqrt(se_free**2 + se_recognition**2)
        z_statistic = difference / se_difference if se_difference > 0 else 0.0
        p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

        log(f"  Difference: {difference:.6f} (SE={se_difference:.6f})")
        log(f"  z-statistic: {z_statistic:.3f}, p={p_uncorrected:.4f}")

        # Interpretation
        if p_uncorrected < 0.05 and difference > 0:
            interpretation = "Recognition shows less negative slope than Free Recall (hypothesis supported)"
            hypothesis_supported = True
        elif p_uncorrected < 0.05 and difference < 0:
            interpretation = "Recognition shows more negative slope than Free Recall (hypothesis not supported)"
            hypothesis_supported = False
        else:
            interpretation = "No significant difference between Recognition and Free Recall slopes"
            hypothesis_supported = False

        log(f"  Interpretation: {interpretation}")

        # Save results
        result = pd.DataFrame({
            'slope_free': [slope_free],
            'slope_recognition': [slope_recognition],
            'difference': [difference],
            'se_difference': [se_difference],
            'z_statistic': [z_statistic],
            'p_uncorrected': [p_uncorrected],
            'interpretation': [interpretation],
            'hypothesis_supported': [hypothesis_supported]
        })

        out_path = RQ_DIR / "data" / "step07_slope_comparison.csv"
        result.to_csv(out_path, index=False, encoding='utf-8')
        log(f"{out_path.name}")

        log("Step 7 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
