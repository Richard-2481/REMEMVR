#!/usr/bin/env python3
"""create_classification: Apply classification criteria to identify false negatives (low RAVLT + normal REMEMVR)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step03_create_classification.log"
INPUT_FILE = RQ_DIR / "data" / "step02_standardized_scores.csv"
OUTPUT_MATRIX = RQ_DIR / "data" / "step03_classification_matrix.csv"
OUTPUT_FALSE_NEG = RQ_DIR / "data" / "step03_false_negatives.csv"

# Classification thresholds (from 4_analysis.yaml)
RAVLT_THRESHOLD = -1.0    # Below this = impaired
REMEMVR_THRESHOLD = -0.5  # Above this = normal

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: create_classification")
        # Load Standardized Scores
        log("Loading standardized scores from step02...")

        df = pd.read_csv(INPUT_FILE)
        log(f"{INPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")
        # Apply Classification Thresholds
        log("Applying classification thresholds...")
        log(f"RAVLT threshold: z < {RAVLT_THRESHOLD} (impaired)")
        log(f"REMEMVR threshold: z > {REMEMVR_THRESHOLD} (normal)")

        # Create binary classification flags
        # RAVLT_low: 1 if impaired (z < -1.0), 0 if normal
        df['RAVLT_low'] = (df['RAVLT_z'] < RAVLT_THRESHOLD).astype(int)

        # REMEMVR_normal: 1 if normal (z > -0.5), 0 if impaired
        df['REMEMVR_normal'] = (df['REMEMVR_z'] > REMEMVR_THRESHOLD).astype(int)

        # Count classifications
        n_ravlt_low = df['RAVLT_low'].sum()
        n_ravlt_normal = len(df) - n_ravlt_low
        n_rememvr_normal = df['REMEMVR_normal'].sum()
        n_rememvr_low = len(df) - n_rememvr_normal

        log(f"RAVLT classification: {n_ravlt_low} impaired, {n_ravlt_normal} normal")
        log(f"REMEMVR classification: {n_rememvr_low} impaired, {n_rememvr_normal} normal")
        # Create 2×2 Contingency Table
        log("Creating 2x2 classification matrix...")

        # Crosstab: rows = RAVLT_low, columns = REMEMVR_normal
        classification_matrix = pd.crosstab(
            index=df['RAVLT_low'],
            columns=df['REMEMVR_normal'],
            margins=True,
            margins_name='Total'
        )

        log(f"Classification matrix:")
        log(f"\n{classification_matrix}")

        # Save classification matrix
        classification_matrix.to_csv(OUTPUT_MATRIX, encoding='utf-8')
        log(f"{OUTPUT_MATRIX}")
        # Extract False Negatives
        log("Identifying false negative cases...")

        # False negatives: RAVLT impaired (low=1) AND REMEMVR normal (normal=1)
        false_negatives = df[(df['RAVLT_low'] == 1) & (df['REMEMVR_normal'] == 1)].copy()

        n_false_neg = len(false_negatives)
        log(f"{n_false_neg} false negative cases")

        if n_false_neg == 0:
            log(f"No false negatives found - check thresholds")
        elif n_false_neg < 3:
            log(f"Very few false negatives ({n_false_neg}) - low statistical power")
        elif n_false_neg > 15:
            log(f"Many false negatives ({n_false_neg}) - threshold may be too liberal")
        else:
            log(f"False negative count ({n_false_neg}) within expected range [3, 15]")

        # Report false negative UIDs
        log(f"False negative UIDs: {false_negatives['UID'].tolist()}")

        # Save false negatives with full data
        false_negatives.to_csv(OUTPUT_FALSE_NEG, index=False, encoding='utf-8')
        log(f"{OUTPUT_FALSE_NEG} ({len(false_negatives)} rows, {len(false_negatives.columns)} columns)")
        # Compute Additional Classification Metrics
        log("Computing classification cell counts...")

        # Extract cell counts from matrix
        # Matrix structure:
        #           REMEMVR_normal
        #              0         1      Total
        # RAVLT_low
        # 0         TN        FP       (RAVLT normal)
        # 1         TP        FN       (RAVLT low)
        # Total   (REMEMVR low) (REMEMVR normal)

        # True Negatives: RAVLT normal (0) AND REMEMVR low (0)
        true_negatives = classification_matrix.loc[0, 0] if (0 in classification_matrix.index and 0 in classification_matrix.columns) else 0

        # False Positives: RAVLT normal (0) AND REMEMVR normal (1)
        false_positives = classification_matrix.loc[0, 1] if (0 in classification_matrix.index and 1 in classification_matrix.columns) else 0

        # True Positives: RAVLT low (1) AND REMEMVR low (0)
        true_positives = classification_matrix.loc[1, 0] if (1 in classification_matrix.index and 0 in classification_matrix.columns) else 0

        # False Negatives: RAVLT low (1) AND REMEMVR normal (1)
        false_negatives_count = classification_matrix.loc[1, 1] if (1 in classification_matrix.index and 1 in classification_matrix.columns) else 0

        log(f"Classification cell counts:")
        log(f"  True Negatives (both normal): {true_negatives}")
        log(f"  False Positives (RAVLT normal, REMEMVR low): {false_positives}")
        log(f"  True Positives (both impaired): {true_positives}")
        log(f"  False Negatives (RAVLT low, REMEMVR normal): {false_negatives_count}")

        # Verify false negative count matches extracted cases
        if false_negatives_count != n_false_neg:
            log(f"Mismatch: matrix shows {false_negatives_count} false negatives but extracted {n_false_neg}")
            raise ValueError(f"False negative count mismatch: {false_negatives_count} vs {n_false_neg}")
        else:
            log(f"False negative count verified: {n_false_neg}")
        # Validate Outputs
        log("Running validate_dataframe_structure...")

        # Validate classification matrix structure
        matrix_validation = validate_dataframe_structure(
            df=classification_matrix,
            expected_rows=(2, 3),  # 2 rows + optional margins row
            expected_columns=['REMEMVR_normal', 'Total'] if 'Total' in classification_matrix.columns else ['REMEMVR_normal']
        )

        if matrix_validation.get('valid', False):
            log(f"Classification matrix structure validated")
        else:
            log(f"Classification matrix validation: {matrix_validation.get('message', 'Unknown issue')}")

        # Validate false negatives structure
        if n_false_neg > 0:
            fn_validation = validate_dataframe_structure(
                df=false_negatives,
                expected_rows=(3, 15),  # Expected range from 4_analysis.yaml
                expected_columns=['UID', 'REMEMVR_theta', 'RAVLT_Total', 'RAVLT_Pct_Ret',
                                'RAVLT_z', 'REMEMVR_z', 'RAVLT_low', 'REMEMVR_normal']
            )

            if fn_validation.get('valid', False):
                log(f"False negatives structure validated")
            else:
                log(f"False negatives validation: {fn_validation.get('message', 'Unknown issue')}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
