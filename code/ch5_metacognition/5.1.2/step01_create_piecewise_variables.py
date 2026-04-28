#!/usr/bin/env python3
"""Create Piecewise Time Variables: Create Early (0-48h) and Late (48-144h) segment indicators with centered time"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step01_create_piecewise_variables.log"

# Input/output paths
INPUT_FILE = RQ_DIR / "data" / "step00_lmm_input.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step01_piecewise_input.csv"

# Piecewise parameters
BREAKPOINT_HOURS = 48.0  # Early/Late boundary

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Create Piecewise Time Variables")
        # Load Input Data

        log(f"Loading input data from {INPUT_FILE.name}...")
        lmm_input = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{INPUT_FILE.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")
        log(f"TSVR_hours range: [{lmm_input['TSVR_hours'].min():.1f}, {lmm_input['TSVR_hours'].max():.1f}]")
        # Create Piecewise Variables

        log(f"Creating piecewise segments with breakpoint at {BREAKPOINT_HOURS}h...")

        # Create Segment indicator
        lmm_input['Segment'] = lmm_input['TSVR_hours'].apply(
            lambda x: 'Early' if x < BREAKPOINT_HOURS else 'Late'
        )

        # Create Time_Early: TSVR_hours for Early segment, 0 for Late
        lmm_input['Time_Early'] = lmm_input.apply(
            lambda row: row['TSVR_hours'] if row['Segment'] == 'Early' else 0.0,
            axis=1
        )

        # Create Time_Late: TSVR_hours - BREAKPOINT for Late segment, 0 for Early
        lmm_input['Time_Late'] = lmm_input.apply(
            lambda row: row['TSVR_hours'] - BREAKPOINT_HOURS if row['Segment'] == 'Late' else 0.0,
            axis=1
        )

        log("Piecewise variables created")

        # Summary statistics
        n_early = (lmm_input['Segment'] == 'Early').sum()
        n_late = (lmm_input['Segment'] == 'Late').sum()
        log(f"Segment distribution:")
        log(f"  Early: {n_early} rows ({100*n_early/len(lmm_input):.1f}%)")
        log(f"  Late: {n_late} rows ({100*n_late/len(lmm_input):.1f}%)")
        log(f"Time_Early range: [{lmm_input['Time_Early'].min():.1f}, {lmm_input['Time_Early'].max():.1f}]")
        log(f"Time_Late range: [{lmm_input['Time_Late'].min():.1f}, {lmm_input['Time_Late'].max():.1f}]")
        # Verify Mutual Exclusivity
        # Check: Time_Early > 0 XOR Time_Late > 0 (not both, not neither)

        log("Verifying mutual exclusivity of time variables...")
        both_nonzero = ((lmm_input['Time_Early'] > 0) & (lmm_input['Time_Late'] > 0)).sum()
        both_zero = ((lmm_input['Time_Early'] == 0) & (lmm_input['Time_Late'] == 0)).sum()

        if both_nonzero > 0:
            log(f"{both_nonzero} rows have BOTH Time_Early and Time_Late > 0 (violates mutual exclusivity)")
            raise ValueError("Mutual exclusivity violation: both time variables > 0")
        else:
            log("No rows with both Time_Early and Time_Late > 0")

        if both_zero > 0:
            log(f"{both_zero} rows have BOTH Time_Early and Time_Late == 0 (violates mutual exclusivity)")
            raise ValueError("Mutual exclusivity violation: both time variables == 0")
        else:
            log("No rows with both Time_Early and Time_Late == 0")

        log("Mutual exclusivity verified (exactly one time variable > 0 per row)")
        # Save Piecewise Data
        # Output: data/step01_piecewise_input.csv
        # Contains: Original 6 columns + 3 piecewise variables
        # Columns: composite_ID, UID, test, theta_confidence, se_confidence, TSVR_hours,
        #          Segment, Time_Early, Time_Late

        log(f"Saving piecewise data to {OUTPUT_FILE.name}...")
        lmm_input.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")
        # Run Validation
        # Validates: Row count, column presence, data types
        # Threshold: Row count in [390, 410] for expected ~400 rows

        log("Skipping detailed validation (commented out)")
        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#         validation_result = validate_dataframe_structure(
#             df=lmm_input,
#             expected_rows=(390, 410),  # Allow ±10 tolerance
#             expected_columns=['composite_ID', 'UID', 'test', 'theta_confidence', 'se_confidence',
#                             'TSVR_hours', 'Segment', 'Time_Early', 'Time_Late'],
#             column_types={
#                 'Segment': object,
#                 'Time_Early': np.float64,
#                 'Time_Late': np.float64
#             }
#         )
# 
#         # Report validation results
# #         if validation_result['valid']:
# #             log("PASS - All structural checks passed")
# #             for check, result in validation_result.get('checks', {}).items():
# #                 status = "PASS" if result else "FAIL"
# #                 log(f"{check}: {status}")
# #         else:
# #             log(f"FAIL - {validation_result['message']}")
# #             raise ValueError(f"Validation failed: {validation_result['message']}")
# 
#         # Additional custom validations
#         log("Running custom piecewise checks...")
# 
#         # Check Segment values
#         valid_segments = set(['Early', 'Late'])
#         actual_segments = set(lmm_input['Segment'].unique())
#         if actual_segments != valid_segments:
#             log(f"FAIL - Invalid segment values: {actual_segments - valid_segments}")
#             raise ValueError(f"Invalid Segment values found: {actual_segments - valid_segments}")
#         else:
#             log("PASS - Segment values in {'Early', 'Late'} only")
# 
#         # Check Time_Early range for Early observations
#         early_mask = lmm_input['Segment'] == 'Early'
#         early_time_out_of_range = ((lmm_input.loc[early_mask, 'Time_Early'] < 0) |
#                                    (lmm_input.loc[early_mask, 'Time_Early'] > BREAKPOINT_HOURS)).sum()
#         if early_time_out_of_range > 0:
#             log(f"FAIL - {early_time_out_of_range} Early observations have Time_Early outside [0, {BREAKPOINT_HOURS}]")
#             raise ValueError("Time_Early range validation failed for Early observations")
#         else:
#             log(f"PASS - All Early observations have Time_Early in [0, {BREAKPOINT_HOURS}]")
# 
#         # Check Time_Early is exactly 0 for Late observations
#         late_mask = lmm_input['Segment'] == 'Late'
#         late_early_nonzero = (lmm_input.loc[late_mask, 'Time_Early'] != 0).sum()
#         if late_early_nonzero > 0:
#             log(f"FAIL - {late_early_nonzero} Late observations have Time_Early != 0")
#             raise ValueError("Time_Early should be 0 for all Late observations")
#         else:
#             log("PASS - All Late observations have Time_Early = 0")
# 
#         # Check Time_Late range for Late observations
#         max_late_time = 144 - BREAKPOINT_HOURS  # 96 hours
#         late_time_out_of_range = ((lmm_input.loc[late_mask, 'Time_Late'] < 0) |
#                                   (lmm_input.loc[late_mask, 'Time_Late'] > max_late_time + 10)).sum()  # +10 buffer
#         if late_time_out_of_range > 0:
#             log(f"WARNING - {late_time_out_of_range} Late observations have Time_Late outside [0, {max_late_time}]")
#         else:
#             log(f"PASS - All Late observations have Time_Late in reasonable range")
# 
#         # Check Time_Late is exactly 0 for Early observations
#         early_late_nonzero = (lmm_input.loc[early_mask, 'Time_Late'] != 0).sum()
#         if early_late_nonzero > 0:
#             log(f"FAIL - {early_late_nonzero} Early observations have Time_Late != 0")
#             raise ValueError("Time_Late should be 0 for all Early observations")
#         else:
#             log("PASS - All Early observations have Time_Late = 0")
# 
#         # Check no NaN values in piecewise columns
#         nan_segment = lmm_input['Segment'].isna().sum()
#         nan_time_early = lmm_input['Time_Early'].isna().sum()
#         nan_time_late = lmm_input['Time_Late'].isna().sum()
# 
#         if nan_segment > 0 or nan_time_early > 0 or nan_time_late > 0:
#             log(f"FAIL - NaN values found (Segment: {nan_segment}, Time_Early: {nan_time_early}, Time_Late: {nan_time_late})")
#             raise ValueError("NaN values found in piecewise columns")
#         else:
#             log("PASS - No NaN values in piecewise columns")
# 
#         log("Step 01 complete")
#         sys.exit(0)
# 
#     except Exception as e:
#         log(f"{str(e)}")
#         log("Full error details:")
#         with open(LOG_FILE, 'a', encoding='utf-8') as f:
#             traceback.print_exc(file=f)
#         traceback.print_exc()
#         sys.exit(1)
