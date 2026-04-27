#!/usr/bin/env python3
"""
===============================================================================
RQ 5.4.3 - Step 01: Merge Data and Prepare LMM Input
===============================================================================

PURPOSE:
    Merge theta scores with TSVR and Age, reshape wide to long format
    (400 -> 1200 rows), center Age, create time transformations for LMM.

INPUTS:
    - data/step00_theta_wide.csv (400 rows, 7 columns)
    - data/step00_tsvr_mapping.csv (400 rows, 4 columns)
    - data/step00_age_data.csv (100 rows, 2 columns)

OUTPUTS:
    - data/step01_lmm_input.csv (1200 rows, 10 columns)

VALIDATION CRITERIA:
    - Exactly 1200 rows (400 x 3 congruence levels)
    - No NaN values in any column
    - congruence contains only {Common, Congruent, Incongruent}
    - Each UID appears exactly 12 times (4 tests x 3 congruence)
    - Age_c mean within +/- 0.1 of 0
    - log_TSVR computed correctly

===============================================================================
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.4.3"
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step01_prepare_lmm_input.log"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

logger = Logger(LOG_FILE)
log = logger.log

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def main():
    log("[START] Step 01: Merge Data and Prepare LMM Input")
    log("")

    # -------------------------------------------------------------------------
    # STEP 1: Load Input Files
    # -------------------------------------------------------------------------
    log("[STEP 1] Load Input Files")
    log("-" * 70)

    theta_wide = pd.read_csv(DATA_DIR / "step00_theta_wide.csv", encoding='utf-8')
    log(f"[LOADED] Theta wide: {len(theta_wide)} rows, {len(theta_wide.columns)} columns")

    tsvr_mapping = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv", encoding='utf-8')
    log(f"[LOADED] TSVR mapping: {len(tsvr_mapping)} rows, {len(tsvr_mapping.columns)} columns")

    age_data = pd.read_csv(DATA_DIR / "step00_age_data.csv", encoding='utf-8')
    log(f"[LOADED] Age data: {len(age_data)} rows, {len(age_data.columns)} columns")
    log("")

    # -------------------------------------------------------------------------
    # STEP 2: Merge Theta with TSVR Mapping
    # -------------------------------------------------------------------------
    log("[STEP 2] Merge Theta with TSVR Mapping")
    log("-" * 70)

    merged = pd.merge(theta_wide, tsvr_mapping, on='composite_ID', how='left')
    log(f"[MERGED] Theta + TSVR: {len(merged)} rows")

    # Check for merge failures
    nan_count = merged['UID'].isna().sum()
    if nan_count > 0:
        log(f"[FAIL] Merge introduced {nan_count} NaN values in UID column")
        return False
    log(f"[PASS] Merge successful, no NaN values introduced")
    log("")

    # -------------------------------------------------------------------------
    # STEP 3: Merge with Age Data
    # -------------------------------------------------------------------------
    log("[STEP 3] Merge with Age Data")
    log("-" * 70)

    merged = pd.merge(merged, age_data, on='UID', how='left')
    log(f"[MERGED] Theta + TSVR + Age: {len(merged)} rows")

    # Check for merge failures
    nan_count = merged['Age'].isna().sum()
    if nan_count > 0:
        log(f"[FAIL] Merge introduced {nan_count} NaN values in Age column")
        return False
    log(f"[PASS] Merge successful, no NaN values introduced")
    log(f"[INFO] Columns after merge: {list(merged.columns)}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 4: Reshape Wide to Long Format
    # -------------------------------------------------------------------------
    log("[STEP 4] Reshape Wide to Long Format")
    log("-" * 70)

    # Melt theta columns
    id_vars = ['composite_ID', 'UID', 'test', 'TSVR_hours', 'Age']
    theta_long = pd.melt(
        merged,
        id_vars=id_vars,
        value_vars=['theta_common', 'theta_congruent', 'theta_incongruent'],
        var_name='congruence_var',
        value_name='theta'
    )

    # Map variable names to congruence labels
    congruence_map = {
        'theta_common': 'Common',
        'theta_congruent': 'Congruent',
        'theta_incongruent': 'Incongruent'
    }
    theta_long['congruence'] = theta_long['congruence_var'].map(congruence_map)

    # Melt SE columns
    se_long = pd.melt(
        merged,
        id_vars=['composite_ID'],
        value_vars=['se_common', 'se_congruent', 'se_incongruent'],
        var_name='se_var',
        value_name='se_theta'
    )

    # Map SE variable names to congruence labels
    se_map = {
        'se_common': 'Common',
        'se_congruent': 'Congruent',
        'se_incongruent': 'Incongruent'
    }
    se_long['congruence'] = se_long['se_var'].map(se_map)

    # Merge theta and SE long dataframes
    lmm_input = pd.merge(
        theta_long[['composite_ID', 'UID', 'test', 'TSVR_hours', 'Age', 'congruence', 'theta']],
        se_long[['composite_ID', 'congruence', 'se_theta']],
        on=['composite_ID', 'congruence'],
        how='left'
    )

    log(f"[RESHAPED] Long format: {len(lmm_input)} rows")
    log(f"[INFO] Expected: 1200 rows (400 x 3 congruence levels)")

    if len(lmm_input) != 1200:
        log(f"[FAIL] Row count mismatch: expected 1200, got {len(lmm_input)}")
        return False
    log(f"[PASS] Row count correct: 1200")
    log("")

    # -------------------------------------------------------------------------
    # STEP 5: Grand-Mean Center Age
    # -------------------------------------------------------------------------
    log("[STEP 5] Grand-Mean Center Age")
    log("-" * 70)

    # Get unique UIDs to compute grand mean across participants (not observations)
    unique_ages = age_data.set_index('UID')['Age']
    age_mean = unique_ages.mean()

    log(f"[INFO] Age mean across 100 participants: {age_mean:.2f}")
    log(f"[INFO] Age SD across 100 participants: {unique_ages.std():.2f}")

    # Create Age_c (centered)
    lmm_input['Age_c'] = lmm_input['Age'] - age_mean

    # Validate centering
    age_c_mean = lmm_input.groupby('UID')['Age_c'].first().mean()
    log(f"[INFO] Age_c mean across participants: {age_c_mean:.4f}")

    if abs(age_c_mean) > 0.1:
        log(f"[FAIL] Age_c mean not approximately 0: {age_c_mean:.4f}")
        return False
    log(f"[PASS] Age_c mean within +/- 0.1 of 0: {age_c_mean:.4f}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 6: Create Time Transformations
    # -------------------------------------------------------------------------
    log("[STEP 6] Create Time Transformations")
    log("-" * 70)

    # Linear time: TSVR_hours (already present)
    log(f"[INFO] TSVR_hours range: [{lmm_input['TSVR_hours'].min():.2f}, {lmm_input['TSVR_hours'].max():.2f}]")

    # Reciprocal time: 1/(TSVR_hours + 1) - Two-process forgetting (rapid component)
    lmm_input['recip_TSVR'] = 1.0 / (lmm_input['TSVR_hours'] + 1)
    log(f"[CREATED] recip_TSVR = 1 / (TSVR_hours + 1)")
    log(f"[INFO] recip_TSVR range: [{lmm_input['recip_TSVR'].min():.4f}, {lmm_input['recip_TSVR'].max():.4f}]")

    # Logarithmic time: log(TSVR_hours + 1) - Two-process forgetting (slow component)
    lmm_input['log_TSVR'] = np.log(lmm_input['TSVR_hours'] + 1)
    log(f"[CREATED] log_TSVR = log(TSVR_hours + 1)")
    log(f"[INFO] log_TSVR range: [{lmm_input['log_TSVR'].min():.2f}, {lmm_input['log_TSVR'].max():.2f}]")
    log(f"[INFO] Two-process forgetting model: Recip+Log per RQ 5.4.1 ROOT")
    log("")

    # -------------------------------------------------------------------------
    # STEP 7: Final Validation
    # -------------------------------------------------------------------------
    log("[STEP 7] Final Validation")
    log("-" * 70)

    # Reorder columns (add recip_TSVR)
    final_columns = ['UID', 'composite_ID', 'test', 'congruence', 'theta', 'se_theta', 'Age', 'Age_c', 'TSVR_hours', 'recip_TSVR', 'log_TSVR']
    lmm_input = lmm_input[final_columns]

    # Check for NaN values
    nan_cols = lmm_input.columns[lmm_input.isna().any()].tolist()
    if nan_cols:
        log(f"[FAIL] NaN values found in columns: {nan_cols}")
        return False
    log(f"[PASS] No NaN values in any column")

    # Check congruence levels
    congruence_levels = set(lmm_input['congruence'].unique())
    expected_levels = {'Common', 'Congruent', 'Incongruent'}
    if congruence_levels != expected_levels:
        log(f"[FAIL] Unexpected congruence levels: {congruence_levels}")
        return False
    log(f"[PASS] Congruence levels correct: {sorted(congruence_levels)}")

    # Check UID counts (each should appear 12 times: 4 tests x 3 congruence)
    uid_counts = lmm_input.groupby('UID').size()
    incorrect_counts = uid_counts[uid_counts != 12]
    if len(incorrect_counts) > 0:
        log(f"[FAIL] Some UIDs don't have 12 observations:")
        log(f"  {incorrect_counts.to_dict()}")
        return False
    log(f"[PASS] Each UID appears exactly 12 times (4 tests x 3 congruence)")
    log("")

    # -------------------------------------------------------------------------
    # STEP 8: Save Output
    # -------------------------------------------------------------------------
    log("[STEP 8] Save Output")
    log("-" * 70)

    output_path = DATA_DIR / "step01_lmm_input.csv"
    lmm_input.to_csv(output_path, index=False, encoding='utf-8')
    log(f"[SAVED] {output_path}")
    log(f"  {len(lmm_input)} rows, {len(lmm_input.columns)} columns")
    log("")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    log("[SUMMARY]")
    log("-" * 70)
    log(f"Input files:")
    log(f"  Theta wide: 400 rows")
    log(f"  TSVR mapping: 400 rows")
    log(f"  Age data: 100 rows")
    log("")
    log(f"Output file:")
    log(f"  LMM input: 1200 rows x 11 columns")
    log("")
    log(f"Transformations applied:")
    log(f"  - Reshaped wide -> long format")
    log(f"  - Grand-mean centered Age (Age_c)")
    log(f"  - Created recip_TSVR = 1 / (TSVR_hours + 1)")
    log(f"  - Created log_TSVR = log(TSVR_hours + 1)")
    log(f"  - Two-process forgetting model: Recip+Log per RQ 5.4.1 ROOT")
    log("")
    log(f"Value ranges:")
    log(f"  theta: [{lmm_input['theta'].min():.2f}, {lmm_input['theta'].max():.2f}]")
    log(f"  Age: [{lmm_input['Age'].min():.0f}, {lmm_input['Age'].max():.0f}]")
    log(f"  Age_c: [{lmm_input['Age_c'].min():.2f}, {lmm_input['Age_c'].max():.2f}]")
    log(f"  TSVR_hours: [{lmm_input['TSVR_hours'].min():.2f}, {lmm_input['TSVR_hours'].max():.2f}]")
    log(f"  recip_TSVR: [{lmm_input['recip_TSVR'].min():.4f}, {lmm_input['recip_TSVR'].max():.4f}]")
    log(f"  log_TSVR: [{lmm_input['log_TSVR'].min():.2f}, {lmm_input['log_TSVR'].max():.2f}]")
    log("")
    log("[SUCCESS] Step 01 complete - LMM input prepared with Recip+Log transformations")

    return True

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        success = main()
        logger.close()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"[ERROR] Unexpected error: {e}")
        log(traceback.format_exc())
        logger.close()
        sys.exit(1)
