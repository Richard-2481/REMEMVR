"""
Step 01: Prepare LMM Input Data
RQ 5.5.3 - Age Effects on Source-Destination Memory

Purpose: Merge theta with TSVR and Age, grand-mean center Age, create log_TSVR,
         reshape to long format with LocationType factor.

Input:
- data/step00_theta_from_rq551.csv (400 rows)
- data/step00_tsvr_from_rq551.csv (400 rows)
- data/step00_age_from_dfdata.csv (100 rows)

Output:
- data/step01_lmm_input.csv (800 rows x 10 columns)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / "step01_prepare_lmm_input.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Step 1: Prepare LMM Input Data")

    # -------------------------------------------------------------------------
    # 1. Load Step 0 outputs
    # -------------------------------------------------------------------------
    logger.info("Loading Step 0 outputs...")

    theta_df = pd.read_csv(DATA_DIR / "step00_theta_from_rq551.csv")
    tsvr_df = pd.read_csv(DATA_DIR / "step00_tsvr_from_rq551.csv")
    age_df = pd.read_csv(DATA_DIR / "step00_age_from_dfdata.csv")

    logger.info(f"theta: {len(theta_df)} rows, {len(theta_df.columns)} cols")
    logger.info(f"tsvr: {len(tsvr_df)} rows, {len(tsvr_df.columns)} cols")
    logger.info(f"age: {len(age_df)} rows, {len(age_df.columns)} cols")

    # -------------------------------------------------------------------------
    # 2. Merge theta with TSVR on composite_ID
    # -------------------------------------------------------------------------
    logger.info("Merging theta with TSVR on composite_ID...")

    merged = pd.merge(theta_df, tsvr_df, on='composite_ID', how='left')
    logger.info(f"theta + TSVR: {len(merged)} rows")

    if len(merged) != 400:
        raise ValueError(f"Expected 400 rows after theta-TSVR merge, got {len(merged)}")

    # -------------------------------------------------------------------------
    # 3. Merge with Age on UID
    # -------------------------------------------------------------------------
    logger.info("Merging with Age on UID...")

    merged = pd.merge(merged, age_df, on='UID', how='left')
    logger.info(f"+ Age: {len(merged)} rows")

    if len(merged) != 400:
        raise ValueError(f"Expected 400 rows after Age merge, got {len(merged)}")

    # Check for NaN in Age (indicates merge failure)
    nan_age = merged['Age'].isna().sum()
    if nan_age > 0:
        raise ValueError(f"Found {nan_age} NaN values in Age after merge - UID mismatch")

    logger.info("No data loss during merges")

    # -------------------------------------------------------------------------
    # 4. Grand-mean center Age
    # -------------------------------------------------------------------------
    logger.info("Grand-mean centering Age...")

    age_mean = merged['Age'].mean()
    merged['Age_c'] = merged['Age'] - age_mean

    age_c_mean = merged['Age_c'].mean()
    logger.info(f"Age mean: {age_mean:.2f}")
    logger.info(f"Age_c mean: {age_c_mean:.6f}")

    if abs(age_c_mean) > 0.01:
        raise ValueError(f"Age_c mean ({age_c_mean:.6f}) exceeds tolerance 0.01")

    logger.info("Age_c grand-mean centered correctly")

    # -------------------------------------------------------------------------
    # 5. Create log_TSVR
    # -------------------------------------------------------------------------
    logger.info("Creating log_TSVR...")

    merged['log_TSVR'] = np.log(merged['TSVR_hours'] + 1)
    logger.info(f"log_TSVR range: [{merged['log_TSVR'].min():.3f}, {merged['log_TSVR'].max():.3f}]")

    # -------------------------------------------------------------------------
    # 6. Reshape wide to long format
    # -------------------------------------------------------------------------
    logger.info("Converting wide to long format...")

    # Create long format by stacking Source and Destination
    rows_long = []

    for _, row in merged.iterrows():
        base_id = row['composite_ID']
        uid = row['UID']
        test = row['test']
        tsvr_hours = row['TSVR_hours']
        log_tsvr = row['log_TSVR']
        age = row['Age']
        age_c = row['Age_c']

        # Source row
        rows_long.append({
            'composite_ID': f"{base_id}_Source",
            'UID': uid,
            'test': test,
            'TSVR_hours': tsvr_hours,
            'log_TSVR': log_tsvr,
            'Age': age,
            'Age_c': age_c,
            'LocationType': 'Source',
            'theta': row['theta_source'],
            'se': row['se_source']
        })

        # Destination row
        rows_long.append({
            'composite_ID': f"{base_id}_Destination",
            'UID': uid,
            'test': test,
            'TSVR_hours': tsvr_hours,
            'log_TSVR': log_tsvr,
            'Age': age,
            'Age_c': age_c,
            'LocationType': 'Destination',
            'theta': row['theta_destination'],
            'se': row['se_destination']
        })

    lmm_input = pd.DataFrame(rows_long)
    logger.info(f"Long format: {len(lmm_input)} rows x {len(lmm_input.columns)} cols")

    # -------------------------------------------------------------------------
    # 7. Validation
    # -------------------------------------------------------------------------
    logger.info("Running validation checks...")

    validation_results = []

    # Check 1: Row count
    if len(lmm_input) == 800:
        validation_results.append(("", "Row count = 800"))
    else:
        validation_results.append(("", f"Row count = {len(lmm_input)}, expected 800"))

    # Check 2: Column count
    expected_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours', 'log_TSVR',
                     'Age', 'Age_c', 'LocationType', 'theta', 'se']
    if len(lmm_input.columns) == 10 and all(c in lmm_input.columns for c in expected_cols):
        validation_results.append(("", "All 10 columns present"))
    else:
        validation_results.append(("", f"Column mismatch: {list(lmm_input.columns)}"))

    # Check 3: LocationType balance
    loc_counts = lmm_input['LocationType'].value_counts()
    if loc_counts.get('Source', 0) == 400 and loc_counts.get('Destination', 0) == 400:
        validation_results.append(("", "LocationType balanced: 400 Source, 400 Destination"))
    else:
        validation_results.append(("", f"LocationType imbalanced: {dict(loc_counts)}"))

    # Check 4: No NaN values
    nan_counts = lmm_input.isna().sum()
    if nan_counts.sum() == 0:
        validation_results.append(("", "No NaN values"))
    else:
        validation_results.append(("", f"NaN found: {dict(nan_counts[nan_counts > 0])}"))

    # Check 5: theta range
    theta_min, theta_max = lmm_input['theta'].min(), lmm_input['theta'].max()
    if -4 <= theta_min and theta_max <= 4:
        validation_results.append(("", f"theta in [-4, 4]: [{theta_min:.3f}, {theta_max:.3f}]"))
    else:
        validation_results.append(("", f"theta out of range: [{theta_min:.3f}, {theta_max:.3f}]"))

    # Check 6: log_TSVR range
    log_min, log_max = lmm_input['log_TSVR'].min(), lmm_input['log_TSVR'].max()
    if 0 <= log_min and log_max <= 6:
        validation_results.append(("", f"log_TSVR in [0, 6]: [{log_min:.3f}, {log_max:.3f}]"))
    else:
        validation_results.append(("", f"log_TSVR out of range: [{log_min:.3f}, {log_max:.3f}]"))

    # Check 7: Age_c mean
    if abs(lmm_input['Age_c'].mean()) <= 0.01:
        validation_results.append(("", f"Age_c centered: mean = {lmm_input['Age_c'].mean():.6f}"))
    else:
        validation_results.append(("", f"Age_c not centered: mean = {lmm_input['Age_c'].mean():.6f}"))

    # Print validation results
    logger.info("Results:")
    all_pass = True
    for status, msg in validation_results:
        logger.info(f"  {status} {msg}")
        if status == "":
            all_pass = False

    if not all_pass:
        raise ValueError("Validation failed - see above for details")

    # -------------------------------------------------------------------------
    # 8. Save output
    # -------------------------------------------------------------------------
    logger.info("Saving LMM input data...")

    # Reorder columns to match specification
    lmm_input = lmm_input[expected_cols]

    output_path = DATA_DIR / "step01_lmm_input.csv"
    lmm_input.to_csv(output_path, index=False)
    logger.info(f"{output_path.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

    # Print summary statistics
    logger.info("Data summary:")
    logger.info(f"  Participants: {lmm_input['UID'].nunique()}")
    logger.info(f"  Tests: {sorted(lmm_input['test'].unique())}")
    logger.info(f"  Location types: {sorted(lmm_input['LocationType'].unique())}")
    logger.info(f"  Age range: [{lmm_input['Age'].min():.0f}, {lmm_input['Age'].max():.0f}]")
    logger.info(f"  Age_c range: [{lmm_input['Age_c'].min():.2f}, {lmm_input['Age_c'].max():.2f}]")
    logger.info(f"  TSVR range: [{lmm_input['TSVR_hours'].min():.1f}, {lmm_input['TSVR_hours'].max():.1f}] hours")
    logger.info(f"  theta range: [{lmm_input['theta'].min():.3f}, {lmm_input['theta'].max():.3f}]")

    logger.info("Step 1 complete - LMM input data prepared")


if __name__ == "__main__":
    main()
