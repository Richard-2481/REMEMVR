#!/usr/bin/env python3
"""Merge Theta Scores with TSVR Time Variable: Create LMM-ready dataset by merging IRT theta scores (source/destination) with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Merge Theta Scores with TSVR Time Variable")
        # Load Input Data

        log("Loading theta scores from Pass 2...")
        # Load step03_theta_scores.csv (400 rows, wide format)
        # Expected columns: composite_ID, theta_source, theta_destination, se_source, se_destination
        theta_data = pd.read_csv(RQ_DIR / "data" / "step03_theta_scores.csv", encoding='utf-8')
        log(f"step03_theta_scores.csv ({len(theta_data)} rows, {len(theta_data.columns)} cols)")
        log(f"Columns: {list(theta_data.columns)}")

        log("Loading TSVR time mapping...")
        # Load step00_tsvr_mapping.csv (400 rows)
        # Expected columns: composite_ID, UID, test, TSVR_hours
        tsvr_data = pd.read_csv(RQ_DIR / "data" / "step00_tsvr_mapping.csv", encoding='utf-8')
        log(f"step00_tsvr_mapping.csv ({len(tsvr_data)} rows, {len(tsvr_data.columns)} cols)")
        log(f"Columns: {list(tsvr_data.columns)}")
        # Merge Theta Scores with TSVR Data

        log("Merging theta scores with TSVR mapping on composite_ID...")
        merged_data = pd.merge(
            theta_data,
            tsvr_data,
            on='composite_ID',
            how='inner'  # All 400 composite_IDs should match
        )
        log(f"Merge complete: {len(merged_data)} rows retained")

        # Validate merge completeness
        if len(merged_data) != 400:
            raise ValueError(f"Merge failed: Expected 400 rows, got {len(merged_data)}. "
                           f"Missing composite_IDs in merge.")
        log("All 400 composite_IDs matched successfully")
        # Reshape from Wide to Long Format for LocationType Factor
        #               to long format (single theta column with LocationType indicator)

        log("Converting wide format to long format for LocationType...")

        # Create two DataFrames: one for source, one for destination
        source_data = merged_data[['composite_ID', 'UID', 'test', 'TSVR_hours',
                                   'theta_source', 'se_source']].copy()
        source_data['LocationType'] = 'source'
        source_data['LocationType_coded'] = 0  # Treatment coding: source = reference (0)
        source_data.rename(columns={'theta_source': 'theta', 'se_source': 'se'}, inplace=True)

        destination_data = merged_data[['composite_ID', 'UID', 'test', 'TSVR_hours',
                                        'theta_destination', 'se_destination']].copy()
        destination_data['LocationType'] = 'destination'
        destination_data['LocationType_coded'] = 1  # Treatment coding: destination = 1
        destination_data.rename(columns={'theta_destination': 'theta', 'se_destination': 'se'}, inplace=True)

        # Concatenate to create long-format DataFrame
        lmm_input = pd.concat([source_data, destination_data], axis=0, ignore_index=True)
        log(f"Reshape complete: {len(lmm_input)} rows (400 x 2 location types)")
        log(f"LocationType value counts:")
        for loctype, count in lmm_input['LocationType'].value_counts().items():
            log(f"  {loctype}: {count} rows")
        # Create Time Transformations (Decision D070)
        # Decision D070: Use TSVR_hours (actual elapsed time) instead of nominal days

        log("Creating time transformations...")

        # Days = TSVR_hours / 24 (convert hours to days)
        lmm_input['Days'] = lmm_input['TSVR_hours'] / 24.0

        # log_Days_plus1 = log(Days + 1) (for logarithmic candidate models)
        lmm_input['log_Days_plus1'] = np.log(lmm_input['Days'] + 1)

        # Days_squared = Days^2 (for quadratic candidate models)
        lmm_input['Days_squared'] = lmm_input['Days'] ** 2

        log("Time transformations created")
        log(f"Days range: [{lmm_input['Days'].min():.2f}, {lmm_input['Days'].max():.2f}]")
        log(f"log_Days_plus1 range: [{lmm_input['log_Days_plus1'].min():.2f}, {lmm_input['log_Days_plus1'].max():.2f}]")
        log(f"Days_squared range: [{lmm_input['Days_squared'].min():.2f}, {lmm_input['Days_squared'].max():.2f}]")
        # Reorder Columns and Save LMM Input
        # Output: data/step04_lmm_input.csv (800 rows)
        # Contains: UID, test, composite_ID, TSVR_hours, Days, log_Days_plus1, Days_squared,
        #           LocationType, LocationType_coded, theta, se

        log("Reordering columns for LMM analysis...")
        lmm_input = lmm_input[['UID', 'test', 'composite_ID', 'TSVR_hours', 'Days',
                               'log_Days_plus1', 'Days_squared', 'LocationType',
                               'LocationType_coded', 'theta', 'se']]

        output_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        lmm_input.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")
        # Run Validation Checks
        # Validates: row count, composite_ID balance, LocationType balance, NaN checks

        log("Running validation checks...")

        validation_errors = []

        # Check 1: 800 total rows
        if len(lmm_input) != 800:
            validation_errors.append(f"Expected 800 rows, got {len(lmm_input)}")
        else:
            log("800 rows (400 composite_IDs x 2 location types)")

        # Check 2: Each composite_ID appears exactly twice
        composite_id_counts = lmm_input['composite_ID'].value_counts()
        if not (composite_id_counts == 2).all():
            bad_ids = composite_id_counts[composite_id_counts != 2]
            validation_errors.append(f"{len(bad_ids)} composite_IDs don't appear exactly twice: {list(bad_ids.index[:5])}")
        else:
            log("Each composite_ID appears exactly twice")

        # Check 3: LocationType balanced (400 source, 400 destination)
        loctype_counts = lmm_input['LocationType'].value_counts()
        if len(loctype_counts) != 2:
            validation_errors.append(f"LocationType should have 2 levels, got {len(loctype_counts)}")
        elif loctype_counts['source'] != 400 or loctype_counts['destination'] != 400:
            validation_errors.append(f"LocationType imbalanced: source={loctype_counts.get('source', 0)}, destination={loctype_counts.get('destination', 0)}")
        else:
            log("LocationType balanced: 400 source, 400 destination")

        # Check 4: No NaN values in critical columns
        nan_counts = lmm_input[['theta', 'TSVR_hours', 'Days', 'log_Days_plus1',
                                'Days_squared', 'LocationType']].isna().sum()
        if nan_counts.sum() > 0:
            validation_errors.append(f"NaN values detected: {nan_counts[nan_counts > 0].to_dict()}")
        else:
            log("No NaN values in critical columns")

        # Check 5: TSVR_hours in valid range [0, 360] (extended for some participants with longer intervals)
        # Note: Some participants had retention intervals up to ~10 days (246 hours)
        if lmm_input['TSVR_hours'].min() < 0 or lmm_input['TSVR_hours'].max() > 360:
            validation_errors.append(f"TSVR_hours out of range [0, 360]: [{lmm_input['TSVR_hours'].min():.2f}, {lmm_input['TSVR_hours'].max():.2f}]")
        else:
            log(f"TSVR_hours in valid range: [{lmm_input['TSVR_hours'].min():.2f}, {lmm_input['TSVR_hours'].max():.2f}]")

        # Check 6: Days in reasonable range (0 to ~15 for extended intervals)
        if lmm_input['Days'].min() < 0 or lmm_input['Days'].max() > 15:
            validation_errors.append(f"Days out of reasonable range [0, 10.5]: [{lmm_input['Days'].min():.2f}, {lmm_input['Days'].max():.2f}]")
        else:
            log(f"Days in reasonable range: [{lmm_input['Days'].min():.2f}, {lmm_input['Days'].max():.2f}]")

        # Check 7: Each UID appears exactly 8 times (4 tests x 2 location types)
        uid_counts = lmm_input['UID'].value_counts()
        if not (uid_counts == 8).all():
            bad_uids = uid_counts[uid_counts != 8]
            validation_errors.append(f"{len(bad_uids)} UIDs don't appear exactly 8 times: {list(bad_uids.index[:5])}")
        else:
            log("Each UID appears exactly 8 times (4 tests x 2 location types)")

        # Report validation results
        if validation_errors:
            log("Validation failed with following errors:")
            for error in validation_errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed: {len(validation_errors)} errors detected")
        else:
            log("All checks passed")
        # Print Descriptive Statistics

        log("Descriptive statistics:")
        log(f"  Total rows: {len(lmm_input)}")
        log(f"  Unique UIDs: {lmm_input['UID'].nunique()}")
        log(f"  Unique composite_IDs: {lmm_input['composite_ID'].nunique()}")
        log(f"  TSVR_hours: mean={lmm_input['TSVR_hours'].mean():.2f}, std={lmm_input['TSVR_hours'].std():.2f}")
        log(f"  Days: mean={lmm_input['Days'].mean():.2f}, std={lmm_input['Days'].std():.2f}")
        log(f"  Theta: mean={lmm_input['theta'].mean():.3f}, std={lmm_input['theta'].std():.3f}, range=[{lmm_input['theta'].min():.3f}, {lmm_input['theta'].max():.3f}]")
        log(f"  SE: mean={lmm_input['se'].mean():.3f}, std={lmm_input['se'].std():.3f}")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
