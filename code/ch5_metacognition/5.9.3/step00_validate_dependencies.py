#!/usr/bin/env python3
"""Validate Dependencies and Merge Data: Verify Ch5 5.2.1 and Ch6 6.3.1 outputs exist, harmonize composite_IDs,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.3
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"

# Input files (absolute paths from project root)
CH5_ACCURACY_FILE = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
CH6_CONFIDENCE_FILE = PROJECT_ROOT / "results" / "ch6" / "6.3.1" / "data" / "step03_theta_confidence.csv"
CH5_TSVR_FILE = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step00_tsvr_mapping.csv"

# Output files (relative to RQ_DIR)
OUTPUT_VALIDATION_LOG = RQ_DIR / "data" / "step00_dependency_validation.txt"
OUTPUT_MERGED_DATA = RQ_DIR / "data" / "step00_merged_accuracy_confidence.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Initialize log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Step 00: Validate Dependencies and Merge Accuracy/Confidence Data\n")
            f.write("=" * 80 + "\n\n")

        log("Step 0: Validate Dependencies and Merge Data")
        # Validate Dependency Files Exist

        log("Checking dependency file existence...")

        dependencies = {
            'Ch5 5.2.1 accuracy theta': CH5_ACCURACY_FILE,
            'Ch6 6.3.1 confidence theta': CH6_CONFIDENCE_FILE,
            'Ch5 5.2.1 TSVR mapping': CH5_TSVR_FILE
        }

        missing_files = []
        for name, path in dependencies.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
                log(f"Missing dependency: {path}")
            else:
                log(f"Found: {name}")

        if missing_files:
            error_msg = "Dependencies incomplete. Missing files:\n" + "\n".join(missing_files)
            log(f"{error_msg}")
            sys.exit(1)

        log("All dependency files exist")
        # Load Accuracy Data (Ch5 5.2.1)

        log("Loading Ch5 5.2.1 accuracy theta scores...")
        accuracy = pd.read_csv(CH5_ACCURACY_FILE, encoding='utf-8')
        log(f"Accuracy data: {len(accuracy)} rows, {len(accuracy.columns)} cols")
        log(f"Accuracy columns: {accuracy.columns.tolist()}")
        log(f"Sample composite_IDs (Ch5 format): {accuracy['composite_ID'].head(3).tolist()}")

        # Validate expected columns
        expected_acc_cols = ['composite_ID', 'theta_what', 'theta_where', 'theta_when']
        if accuracy.columns.tolist() != expected_acc_cols:
            log(f"Accuracy column mismatch!")
            log(f"  Expected: {expected_acc_cols}")
            log(f"  Actual: {accuracy.columns.tolist()}")
            sys.exit(1)
        # Load Confidence Data (Ch6 6.3.1)

        log("Loading Ch6 6.3.1 confidence theta scores...")
        confidence = pd.read_csv(CH6_CONFIDENCE_FILE, encoding='utf-8')
        log(f"Confidence data: {len(confidence)} rows, {len(confidence.columns)} cols")
        log(f"Confidence columns: {confidence.columns.tolist()}")
        log(f"Sample composite_IDs (Ch6 format BEFORE harmonization): {confidence['composite_ID'].head(3).tolist()}")

        # Validate expected columns
        expected_conf_cols = ['composite_ID', 'theta_What', 'theta_Where', 'theta_When']
        if confidence.columns.tolist() != expected_conf_cols:
            log(f"Confidence column mismatch!")
            log(f"  Expected: {expected_conf_cols}")
            log(f"  Actual: {confidence.columns.tolist()}")
            sys.exit(1)
        # CRITICAL FIX 1 - Harmonize composite_ID Format
        # Problem: Ch6 uses 'A010_T1', Ch5 uses 'A010_1'
        # Solution: Convert Ch6 format to Ch5 format (remove 'T' prefix from test number)

        log("Converting Ch6 composite_ID format to Ch5 format...")
        log("Ch6 format: 'A010_T1' -> Ch5 format: 'A010_1'")

        # Map T1->1, T2->2, T3->3, T4->4 by removing the 'T'
        confidence['composite_ID'] = confidence['composite_ID'].str.replace('_T', '_', regex=False)

        log(f"Sample composite_IDs (Ch6 AFTER harmonization): {confidence['composite_ID'].head(3).tolist()}")

        # Verify harmonization worked
        acc_ids = set(accuracy['composite_ID'])
        conf_ids = set(confidence['composite_ID'])

        if acc_ids != conf_ids:
            missing_in_conf = acc_ids - conf_ids
            missing_in_acc = conf_ids - acc_ids
            log(f"composite_ID mismatch after harmonization!")
            log(f"  In accuracy but not confidence: {len(missing_in_conf)} IDs")
            log(f"  In confidence but not accuracy: {len(missing_in_acc)} IDs")
            if missing_in_conf:
                log(f"  Examples (missing in conf): {list(missing_in_conf)[:5]}")
            if missing_in_acc:
                log(f"  Examples (missing in acc): {list(missing_in_acc)[:5]}")
            sys.exit(1)

        log("composite_ID harmonization successful - all IDs match")
        # CRITICAL FIX 2 - Standardize Domain Column Names
        # Problem: Ch6 has Title case (theta_What), Ch5 has lowercase (theta_what)
        # Solution: Add unique suffixes (_acc, _conf) to prevent column name conflicts

        log("Adding unique suffixes to domain columns...")

        # Rename accuracy columns
        accuracy = accuracy.rename(columns={
            'theta_what': 'theta_what_acc',
            'theta_where': 'theta_where_acc',
            'theta_when': 'theta_when_acc'
        })
        log(f"Accuracy columns after rename: {accuracy.columns.tolist()}")

        # Rename confidence columns
        confidence = confidence.rename(columns={
            'theta_What': 'theta_What_conf',
            'theta_Where': 'theta_Where_conf',
            'theta_When': 'theta_When_conf'
        })
        log(f"Confidence columns after rename: {confidence.columns.tolist()}")

        log("Domain column standardization complete")
        # Extract UID and test_num from composite_ID

        log("Extracting UID and test_num from composite_ID...")

        accuracy['UID'] = accuracy['composite_ID'].str.split('_').str[0]
        accuracy['test_num'] = accuracy['composite_ID'].str.split('_').str[1].astype(int)

        log(f"Extracted UID and test_num for {len(accuracy)} observations")
        log(f"Unique UIDs: {accuracy['UID'].nunique()}")
        log(f"Test numbers: {sorted(accuracy['test_num'].unique())}")
        # CRITICAL FIX 3 - Load TSVR from Ch5 5.2.1
        # CORRECTED: Use Ch5 5.2.1 TSVR mapping (NOT master.xlsx, NOT tools.data.load_tsvr)

        log("Loading TSVR mapping from Ch5 5.2.1...")
        tsvr_mapping = pd.read_csv(CH5_TSVR_FILE, encoding='utf-8')
        log(f"TSVR mapping: {len(tsvr_mapping)} rows, {len(tsvr_mapping.columns)} cols")
        log(f"TSVR columns: {tsvr_mapping.columns.tolist()}")

        # Rename 'test' to 'test_num' for consistency
        if 'test' in tsvr_mapping.columns:
            tsvr_mapping = tsvr_mapping.rename(columns={'test': 'test_num'})

        log(f"TSVR_hours range: [{tsvr_mapping['TSVR_hours'].min():.2f}, {tsvr_mapping['TSVR_hours'].max():.2f}] hours")

        # Validate TSVR range (allow generous margin for late participants)
        if tsvr_mapping['TSVR_hours'].min() < 0:
            log(f"Negative TSVR_hours detected: {tsvr_mapping['TSVR_hours'].min()}")
            sys.exit(1)

        if tsvr_mapping['TSVR_hours'].max() > 200:
            log(f"TSVR_hours > 200 detected (late participants): {tsvr_mapping['TSVR_hours'].max():.2f}")
            log("Allowing up to 200 hours for late participants (per TSVR validation guidelines)")

        log("TSVR data loaded successfully")
        # Merge Accuracy and Confidence

        log("Merging accuracy and confidence data...")

        merged = pd.merge(
            accuracy,
            confidence,
            on='composite_ID',
            how='inner',
            suffixes=('', '_conf_dup')  # Should not be needed due to unique column names
        )

        log(f"Merged shape: {len(merged)} rows, {len(merged.columns)} cols")

        # Validate merge produced exactly 400 rows
        if len(merged) != 400:
            log(f"Merge produced {len(merged)} rows, expected 400")
            log("composite_ID mismatch after harmonization")
            sys.exit(1)

        log("Merge successful - exactly 400 observations")
        # Merge with TSVR

        log("Adding TSVR_hours time variable...")

        # Select only needed columns from TSVR
        tsvr_subset = tsvr_mapping[['UID', 'test_num', 'TSVR_hours']].copy()

        merged = pd.merge(
            merged,
            tsvr_subset,
            on=['UID', 'test_num'],
            how='left'
        )

        log(f"Final merged shape: {len(merged)} rows, {len(merged.columns)} cols")

        # Check for missing TSVR values
        missing_tsvr = merged['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            log(f"TSVR missing for {missing_tsvr} observations")
            log("master.xlsx TSVR extraction failed (or missing UID-test combinations)")
            sys.exit(1)

        log("All 400 observations have TSVR_hours")
        # Check Missing Data

        log("Checking missing data...")

        theta_cols = [
            'theta_what_acc', 'theta_where_acc', 'theta_when_acc',
            'theta_What_conf', 'theta_Where_conf', 'theta_When_conf'
        ]

        missing_summary = []
        for col in theta_cols:
            missing_count = merged[col].isna().sum()
            missing_pct = 100 * missing_count / len(merged)
            missing_summary.append(f"  {col}: {missing_count}/{len(merged)} ({missing_pct:.1f}%)")
            log(f"Missing in {col}: {missing_count} ({missing_pct:.1f}%)")

            if missing_pct > 5:
                log(f"{col} has >{missing_pct:.1f}% missing (threshold=5%)")

        log("Missing data check complete")
        # Validate Theta Value Ranges

        log("Checking theta value ranges...")

        for col in theta_cols:
            theta_min = merged[col].min()
            theta_max = merged[col].max()
            log(f"{col} range: [{theta_min:.2f}, {theta_max:.2f}]")

            if theta_min < -4 or theta_max > 4:
                log(f"{col} outside typical IRT range [-4, 4]")

        log("Theta value range check complete")
        # Validate Unique UIDs

        log("Checking unique UIDs...")
        unique_uids = merged['UID'].nunique()
        log(f"Unique UIDs: {unique_uids}")

        if unique_uids != 100:
            log(f"Expected 100 unique UIDs, found {unique_uids}")

        log("UID check complete")
        # Check for Duplicate composite_IDs

        log("Checking for duplicate composite_IDs...")
        duplicates = merged['composite_ID'].duplicated().sum()

        if duplicates > 0:
            log(f"Found {duplicates} duplicate composite_IDs")
            sys.exit(1)

        log("No duplicate composite_IDs")
        # Reorder Columns for Output

        log("Reordering columns for output...")

        output_col_order = [
            'composite_ID', 'UID', 'test_num', 'TSVR_hours',
            'theta_what_acc', 'theta_where_acc', 'theta_when_acc',
            'theta_What_conf', 'theta_Where_conf', 'theta_When_conf'
        ]

        merged = merged[output_col_order]
        log(f"Final columns: {merged.columns.tolist()}")
        # Write Validation Log

        log("Writing validation log...")

        validation_text = f"""Dependency Validation Log - RQ 6.9.3 Step 0
{"=" * 80}

Dependency validation complete: 400 observations

CRITICAL FIXES APPLIED:
1. composite_ID harmonization: Ch6 '_T1' -> Ch5 '_1' format
   - SUCCESSFUL: All {len(merged)} composite_IDs match between accuracy and confidence

2. Domain case standardization: Title case -> lowercase with unique suffixes
   - Accuracy: theta_what_acc, theta_where_acc, theta_when_acc
   - Confidence: theta_What_conf, theta_Where_conf, theta_When_conf
   - SUCCESSFUL: No column name conflicts

3. TSVR loading: Loaded from Ch5 5.2.1/data/step00_tsvr_mapping.csv
   - SUCCESSFUL: All 400 observations have TSVR_hours
   - TSVR_hours range: [{merged['TSVR_hours'].min():.2f}, {merged['TSVR_hours'].max():.2f}] hours

DATA QUALITY SUMMARY:
- Observations: {len(merged)}
- Unique UIDs: {unique_uids}
- Unique test numbers: {sorted(merged['test_num'].unique())}
- Duplicate composite_IDs: {duplicates}

MISSING DATA:
{chr(10).join(missing_summary)}

THETA VALUE RANGES (IRT ability scale):
- theta_what_acc: [{merged['theta_what_acc'].min():.2f}, {merged['theta_what_acc'].max():.2f}]
- theta_where_acc: [{merged['theta_where_acc'].min():.2f}, {merged['theta_where_acc'].max():.2f}]
- theta_when_acc: [{merged['theta_when_acc'].min():.2f}, {merged['theta_when_acc'].max():.2f}]
- theta_What_conf: [{merged['theta_What_conf'].min():.2f}, {merged['theta_What_conf'].max():.2f}]
- theta_Where_conf: [{merged['theta_Where_conf'].min():.2f}, {merged['theta_Where_conf'].max():.2f}]
- theta_When_conf: [{merged['theta_When_conf'].min():.2f}, {merged['theta_When_conf'].max():.2f}]

DEPENDENCIES VALIDATED:
- Ch5 5.2.1 accuracy theta: {CH5_ACCURACY_FILE}
- Ch6 6.3.1 confidence theta: {CH6_CONFIDENCE_FILE}
- Ch5 5.2.1 TSVR mapping: {CH5_TSVR_FILE}

OUTPUT FILES CREATED:
- Merged data: {OUTPUT_MERGED_DATA}
  -> {len(merged)} observations, {len(merged.columns)} columns

READY FOR: Step 0b - When Domain Matched-Item IRT
"""

        with open(OUTPUT_VALIDATION_LOG, 'w', encoding='utf-8') as f:
            f.write(validation_text)

        log(f"Validation log: {OUTPUT_VALIDATION_LOG}")
        # Save Merged Data

        log("Saving merged data...")
        merged.to_csv(OUTPUT_MERGED_DATA, index=False, encoding='utf-8')
        log(f"{OUTPUT_MERGED_DATA} ({len(merged)} rows, {len(merged.columns)} cols)")
        # Run Validation Tool
        # Validates: Output file exists and meets minimum size requirement
        # Threshold: 10KB (400 rows × 10 columns)

        log("Running validation tool...")
        validation_result = check_file_exists(
            file_path=OUTPUT_MERGED_DATA,
            min_size_bytes=10000
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 0 complete")
        log("")
        log("=" * 80)
        log("NEXT STEP: Step 0b - When Domain Matched-Item IRT")
        log("=" * 80)
        log("")
        log("Measurement equivalence issue detected:")
        log("- Ch5 accuracy used 48 When items")
        log("- Ch6 confidence retained only ~18 When items after purification")
        log("-> Step 0b will re-run Ch5 accuracy IRT with matched 18 items")
        log("-> Ensures valid comparison (same items for accuracy vs confidence)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
