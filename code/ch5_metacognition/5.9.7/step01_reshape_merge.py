#!/usr/bin/env python3
"""reshape_merge: Reshape confidence data from wide (400 rows, 3 paradigm columns) to long format"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.7
LOG_FILE = RQ_DIR / "logs" / "step01_reshape_merge.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: reshape_merge")
        # Load Input Data

        log("Loading validated input data from Step 0...")

        # Load accuracy data
        accuracy_path = RQ_DIR / "data" / "step00_accuracy_raw.csv"
        df_accuracy = pd.read_csv(accuracy_path, encoding='utf-8')
        log(f"{accuracy_path.name} ({len(df_accuracy)} rows, {len(df_accuracy.columns)} cols)")

        # Load confidence data
        confidence_path = RQ_DIR / "data" / "step00_confidence_raw.csv"
        df_confidence = pd.read_csv(confidence_path, encoding='utf-8')
        log(f"{confidence_path.name} ({len(df_confidence)} rows, {len(df_confidence.columns)} cols)")
        # Reshape Confidence from Wide to Long
        # Transformation: 400 rows × 3 paradigm columns → 1200 rows × 1 paradigm column

        log("Converting confidence from wide to long format...")

        df_confidence_long = pd.melt(
            df_confidence,
            id_vars=['composite_ID'],
            value_vars=['theta_IFR', 'theta_ICR', 'theta_IRE'],
            var_name='paradigm_raw',
            value_name='theta_confidence'
        )

        log(f"Confidence data: {len(df_confidence)} -> {len(df_confidence_long)} rows")

        # Standardize paradigm naming to match accuracy
        paradigm_mapping = {
            'theta_IFR': 'free_recall',
            'theta_ICR': 'cued_recall',
            'theta_IRE': 'recognition'
        }
        df_confidence_long['paradigm'] = df_confidence_long['paradigm_raw'].map(paradigm_mapping)
        df_confidence_long = df_confidence_long.drop(columns=['paradigm_raw'])

        log(f"Paradigm names: {sorted(df_confidence_long['paradigm'].unique())}")
        # Standardize composite_ID Format
        # Accuracy: 'A010_1' → 'A010_T1'
        # Confidence: 'A010_T1' (already correct)

        log("Aligning composite_ID formats...")

        # Add 'T' prefix to test number in accuracy composite_IDs
        df_accuracy['composite_ID'] = df_accuracy['composite_ID'].str.replace(
            r'_(\d+)$', r'_T\1', regex=True
        )

        log(f"Accuracy composite_IDs: {df_accuracy['composite_ID'].head(3).tolist()}")
        log(f"Confidence composite_IDs: {df_confidence_long['composite_ID'].head(3).tolist()}")
        # Standardize Accuracy Paradigm Naming
        # Accuracy uses domain_name (free_recall, cued_recall, recognition)
        # Rename to 'paradigm' for consistent merge

        df_accuracy = df_accuracy.rename(columns={
            'domain_name': 'paradigm',
            'theta': 'theta_accuracy'
        })

        log(f"Accuracy paradigm column renamed")
        # Merge Accuracy and Confidence
        # Merge on: composite_ID + paradigm (inner join, expect exact 1200 rows)

        log("Merging accuracy and confidence by composite_ID + paradigm...")

        df_merged = pd.merge(
            df_accuracy,
            df_confidence_long,
            on=['composite_ID', 'paradigm'],
            how='inner'
        )

        log(f"Result: {len(df_merged)} rows, {len(df_merged.columns)} columns")

        # Validate merge completeness
        if len(df_merged) != 1200:
            log(f"Merge produced {len(df_merged)} rows, expected 1200")
            log("MERGE INCOMPLETE: Some accuracy/confidence rows did not match")
            sys.exit(1)

        # Check for missing values after merge
        missing_accuracy = df_merged['theta_accuracy'].isna().sum()
        missing_confidence = df_merged['theta_confidence'].isna().sum()
        if missing_accuracy > 0 or missing_confidence > 0:
            log(f"Missing values after merge: accuracy={missing_accuracy}, confidence={missing_confidence}")
            sys.exit(1)

        log("Merge complete with no missing values")
        # Extract UID and Test from composite_ID
        # composite_ID format: 'P001_T1' → UID='P001', test='T1'

        log("Extracting UID and test from composite_ID...")

        df_merged[['UID', 'test']] = df_merged['composite_ID'].str.split('_', expand=True)

        log(f"{len(df_merged['UID'].unique())} unique UIDs, {len(df_merged['test'].unique())} unique tests")

        # Validate each UID has 12 rows (4 tests × 3 paradigms)
        rows_per_uid = df_merged.groupby('UID').size()
        incorrect_uids = rows_per_uid[rows_per_uid != 12]
        if len(incorrect_uids) > 0:
            log(f"{len(incorrect_uids)} UIDs have incorrect row counts:")
            for uid, count in incorrect_uids.head(10).items():
                log(f"  {uid}: {count} rows (expected 12)")
            sys.exit(1)

        log("All UIDs have exactly 12 rows (4 tests × 3 paradigms)")
        # Add TSVR_hours Time Variable
        # Use nominal time mapping for now (actual TSVR may be added later)

        log("Adding TSVR_hours time variable...")

        # Nominal time mapping: T1=0, T2=24, T3=72, T4=144 hours
        time_mapping = {'T1': 0.0, 'T2': 24.0, 'T3': 72.0, 'T4': 144.0}
        df_merged['TSVR_hours'] = df_merged['test'].map(time_mapping)

        # Verify mapping succeeded
        missing_time = df_merged['TSVR_hours'].isna().sum()
        if missing_time > 0:
            log(f"{missing_time} rows have missing TSVR_hours (invalid test values)")
            sys.exit(1)

        log(f"TSVR_hours using nominal time mapping")
        log(f"  Time range: {df_merged['TSVR_hours'].min():.1f} to {df_merged['TSVR_hours'].max():.1f} hours")
        # Validate and Save Merged Data

        log("Validating merged dataset structure...")

        # Validate required columns
        validation_result = validate_data_columns(
            df_merged,
            required_columns=['composite_ID', 'UID', 'paradigm', 'theta_accuracy', 'theta_confidence']
        )

        if not validation_result.get('valid', False):
            log(f"Column validation failed: {validation_result.get('message', 'Unknown error')}")
            sys.exit(1)

        log("All required columns present")

        # Check paradigm distribution (should be balanced: 400 rows each)
        paradigm_counts = df_merged['paradigm'].value_counts()
        log("Paradigm distribution:")
        for paradigm, count in paradigm_counts.items():
            log(f"  {paradigm}: {count} rows")
            if count != 400:
                log(f"Expected 400 rows for {paradigm}, found {count}")

        # Check test distribution (should be balanced: 300 rows each)
        test_counts = df_merged['test'].value_counts()
        log("Test distribution:")
        for test, count in test_counts.items():
            log(f"  {test}: {count} rows")

        # Save merged data
        output_path = RQ_DIR / "data" / "step01_merged_calibration_data.csv"
        df_merged.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_merged)} rows, {len(df_merged.columns)} cols)")

        # Display final column structure
        log(f"Final columns: {list(df_merged.columns)}")

        log("Step 1 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
