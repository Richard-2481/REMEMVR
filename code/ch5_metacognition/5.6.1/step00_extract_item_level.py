#!/usr/bin/env python3
"""Extract Item-Level Confidence-Accuracy Data: Extract paired TC_* (confidence) and TQ_* (accuracy) items from dfData.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_item_level.log"


# Paradigm filtering configuration
PARADIGMS_INCLUDE = ["IFR", "ICR", "IRE"]  # Interactive paradigms with confidence data
PARADIGMS_EXCLUDE = ["RFR", "TCR", "RRE"]  # Exclude: RFR (no confidence), TCR/RRE (text-based)

# Column prefixes
CONFIDENCE_PREFIX = "TC_"
ACCURACY_PREFIX = "TQ_"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract Item-Level Confidence-Accuracy Data")
        # Load Input Data

        log("Loading dfData.csv from data/cache/...")
        input_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_data = pd.read_csv(input_path)
        log(f"dfData.csv ({len(df_data)} rows, {len(df_data.columns)} cols)")

        # Check metadata columns present
        required_meta = ["UID", "TEST", "TSVR"]
        missing_meta = [col for col in required_meta if col not in df_data.columns]
        if missing_meta:
            raise ValueError(f"Missing required metadata columns: {missing_meta}")
        log(f"Metadata columns present: {required_meta}")
        # Extract Confidence and Accuracy Columns

        log("Identifying confidence (TC_*) and accuracy (TQ_*) columns...")

        # Find all TC_* and TQ_* columns
        conf_cols = [col for col in df_data.columns if col.startswith(CONFIDENCE_PREFIX)]
        acc_cols = [col for col in df_data.columns if col.startswith(ACCURACY_PREFIX)]

        log(f"{len(conf_cols)} confidence columns (TC_*)")
        log(f"{len(acc_cols)} accuracy columns (TQ_*)")

        # Extract paradigm from column names
        # Column format: TC_PARADIGM-DOMAIN-ITEM (e.g., TC_IFR-N-i1)
        # Need to extract PARADIGM part (between prefix and first hyphen)

        def extract_paradigm(col_name):
            """Extract paradigm from column name (e.g., TC_IFR-N-i1 -> IFR)

            Column format: PREFIX_PARADIGM-DOMAIN-ITEM
            - Remove prefix (TC_ or TQ_)
            - Split by hyphen
            - First part is paradigm code
            """
            # Remove prefix (TC_ or TQ_)
            if col_name.startswith(CONFIDENCE_PREFIX):
                suffix = col_name[len(CONFIDENCE_PREFIX):]
            elif col_name.startswith(ACCURACY_PREFIX):
                suffix = col_name[len(ACCURACY_PREFIX):]
            else:
                return None

            # Split by hyphen and take first part (paradigm code)
            parts = suffix.split("-")
            if len(parts) >= 1:
                return parts[0]  # First part after prefix is paradigm (IFR, ICR, IRE, RFR, etc.)
            return None

        # Filter confidence columns to interactive paradigms
        conf_cols_filtered = []
        for col in conf_cols:
            paradigm = extract_paradigm(col)
            if paradigm in PARADIGMS_INCLUDE:
                conf_cols_filtered.append(col)

        # Filter accuracy columns to interactive paradigms
        acc_cols_filtered = []
        for col in acc_cols:
            paradigm = extract_paradigm(col)
            if paradigm in PARADIGMS_INCLUDE:
                acc_cols_filtered.append(col)

        log(f"{len(conf_cols_filtered)} confidence columns after paradigm filter (IFR, ICR, IRE)")
        log(f"{len(acc_cols_filtered)} accuracy columns after paradigm filter (IFR, ICR, IRE)")
        log(f"Included: {PARADIGMS_INCLUDE}")
        log(f"Excluded: {PARADIGMS_EXCLUDE}")
        # Reshape to Long Format

        log("Converting to long format (one row per item-response)...")

        # Melt confidence columns
        df_conf_long = df_data[required_meta + conf_cols_filtered].melt(
            id_vars=required_meta,
            value_vars=conf_cols_filtered,
            var_name='item_code_conf',
            value_name='confidence'
        )

        # Extract item code suffix (e.g., TC_IFR-N-i1 -> IFR-N-i1)
        df_conf_long['item_code'] = df_conf_long['item_code_conf'].str[len(CONFIDENCE_PREFIX):]
        df_conf_long = df_conf_long.drop(columns=['item_code_conf'])

        log(f"Confidence data: {len(df_conf_long)} rows")

        # Melt accuracy columns
        df_acc_long = df_data[required_meta + acc_cols_filtered].melt(
            id_vars=required_meta,
            value_vars=acc_cols_filtered,
            var_name='item_code_acc',
            value_name='accuracy'
        )

        # Extract item code suffix (e.g., TQ_IFR-N-i1 -> IFR-N-i1)
        df_acc_long['item_code'] = df_acc_long['item_code_acc'].str[len(ACCURACY_PREFIX):]
        df_acc_long = df_acc_long.drop(columns=['item_code_acc'])

        log(f"Accuracy data: {len(df_acc_long)} rows")
        # Match Confidence-Accuracy Pairs

        log("Matching confidence-accuracy pairs by item_code...")

        df_item_level = pd.merge(
            df_conf_long,
            df_acc_long[['UID', 'TEST', 'item_code', 'accuracy']],
            on=['UID', 'TEST', 'item_code'],
            how='inner'  # Only keep items with BOTH confidence and accuracy
        )

        log(f"{len(df_item_level)} item-response pairs created")

        # Reorder columns for clarity
        df_item_level = df_item_level[['UID', 'TEST', 'TSVR', 'item_code', 'confidence', 'accuracy']]

        # Report basic statistics
        n_participants = df_item_level['UID'].nunique()
        n_tests = df_item_level['TEST'].nunique()
        n_items = df_item_level['item_code'].nunique()

        log(f"Extracted {len(df_item_level)} item-responses from {n_participants} participants")
        log(f"{n_tests} test sessions, {n_items} unique items")
        log(f"Paradigms included: {', '.join(PARADIGMS_INCLUDE)}")
        # Save Analysis Output
        # Output: data/step00_item_level.csv
        # Contains: Item-level confidence-accuracy pairs for HCE computation

        output_path = RQ_DIR / "data" / "step00_item_level.csv"
        log(f"Saving {output_path}...")

        df_item_level.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_item_level)} rows, {len(df_item_level.columns)} cols)")
        # Run Validation Tool
        # Validates: Column presence (first pass - basic structure check)
        # Criteria: All required columns present

        log("Running validate_data_format...")

        required_cols = ['UID', 'TEST', 'TSVR', 'item_code', 'confidence', 'accuracy']
        validation_result = validate_data_format(df=df_item_level, required_cols=required_cols)

        if not validation_result['valid']:
            raise ValueError(f"VALIDATION FAILED: {validation_result['message']}")

        log(f"PASS - Column check: {validation_result['message']}")
        # Additional Validation Checks (Per 4_analysis.yaml Criteria)
        # Check value ranges and data quality as specified in validation criteria

        log("Checking value ranges and data quality...")

        # Check confidence values (must be 5-point Likert scale: 0.2, 0.4, 0.6, 0.8, 1.0)
        valid_confidence = {0.2, 0.4, 0.6, 0.8, 1.0}
        conf_non_null = df_item_level['confidence'].dropna()
        invalid_conf = conf_non_null[~conf_non_null.isin(valid_confidence)]

        if len(invalid_conf) > 0:
            unique_invalid = invalid_conf.unique()
            raise ValueError(
                f"VALIDATION FAILED: Found {len(invalid_conf)} confidence values not in "
                f"{{0.2, 0.4, 0.6, 0.8, 1.0}}. Invalid values: {unique_invalid}"
            )
        log("PASS - confidence values in {0.2, 0.4, 0.6, 0.8, 1.0}")

        # Check accuracy values (partial credit scoring: 0, 0.25, 0.5, 1.0)
        valid_accuracy = {0.0, 0.25, 0.5, 1.0}
        acc_non_null = df_item_level['accuracy'].dropna()
        invalid_acc = acc_non_null[~acc_non_null.isin(valid_accuracy)]

        if len(invalid_acc) > 0:
            unique_invalid = invalid_acc.unique()
            raise ValueError(
                f"VALIDATION FAILED: Found {len(invalid_acc)} accuracy values not in {{0, 0.25, 0.5, 1.0}}. "
                f"Invalid values: {unique_invalid}"
            )
        log("PASS - accuracy values in {0, 0.25, 0.5, 1.0}")

        # Check TSVR range (allow up to 300 hours for realistic retention intervals)
        tsvr_min = df_item_level['TSVR'].min()
        tsvr_max = df_item_level['TSVR'].max()

        if tsvr_min < 0 or tsvr_max > 300:
            raise ValueError(
                f"VALIDATION FAILED: TSVR out of range [0, 300]. "
                f"Found: min={tsvr_min:.2f}, max={tsvr_max:.2f}"
            )
        log(f"PASS - TSVR in reasonable range (min={tsvr_min:.2f}, max={tsvr_max:.2f} hours)")

        # Check expected row count
        expected_min = 26000
        expected_max = 28000
        actual_rows = len(df_item_level)

        if actual_rows < expected_min or actual_rows > expected_max:
            log(f"WARNING - Expected rows: {expected_min}-{expected_max}, found: {actual_rows}")
            log(f"This may be acceptable if some items have missing data")
        else:
            log(f"PASS - Row count within expected range: {actual_rows}")

        # Check missing data tolerance (<5% NaN in confidence/accuracy)
        conf_missing_pct = (df_item_level['confidence'].isna().sum() / len(df_item_level)) * 100
        acc_missing_pct = (df_item_level['accuracy'].isna().sum() / len(df_item_level)) * 100

        log(f"Missing data: confidence={conf_missing_pct:.2f}%, accuracy={acc_missing_pct:.2f}%")

        if conf_missing_pct > 5.0:
            raise ValueError(
                f"VALIDATION FAILED: Confidence missing data exceeds 5% tolerance ({conf_missing_pct:.2f}%)"
            )
        if acc_missing_pct > 5.0:
            raise ValueError(
                f"VALIDATION FAILED: Accuracy missing data exceeds 5% tolerance ({acc_missing_pct:.2f}%)"
            )
        log("PASS - Missing data < 5% tolerance")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
