#!/usr/bin/env python3
"""Compare to Ch5 5.1.1 Accuracy Model Selection: Compare confidence functional form (this RQ 6.1.1) to accuracy functional form"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_compare_to_ch5.log"
CH5_RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.1.1"  # Ch5 RQ 5.1.1 directory


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Compare to Ch5 5.1.1 Accuracy Model Selection")
        # Load Confidence Model Comparison (This RQ)

        log("Loading confidence model comparison from this RQ...")
        confidence_path = RQ_DIR / "data" / "step06_aic_comparison.csv"
        df_confidence = pd.read_csv(confidence_path, encoding='utf-8')
        log(f"{confidence_path.name} ({len(df_confidence)} rows, {len(df_confidence.columns)} cols)")
        log(f"Confidence models: {df_confidence['model_name'].tolist()}")

        # Extract best confidence model
        best_confidence = df_confidence[df_confidence['is_best'] == True]['model_name'].values
        if len(best_confidence) == 1:
            log(f"Best confidence model: {best_confidence[0]}")
        else:
            log(f"Expected 1 best model, found {len(best_confidence)}")
        # Attempt to Load Accuracy Model Comparison (Ch5 5.1.1)
        # NOTE: SOFT DEPENDENCY - file may not exist yet

        log("Attempting to load accuracy model comparison from Ch5 5.1.1...")
        accuracy_path = CH5_RQ_DIR / "data" / "step06_aic_comparison.csv"

        ch5_exists = False
        df_accuracy = None

        try:
            if accuracy_path.exists():
                df_accuracy = pd.read_csv(accuracy_path, encoding='utf-8')
                ch5_exists = True
                log(f"{accuracy_path.name} ({len(df_accuracy)} rows, {len(df_accuracy.columns)} cols)")
                log(f"Accuracy models: {df_accuracy['model_name'].tolist()}")

                # Extract best accuracy model
                best_accuracy = df_accuracy[df_accuracy['is_best'] == True]['model_name'].values
                if len(best_accuracy) == 1:
                    log(f"Best accuracy model: {best_accuracy[0]}")
                else:
                    log(f"Expected 1 best model, found {len(best_accuracy)}")
            else:
                log(f"Ch5 5.1.1 file not found: {accuracy_path}")
                log(f"This is expected if Ch5 5.1.1 hasn't run yet (soft dependency)")
                log(f"Will create placeholder output with NaN accuracy values")
        except Exception as e:
            log(f"Failed to load Ch5 5.1.1 data: {str(e)}")
            log(f"Will create placeholder output with NaN accuracy values")
        # Create Comparison Table

        log("Creating comparison table...")

        # Start with confidence data
        comparison = df_confidence[['model_name', 'akaike_weight', 'is_best']].copy()
        comparison.rename(columns={
            'akaike_weight': 'confidence_weight',
            'is_best': 'best_in_confidence'
        }, inplace=True)

        if ch5_exists and df_accuracy is not None:
            # Merge with accuracy data
            accuracy_subset = df_accuracy[['model_name', 'akaike_weight', 'is_best']].copy()
            accuracy_subset.rename(columns={
                'akaike_weight': 'accuracy_weight',
                'is_best': 'best_in_accuracy'
            }, inplace=True)

            comparison = comparison.merge(accuracy_subset, on='model_name', how='left')

            # Compute weight difference
            comparison['weight_difference'] = (
                comparison['confidence_weight'] - comparison['accuracy_weight']
            )

            log(f"Merged confidence and accuracy data")
        else:
            # Create placeholder columns with NaN
            comparison['accuracy_weight'] = np.nan
            comparison['weight_difference'] = np.nan
            comparison['best_in_accuracy'] = np.nan

            log(f"Created placeholder columns (Ch5 data not available)")

        log("Comparison table created")
        # Save Comparison Output
        # Output: data/step07_ch5_comparison.csv
        # Contains: Cross-RQ model weight comparison with difference metrics
        # Downstream usage: Final results report, thesis figure generation

        output_path = RQ_DIR / "data" / "step07_ch5_comparison.csv"
        log(f"Saving {output_path.name}...")
        comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(comparison)} rows, {len(comparison.columns)} cols)")

        # Log comparison summary
        log("Model Comparison Results:")
        for _, row in comparison.iterrows():
            conf_best = "[BEST CONF]" if row['best_in_confidence'] else ""
            if ch5_exists:
                acc_best = "[BEST ACC]" if row['best_in_accuracy'] else ""
                log(f"  {row['model_name']:25s} | Conf: {row['confidence_weight']:.4f} | "
                    f"Acc: {row['accuracy_weight']:.4f} | Diff: {row['weight_difference']:+.4f} | "
                    f"{conf_best} {acc_best}")
            else:
                log(f"  {row['model_name']:25s} | Conf: {row['confidence_weight']:.4f} | "
                    f"Acc: N/A | {conf_best}")

        # Determine conclusion
        if ch5_exists:
            best_conf = comparison[comparison['best_in_confidence'] == True]['model_name'].values[0]
            best_acc = comparison[comparison['best_in_accuracy'] == True]['model_name'].values[0]

            if best_conf == best_acc:
                log(f"Confidence PARALLELS accuracy (both best: {best_conf})")
            else:
                log(f"Confidence DIVERGES from accuracy (Conf: {best_conf}, Acc: {best_acc})")
        else:
            best_conf = comparison[comparison['best_in_confidence'] == True]['model_name'].values[0]
            log(f"Confidence best model: {best_conf} (accuracy comparison pending)")
        # Run Validation Tool
        # Validates: Required columns present in output
        # Threshold: All required columns must exist

        log("Running validate_data_columns...")
        required_columns = [
            'model_name', 'confidence_weight', 'accuracy_weight',
            'weight_difference', 'best_in_confidence', 'best_in_accuracy'
        ]
        validation_result = validate_data_columns(
            df=comparison,
            required_columns=required_columns
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - All required columns present")
            log(f"Expected: {validation_result['n_required']} columns")
            log(f"Missing: {validation_result['n_missing']} columns")
        else:
            log(f"FAIL - Missing columns detected")
            log(f"Missing: {validation_result['missing_columns']}")
            raise ValueError(f"Validation failed: {validation_result}")

        # Additional validation checks
        log("Additional checks...")

        # Check row count
        if len(comparison) != 5:
            log(f"FAIL - Expected 5 rows, got {len(comparison)}")
            raise ValueError(f"Expected 5 models, got {len(comparison)}")
        log(f"PASS - Row count (5 models)")

        # Check exactly one best_in_confidence
        n_best_conf = comparison['best_in_confidence'].sum()
        if n_best_conf != 1:
            log(f"FAIL - Expected 1 best confidence model, got {n_best_conf}")
            raise ValueError(f"Expected 1 best confidence model, got {n_best_conf}")
        log(f"PASS - Exactly one best_in_confidence=True")

        # If Ch5 exists, validate accuracy data
        if ch5_exists:
            # Check exactly one best_in_accuracy
            n_best_acc = comparison['best_in_accuracy'].sum()
            if n_best_acc != 1:
                log(f"FAIL - Expected 1 best accuracy model, got {n_best_acc}")
                raise ValueError(f"Expected 1 best accuracy model, got {n_best_acc}")
            log(f"PASS - Exactly one best_in_accuracy=True")

            # Check confidence weights sum to 1.0
            conf_sum = comparison['confidence_weight'].sum()
            if not (0.99 <= conf_sum <= 1.01):
                log(f"FAIL - Confidence weights sum to {conf_sum:.4f} (expected ~1.0)")
                raise ValueError(f"Confidence weights sum to {conf_sum:.4f}")
            log(f"PASS - Confidence weights sum to {conf_sum:.4f}")

            # Check accuracy weights sum to 1.0
            acc_sum = comparison['accuracy_weight'].sum()
            if not (0.99 <= acc_sum <= 1.01):
                log(f"FAIL - Accuracy weights sum to {acc_sum:.4f} (expected ~1.0)")
                raise ValueError(f"Accuracy weights sum to {acc_sum:.4f}")
            log(f"PASS - Accuracy weights sum to {acc_sum:.4f}")
        else:
            # Ch5 missing - validate NaN values are present
            if not comparison['accuracy_weight'].isna().all():
                log(f"FAIL - Expected all NaN in accuracy_weight (Ch5 missing)")
                raise ValueError("Expected NaN accuracy values when Ch5 file missing")
            log(f"PASS - Accuracy columns contain NaN (Ch5 not available)")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
