#!/usr/bin/env python3
"""Compare Model Fit (AIC, BIC): Compare AIC and BIC between IRT-based and CTT-based Linear Mixed Models to assess"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compare_lmm_fit_aic_bic

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_compare_model_fit.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Compare Model Fit (AIC, BIC)")
        # Load Model Metadata (AIC, BIC from Step 3)

        log("Loading model metadata from step03_model_metadata.yaml...")
        metadata_path = RQ_DIR / "data" / "step03_model_metadata.yaml"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        # Extract AIC and BIC for both models
        irt_aic = metadata['irt_model']['aic']
        irt_bic = metadata['irt_model']['bic']
        ctt_aic = metadata['ctt_model']['aic']
        ctt_bic = metadata['ctt_model']['bic']

        log(f"IRT model: AIC={irt_aic:.2f}, BIC={irt_bic:.2f}")
        log(f"CTT model: AIC={ctt_aic:.2f}, BIC={ctt_bic:.2f}")
        # Compare AIC/BIC using Analysis Tool
        #               applies Burnham & Anderson (2002) interpretation thresholds

        log("Running compare_lmm_fit_aic_bic...")
        tool_output = compare_lmm_fit_aic_bic(
            aic_model1=irt_aic,           # IRT model AIC (unbounded scale)
            bic_model1=irt_bic,           # IRT model BIC
            aic_model2=ctt_aic,           # CTT model AIC (bounded scale [0,1])
            bic_model2=ctt_bic,           # CTT model BIC
            model1_name='IRT',            # Label for IRT model
            model2_name='CTT'             # Label for CTT model
        )
        log("AIC/BIC comparison complete")

        # Tool returns 2 rows (AIC, BIC) with columns: metric, IRT, CTT, delta, interpretation
        # Reshape to single-row format per spec
        aic_row = tool_output[tool_output['metric'] == 'AIC'].iloc[0]
        bic_row = tool_output[tool_output['metric'] == 'BIC'].iloc[0]

        model_fit_comparison = pd.DataFrame([{
            'irt_aic': aic_row['IRT'],
            'irt_bic': bic_row['IRT'],
            'ctt_aic': aic_row['CTT'],
            'ctt_bic': bic_row['CTT'],
            'delta_aic': aic_row['delta'],
            'delta_bic': bic_row['delta'],
            'aic_interpretation': aic_row['interpretation'],
            'bic_interpretation': bic_row['interpretation']
        }])

        # Log comparison results
        delta_aic = model_fit_comparison['delta_aic'].iloc[0]
        delta_bic = model_fit_comparison['delta_bic'].iloc[0]
        log(f"Delta AIC (CTT - IRT): {delta_aic:.2f}")
        log(f"Delta BIC (CTT - IRT): {delta_bic:.2f}")
        log(f"AIC interpretation: {model_fit_comparison['aic_interpretation'].iloc[0]}")
        log(f"BIC interpretation: {model_fit_comparison['bic_interpretation'].iloc[0]}")
        # Save Comparison Output
        # Output: data/step06_model_fit_comparison.csv
        # Contains: Single row with AIC/BIC values, deltas, and interpretations
        # Columns: irt_aic, irt_bic, ctt_aic, ctt_bic, delta_aic, delta_bic,
        #          aic_interpretation, bic_interpretation

        output_path = RQ_DIR / "data" / "step06_model_fit_comparison.csv"
        log(f"Saving model fit comparison to {output_path.name}...")
        model_fit_comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(model_fit_comparison)} rows, {len(model_fit_comparison.columns)} cols)")
        # Run Validation Tool
        # Validates: Row count (1), required columns (8), structure integrity
        # Threshold: All AIC/BIC values must be finite (not NaN, not inf)

        log("Running validate_dataframe_structure...")

        expected_columns = [
            'irt_aic', 'irt_bic', 'ctt_aic', 'ctt_bic',
            'delta_aic', 'delta_bic', 'aic_interpretation', 'bic_interpretation'
        ]

        validation_result = validate_dataframe_structure(
            df=model_fit_comparison,
            expected_rows=1,                    # Single comparison row
            expected_columns=expected_columns,  # All 8 required columns
            column_types=None                   # No type checking (mixed str/float)
        )

        # Report validation results
        if validation_result['valid']:
            log("Structure validation PASSED")
            for check_name, check_result in validation_result['checks'].items():
                status = "PASS" if check_result else "FAIL"
                log(f"{check_name}: [{status}]")
        else:
            log("Structure validation FAILED")
            log(f"{validation_result['message']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation: Check all AIC/BIC values are finite
        log("Checking AIC/BIC values are finite...")
        numeric_cols = ['irt_aic', 'irt_bic', 'ctt_aic', 'ctt_bic', 'delta_aic', 'delta_bic']
        for col in numeric_cols:
            value = model_fit_comparison[col].iloc[0]
            if pd.isna(value) or not pd.api.types.is_number(value):
                raise ValueError(f"Column {col} has invalid value: {value} (expected finite number)")
            if not pd.api.types.is_float(value) and not pd.api.types.is_integer(value):
                raise ValueError(f"Column {col} has non-numeric value: {value}")
        log("All AIC/BIC values are finite")

        # Verify delta computation
        computed_delta_aic = model_fit_comparison['ctt_aic'].iloc[0] - model_fit_comparison['irt_aic'].iloc[0]
        computed_delta_bic = model_fit_comparison['ctt_bic'].iloc[0] - model_fit_comparison['irt_bic'].iloc[0]
        stored_delta_aic = model_fit_comparison['delta_aic'].iloc[0]
        stored_delta_bic = model_fit_comparison['delta_bic'].iloc[0]

        if not abs(computed_delta_aic - stored_delta_aic) < 0.01:
            raise ValueError(f"Delta AIC mismatch: computed={computed_delta_aic:.2f}, stored={stored_delta_aic:.2f}")
        if not abs(computed_delta_bic - stored_delta_bic) < 0.01:
            raise ValueError(f"Delta BIC mismatch: computed={computed_delta_bic:.2f}, stored={stored_delta_bic:.2f}")
        log("Delta computations verified (CTT - IRT)")

        log("Step 6 complete")
        log("")
        log("Model Fit Comparison Results:")
        log(f"  IRT Model: AIC={model_fit_comparison['irt_aic'].iloc[0]:.2f}, BIC={model_fit_comparison['irt_bic'].iloc[0]:.2f}")
        log(f"  CTT Model: AIC={model_fit_comparison['ctt_aic'].iloc[0]:.2f}, BIC={model_fit_comparison['ctt_bic'].iloc[0]:.2f}")
        log(f"  Delta AIC: {model_fit_comparison['delta_aic'].iloc[0]:.2f} ({model_fit_comparison['aic_interpretation'].iloc[0]})")
        log(f"  Delta BIC: {model_fit_comparison['delta_bic'].iloc[0]:.2f} ({model_fit_comparison['bic_interpretation'].iloc[0]})")
        log("")
        log("CTT has bounded outcome [0,1], IRT has unbounded scale.")
        log("Direct AIC/BIC comparison NOT about which model is 'better',")
        log("but whether models converge on similar structural conclusions.")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
