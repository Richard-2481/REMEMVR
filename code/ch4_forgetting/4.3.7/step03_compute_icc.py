#!/usr/bin/env python3
"""Compute Intraclass Correlation Coefficients (ICC) Per Paradigm: Compute three types of ICC estimates per paradigm (Free Recall, Cued Recall, Recognition)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.7
LOG_FILE = RQ_DIR / "logs" / "step03_compute_icc.log"

# ICC interpretation thresholds (per 4_analysis.yaml specification)
ICC_THRESHOLDS = {
    'low': 0.20,
    'moderate': 0.40
}

# Day 6 timepoint for conditional ICC
# TSVR ~144 hours → log(TSVR_hours + 1) = log(145) ≈ 4.976
DAY_6_TSVR_HOURS = 144.0
LOG_TSVR_DAY6 = np.log(DAY_6_TSVR_HOURS + 1)


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# ICC Computation Functions

def interpret_icc(icc: float) -> str:
    """
    Interpret ICC value using thresholds from 4_analysis.yaml.

    Thresholds:
    - Low: ICC < 0.20
    - Moderate: 0.20 <= ICC < 0.40
    - Substantial: ICC >= 0.40
    """
    if icc < ICC_THRESHOLDS['low']:
        return "Low"
    elif icc < ICC_THRESHOLDS['moderate']:
        return "Moderate"
    else:
        return "Substantial"

def compute_icc_for_paradigm(
    paradigm: str,
    variance_components: pd.DataFrame,
    timepoint_log_tsvr: float
) -> list:
    """
    Compute 3 types of ICC for a single paradigm.

    Parameters
    ----------
    paradigm : str
        Paradigm name (free_recall, cued_recall, recognition)
    variance_components : DataFrame
        Variance components with columns ['paradigm', 'component', 'estimate']
    timepoint_log_tsvr : float
        Log-transformed TSVR value for conditional ICC (e.g., log(145) ≈ 4.976)

    Returns
    -------
    list
        List of 3 dicts, one per ICC type
    """
    # Filter to this paradigm
    paradigm_data = variance_components[variance_components['paradigm'] == paradigm]

    # Extract variance components
    components = {row['component']: row['estimate']
                  for _, row in paradigm_data.iterrows()}

    var_intercept = components.get('var_intercept', 0.0)
    var_slope = components.get('var_slope', 0.0)
    cov_int_slope = components.get('cov_int_slope', 0.0)
    var_residual = components.get('var_residual', 0.0)

    results = []

    # ICC 1: Intercept-only ICC (between-person variance in baseline)
    total_var_intercept = var_intercept + var_residual
    if total_var_intercept > 0:
        icc_intercept = var_intercept / total_var_intercept
    else:
        icc_intercept = 0.0

    results.append({
        'paradigm': paradigm,
        'icc_type': 'intercept',
        'icc_value': icc_intercept,
        'interpretation': interpret_icc(icc_intercept)
    })

    # ICC 2: Simple slope ICC (between-person variance in slopes)
    total_var_slope = var_slope + var_residual
    if total_var_slope > 0:
        icc_slope_simple = var_slope / total_var_slope
    else:
        icc_slope_simple = 0.0

    results.append({
        'paradigm': paradigm,
        'icc_type': 'slope_simple',
        'icc_value': icc_slope_simple,
        'interpretation': interpret_icc(icc_slope_simple)
    })

    # ICC 3: Conditional ICC at specific timepoint
    # Formula: [var_int + 2*cov*T + var_slope*T²] / [var_int + 2*cov*T + var_slope*T² + var_res]
    T = timepoint_log_tsvr

    var_at_time = (var_intercept +
                   2 * T * cov_int_slope +
                   (T ** 2) * var_slope)

    total_var_at_time = var_at_time + var_residual

    if total_var_at_time > 0:
        icc_conditional = var_at_time / total_var_at_time
    else:
        icc_conditional = 0.0

    # Ensure ICC is in valid [0, 1] range (negative variance estimates can occur)
    icc_conditional = max(0.0, min(1.0, icc_conditional))

    results.append({
        'paradigm': paradigm,
        'icc_type': 'slope_conditional',
        'icc_value': icc_conditional,
        'interpretation': interpret_icc(icc_conditional)
    })

    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Compute Intraclass Correlation Coefficients (ICC) Per Paradigm")
        # Load Variance Components

        log("Loading variance components from step02...")
        variance_components = pd.read_csv(RQ_DIR / "data" / "step02_variance_components.csv")
        log(f"{len(variance_components)} variance components from {variance_components['paradigm'].nunique()} paradigms")

        # Validate structure
        expected_paradigms = {'free_recall', 'cued_recall', 'recognition'}
        actual_paradigms = set(variance_components['paradigm'].unique())

        if actual_paradigms != expected_paradigms:
            raise ValueError(f"Expected paradigms {expected_paradigms}, got {actual_paradigms}")

        log(f"All 3 paradigms present: {sorted(actual_paradigms)}")
        # Compute ICC for Each Paradigm

        log(f"Computing ICC estimates (timepoint: Day 6, log_TSVR={LOG_TSVR_DAY6:.3f})...")

        all_icc_results = []

        for paradigm in sorted(expected_paradigms):
            log(f"Computing ICC for {paradigm}...")

            paradigm_iccs = compute_icc_for_paradigm(
                paradigm=paradigm,
                variance_components=variance_components,
                timepoint_log_tsvr=LOG_TSVR_DAY6
            )

            all_icc_results.extend(paradigm_iccs)

            # Log results for this paradigm
            for result in paradigm_iccs:
                log(f"  {result['icc_type']:20s}: {result['icc_value']:.3f} ({result['interpretation']})")

        log("ICC computation complete")
        # Save ICC Estimates to CSV
        # Output: data/step03_icc_estimates.csv (9 rows × 4 columns)

        log("Saving ICC estimates...")
        icc_df = pd.DataFrame(all_icc_results)

        output_csv = RQ_DIR / "data" / "step03_icc_estimates.csv"
        icc_df.to_csv(output_csv, index=False, encoding='utf-8')
        log(f"{output_csv} ({len(icc_df)} rows, {len(icc_df.columns)} cols)")
        # Generate ICC Summary Report
        # Output: data/step03_icc_summary.txt

        log("Generating ICC interpretation report...")

        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("INTRACLASS CORRELATION COEFFICIENT (ICC) SUMMARY")
        summary_lines.append("RQ 5.3.7 - Paradigm-Stratified Variance Decomposition")
        summary_lines.append("="*80)
        summary_lines.append("")
        summary_lines.append("INTERPRETATION THRESHOLDS:")
        summary_lines.append("  Low:         ICC < 0.20")
        summary_lines.append("  Moderate:    0.20 <= ICC < 0.40")
        summary_lines.append("  Substantial: ICC >= 0.40")
        summary_lines.append("")
        summary_lines.append("TIMEPOINT FOR CONDITIONAL ICC:")
        summary_lines.append(f"  Day 6 (TSVR ~{DAY_6_TSVR_HOURS:.0f} hours)")
        summary_lines.append(f"  Log-transformed: log({DAY_6_TSVR_HOURS + 1:.0f}) = {LOG_TSVR_DAY6:.3f}")
        summary_lines.append("")
        summary_lines.append("-"*80)

        for paradigm in sorted(expected_paradigms):
            paradigm_data = icc_df[icc_df['paradigm'] == paradigm]

            summary_lines.append(f"\nPARADIGM: {paradigm.upper()}")
            summary_lines.append("-"*80)

            for _, row in paradigm_data.iterrows():
                icc_type = row['icc_type']
                icc_value = row['icc_value']
                interpretation = row['interpretation']

                # Format ICC type name
                if icc_type == 'intercept':
                    type_name = "ICC Intercept (baseline memory)"
                elif icc_type == 'slope_simple':
                    type_name = "ICC Slope Simple (forgetting rate)"
                else:
                    type_name = f"ICC Conditional at Day 6 (memory at {DAY_6_TSVR_HOURS:.0f}h)"

                summary_lines.append(f"  {type_name:50s}: {icc_value:.3f} ({interpretation})")

            summary_lines.append("")

        summary_lines.append("-"*80)
        summary_lines.append("\nINTERPRETATION NOTES:")
        summary_lines.append("- ICC quantifies proportion of variance between participants")
        summary_lines.append("- Higher ICC = more individual differences = better reliability")
        summary_lines.append("- Conditional ICC at Day 6 most relevant for retention research")
        summary_lines.append("- Intercept ICC captures individual differences in baseline encoding")
        summary_lines.append("- Slope ICC captures individual differences in forgetting rates")
        summary_lines.append("")
        summary_lines.append("="*80)

        summary_text = "\n".join(summary_lines)

        output_txt = RQ_DIR / "data" / "step03_icc_summary.txt"
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        log(f"{output_txt}")
        # Run Validation
        # Validation: All ICC values in [0, 1], no NaN, correct structure

        log("Validating ICC estimates...")

        validation_errors = []

        # Check 1: Exactly 9 rows (3 paradigms × 3 ICC types)
        if len(icc_df) != 9:
            validation_errors.append(f"Expected 9 rows, got {len(icc_df)}")

        # Check 2: All ICC values in [0, 1]
        invalid_iccs = icc_df[(icc_df['icc_value'] < 0) | (icc_df['icc_value'] > 1)]
        if len(invalid_iccs) > 0:
            validation_errors.append(f"Found {len(invalid_iccs)} ICC values outside [0, 1] range")

        # Check 3: No NaN values
        nan_count = icc_df['icc_value'].isna().sum()
        if nan_count > 0:
            validation_errors.append(f"Found {nan_count} NaN ICC values")

        # Check 4: All paradigms present
        if set(icc_df['paradigm'].unique()) != expected_paradigms:
            validation_errors.append(f"Missing paradigms in ICC results")

        # Check 5: All ICC types present per paradigm
        for paradigm in expected_paradigms:
            paradigm_types = set(icc_df[icc_df['paradigm'] == paradigm]['icc_type'])
            expected_types = {'intercept', 'slope_simple', 'slope_conditional'}
            if paradigm_types != expected_types:
                validation_errors.append(f"Missing ICC types for {paradigm}: expected {expected_types}, got {paradigm_types}")

        # Check 6: Interpretation labels match thresholds
        for _, row in icc_df.iterrows():
            expected_interp = interpret_icc(row['icc_value'])
            if row['interpretation'] != expected_interp:
                validation_errors.append(
                    f"Interpretation mismatch for {row['paradigm']} {row['icc_type']}: "
                    f"expected '{expected_interp}', got '{row['interpretation']}'"
                )

        if validation_errors:
            log("FAILED with errors:")
            for error in validation_errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed: {validation_errors}")

        log("PASSED - All criteria met:")
        log(f"  Row count: {len(icc_df)} rows (3 paradigms x 3 ICC types)")
        log(f"  ICC bounds: All values in [0, 1] (range: {icc_df['icc_value'].min():.3f} to {icc_df['icc_value'].max():.3f})")
        log(f"  No NaN values: {nan_count} NaN found")
        log(f"  All paradigms present: {sorted(icc_df['paradigm'].unique())}")
        log(f"  All ICC types present per paradigm")
        log(f"  Interpretation labels match thresholds")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
