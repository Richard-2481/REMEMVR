#!/usr/bin/env python3
"""compute_clinical_metrics: Calculate diagnostic performance metrics (sensitivity, specificity, PPV, NPV)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step06_compute_clinical_metrics.log"
INPUT_MATRIX = RQ_DIR / "data" / "step03_classification_matrix.csv"
INPUT_ALL = RQ_DIR / "data" / "step02_standardized_scores.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step06_clinical_metrics.csv"

CONFIDENCE_LEVEL = 0.95

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Wilson CI Function

def wilson_ci(successes, trials, confidence=0.95):
    """
    Compute Wilson confidence interval for a proportion.

    More accurate than Wald (normal approximation) for small sample sizes
    or proportions near 0 or 1.

    Parameters:
    - successes: int, number of successes
    - trials: int, total number of trials
    - confidence: float, confidence level (e.g., 0.95)

    Returns:
    - ci_lower: float, lower bound
    - ci_upper: float, upper bound
    """
    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    z = 1.96 if confidence == 0.95 else 1.645  # Simplified for common values

    denominator = 1 + (z**2 / trials)
    center = (p + (z**2 / (2 * trials))) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / trials) + (z**2 / (4 * trials**2)))

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    return ci_lower, ci_upper

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: compute_clinical_metrics")
        # Load Classification Matrix
        log("Loading classification matrix from step03...")

        df_matrix = pd.read_csv(INPUT_MATRIX, index_col=0)
        log(f"Classification matrix:")
        log(f"\n{df_matrix}")
        # Extract Cell Counts
        log("Extracting 2x2 cell counts...")

        # Matrix structure (ACTUAL):
        #           Column: REMEMVR (0=impaired, 1=normal)
        #                0         1
        # Row: RAVLT
        # 0         19        65       (RAVLT normal)
        # 1          9         7       (RAVLT low)

        # For RQ 7.7.4 "False Negatives" framing:
        # - False Negative (FN): RAVLT low BUT REMEMVR normal → matrix[1, 1] = 7
        # - True Positive (both impaired): RAVLT low AND REMEMVR low → matrix[1, 0] = 9
        # - True Negative (both normal): RAVLT normal AND REMEMVR normal → matrix[0, 1] = 65
        # - False Positive: RAVLT normal BUT REMEMVR low → matrix[0, 0] = 19

        # Extract cell counts using iloc (position-based, robust to index types)
        # Matrix rows: [RAVLT_low=0, RAVLT_low=1, Total] → Use rows 0 and 1
        # Matrix cols: [0, 1, Total] → Use cols 0 and 1

        # Remove 'Total' row/column if present
        df_data = df_matrix.iloc[:2, :2]  # First 2 rows, first 2 columns

        false_negatives = int(df_data.iloc[1, 1])  # Row 1 (RAVLT low), Col 1 (REMEMVR normal)
        true_positives = int(df_data.iloc[1, 0])   # Row 1 (RAVLT low), Col 0 (REMEMVR low)
        true_negatives = int(df_data.iloc[0, 1])   # Row 0 (RAVLT normal), Col 1 (REMEMVR normal)
        false_positives = int(df_data.iloc[0, 0])  # Row 0 (RAVLT normal), Col 0 (REMEMVR low)

        log(f"Cell counts:")
        log(f"  True Positives (both impaired): {true_positives}")
        log(f"  False Negatives (RAVLT low, REMEMVR normal): {false_negatives}")
        log(f"  True Negatives (both normal): {true_negatives}")
        log(f"  False Positives (RAVLT normal, REMEMVR low): {false_positives}")

        total = true_positives + false_negatives + true_negatives + false_positives
        log(f"Total participants: {total}")
        # Compute Diagnostic Metrics
        log("Computing diagnostic performance metrics...")

        metrics_list = []

        # Sensitivity: TP / (TP + FN) = proportion of RAVLT-impaired correctly identified
        sensitivity_denom = true_positives + false_negatives
        if sensitivity_denom > 0:
            sensitivity = true_positives / sensitivity_denom
            sens_ci_lower, sens_ci_upper = wilson_ci(true_positives, sensitivity_denom, CONFIDENCE_LEVEL)
        else:
            sensitivity = np.nan
            sens_ci_lower, sens_ci_upper = np.nan, np.nan

        log(f"Sensitivity: {sensitivity:.3f} (95% CI: [{sens_ci_lower:.3f}, {sens_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'Sensitivity',
            'Value': sensitivity,
            'CI_Lower': sens_ci_lower,
            'CI_Upper': sens_ci_upper,
            'Interpretation': 'Proportion of RAVLT-impaired correctly identified by REMEMVR'
        })

        # Specificity: TN / (TN + FP) = proportion of RAVLT-normal correctly identified
        specificity_denom = true_negatives + false_positives
        if specificity_denom > 0:
            specificity = true_negatives / specificity_denom
            spec_ci_lower, spec_ci_upper = wilson_ci(true_negatives, specificity_denom, CONFIDENCE_LEVEL)
        else:
            specificity = np.nan
            spec_ci_lower, spec_ci_upper = np.nan, np.nan

        log(f"Specificity: {specificity:.3f} (95% CI: [{spec_ci_lower:.3f}, {spec_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'Specificity',
            'Value': specificity,
            'CI_Lower': spec_ci_lower,
            'CI_Upper': spec_ci_upper,
            'Interpretation': 'Proportion of RAVLT-normal correctly identified by REMEMVR'
        })

        # Positive Predictive Value (PPV): TP / (TP + FP) = if REMEMVR low, prob RAVLT low
        ppv_denom = true_positives + false_positives
        if ppv_denom > 0:
            ppv = true_positives / ppv_denom
            ppv_ci_lower, ppv_ci_upper = wilson_ci(true_positives, ppv_denom, CONFIDENCE_LEVEL)
        else:
            ppv = np.nan
            ppv_ci_lower, ppv_ci_upper = np.nan, np.nan

        log(f"PPV: {ppv:.3f} (95% CI: [{ppv_ci_lower:.3f}, {ppv_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'PPV',
            'Value': ppv,
            'CI_Lower': ppv_ci_lower,
            'CI_Upper': ppv_ci_upper,
            'Interpretation': 'If REMEMVR low, probability RAVLT also impaired'
        })

        # Negative Predictive Value (NPV): TN / (TN + FN) = if REMEMVR normal, prob RAVLT normal
        npv_denom = true_negatives + false_negatives
        if npv_denom > 0:
            npv = true_negatives / npv_denom
            npv_ci_lower, npv_ci_upper = wilson_ci(true_negatives, npv_denom, CONFIDENCE_LEVEL)
        else:
            npv = np.nan
            npv_ci_lower, npv_ci_upper = np.nan, np.nan

        log(f"NPV: {npv:.3f} (95% CI: [{npv_ci_lower:.3f}, {npv_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'NPV',
            'Value': npv,
            'CI_Lower': npv_ci_lower,
            'CI_Upper': npv_ci_upper,
            'Interpretation': 'If REMEMVR normal, probability RAVLT also normal'
        })
        # Compute Base Rates
        log("Computing base rates...")

        # RAVLT impairment rate (prevalence)
        ravlt_impairment_rate = (true_positives + false_negatives) / total
        ravlt_ci_lower, ravlt_ci_upper = wilson_ci(true_positives + false_negatives, total, CONFIDENCE_LEVEL)

        log(f"RAVLT impairment rate: {ravlt_impairment_rate:.3f} (95% CI: [{ravlt_ci_lower:.3f}, {ravlt_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'RAVLT_Impairment_Rate',
            'Value': ravlt_impairment_rate,
            'CI_Lower': ravlt_ci_lower,
            'CI_Upper': ravlt_ci_upper,
            'Interpretation': 'Base rate of RAVLT impairment (z < -1.0) in sample'
        })

        # REMEMVR normal rate
        rememvr_normal_rate = (true_negatives + false_negatives) / total
        rememvr_ci_lower, rememvr_ci_upper = wilson_ci(true_negatives + false_negatives, total, CONFIDENCE_LEVEL)

        log(f"REMEMVR normal rate: {rememvr_normal_rate:.3f} (95% CI: [{rememvr_ci_lower:.3f}, {rememvr_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'REMEMVR_Normal_Rate',
            'Value': rememvr_normal_rate,
            'CI_Lower': rememvr_ci_lower,
            'CI_Upper': rememvr_ci_upper,
            'Interpretation': 'Proportion classified as REMEMVR normal (z > -0.5)'
        })

        # False negative rate (among RAVLT impaired)
        if sensitivity_denom > 0:
            false_negative_rate = false_negatives / sensitivity_denom
            fn_ci_lower, fn_ci_upper = wilson_ci(false_negatives, sensitivity_denom, CONFIDENCE_LEVEL)
        else:
            false_negative_rate = np.nan
            fn_ci_lower, fn_ci_upper = np.nan, np.nan

        log(f"False negative rate: {false_negative_rate:.3f} (95% CI: [{fn_ci_lower:.3f}, {fn_ci_upper:.3f}])")
        metrics_list.append({
            'Metric': 'False_Negative_Rate',
            'Value': false_negative_rate,
            'CI_Lower': fn_ci_lower,
            'CI_Upper': fn_ci_upper,
            'Interpretation': 'Proportion of RAVLT-impaired missed by REMEMVR (1 - Sensitivity)'
        })
        # Create Metrics DataFrame
        log("Creating clinical metrics table...")

        df_metrics = pd.DataFrame(metrics_list)
        log(f"Metrics table: {len(df_metrics)} metrics")
        # Save Metrics
        log("Saving clinical metrics...")

        df_metrics.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE}")
        # Validate Metrics
        log("Running validate_probability_range...")

        prob_columns = ['Value', 'CI_Lower', 'CI_Upper']
        validation_result = validate_probability_range(
            probability_df=df_metrics,
            prob_columns=prob_columns
        )

        if validation_result.get('valid', False):
            log(f"All metric values in valid range [0, 1]")
        else:
            log(f"Validation: {validation_result.get('message', 'Unknown issue')}")

        # Check expected metrics present
        expected_metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV',
                          'RAVLT_Impairment_Rate', 'REMEMVR_Normal_Rate', 'False_Negative_Rate']

        actual_metrics = df_metrics['Metric'].tolist()
        missing_metrics = [m for m in expected_metrics if m not in actual_metrics]

        if missing_metrics:
            log(f"Missing expected metrics: {missing_metrics}")
        else:
            log(f"All {len(expected_metrics)} expected metrics present")

        # Check CI validity (Lower <= Value <= Upper)
        ci_violations = []
        for _, row in df_metrics.iterrows():
            if pd.notna(row['Value']):  # Skip NaN values
                if not (row['CI_Lower'] <= row['Value'] <= row['CI_Upper']):
                    ci_violations.append(row['Metric'])

        if ci_violations:
            log(f"CI violations (Value outside CI bounds): {ci_violations}")
        else:
            log(f"All CIs valid (CI_Lower <= Value <= CI_Upper)")

        # Check for missing values
        missing_count = df_metrics.isnull().sum().sum()
        if missing_count > 0:
            log(f"{missing_count} missing values (may be expected if denominators=0)")
        else:
            log(f"No missing values")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
