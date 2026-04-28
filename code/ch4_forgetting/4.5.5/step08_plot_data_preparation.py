#!/usr/bin/env python3
"""plot_data_preparation: Create plot source CSV files for visualization of correlation comparison and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_plot_data_preparation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def fisher_z_transform(r: float) -> float:
    """Convert correlation r to Fisher's z."""
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z: float) -> float:
    """Convert Fisher's z back to correlation r."""
    exp_2z = np.exp(2 * z)
    return (exp_2z - 1) / (exp_2z + 1)

def compute_correlation_ci(r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for correlation using Fisher's z-transformation.

    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    Tuple[float, float]
        CI_lower, CI_upper
    """
    # Fisher's z-transformation
    z = fisher_z_transform(r)

    # Standard error of z
    se_z = 1.0 / np.sqrt(n - 3)

    # Z-score for confidence level (1.96 for 95% CI)
    z_score = 1.96 if confidence == 0.95 else 1.645  # 90% CI fallback

    # Confidence interval in z-space
    z_lower = z - z_score * se_z
    z_upper = z + z_score * se_z

    # Back-transform to correlation space
    r_lower = inverse_fisher_z(z_lower)
    r_upper = inverse_fisher_z(z_upper)

    return r_lower, r_upper

def interpret_delta_aic(delta: float) -> str:
    """
    Interpret delta AIC per Burnham & Anderson (2002).

    Delta AIC = AIC(model) - AIC(reference)
    Negative delta = model is better than reference
    Positive delta = reference is better than model

    Parameters
    ----------
    delta : float
        Delta AIC value

    Returns
    -------
    str
        Interpretation string
    """
    abs_delta = abs(delta)

    if abs_delta <= 2:
        return "No difference"
    elif delta > 2 and delta <= 10:
        return "Substantial (favoring Full CTT)"
    elif delta > 10:
        return "Decisive (favoring Full CTT)"
    elif delta < -2 and delta >= -10:
        return "Substantial (favoring this model)"
    else:  # delta < -10
        return "Decisive (favoring this model)"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 8: Plot Data Preparation")
        # Load Input Data

        log("Loading input data...")

        # Load reliability assessment
        reliability_file = RQ_DIR / "data" / "step04_reliability_assessment.csv"
        df_reliability = pd.read_csv(reliability_file, encoding='utf-8')
        log(f"step04_reliability_assessment.csv ({len(df_reliability)} rows, {len(df_reliability.columns)} cols)")

        # Load correlation analysis
        correlation_file = RQ_DIR / "data" / "step05_correlation_analysis.csv"
        df_correlation = pd.read_csv(correlation_file, encoding='utf-8')
        log(f"step05_correlation_analysis.csv ({len(df_correlation)} rows, {len(df_correlation.columns)} cols)")

        # Load LMM model comparison
        lmm_file = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        df_lmm = pd.read_csv(lmm_file, encoding='utf-8')
        log(f"step07_lmm_model_comparison.csv ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")
        # Create Correlation Comparison Plot Data

        log("Creating correlation comparison plot data...")

        # Initialize list to collect rows
        correlation_plot_rows = []

        # Process each location type
        for _, row in df_correlation.iterrows():
            location_type = row['location_type']
            n = int(row['n'])

            # Full CTT correlation
            r_full = row['r_full']
            ci_lower_full, ci_upper_full = compute_correlation_ci(r_full, n)
            correlation_plot_rows.append({
                'location_type': location_type,
                'version': 'full',
                'r': r_full,
                'CI_lower': ci_lower_full,
                'CI_upper': ci_upper_full
            })

            # Purified CTT correlation
            r_purified = row['r_purified']
            ci_lower_purified, ci_upper_purified = compute_correlation_ci(r_purified, n)
            correlation_plot_rows.append({
                'location_type': location_type,
                'version': 'purified',
                'r': r_purified,
                'CI_lower': ci_lower_purified,
                'CI_upper': ci_upper_purified
            })

        # Create DataFrame
        df_correlation_plot = pd.DataFrame(correlation_plot_rows)
        log(f"Correlation comparison plot data ({len(df_correlation_plot)} rows)")
        # Create AIC Comparison Plot Data

        log("Creating AIC comparison plot data...")

        # Initialize list to collect rows
        aic_plot_rows = []

        # Process each location type
        for _, row in df_lmm.iterrows():
            location_type = row['location_type']
            aic_irt = row['aic_irt']
            aic_full = row['aic_ctt_full']
            aic_purified = row['aic_ctt_purified']

            # Reference model = Full CTT per location type
            # Delta AIC = AIC(model) - AIC(Full CTT)

            # IRT model
            delta_irt = aic_irt - aic_full
            aic_plot_rows.append({
                'location_type': location_type,
                'model': 'IRT',
                'aic': aic_irt,
                'delta_aic': delta_irt,
                'interpretation': interpret_delta_aic(delta_irt)
            })

            # Full CTT model (reference)
            aic_plot_rows.append({
                'location_type': location_type,
                'model': 'Full_CTT',
                'aic': aic_full,
                'delta_aic': 0.0,
                'interpretation': 'Reference'
            })

            # Purified CTT model
            delta_purified = aic_purified - aic_full
            aic_plot_rows.append({
                'location_type': location_type,
                'model': 'Purified_CTT',
                'aic': aic_purified,
                'delta_aic': delta_purified,
                'interpretation': interpret_delta_aic(delta_purified)
            })

        # Create DataFrame
        df_aic_plot = pd.DataFrame(aic_plot_rows)
        log(f"AIC comparison plot data ({len(df_aic_plot)} rows)")
        # Save Plot Data
        # These outputs will be used by: plotting pipeline for visualization

        log("Saving plot data...")

        # Ensure plots/ directory exists
        plots_dir = RQ_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Save correlation comparison plot data
        correlation_plot_file = plots_dir / "step08_correlation_comparison_data.csv"
        df_correlation_plot.to_csv(correlation_plot_file, index=False, encoding='utf-8')
        log(f"plots/step08_correlation_comparison_data.csv ({len(df_correlation_plot)} rows, {len(df_correlation_plot.columns)} cols)")

        # Save AIC comparison plot data
        aic_plot_file = plots_dir / "step08_aic_comparison_data.csv"
        df_aic_plot.to_csv(aic_plot_file, index=False, encoding='utf-8')
        log(f"plots/step08_aic_comparison_data.csv ({len(df_aic_plot)} rows, {len(df_aic_plot.columns)} cols)")
        # Validate Output Structure
        # Validates: Row counts, column presence, CI validity, no NaN values

        log("Validating plot data...")

        validation_errors = []

        # Validate correlation plot data
        if len(df_correlation_plot) != 4:
            validation_errors.append(f"Correlation plot: Expected 4 rows, got {len(df_correlation_plot)}")

        if not all(df_correlation_plot['CI_upper'] > df_correlation_plot['CI_lower']):
            validation_errors.append("Correlation plot: CI_upper must be > CI_lower for all rows")

        if df_correlation_plot.isnull().any().any():
            validation_errors.append("Correlation plot: Contains NaN values")

        required_corr_cols = ['location_type', 'version', 'r', 'CI_lower', 'CI_upper']
        missing_corr_cols = [col for col in required_corr_cols if col not in df_correlation_plot.columns]
        if missing_corr_cols:
            validation_errors.append(f"Correlation plot: Missing columns {missing_corr_cols}")

        # Validate AIC plot data
        if len(df_aic_plot) != 6:
            validation_errors.append(f"AIC plot: Expected 6 rows, got {len(df_aic_plot)}")

        expected_combinations = [
            ('source', 'IRT'), ('source', 'Full_CTT'), ('source', 'Purified_CTT'),
            ('destination', 'IRT'), ('destination', 'Full_CTT'), ('destination', 'Purified_CTT')
        ]
        actual_combinations = set(zip(df_aic_plot['location_type'], df_aic_plot['model']))
        expected_combinations_set = set(expected_combinations)
        missing_combinations = expected_combinations_set - actual_combinations
        if missing_combinations:
            validation_errors.append(f"AIC plot: Missing combinations {missing_combinations}")

        if df_aic_plot.isnull().any().any():
            validation_errors.append("AIC plot: Contains NaN values")

        required_aic_cols = ['location_type', 'model', 'aic', 'delta_aic', 'interpretation']
        missing_aic_cols = [col for col in required_aic_cols if col not in df_aic_plot.columns]
        if missing_aic_cols:
            validation_errors.append(f"AIC plot: Missing columns {missing_aic_cols}")

        # Report validation results
        if validation_errors:
            log("FAILED - Errors detected:")
            for error in validation_errors:
                log(f"  - {error}")
            sys.exit(1)
        else:
            log("PASSED - All checks successful")
            log(f"  - Correlation plot: 4 rows, all CIs valid, no NaN")
            log(f"  - AIC plot: 6 rows, all combinations present, no NaN")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
