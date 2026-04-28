#!/usr/bin/env python3
"""AIC Model Selection: Compute AIC-based model selection metrics: delta AIC (difference from minimum),"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any
import traceback

# parents[4] = REMEMVR/ (code -> rq7 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_irt_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
LOG_FILE = RQ_DIR / "logs" / "step06_model_selection.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: AIC Model Selection")
        # Load Model Comparison Results

        log("Loading model comparison table...")
        comparison_path = RQ_DIR / "results" / "step05_model_comparison.csv"

        if not comparison_path.exists():
            raise FileNotFoundError(f"Model comparison table missing: {comparison_path}\n"
                                     "Run step05_fit_5_candidate_lmms.py first")

        model_comparison = pd.read_csv(comparison_path, encoding='utf-8')
        log(f"{comparison_path.name} ({len(model_comparison)} models)")
        log(f"  Columns: {model_comparison.columns.tolist()}")
        log(f"  AIC range: [{model_comparison['AIC'].min():.2f}, {model_comparison['AIC'].max():.2f}]")

        log("Skipping model pickle loading (patsy unpickling issue)")
        # Note: Model pickles exist but cannot be unpickled due to patsy formula issue
        # We'll skip saving best_model.pkl and use limited summary info
        model_fits = None  # Not loaded to avoid unpickling errors
        # Compute Delta AIC
        # Formula: delta_AIC_i = AIC_i - min(AIC)
        # Interpretation: <2 (substantial), 2-7 (positive), 7-10 (weak), >10 (none)

        log("Computing delta AIC...")
        aic_min = model_comparison['AIC'].min()
        model_comparison['delta_AIC'] = model_comparison['AIC'] - aic_min

        log(f"  Minimum AIC: {aic_min:.2f}")
        log(f"  delta_AIC range: [{model_comparison['delta_AIC'].min():.2f}, {model_comparison['delta_AIC'].max():.2f}]")
        # Compute Akaike Weights
        # Formula: w_i = exp(-0.5 * delta_AIC_i) / sum(exp(-0.5 * delta_AIC_j))
        # Properties: Sum to 1.0, all in (0, 1), higher weight = better support

        log("Computing Akaike weights...")
        model_comparison['akaike_weight'] = np.exp(-0.5 * model_comparison['delta_AIC'])
        weight_sum = model_comparison['akaike_weight'].sum()
        model_comparison['akaike_weight'] = model_comparison['akaike_weight'] / weight_sum

        log(f"  Weight sum: {model_comparison['akaike_weight'].sum():.6f} (should be 1.0)")
        log(f"  Weight range: [{model_comparison['akaike_weight'].min():.4f}, {model_comparison['akaike_weight'].max():.4f}]")
        # Sort by AIC and Compute Cumulative Weights
        # Cumulative weight: Running sum of weights (monotonic increasing to 1.0)

        log("Sorting by AIC ascending...")
        model_comparison = model_comparison.sort_values('AIC').reset_index(drop=True)

        log("Computing cumulative weights...")
        model_comparison['cumulative_weight'] = model_comparison['akaike_weight'].cumsum()

        log("")
        log("AIC Comparison (sorted by AIC):")
        log(model_comparison[['model_name', 'AIC', 'delta_AIC', 'akaike_weight', 'cumulative_weight']].to_string(index=False))
        log("")
        # Identify Best Model
        # Best model: Row with delta_AIC = 0 (minimum AIC)

        best_model_name = model_comparison.iloc[0]['model_name']
        best_model_aic = model_comparison.iloc[0]['AIC']
        best_model_weight = model_comparison.iloc[0]['akaike_weight']

        log(f"[BEST MODEL] {best_model_name}")
        log(f"  AIC: {best_model_aic:.2f}")
        log(f"  Akaike weight: {best_model_weight:.4f}")

        # Categorize uncertainty
        if best_model_weight > 0.90:
            uncertainty = "Very strong"
            interpretation = ">90% probability this is the best model"
        elif best_model_weight >= 0.60:
            uncertainty = "Strong"
            interpretation = "60-90% probability this is the best model"
        elif best_model_weight >= 0.30:
            uncertainty = "Moderate"
            interpretation = "30-60% probability - substantial uncertainty"
        else:
            uncertainty = "High"
            interpretation = "<30% probability - weak support, consider model averaging"

        log(f"  Uncertainty: {uncertainty} ({interpretation})")
        # Save AIC Comparison Results
        # Output will be used by: Step 7 (plot preparation), final RQ report

        # Save AIC comparison table
        output_path = RQ_DIR / "results" / "step06_aic_comparison.csv"
        log(f"Saving AIC comparison to {output_path.name}...")
        model_comparison[['model_name', 'AIC', 'delta_AIC', 'akaike_weight', 'cumulative_weight']].to_csv(
            output_path,
            index=False,
            encoding='utf-8'
        )
        log(f"{output_path.name}")

        # Skip saving best model pickle (unpickling issue)
        log(f"Not saving best_model.pkl (model pickle exists as data/lmm_{best_model_name}.pkl)")

        # Save best model summary text (without full model summary)
        summary_path = RQ_DIR / "results" / "step06_best_model_summary.txt"
        log(f"Saving best model summary to {summary_path.name}...")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BEST MODEL SUMMARY - AIC Model Selection\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"AIC: {best_model_aic:.2f}\n")
            f.write(f"Akaike Weight: {best_model_weight:.4f}\n")
            f.write(f"Uncertainty: {uncertainty}\n")
            f.write(f"Interpretation: {interpretation}\n\n")
            f.write("NOTE: Full model summary not available due to pickle unpickling issue.\n")
            f.write(f"Model pickle file available at: data/lmm_{best_model_name}.pkl\n\n")
            f.write("=" * 80 + "\n")
            f.write("All Models Comparison:\n")
            f.write("=" * 80 + "\n\n")
            f.write(model_comparison[['model_name', 'AIC', 'delta_AIC', 'akaike_weight', 'cumulative_weight']].to_string(index=False))
            f.write("\n")

        log(f"{summary_path.name}")
        # Validate AIC Comparison
        # Validates: Weights sum to 1.0, weights in (0,1), delta_AIC correct,
        #            cumulative_weight monotonic

        log("Validating AIC comparison metrics...")

        # Check weights sum to 1.0
        weight_sum = model_comparison['akaike_weight'].sum()
        if 0.999 <= weight_sum <= 1.001:
            log(f"Akaike weights sum to 1.0 ({weight_sum:.6f})")
        else:
            raise ValueError(f"Akaike weights sum incorrect: {weight_sum:.6f} (expected 1.0)")

        # Check all weights in (0, 1)
        if (model_comparison['akaike_weight'] > 0).all() and (model_comparison['akaike_weight'] < 1).all():
            log("All Akaike weights in (0, 1) exclusive")
        else:
            raise ValueError("Some Akaike weights outside (0, 1) range")

        # Check delta_AIC correct
        if model_comparison.iloc[0]['delta_AIC'] == 0.0:
            log("Best model has delta_AIC = 0")
        else:
            raise ValueError(f"Best model delta_AIC incorrect: {model_comparison.iloc[0]['delta_AIC']} (expected 0)")

        if (model_comparison['delta_AIC'] >= 0).all():
            log("All delta_AIC >= 0")
        else:
            raise ValueError("Some delta_AIC values negative")

        # Check cumulative_weight monotonic increasing
        if (model_comparison['cumulative_weight'].diff().dropna() >= 0).all():
            log("cumulative_weight monotonic increasing")
        else:
            raise ValueError("cumulative_weight not monotonic increasing")

        # Check cumulative_weight ends at 1.0
        cum_weight_final = model_comparison.iloc[-1]['cumulative_weight']
        if 0.999 <= cum_weight_final <= 1.001:
            log(f"cumulative_weight ends at 1.0 ({cum_weight_final:.6f})")
        else:
            raise ValueError(f"cumulative_weight final value incorrect: {cum_weight_final:.6f} (expected 1.0)")

        # Invoke generic validation
        validation_result = validate_irt_convergence(
            results={
                "data": model_comparison
            }
        )

        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
