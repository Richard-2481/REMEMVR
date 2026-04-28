#!/usr/bin/env python3
"""select_best_model: Compute Akaike weights (model probabilities) from AIC values and identify the"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_select_best_model.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Select Best Model via AIC")
        # Load Input Data

        log("Loading model comparison results...")
        input_path = RQ_DIR / "data" / "step05_model_comparison.csv"
        df_models = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df_models)} rows, {len(df_models.columns)} cols)")

        # Verify expected structure
        expected_models = {"Linear", "Quadratic", "Logarithmic", "Linear+Logarithmic", "Quadratic+Logarithmic"}
        actual_models = set(df_models['model_name'].tolist())
        if actual_models != expected_models:
            log(f"Model names mismatch:")
            log(f"  Expected: {expected_models}")
            log(f"  Actual:   {actual_models}")

        log(f"AIC range: {df_models['AIC'].min():.2f} - {df_models['AIC'].max():.2f}")
        # Compute Akaike Weights

        log("Computing Akaike weights...")

        # Step 2a: Compute delta_AIC (difference from best model)
        min_aic = df_models['AIC'].min()
        df_models['delta_AIC'] = df_models['AIC'] - min_aic
        log(f"delta_AIC (minimum AIC = {min_aic:.4f})")

        # Step 2b: Compute relative likelihood
        # Formula: exp(-0.5 * delta_AIC)
        # Interpretation: How likely each model is relative to best model
        df_models['relative_likelihood'] = np.exp(-0.5 * df_models['delta_AIC'])
        log(f"relative_likelihood (exponential transformation)")

        # Step 2c: Compute Akaike weights (normalize to probabilities)
        # Formula: relative_likelihood / sum(all relative_likelihoods)
        # Interpretation: Probability that each model is the best model
        sum_likelihoods = df_models['relative_likelihood'].sum()
        df_models['akaike_weight'] = df_models['relative_likelihood'] / sum_likelihoods
        log(f"akaike_weight (normalized to sum=1.0)")

        # Step 2d: Identify best model (lowest AIC, highest akaike_weight)
        df_models['is_best'] = df_models['AIC'] == min_aic
        best_model_name = df_models.loc[df_models['is_best'], 'model_name'].iloc[0]
        best_aic = df_models.loc[df_models['is_best'], 'AIC'].iloc[0]
        best_weight = df_models.loc[df_models['is_best'], 'akaike_weight'].iloc[0]

        log(f"Best model: {best_model_name}")
        log(f"  AIC: {best_aic:.4f}")
        log(f"  Akaike weight: {best_weight:.4f}")

        # Sort by AIC ascending (best model first)
        df_models = df_models.sort_values('AIC', ascending=True).reset_index(drop=True)
        log("Models by AIC ascending (best first)")
        # Save Analysis Outputs
        # These outputs will be used by: Step 7 (Ch5 comparison), results analysis (final report)

        log("Saving AIC comparison table...")
        # Output: step06_aic_comparison.csv
        # Contains: Model names, AIC values, Akaike weights, best model flag
        # Columns: model_name, AIC, delta_AIC, relative_likelihood, akaike_weight, is_best
        output_path = RQ_DIR / "data" / "step06_aic_comparison.csv"
        df_models.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_models)} rows, {len(df_models.columns)} cols)")

        # Print summary table for log
        log("AIC Comparison:")
        for _, row in df_models.iterrows():
            best_marker = " <- BEST" if row['is_best'] else ""
            log(f"  {row['model_name']:25s} AIC={row['AIC']:7.2f}  delta={row['delta_AIC']:6.2f}  weight={row['akaike_weight']:.4f}{best_marker}")

        # Copy best model pkl file to canonical name
        log("Copying best model to canonical filename...")

        # Map model names to step05 pkl filenames
        model_file_mapping = {
            "Linear": "step05_model1_linear.pkl",
            "Quadratic": "step05_model2_quadratic.pkl",
            "Logarithmic": "step05_model3_logarithmic.pkl",
            "Linear+Logarithmic": "step05_model4_linear_logarithmic.pkl",
            "Quadratic+Logarithmic": "step05_model5_quadratic_logarithmic.pkl"
        }

        source_pkl = RQ_DIR / "data" / model_file_mapping[best_model_name]
        dest_pkl = RQ_DIR / "data" / "step06_best_model.pkl"

        if not source_pkl.exists():
            log(f"Source model file not found: {source_pkl.name}")
            raise FileNotFoundError(f"Expected source file {source_pkl} does not exist")

        shutil.copy2(source_pkl, dest_pkl)
        log(f"{dest_pkl.name} (copy of {source_pkl.name})")
        # Run Validation Tool
        # Validates: Akaike weights are valid probabilities, delta_AIC non-negative
        # Threshold: Weights sum to 1.0 +/- 0.01

        log("Running validate_numeric_range...")

        # Validation 1: Exactly one is_best=True
        n_best = df_models['is_best'].sum()
        if n_best == 1:
            log(f"Exactly one best model identified: PASS")
        else:
            log(f"Expected 1 best model, found {n_best}: FAIL")
            raise ValueError(f"Expected exactly 1 is_best=True, found {n_best}")

        # Validation 2: Akaike weights sum to 1.0 +/- 0.01
        weight_sum = df_models['akaike_weight'].sum()
        if abs(weight_sum - 1.0) <= 0.01:
            log(f"Akaike weights sum to 1.0: PASS (sum={weight_sum:.6f})")
        else:
            log(f"Akaike weights sum incorrect: FAIL (sum={weight_sum:.6f})")
            raise ValueError(f"Akaike weights must sum to 1.0 +/- 0.01, got {weight_sum:.6f}")

        # Validation 3: Best model has delta_AIC=0
        best_delta = df_models.loc[df_models['is_best'], 'delta_AIC'].iloc[0]
        if abs(best_delta) < 1e-10:  # Floating point tolerance
            log(f"Best model has delta_AIC=0: PASS")
        else:
            log(f"Best model delta_AIC={best_delta:.6f}: FAIL")
            raise ValueError(f"Best model must have delta_AIC=0, got {best_delta:.6f}")

        # Validation 4: All AIC values finite
        if df_models['AIC'].notna().all() and np.isfinite(df_models['AIC']).all():
            log(f"All AIC values finite: PASS")
        else:
            log(f"Non-finite AIC values detected: FAIL")
            raise ValueError("AIC values contain NaN or Inf")

        # Validation 5: delta_AIC in [0, Inf)
        val_result = validate_numeric_range(
            data=df_models['delta_AIC'].values,
            min_val=0.0,
            max_val=np.inf,
            column_name='delta_AIC'
        )
        if val_result['valid']:
            log(f"delta_AIC in [0, Inf): PASS")
        else:
            log(f"delta_AIC range check: FAIL")
            log(f"  {val_result['message']}")
            raise ValueError(val_result['message'])

        # Validation 6: relative_likelihood in (0, 1]
        if (df_models['relative_likelihood'] > 0).all() and (df_models['relative_likelihood'] <= 1.0).all():
            log(f"relative_likelihood in (0, 1]: PASS")
        else:
            log(f"relative_likelihood range check: FAIL")
            raise ValueError("relative_likelihood must be in (0, 1]")

        # Validation 7: akaike_weight in (0, 1)
        if (df_models['akaike_weight'] > 0).all() and (df_models['akaike_weight'] < 1.0).all():
            log(f"akaike_weight in (0, 1): PASS")
        else:
            log(f"akaike_weight range check: FAIL")
            raise ValueError("akaike_weight must be in (0, 1)")

        log("Step 06 complete - Best model identified: {0}".format(best_model_name))
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
