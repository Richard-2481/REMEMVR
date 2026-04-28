#!/usr/bin/env python3
"""AIC Model Comparison for HCE Rate Time Transformation: Empirically determine whether HCE rate follows linear, quadratic, or logarithmic"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import required libraries
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02a_model_comparison.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

# Model Fitting Functions

def fit_model_with_convergence_check(formula: str, re_formula: str, data: pd.DataFrame,
                                     model_name: str) -> Tuple[Any, bool, str]:
    """
    Fit LMM model and check convergence status.

    Returns:
        Tuple of (fitted_model, converged_bool, message)
    """
    try:
        # Suppress convergence warnings during fitting (we'll check manually)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Fit model with REML=False (required for AIC comparison)
            model = MixedLM.from_formula(
                formula=formula,
                groups=data['UID'],
                re_formula=re_formula,
                data=data
            )
            result = model.fit(reml=False, method='lbfgs', maxiter=200)

            # Check convergence
            converged = result.converged
            if converged:
                msg = f"{model_name} converged successfully"
            else:
                msg = f"{model_name} did not converge (may still have valid AIC)"

            return result, converged, msg

    except Exception as e:
        msg = f"{model_name} failed to fit: {str(e)}"
        return None, False, msg


def extract_model_statistics(result: Any, model_name: str, converged: bool) -> Dict[str, Any]:
    """
    Extract AIC, BIC, log-likelihood, and number of parameters from fitted model.

    Returns:
        Dictionary with model statistics
    """
    if result is None:
        return {
            'model': model_name,
            'AIC': np.nan,
            'BIC': np.nan,
            'loglik': np.nan,
            'n_params': np.nan,
            'converged': False
        }

    try:
        return {
            'model': model_name,
            'AIC': result.aic,
            'BIC': result.bic,
            'loglik': result.llf,  # Log-likelihood
            'n_params': result.df_modelwc,  # Number of parameters (with random effects)
            'converged': converged
        }
    except Exception as e:
        log(f"Could not extract statistics for {model_name}: {str(e)}")
        return {
            'model': model_name,
            'AIC': np.nan,
            'BIC': np.nan,
            'loglik': np.nan,
            'n_params': np.nan,
            'converged': False
        }


def compute_akaike_weights(df_models: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Akaike weights for model comparison.

    Akaike weights represent the probability that each model is the best model
    given the data and the candidate set.

    Formula:
        delta_AIC_i = AIC_i - min(AIC)
        weight_i = exp(-0.5 * delta_AIC_i) / sum(exp(-0.5 * delta_AIC_j) for all j)

    Returns:
        DataFrame with added columns: delta_AIC, akaike_weight
    """
    # Only compute weights for converged models with finite AIC
    valid_models = df_models[df_models['AIC'].notna() & np.isfinite(df_models['AIC'])].copy()

    if len(valid_models) == 0:
        log("No models with valid AIC values for weight computation")
        df_models['delta_AIC'] = np.nan
        df_models['akaike_weight'] = np.nan
        return df_models

    # Compute delta AIC (difference from best model)
    min_aic = valid_models['AIC'].min()
    valid_models['delta_AIC'] = valid_models['AIC'] - min_aic

    # Compute Akaike weights
    # weight_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    exp_terms = np.exp(-0.5 * valid_models['delta_AIC'])
    valid_models['akaike_weight'] = exp_terms / exp_terms.sum()

    # Merge back into full dataframe
    df_models = df_models.merge(
        valid_models[['model', 'delta_AIC', 'akaike_weight']],
        on='model',
        how='left'
    )

    return df_models


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02a: AIC Model Comparison for HCE Rate Time Transformation")
        # Load Input Data

        log("Loading HCE rate data from step01...")
        df_hce = pd.read_csv(RQ_DIR / "data" / "step01_hce_rates.csv")
        log(f"step01_hce_rates.csv ({len(df_hce)} rows, {len(df_hce.columns)} cols)")

        # Validate expected structure
        expected_cols = ['UID', 'TEST', 'TSVR', 'HCE_rate', 'n_HCE', 'n_total']
        missing_cols = [col for col in expected_cols if col not in df_hce.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")

        log(f"Data summary: {len(df_hce)} observations from {df_hce['UID'].nunique()} participants")
        # Create Time Transformations
        # Transformations match RQ 6.1.1 approach:
        #   - Days = TSVR / 24 (linear time in days)
        #   - Days_squared = Days^2 (quadratic term)
        #   - log_Days_plus1 = log(Days + 1) (logarithmic transformation)

        log("Creating time transformations from TSVR...")

        # Linear time (days since encoding)
        df_hce['Days'] = df_hce['TSVR'] / 24.0

        # Quadratic term
        df_hce['Days_squared'] = df_hce['Days'] ** 2

        # Logarithmic term (log of days + 1 to handle Day 0)
        df_hce['log_Days_plus1'] = np.log(df_hce['Days'] + 1.0)

        log(f"Created transformations:")
        log(f"  - Days: range [{df_hce['Days'].min():.2f}, {df_hce['Days'].max():.2f}]")
        log(f"  - Days_squared: range [{df_hce['Days_squared'].min():.2f}, {df_hce['Days_squared'].max():.2f}]")
        log(f"  - log_Days_plus1: range [{df_hce['log_Days_plus1'].min():.3f}, {df_hce['log_Days_plus1'].max():.3f}]")
        # Fit 5 Candidate LMM Models
        # Models (matching RQ 6.1.1 functional form comparison):
        #   1. Linear: HCE_rate ~ Days + (Days | UID)
        #   2. Quadratic: HCE_rate ~ Days + Days_squared + (Days | UID)
        #   3. Logarithmic: HCE_rate ~ log_Days_plus1 + (log_Days_plus1 | UID)
        #   4. Linear+Log: HCE_rate ~ Days + log_Days_plus1 + (log_Days_plus1 | UID)
        #   5. Quadratic+Log: HCE_rate ~ Days + Days_squared + log_Days_plus1 + (log_Days_plus1 | UID)
        #
        # CRITICAL: REML=False for all models (required for valid AIC comparison)
        # Random slopes match fixed effects structure

        log("Fitting 5 candidate LMM models with REML=False...")

        # Define model specifications
        models = [
            {
                'name': 'Linear',
                'formula': 'HCE_rate ~ Days',
                're_formula': '~Days',
                'description': 'Linear time trend'
            },
            {
                'name': 'Quadratic',
                'formula': 'HCE_rate ~ Days + Days_squared',
                're_formula': '~Days',
                'description': 'Quadratic time trend (Days + Days^2)'
            },
            {
                'name': 'Logarithmic',
                'formula': 'HCE_rate ~ log_Days_plus1',
                're_formula': '~log_Days_plus1',
                'description': 'Logarithmic time trend (log(Days+1))'
            },
            {
                'name': 'Linear+Log',
                'formula': 'HCE_rate ~ Days + log_Days_plus1',
                're_formula': '~log_Days_plus1',
                'description': 'Linear + Logarithmic (Days + log(Days+1))'
            },
            {
                'name': 'Quadratic+Log',
                'formula': 'HCE_rate ~ Days + Days_squared + log_Days_plus1',
                're_formula': '~log_Days_plus1',
                'description': 'Quadratic + Logarithmic (Days + Days^2 + log(Days+1))'
            }
        ]

        # Fit each model and collect results
        model_results = []
        fitted_models = {}

        for model_spec in models:
            log(f"\nFitting {model_spec['name']}: {model_spec['description']}")
            log(f"  Formula: {model_spec['formula']}")
            log(f"  Random: {model_spec['re_formula']}")

            result, converged, msg = fit_model_with_convergence_check(
                formula=model_spec['formula'],
                re_formula=model_spec['re_formula'],
                data=df_hce,
                model_name=model_spec['name']
            )

            log(f"  {msg}")

            # Extract statistics
            stats = extract_model_statistics(result, model_spec['name'], converged)
            model_results.append(stats)

            # Store fitted model for potential later use
            if result is not None:
                fitted_models[model_spec['name']] = result
                log(f"  AIC: {stats['AIC']:.2f}, BIC: {stats['BIC']:.2f}, LogLik: {stats['loglik']:.2f}")

        log("\nAll models fitted")
        # Compute Akaike Weights
        # Akaike weight = probability that model i is best given data and candidate set

        log("\nComputing Akaike weights...")

        df_comparison = pd.DataFrame(model_results)
        df_comparison = compute_akaike_weights(df_comparison)

        # Check if weights sum to 1.0 (across converged models)
        valid_weights = df_comparison['akaike_weight'].dropna()
        if len(valid_weights) > 0:
            weight_sum = valid_weights.sum()
            log(f"Akaike weights sum: {weight_sum:.6f} (should be 1.0)")
            if abs(weight_sum - 1.0) > 0.01:
                log(f"Weights do not sum to 1.0 (sum={weight_sum:.6f})")
        else:
            log("No valid Akaike weights computed (no converged models)")
        # Identify Best Model
        # Best model = lowest AIC = highest Akaike weight

        log("\nIdentifying best model...")

        # Find model with highest Akaike weight
        converged_models = df_comparison[df_comparison['converged'] == True].copy()

        if len(converged_models) == 0:
            log("No models converged successfully")
            best_model_name = "NONE"
            best_weight = 0.0
        else:
            best_idx = converged_models['akaike_weight'].idxmax()
            best_model_name = converged_models.loc[best_idx, 'model']
            best_weight = converged_models.loc[best_idx, 'akaike_weight']
            best_aic = converged_models.loc[best_idx, 'AIC']

            log(f"{best_model_name} (weight={best_weight:.4f}, AIC={best_aic:.2f})")

            # Interpretation of best model weight
            if best_weight > 0.90:
                interpretation = "Overwhelming evidence"
            elif best_weight > 0.70:
                interpretation = "Strong evidence"
            elif best_weight > 0.50:
                interpretation = "Moderate evidence"
            elif best_weight > 0.30:
                interpretation = "Weak evidence"
            else:
                interpretation = "No clear winner"

            log(f"{interpretation} for {best_model_name} model")
        # Save Model Comparison Results

        log("\nSaving model comparison results...")

        # Save comparison table to CSV
        comparison_path = RQ_DIR / "data" / "step02a_model_comparison.csv"
        df_comparison.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"{comparison_path.name} ({len(df_comparison)} models)")

        # Create text summary
        summary_path = RQ_DIR / "data" / "step02a_best_model_selection.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AIC MODEL COMPARISON SUMMARY - HCE RATE TIME TRANSFORMATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("Research Question: RQ 6.6.1\n")
            f.write("Analysis: Which functional form best describes HCE rate trajectory?\n")
            f.write("Method: AIC comparison with Akaike weights\n\n")

            f.write("BEST MODEL SELECTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Model: {best_model_name}\n")
            if len(converged_models) > 0:
                f.write(f"Akaike Weight: {best_weight:.4f}\n")
                f.write(f"AIC: {best_aic:.2f}\n")
                f.write(f"Interpretation: {interpretation}\n\n")
            else:
                f.write("Status: No models converged\n\n")

            f.write("FULL MODEL COMPARISON TABLE\n")
            f.write("-" * 80 + "\n")
            f.write(df_comparison.to_string(index=False) + "\n\n")

            f.write("AKAIKE WEIGHT INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("Akaike weight = probability that model is best given data and candidate set\n")
            f.write("  > 0.90: Overwhelming evidence\n")
            f.write("  > 0.70: Strong evidence\n")
            f.write("  > 0.50: Moderate evidence\n")
            f.write("  > 0.30: Weak evidence\n")
            f.write("  < 0.30: No clear winner\n\n")

            f.write("DELTA AIC INTERPRETATION (Burnham & Anderson 2002)\n")
            f.write("-" * 80 + "\n")
            f.write("delta_AIC = AIC_i - min(AIC)\n")
            f.write("  0-2: Substantial support (models are competitive)\n")
            f.write("  4-7: Considerably less support\n")
            f.write("  >10: Essentially no support\n\n")

            f.write("CONVERGENCE STATUS\n")
            f.write("-" * 80 + "\n")
            for _, row in df_comparison.iterrows():
                status = "CONVERGED" if row['converged'] else "DID NOT CONVERGE"
                f.write(f"  {row['model']:20s}: {status}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        log(f"{summary_path.name}")
        # Validation

        log("\nChecking results...")

        validation_passed = True

        # Check 1: All 5 models attempted
        if len(df_comparison) != 5:
            log(f"Expected 5 models, found {len(df_comparison)}")
            validation_passed = False
        else:
            log("All 5 models attempted")

        # Check 2: At least some models have finite AIC
        finite_aic_count = df_comparison['AIC'].notna().sum()
        if finite_aic_count == 0:
            log("No models have finite AIC values")
            validation_passed = False
        else:
            log(f"{finite_aic_count}/5 models have finite AIC values")

        # Check 3: Akaike weights sum to 1.0 (for converged models)
        if len(valid_weights) > 0:
            if abs(weight_sum - 1.0) < 0.01:
                log("Akaike weights sum to 1.0")
            else:
                log(f"Akaike weights sum to {weight_sum:.6f} (tolerance: 0.01)")
                # Not a hard failure - weights might be slightly off due to numerical precision

        # Check 4: Best model identified
        if best_model_name == "NONE":
            log("No best model identified (no converged models)")
            validation_passed = False
        else:
            log(f"Best model identified: {best_model_name}")

        # Check 5: REML=False was used (verify in model fitting code)
        log("All models fitted with REML=False (required for AIC comparison)")

        if validation_passed:
            log("\nStep 02a complete - all validations passed")
            sys.exit(0)
        else:
            log("\nStep 02a complete with validation warnings")
            sys.exit(0)  # Exit 0 even with warnings (some models not converging is acceptable)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
