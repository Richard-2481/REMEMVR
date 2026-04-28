#!/usr/bin/env python3
"""Fit LMM with 5 Candidate Time Transformations (Model Selection): Fit 5 candidate Linear Mixed Models with LocationType × Time interactions using"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: LMM Model Selection (5 Candidate Time Transformations)")
        # Load LMM Input Data

        log("Loading LMM input data...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv", encoding='utf-8')
        log(f"step04_lmm_input.csv ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Verify required columns present
        required_cols = ['UID', 'Days', 'log_Days_plus1', 'Days_squared', 'LocationType', 'theta']
        missing_cols = [col for col in required_cols if col not in lmm_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")

        # Log data summary
        log(f"Data summary:")
        log(f"  - UIDs: {lmm_input['UID'].nunique()}")
        log(f"  - Location types: {lmm_input['LocationType'].unique().tolist()}")
        log(f"  - Location type counts: {lmm_input['LocationType'].value_counts().to_dict()}")
        log(f"  - Days range: [{lmm_input['Days'].min():.2f}, {lmm_input['Days'].max():.2f}]")
        log(f"  - Theta range: [{lmm_input['theta'].min():.2f}, {lmm_input['theta'].max():.2f}]")
        # Define 5 Candidate LMM Models
        # Models test different time transformation functions to capture
        # forgetting trajectory shape:
        #   1. Linear: theta ~ Days × LocationType
        #   2. Quadratic: theta ~ (Days + Days²) × LocationType
        #   3. Logarithmic: theta ~ log(Days+1) × LocationType
        #   4. Linear+Log: theta ~ (Days + log(Days+1)) × LocationType
        #   5. Quadratic+Log: theta ~ (Days + Days² + log(Days+1)) × LocationType
        #
        # All models include:
        #   - LocationType main effect (source vs destination)
        #   - Time main effect(s) (depends on transformation)
        #   - LocationType × Time interaction (differential forgetting rates)
        #   - Random intercepts and slopes by UID (individual differences)
        #
        # REML=False required for valid AIC comparison (ML estimation)

        log("Defining 5 candidate models...")

        candidate_models = [
            {
                'name': 'Linear',
                'formula': 'theta ~ Days * LocationType',
                're_formula': '~Days',
                'description': 'Linear time × LocationType interaction'
            },
            {
                'name': 'Quadratic',
                'formula': 'theta ~ (Days + Days_squared) * LocationType',
                're_formula': '~Days',
                'description': 'Quadratic time (Days + Days²) × LocationType interaction'
            },
            {
                'name': 'Logarithmic',
                'formula': 'theta ~ log_Days_plus1 * LocationType',
                're_formula': '~log_Days_plus1',
                'description': 'Logarithmic time × LocationType interaction'
            },
            {
                'name': 'Linear+Logarithmic',
                'formula': 'theta ~ (Days + log_Days_plus1) * LocationType',
                're_formula': '~Days',
                'description': 'Linear + Logarithmic time × LocationType interaction'
            },
            {
                'name': 'Quadratic+Logarithmic',
                'formula': 'theta ~ (Days + Days_squared + log_Days_plus1) * LocationType',
                're_formula': '~Days',
                'description': 'Quadratic + Logarithmic time × LocationType interaction'
            }
        ]

        log(f"Candidate models defined:")
        for model_spec in candidate_models:
            log(f"  - {model_spec['name']}: {model_spec['description']}")
        # Fit All 5 Candidate Models
        # Fit each model with ML estimation (REML=False) for AIC comparison
        # Track convergence status and AIC values

        log("Fitting 5 candidate models (REML=False for AIC comparison)...")

        fitted_models = {}
        aic_values = {}
        convergence_status = {}

        for i, model_spec in enumerate(candidate_models, 1):
            model_name = model_spec['name']
            log(f"Model {i}/5: {model_name}")
            log(f"  Formula: {model_spec['formula']}")
            log(f"  Random effects: {model_spec['re_formula']}")

            try:
                # Fit MixedLM with statsmodels
                model = smf.mixedlm(
                    formula=model_spec['formula'],
                    data=lmm_input,
                    groups=lmm_input['UID'],
                    re_formula=model_spec['re_formula']
                )

                result = model.fit(reml=False)  # ML estimation for AIC

                # Store results
                fitted_models[model_name] = result
                aic_values[model_name] = result.aic
                convergence_status[model_name] = result.converged

                log(f"  AIC = {result.aic:.2f}")

            except Exception as e:
                log(f"  Model failed to fit: {str(e)}")
                fitted_models[model_name] = None
                aic_values[model_name] = np.nan
                convergence_status[model_name] = False

        log(f"All 5 models fitted")
        # Check Convergence for All Models
        # All models must converge for valid AIC comparison

        log("Verifying all models converged...")

        failed_models = [name for name, status in convergence_status.items() if not status]

        if failed_models:
            raise ValueError(f"Models failed to converge: {failed_models}")

        log("All 5 models converged successfully")
        # Compute AIC Comparison Table
        # Calculate delta_AIC and Akaike weights for model comparison
        # delta_AIC = AIC - min(AIC)
        # Akaike weight = exp(-0.5 × delta_AIC) / sum(exp(-0.5 × delta_AIC))

        log("Computing AIC comparison table...")

        # Find minimum AIC
        min_aic = min(aic_values.values())
        log(f"Minimum AIC: {min_aic:.2f}")

        # Compute delta_AIC for each model
        delta_aic = {name: aic - min_aic for name, aic in aic_values.items()}

        # Compute Akaike weights
        # weight_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j) for all j)
        exp_delta = {name: np.exp(-0.5 * delta) for name, delta in delta_aic.items()}
        sum_exp_delta = sum(exp_delta.values())
        weights = {name: exp_val / sum_exp_delta for name, exp_val in exp_delta.items()}

        # Create comparison DataFrame
        comparison_data = []
        for model_name in candidate_models:
            name = model_name['name']
            comparison_data.append({
                'model_name': name,
                'AIC': aic_values[name],
                'delta_AIC': delta_aic[name],
                'weight': weights[name]
            })

        model_comparison = pd.DataFrame(comparison_data)

        # Sort by AIC (best model first)
        model_comparison = model_comparison.sort_values('AIC').reset_index(drop=True)

        log("AIC comparison table:")
        for _, row in model_comparison.iterrows():
            log(f"  {row['model_name']:25s} AIC={row['AIC']:8.2f}  delta={row['delta_AIC']:6.2f}  weight={row['weight']:.4f}")

        # Check Akaike weights sum to 1.0
        weight_sum = model_comparison['weight'].sum()
        log(f"Akaike weights sum: {weight_sum:.6f} (should be ~1.0)")

        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Akaike weights sum to {weight_sum:.6f}, expected 1.0 +/- 0.01")
        # Select Best Model
        # Best model = lowest AIC (delta_AIC = 0)
        # Quality check: best model weight should be > 0.30 for clear winner

        log("Selecting best model...")

        best_model_name = model_comparison.iloc[0]['model_name']
        best_model_weight = model_comparison.iloc[0]['weight']
        best_model_aic = model_comparison.iloc[0]['AIC']

        log(f"{best_model_name}")
        log(f"  AIC: {best_model_aic:.2f}")
        log(f"  Weight: {best_model_weight:.4f}")

        # Quality threshold check
        if best_model_weight < 0.30:
            log(f"Best model weight ({best_model_weight:.4f}) < 0.30 (quality threshold)")
            log(f"       Model selection uncertainty is high - consider reporting multiple models")
        else:
            log(f"Best model weight ({best_model_weight:.4f}) > 0.30 (quality threshold met)")

        best_model = fitted_models[best_model_name]
        # Save Analysis Outputs
        # Save: (1) model comparison table, (2) best-fitting model (pickle),
        #       (3) summary text file

        log("Saving analysis outputs...")

        # Save model comparison table
        comparison_path = RQ_DIR / "data" / "step05_model_comparison.csv"
        model_comparison.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"{comparison_path.name} ({len(model_comparison)} rows)")

        # Save best-fitting model (pickle)
        model_path = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"
        best_model.save(str(model_path))
        log(f"{model_path.name} ({best_model_name} model)")

        # Save coefficients table separately (for Step 6 post-hoc)
        # This avoids patsy environment issues when loading pickled model
        coef_df = pd.DataFrame({
            'parameter': best_model.params.index,
            'coefficient': best_model.params.values,
            'std_error': best_model.bse.values,
            'z_value': best_model.tvalues.values,
            'p_value': best_model.pvalues.values,
            'ci_lower': best_model.conf_int().iloc[:, 0].values,
            'ci_upper': best_model.conf_int().iloc[:, 1].values
        })
        coef_path = RQ_DIR / "data" / "step05_lmm_coefficients.csv"
        coef_df.to_csv(coef_path, index=False, encoding='utf-8')
        log(f"{coef_path.name} ({len(coef_df)} parameters)")

        # Save random effects variance
        random_effects = {
            'group_var': best_model.cov_re.iloc[0, 0] if hasattr(best_model.cov_re, 'iloc') else best_model.cov_re[0, 0],
            'residual_var': best_model.scale
        }
        random_path = RQ_DIR / "data" / "step05_lmm_random_effects.csv"
        pd.DataFrame([random_effects]).to_csv(random_path, index=False, encoding='utf-8')
        log(f"{random_path.name}")

        # Save summary text file
        summary_path = RQ_DIR / "data" / "step05_lmm_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LMM MODEL SELECTION SUMMARY - RQ 5.5.1\n")
            f.write("=" * 80 + "\n\n")

            f.write("CANDIDATE MODELS (5 total):\n")
            f.write("-" * 80 + "\n")
            for model_spec in candidate_models:
                f.write(f"\n{model_spec['name']}:\n")
                f.write(f"  Formula: {model_spec['formula']}\n")
                f.write(f"  Random: {model_spec['re_formula']}\n")
                f.write(f"  Description: {model_spec['description']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("AIC COMPARISON:\n")
            f.write("=" * 80 + "\n\n")
            f.write(model_comparison.to_string(index=False))

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("BEST MODEL SELECTED:\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {best_model_name}\n")
            f.write(f"AIC: {best_model_aic:.2f}\n")
            f.write(f"Weight: {best_model_weight:.4f}\n\n")

            f.write("Model Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(str(best_model.summary()))

        log(f"{summary_path.name}")
        # Validation
        # Validate best model convergence status and comparison table quality

        log("Validating model selection results...")

        # Check 1: Best model converged
        if not best_model.converged:
            raise ValueError("Best model did not converge")
        log("Best model converged")

        # Check 2: All AIC values finite
        if not all(np.isfinite(list(aic_values.values()))):
            raise ValueError("Some AIC values are NaN or inf")
        log("All AIC values finite")

        # Check 3: Akaike weights sum to 1.0 +/- 0.01
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Akaike weights sum to {weight_sum}, expected 1.0")
        log("Akaike weights sum to 1.0")

        # Check 4: Exactly one model has delta_AIC = 0
        zero_delta_count = (model_comparison['delta_AIC'] == 0).sum()
        if zero_delta_count != 1:
            raise ValueError(f"Expected 1 model with delta_AIC=0, found {zero_delta_count}")
        log("Exactly one model has delta_AIC = 0")

        # Check 5: Best model weight > 0.30 (quality threshold)
        if best_model_weight <= 0.30:
            log(f"Best model weight ({best_model_weight:.4f}) <= 0.30")
            log(f"       This is not a validation failure, but indicates model selection uncertainty")

        log("All validation checks passed")

        log("Step 5 complete")
        log(f"Best model: {best_model_name} (AIC={best_model_aic:.2f}, weight={best_model_weight:.4f})")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
