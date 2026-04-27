"""
RQ 6.8.1 - Step 05b: Model Averaging for Source-Destination Confidence Trajectories

PURPOSE:
Implement Burnham & Anderson (2002) model averaging across competitive models
(ΔAIC < 7) to properly characterize confidence trajectories accounting for
extreme model uncertainty (best model has only 4.2% weight).

INPUT:
- data/step04_lmm_input.csv (800 rows: 100 participants × 4 tests × 2 locations)
- data/step05_model_comparison.csv (66 models with AIC, weights)

OUTPUT:
- data/step05b_competitive_models.csv (models with ΔAIC < 7)
- data/step05b_model_averaged_predictions.csv (MA predictions by location)
- data/step05b_model_averaged_theta.csv (MA theta for derivative RQs)
- data/step05b_model_averaged_random_effects.csv (for ICC/clustering derivatives)
- data/step05b_unconditional_se.csv (prediction uncertainty)
- data/step05b_metadata.csv (summary statistics)

METHODOLOGY:
- ΔAIC threshold: 7 (includes models with weak-to-substantial support)
- Weight renormalization among competitive models
- Model-averaged predictions: ŷ_MA = Σ w_i * ŷ_i
- Unconditional variance: Var(ŷ_MA) = Σ w_i * [Var(ŷ|M_i) + (ŷ_i - ŷ_MA)²]

Author: REMEMVR Team
Date: 2025-12-13
RQ: ch6/6.8.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_averaging import (
    identify_competitive_models,
    compute_unconditional_variance,
    _create_transformations,
    _build_formula,
    _get_primary_time_term,
)

RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "step05b_model_averaging.log"

# Ensure logs directory exists
(RQ_DIR / "logs").mkdir(exist_ok=True)


def log(msg):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)


def fit_model_with_location_interaction(
    model_name: str,
    data: pd.DataFrame,
    outcome_var: str = 'theta',
    groups_var: str = 'UID',
    location_var: str = 'location',
    include_random_slope: bool = False,
):
    """
    Fit a single LMM with location interaction.

    Formula pattern: theta ~ time_transform * location
    This tests main effect of time, main effect of location,
    and time × location interaction.
    """
    # Get the time predictor formula component
    time_formula = _build_formula(model_name, outcome_var, data)
    # Extract just the predictors (after ~)
    time_predictors = time_formula.split('~')[1].strip()

    # Build full formula with location interaction
    # For each time term, we need term + term:location
    terms = [t.strip() for t in time_predictors.split('+')]
    interaction_terms = []
    for term in terms:
        interaction_terms.append(term)
        interaction_terms.append(f"{term}:{location_var}")

    # Add main effect of location
    full_predictors = ' + '.join(interaction_terms) + f' + {location_var}'
    full_formula = f"{outcome_var} ~ {full_predictors}"

    # Random effects structure
    if include_random_slope:
        primary_time = _get_primary_time_term(model_name)
        re_formula = f"1 + {primary_time}"
    else:
        re_formula = "1"

    # Fit model
    model = smf.mixedlm(
        full_formula,
        data,
        groups=data[groups_var],
        re_formula=re_formula
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(reml=False, method='powell')

    return fitted, full_formula


def run_model_averaging_with_location(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    delta_aic_threshold: float = 7.0,
    include_random_slopes: bool = False,
):
    """
    Run model averaging for models with location interaction.
    """
    log("=" * 70)
    log("MODEL AVERAGING WITH LOCATION INTERACTION")
    log("=" * 70)

    # Step 1: Identify competitive models
    log(f"\n[STEP 1] Identifying competitive models (ΔAIC < {delta_aic_threshold})...")
    competitive = identify_competitive_models(
        comparison,
        delta_aic_threshold=delta_aic_threshold
    )
    n_competitive = len(competitive)
    log(f"  ✓ {n_competitive} competitive models identified")
    log(f"  ✓ Total original weight: {competitive['akaike_weight'].sum():.1%}")

    # Create time transformations
    log("\n[STEP 2] Creating time transformations...")
    data_trans = _create_transformations(data, 'TSVR_hours')
    log(f"  ✓ {len([c for c in data_trans.columns if 'TSVR' in c])} time transformations created")

    # Step 2: Fit each competitive model and collect predictions
    log(f"\n[STEP 3] Fitting {n_competitive} models with location interaction...")
    predictions_matrix = []
    random_effects_list = []
    weights = []
    model_names = []
    fitted_models = {}

    for idx, row in competitive.iterrows():
        model_name = row['model_name']
        weight = row['renorm_weight']

        try:
            fitted, formula = fit_model_with_location_interaction(
                model_name=model_name,
                data=data_trans,
                include_random_slope=include_random_slopes,
            )

            # Get predictions
            preds = fitted.predict(data_trans)
            predictions_matrix.append(preds.values)

            # Extract random effects
            re_df = pd.DataFrame(fitted.random_effects).T
            re_df.index.name = 'UID'
            re_df = re_df.reset_index()
            random_effects_list.append((model_name, weight, re_df))

            weights.append(weight)
            model_names.append(model_name)
            fitted_models[model_name] = fitted

            log(f"  [{len(model_names):2d}/{n_competitive}] {model_name:25s} w={weight:.4f} ✓")

        except Exception as e:
            log(f"  [{idx+1:2d}/{n_competitive}] {model_name:25s} FAILED: {str(e)[:50]}")
            continue

    if len(model_names) == 0:
        raise ValueError("All models failed to fit")

    # Renormalize weights for fitted models
    weights = np.array(weights)
    weights = weights / weights.sum()

    log(f"\n[STEP 4] Computing model-averaged predictions...")
    predictions_matrix = np.array(predictions_matrix)  # (n_models, n_obs)

    # Model-averaged predictions
    ma_predictions = np.average(predictions_matrix, axis=0, weights=weights)

    # Unconditional variance
    uncond_var = compute_unconditional_variance(
        predictions_matrix, ma_predictions, weights
    )

    log(f"  ✓ Predictions: mean={ma_predictions.mean():.4f}, SD={ma_predictions.std():.4f}")
    log(f"  ✓ Uncond. variance: mean={uncond_var.mean():.6f}, max={uncond_var.max():.6f}")

    # Step 4: Model-averaged random effects
    log(f"\n[STEP 5] Computing model-averaged random effects...")

    # Get UIDs from first model's random effects
    uids = random_effects_list[0][2]['UID'].values
    n_uids = len(uids)

    # Compute weighted average of intercepts
    ma_intercepts = np.zeros(n_uids)
    for model_name, orig_weight, re_df in random_effects_list:
        # Find this model's index in fitted list
        model_idx = model_names.index(model_name)
        weight = weights[model_idx]

        re_df_sorted = re_df.set_index('UID').loc[uids]
        # First column after UID is intercept (named 'Group' or similar)
        intercept_col = re_df_sorted.columns[0]
        ma_intercepts += weight * re_df_sorted[intercept_col].values

    log(f"  ✓ Random intercepts: mean={ma_intercepts.mean():.4f}, SD={ma_intercepts.std():.4f}")

    # Effective N models
    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))
    log(f"  ✓ Effective N models: {effective_n:.2f}")

    # Prepare output DataFrames
    log(f"\n[STEP 6] Preparing output files...")

    # Predictions with original data
    pred_df = data.copy()
    pred_df['ma_prediction'] = ma_predictions
    pred_df['ma_pred_variance'] = uncond_var
    pred_df['ma_pred_se'] = np.sqrt(uncond_var)

    # Random effects
    re_output = pd.DataFrame({
        'UID': uids,
        'ma_intercept': ma_intercepts,
    })

    # Model-averaged theta (same as predictions for this RQ)
    theta_df = data[['composite_ID', 'UID', 'TEST', 'location', 'TSVR_hours']].copy()
    theta_df['ma_theta'] = ma_predictions
    theta_df['ma_theta_se'] = np.sqrt(uncond_var)

    # Metadata
    meta = pd.DataFrame([{
        'n_competitive_models': n_competitive,
        'n_models_fitted': len(model_names),
        'delta_aic_threshold': delta_aic_threshold,
        'effective_n_models': effective_n,
        'total_original_weight': competitive['akaike_weight'].sum(),
        'top_model': competitive.iloc[0]['model_name'],
        'top_model_renorm_weight': weights[0],
        'top_model_original_weight': competitive.iloc[0]['akaike_weight'],
        'prediction_mean': ma_predictions.mean(),
        'prediction_std': ma_predictions.std(),
        'max_model_variance': uncond_var.max(),
        'mean_model_variance': uncond_var.mean(),
    }])

    return {
        'competitive_models': competitive,
        'predictions': pred_df,
        'theta': theta_df,
        'random_effects': re_output,
        'metadata': meta,
        'weights': dict(zip(model_names, weights)),
        'effective_n_models': effective_n,
    }


if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 05b: Model Averaging for RQ 6.8.1")
        log("=" * 80)

        # Load data
        log("\n[LOAD] Loading input data...")
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv")
        comparison = pd.read_csv(DATA_DIR / "step05_model_comparison.csv")

        log(f"  ✓ LMM input: {len(lmm_input)} rows")
        log(f"  ✓ Model comparison: {len(comparison)} models")
        log(f"  ✓ Best model: {comparison.iloc[0]['model_name']} (weight={comparison.iloc[0]['akaike_weight']:.1%})")

        # Run model averaging
        results = run_model_averaging_with_location(
            data=lmm_input,
            comparison=comparison,
            delta_aic_threshold=7.0,
            include_random_slopes=False,  # Start with intercepts only
        )

        # Save outputs
        log("\n[SAVE] Saving output files...")

        results['competitive_models'].to_csv(
            DATA_DIR / "step05b_competitive_models.csv", index=False
        )
        log(f"  ✓ step05b_competitive_models.csv ({len(results['competitive_models'])} models)")

        results['predictions'].to_csv(
            DATA_DIR / "step05b_model_averaged_predictions.csv", index=False
        )
        log(f"  ✓ step05b_model_averaged_predictions.csv ({len(results['predictions'])} rows)")

        results['theta'].to_csv(
            DATA_DIR / "step05b_model_averaged_theta.csv", index=False
        )
        log(f"  ✓ step05b_model_averaged_theta.csv ({len(results['theta'])} rows)")

        results['random_effects'].to_csv(
            DATA_DIR / "step05b_model_averaged_random_effects.csv", index=False
        )
        log(f"  ✓ step05b_model_averaged_random_effects.csv ({len(results['random_effects'])} UIDs)")

        results['metadata'].to_csv(
            DATA_DIR / "step05b_metadata.csv", index=False
        )
        log(f"  ✓ step05b_metadata.csv")

        # Summary
        log("\n" + "=" * 80)
        log("[SUCCESS] Model Averaging Complete")
        log("=" * 80)
        log(f"  Models in competitive set: {len(results['competitive_models'])}")
        log(f"  Effective N models: {results['effective_n_models']:.2f}")
        log(f"  Top 3 models by renormalized weight:")
        for i, (name, w) in enumerate(list(results['weights'].items())[:3]):
            log(f"    {i+1}. {name}: {w:.3f}")
        log("=" * 80)

    except Exception as e:
        log(f"\n[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
