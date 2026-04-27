"""
RQ 6.1.1 - Step 05b: Model Averaging for Overall Confidence Trajectories

PURPOSE:
Implement Burnham & Anderson (2002) model averaging across competitive models
(ΔAIC < 7) to properly characterize confidence decline trajectories accounting
for model uncertainty (best model has only 21.7% weight).

INPUT:
- data/step04_lmm_input.csv (400 rows: 100 participants × 4 tests)
- data/step05_model_comparison.csv (65 models with AIC, weights)

OUTPUT:
- data/step05b_competitive_models.csv (models with ΔAIC < 7)
- data/step05b_model_averaged_predictions.csv (MA predictions)
- data/step05b_model_averaged_theta.csv (MA theta for derivative RQs)
- data/step05b_model_averaged_random_effects.csv (for ICC/clustering)
- data/step05b_metadata.csv (summary statistics)

METHODOLOGY:
- ΔAIC threshold: 7 (includes models with weak-to-substantial support)
- Weight renormalization among competitive models
- Model-averaged predictions: ŷ_MA = Σ w_i * ŷ_i
- Model-averaged random effects for derivative RQs (6.1.2-6.1.5)

Author: REMEMVR Team
Date: 2025-12-13
RQ: ch6/6.1.1
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


def run_model_averaging(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    outcome_var: str = 'theta_All',
    tsvr_var: str = 'TSVR_hours',
    groups_var: str = 'UID',
    delta_aic_threshold: float = 7.0,
    include_random_slopes: bool = True,  # Need slopes for ICC derivatives
):
    """
    Run model averaging for simple trajectory models (no interaction terms).
    """
    log("=" * 70)
    log("MODEL AVERAGING FOR CONFIDENCE TRAJECTORIES")
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
    data_trans = _create_transformations(data, tsvr_var)
    log(f"  ✓ {len([c for c in data_trans.columns if 'TSVR' in c])} time transformations created")

    # Step 2: Fit each competitive model and collect predictions/random effects
    log(f"\n[STEP 3] Fitting {n_competitive} models...")
    predictions_matrix = []
    intercepts_by_model = {}
    slopes_by_model = {}
    weights = []
    model_names = []

    for idx, row in competitive.iterrows():
        model_name = row['model_name']
        weight = row['renorm_weight']

        try:
            # Get formula
            formula = _build_formula(model_name, outcome_var, data_trans)

            # Random effects structure
            if include_random_slopes:
                primary_time = _get_primary_time_term(model_name)
                if primary_time and primary_time in data_trans.columns:
                    re_formula = f"1 + {primary_time}"
                else:
                    re_formula = "1"
            else:
                re_formula = "1"

            # Fit model
            model = smf.mixedlm(
                formula,
                data_trans,
                groups=data_trans[groups_var],
                re_formula=re_formula
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(reml=False, method='powell')

            # Get predictions WITH random effects
            # CRITICAL FIX: Use fittedvalues instead of predict() to include random effects
            # predict() only gives fixed effects, fittedvalues includes both fixed + random
            preds = fitted.fittedvalues
            predictions_matrix.append(preds.values)

            # Extract random effects
            re_df = pd.DataFrame(fitted.random_effects).T
            re_df.index.name = groups_var
            re_df = re_df.reset_index()

            # Store intercepts
            if 'Group' in re_df.columns:
                re_df = re_df.rename(columns={'Group': 'Intercept'})
            intercept_col = [c for c in re_df.columns if c != groups_var][0]
            intercepts_by_model[model_name] = re_df[[groups_var, intercept_col]].rename(
                columns={intercept_col: 'Intercept'}
            )

            # Store slopes if present
            if re_df.shape[1] > 2:
                slope_col = [c for c in re_df.columns if c not in [groups_var, intercept_col]][0]
                slopes_by_model[model_name] = re_df[[groups_var, slope_col]].rename(
                    columns={slope_col: 'Slope'}
                )

            weights.append(weight)
            model_names.append(model_name)
            slope_status = "with slope" if model_name in slopes_by_model else "intercept only"
            log(f"  [{len(model_names):2d}/{n_competitive}] {model_name:25s} w={weight:.4f} ({slope_status}) ✓")

        except Exception as e:
            log(f"  [{idx+1:2d}/{n_competitive}] {model_name:25s} FAILED: {str(e)[:50]}")
            continue

    if len(model_names) == 0:
        raise ValueError("All models failed to fit")

    # Renormalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    log(f"\n[STEP 4] Computing model-averaged predictions...")
    predictions_matrix = np.array(predictions_matrix)

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

    uids = intercepts_by_model[model_names[0]][groups_var].values
    n_uids = len(uids)

    # Compute weighted average of intercepts
    ma_intercepts = np.zeros(n_uids)
    for model_name, weight in zip(model_names, weights):
        int_df = intercepts_by_model[model_name].set_index(groups_var)
        ma_intercepts += weight * int_df.loc[uids, 'Intercept'].values

    # Compute weighted average of slopes (only from models that have them)
    ma_slopes = None
    if len(slopes_by_model) > 0:
        ma_slopes = np.zeros(n_uids)
        slope_weight_sum = 0
        for model_name, weight in zip(model_names, weights):
            if model_name in slopes_by_model:
                slope_df = slopes_by_model[model_name].set_index(groups_var)
                ma_slopes += weight * slope_df.loc[uids, 'Slope'].values
                slope_weight_sum += weight
        # Renormalize slopes
        if slope_weight_sum > 0:
            ma_slopes = ma_slopes / slope_weight_sum
            log(f"  ✓ Slopes from {len(slopes_by_model)} models (weight sum = {slope_weight_sum:.2%})")

    log(f"  ✓ Random intercepts: mean={ma_intercepts.mean():.4f}, SD={ma_intercepts.std():.4f}")
    if ma_slopes is not None:
        log(f"  ✓ Random slopes: mean={ma_slopes.mean():.6f}, SD={ma_slopes.std():.6f}")

    # Effective N models
    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))
    log(f"  ✓ Effective N models: {effective_n:.2f}")

    # Prepare outputs
    log(f"\n[STEP 6] Preparing output files...")

    # Predictions with original data
    pred_df = data.copy()
    pred_df['ma_prediction'] = ma_predictions
    pred_df['ma_pred_variance'] = uncond_var
    pred_df['ma_pred_se'] = np.sqrt(uncond_var)

    # Random effects
    re_output = pd.DataFrame({
        groups_var: uids,
        'ma_intercept': ma_intercepts,
    })
    if ma_slopes is not None:
        re_output['ma_slope'] = ma_slopes

    # Model-averaged theta
    theta_df = data[['composite_ID', 'UID', 'test', 'TSVR_hours']].copy()
    theta_df['ma_theta'] = ma_predictions
    theta_df['ma_theta_se'] = np.sqrt(uncond_var)

    # Metadata
    meta = pd.DataFrame([{
        'n_competitive_models': n_competitive,
        'n_models_fitted': len(model_names),
        'n_models_with_slopes': len(slopes_by_model),
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
        'intercept_sd': ma_intercepts.std(),
        'slope_sd': ma_slopes.std() if ma_slopes is not None else None,
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
        log("[START] Step 05b: Model Averaging for RQ 6.1.1")
        log("=" * 80)

        # Load data
        log("\n[LOAD] Loading input data...")
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv")
        comparison = pd.read_csv(DATA_DIR / "step05_model_comparison.csv")

        log(f"  ✓ LMM input: {len(lmm_input)} rows")
        log(f"  ✓ Model comparison: {len(comparison)} models")

        # Run model averaging
        results = run_model_averaging(
            data=lmm_input,
            comparison=comparison,
            delta_aic_threshold=7.0,
            include_random_slopes=True,  # Need for ICC derivatives
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
        if 'ma_slope' in results['random_effects'].columns:
            log(f"      (includes slopes from {results['metadata']['n_models_with_slopes'].iloc[0]} models)")

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
