"""
RQ 6.4.1 - Step 05b: Model Averaging for Paradigm-Specific Confidence Trajectories

PURPOSE:
Implement Burnham & Anderson (2002) model averaging for paradigm (IFR/ICR/IRE)
confidence trajectories with interaction terms.

INPUT:
- data/step04_lmm_input.csv (1200 rows: 100 participants × 4 tests × 3 paradigms)
- data/step05_model_comparison.csv

INTERACTION FACTOR: paradigm (IFR, ICR, IRE)

NOTE: 6.4.1 has concentrated weights (Linear/Exponential_proxy tied at 50% each)
so model averaging will have limited impact, but ensures consistency.

Author: REMEMVR Team
Date: 2025-12-13
RQ: ch6/6.4.1
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
)

RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "step05b_model_averaging.log"
(RQ_DIR / "logs").mkdir(exist_ok=True)


def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)


def fit_model_with_factor_interaction(
    model_name: str,
    data: pd.DataFrame,
    outcome_var: str = 'theta',
    factor_var: str = 'paradigm',
    groups_var: str = 'UID',
):
    """Fit LMM with factor interaction: theta ~ time_transform * factor."""
    time_formula = _build_formula(model_name, outcome_var, data)
    time_predictors = time_formula.split('~')[1].strip()

    terms = [t.strip() for t in time_predictors.split('+')]
    interaction_terms = []
    for term in terms:
        interaction_terms.append(term)
        interaction_terms.append(f"{term}:{factor_var}")

    full_predictors = ' + '.join(interaction_terms) + f' + {factor_var}'
    full_formula = f"{outcome_var} ~ {full_predictors}"

    model = smf.mixedlm(
        full_formula,
        data,
        groups=data[groups_var],
        re_formula="1"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(reml=False, method='powell')

    return fitted, full_formula


def run_model_averaging_with_factor(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    factor_var: str = 'paradigm',
    outcome_var: str = 'theta',
    tsvr_var: str = 'TSVR_hours',
    groups_var: str = 'UID',
    delta_aic_threshold: float = 7.0,
):
    """Run model averaging for models with factor interaction."""
    log("=" * 70)
    log(f"MODEL AVERAGING WITH {factor_var.upper()} INTERACTION")
    log("=" * 70)

    log(f"\n[STEP 1] Identifying competitive models (ΔAIC < {delta_aic_threshold})...")
    competitive = identify_competitive_models(comparison, delta_aic_threshold=delta_aic_threshold)
    n_competitive = len(competitive)
    log(f"  ✓ {n_competitive} competitive models identified")

    log("\n[STEP 2] Creating time transformations...")
    data_trans = _create_transformations(data, tsvr_var)

    log(f"\n[STEP 3] Fitting {n_competitive} models with {factor_var} interaction...")
    predictions_matrix = []
    weights = []
    model_names = []

    for idx, row in competitive.iterrows():
        model_name = row['model_name']
        weight = row['renorm_weight']

        try:
            fitted, formula = fit_model_with_factor_interaction(
                model_name=model_name,
                data=data_trans,
                outcome_var=outcome_var,
                factor_var=factor_var,
                groups_var=groups_var,
            )
            preds = fitted.predict(data_trans)
            predictions_matrix.append(preds.values)
            weights.append(weight)
            model_names.append(model_name)
            log(f"  [{len(model_names):2d}/{n_competitive}] {model_name:25s} w={weight:.4f} ✓")
        except Exception as e:
            log(f"  [{idx+1:2d}/{n_competitive}] {model_name:25s} FAILED: {str(e)[:50]}")
            continue

    if len(model_names) == 0:
        raise ValueError("All models failed to fit")

    weights = np.array(weights)
    weights = weights / weights.sum()

    log(f"\n[STEP 4] Computing model-averaged predictions...")
    predictions_matrix = np.array(predictions_matrix)
    ma_predictions = np.average(predictions_matrix, axis=0, weights=weights)
    uncond_var = compute_unconditional_variance(predictions_matrix, ma_predictions, weights)

    log(f"  ✓ Predictions: mean={ma_predictions.mean():.4f}, SD={ma_predictions.std():.4f}")
    log(f"  ✓ Uncond. variance: mean={uncond_var.mean():.6f}, max={uncond_var.max():.6f}")

    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))
    log(f"  ✓ Effective N models: {effective_n:.2f}")

    pred_df = data.copy()
    pred_df['ma_prediction'] = ma_predictions
    pred_df['ma_pred_variance'] = uncond_var
    pred_df['ma_pred_se'] = np.sqrt(uncond_var)

    theta_df = data[['composite_ID', 'UID', 'test', factor_var, 'TSVR_hours']].copy()
    theta_df['ma_theta'] = ma_predictions
    theta_df['ma_theta_se'] = np.sqrt(uncond_var)

    meta = pd.DataFrame([{
        'n_competitive_models': n_competitive,
        'n_models_fitted': len(model_names),
        'delta_aic_threshold': delta_aic_threshold,
        'effective_n_models': effective_n,
        'total_original_weight': competitive['akaike_weight'].sum(),
        'top_model': competitive.iloc[0]['model_name'],
        'top_model_renorm_weight': weights[0],
        'prediction_mean': ma_predictions.mean(),
        'prediction_std': ma_predictions.std(),
        'factor_variable': factor_var,
    }])

    return {
        'competitive_models': competitive,
        'predictions': pred_df,
        'theta': theta_df,
        'metadata': meta,
        'weights': dict(zip(model_names, weights)),
        'effective_n_models': effective_n,
    }


if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 05b: Model Averaging for RQ 6.4.1 (Paradigm)")
        log("=" * 80)

        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv")
        comparison = pd.read_csv(DATA_DIR / "step05_model_comparison.csv")

        log(f"\nLMM input: {len(lmm_input)} rows, {lmm_input['paradigm'].nunique()} paradigms")
        log(f"  Best model: {comparison.iloc[0]['model_name']} (weight={comparison.iloc[0]['akaike_weight']:.1%})")

        results = run_model_averaging_with_factor(
            data=lmm_input,
            comparison=comparison,
            factor_var='paradigm',
            delta_aic_threshold=7.0,
        )

        log("\nSaving output files...")
        results['competitive_models'].to_csv(DATA_DIR / "step05b_competitive_models.csv", index=False)
        results['predictions'].to_csv(DATA_DIR / "step05b_model_averaged_predictions.csv", index=False)
        results['theta'].to_csv(DATA_DIR / "step05b_model_averaged_theta.csv", index=False)
        results['metadata'].to_csv(DATA_DIR / "step05b_metadata.csv", index=False)

        log(f"\nModel averaging complete: {len(results['competitive_models'])} models, effective N = {results['effective_n_models']:.2f}")

    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
