"""
Model Averaging for LMM Trajectory Analysis

Implements Burnham & Anderson (2002) multi-model inference approach.
When best model has low Akaike weight (<30%), model averaging provides
more robust predictions accounting for model selection uncertainty.

Author: REMEMVR Team
Date: 2025-12-08
Updated: 2025-12-13 - Added functions for complete model averaging workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statsmodels.formula.api as smf
import warnings


def identify_competitive_models(
    comparison_df: pd.DataFrame,
    delta_aic_threshold: float = 7.0,
    min_weight: float = 0.001,
) -> pd.DataFrame:
    """
    Identify competitive models using ΔAIC threshold.

    Following Burnham & Anderson (2002):
    - ΔAIC < 2: Substantial support
    - ΔAIC 2-4: Moderate support
    - ΔAIC 4-7: Weak support
    - ΔAIC > 7: Essentially no support

    Parameters
    ----------
    comparison_df : DataFrame
        Model comparison table with columns: model_name, AIC, delta_AIC, akaike_weight
    delta_aic_threshold : float
        Include models with ΔAIC < threshold (default=7.0 for conservative inclusion)
    min_weight : float
        Minimum Akaike weight to include (filters numerical noise)

    Returns
    -------
    DataFrame
        Competitive models with renormalized weights (renorm_weight column added)
    """
    # Filter by ΔAIC threshold
    competitive = comparison_df[
        (comparison_df['delta_AIC'] < delta_aic_threshold) &
        (comparison_df['akaike_weight'] >= min_weight)
    ].copy()

    if len(competitive) == 0:
        raise ValueError(f"No models with ΔAIC < {delta_aic_threshold}")

    # Renormalize weights to sum to 1.0
    total_weight = competitive['akaike_weight'].sum()
    competitive['renorm_weight'] = competitive['akaike_weight'] / total_weight

    # Compute effective number of models (Shannon entropy-based)
    weights = competitive['renorm_weight'].values
    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))

    print(f"[COMPETITIVE MODELS] ΔAIC < {delta_aic_threshold}")
    print(f"  Models included: {len(competitive)}")
    print(f"  Total original weight: {total_weight:.1%}")
    print(f"  Effective N models: {effective_n:.2f}")
    print(f"  Top 3 models:")
    for i, (_, row) in enumerate(competitive.head(3).iterrows()):
        print(f"    {i+1}. {row['model_name']:20s} w={row['renorm_weight']:.3f} (ΔAIC={row['delta_AIC']:.2f})")

    return competitive


def compute_unconditional_variance(
    predictions_matrix: np.ndarray,
    model_averaged_pred: np.ndarray,
    weights: np.ndarray,
    within_model_variances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute unconditional variance incorporating model selection uncertainty.

    Burnham & Anderson (2002) Equation 4.9:
    Var_unconditional = Σ w_i * [Var(ŷ|M_i) + (ŷ_i - ŷ_MA)²]

    Parameters
    ----------
    predictions_matrix : ndarray
        Shape (n_models, n_obs) - predictions from each model
    model_averaged_pred : ndarray
        Shape (n_obs,) - model-averaged predictions
    weights : ndarray
        Shape (n_models,) - renormalized Akaike weights
    within_model_variances : ndarray, optional
        Shape (n_models, n_obs) - prediction variance from each model
        If None, only between-model variance is computed

    Returns
    -------
    ndarray
        Shape (n_obs,) - unconditional variance at each observation
    """
    n_models, n_obs = predictions_matrix.shape

    # Between-model variance: weighted variance of predictions across models
    # This captures model selection uncertainty
    between_var = np.zeros(n_obs)
    for i in range(n_models):
        deviation_sq = (predictions_matrix[i, :] - model_averaged_pred) ** 2
        between_var += weights[i] * deviation_sq

    # Total unconditional variance
    if within_model_variances is not None:
        # Full Burnham & Anderson formula
        within_var = np.zeros(n_obs)
        for i in range(n_models):
            within_var += weights[i] * within_model_variances[i, :]
        unconditional_var = within_var + between_var
    else:
        # Simplified: only between-model variance (model selection uncertainty)
        unconditional_var = between_var

    return unconditional_var


def compute_model_averaged_predictions(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    outcome_var: str,
    tsvr_var: str,
    groups_var: str,
    delta_aic_threshold: float = 2.0,
    prediction_grid: pd.DataFrame = None,
    reml: bool = False,
) -> Dict:
    """
    Compute model-averaged predictions using Akaike weights.
    
    Parameters
    ----------
    data : DataFrame
        Training data with all time transformations
    comparison : DataFrame
        Model comparison table from kitchen_sink (has model_name, akaike_weight)
    outcome_var : str
        Outcome variable name (e.g., 'theta')
    tsvr_var : str
        Continuous TSVR variable (e.g., 'TSVR_hours')
    groups_var : str
        Grouping variable for random effects (e.g., 'UID')
    delta_aic_threshold : float
        Only average models with ΔAIC < threshold (default=2.0)
    prediction_grid : DataFrame, optional
        New data for predictions. If None, uses training data.
    reml : bool
        Use REML (True) or ML (False) for final fits
        
    Returns
    -------
    dict
        {
            'averaged_predictions': Series with model-averaged predictions,
            'models_used': list of model names included,
            'weights_normalized': dict of renormalized weights,
            'prediction_variance': Series with prediction uncertainty,
            'effective_n_models': float (effective number of models),
        }
    """
    
    print("[MODEL AVERAGING] Starting multi-model inference...")
    
    # Filter to competitive models
    competitive = comparison[comparison['delta_AIC'] < delta_aic_threshold].copy()
    print(f"  Competitive models (ΔAIC < {delta_aic_threshold}): {len(competitive)}")
    
    if len(competitive) == 0:
        raise ValueError(f"No models with ΔAIC < {delta_aic_threshold}")
    
    # Renormalize weights (sum competitive weights to 1.0)
    competitive['weight_normalized'] = (
        competitive['akaike_weight'] / competitive['akaike_weight'].sum()
    )
    
    print(f"  Total weight of competitive set: {competitive['akaike_weight'].sum():.1%}")
    print(f"  Renormalized to 100% across {len(competitive)} models")
    
    # Create time transformations
    data_trans = _create_transformations(data, tsvr_var)
    if prediction_grid is not None:
        pred_trans = _create_transformations(prediction_grid, tsvr_var)
    else:
        pred_trans = data_trans
    
    # Fit models and collect predictions
    predictions_matrix = []
    weights = []
    model_names = []
    
    for idx, row in competitive.iterrows():
        model_name = row['model_name']
        weight = row['weight_normalized']
        
        try:
            # Get formula for this model
            formula = _build_formula(model_name, outcome_var, data_trans)
            
            # Fit model
            model = smf.mixedlm(formula, data_trans, groups=data_trans[groups_var])
            fitted = model.fit(reml=reml, method='powell')
            
            # Get predictions
            preds = fitted.predict(pred_trans)
            
            predictions_matrix.append(preds.values)
            weights.append(weight)
            model_names.append(model_name)
            
            print(f"    [{idx+1}/{len(competitive)}] {model_name:20s} w={weight:.4f} ✓")
            
        except Exception as e:
            print(f"    [{idx+1}/{len(competitive)}] {model_name:20s} FAILED: {e}")
            continue
    
    if len(predictions_matrix) == 0:
        raise ValueError("All models failed to fit")
    
    # Convert to array
    predictions_matrix = np.array(predictions_matrix)  # (n_models, n_obs)
    weights = np.array(weights)
    
    # Renormalize weights (some models may have failed)
    weights = weights / weights.sum()
    
    # Compute model-averaged predictions
    averaged = np.average(predictions_matrix, axis=0, weights=weights)
    
    # Compute prediction variance (unconditional variance)
    # Var(avg) = E[Var(Y|model)] + Var(E[Y|model])
    # First term: within-model variance (ignore for now, constant SE)
    # Second term: between-model variance
    model_means = predictions_matrix
    grand_mean = averaged
    between_var = np.average((model_means - grand_mean[np.newaxis, :])**2, 
                             axis=0, weights=weights)
    
    # Effective number of models (Shannon diversity)
    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))
    
    print(f"\nModel averaging complete:")
    print(f"  Models used: {len(model_names)}/{len(competitive)}")
    print(f"  Effective N models: {effective_n:.2f}")
    print(f"  Prediction variance range: [{np.min(between_var):.4f}, {np.max(between_var):.4f}]")
    
    return {
        'averaged_predictions': pd.Series(averaged, index=pred_trans.index),
        'models_used': model_names,
        'weights_normalized': dict(zip(model_names, weights)),
        'prediction_variance': pd.Series(between_var, index=pred_trans.index),
        'effective_n_models': effective_n,
    }


def _create_transformations(df: pd.DataFrame, tsvr_var: str) -> pd.DataFrame:
    """Create all time transformations needed for model averaging."""
    df = df.copy()
    
    # Convert TSVR_hours to days
    df['TSVR'] = df[tsvr_var] / 24.0
    
    # Polynomial
    df['TSVR_sq'] = df['TSVR'] ** 2
    df['TSVR_cub'] = df['TSVR'] ** 3
    df['TSVR_4th'] = df['TSVR'] ** 4
    
    # Logarithmic
    df['log_TSVR'] = np.log(df['TSVR'] + 1)
    df['log2_TSVR'] = np.log2(df['TSVR'] + 1)
    df['log10_TSVR'] = np.log10(df['TSVR'] + 1)
    df['log_log_TSVR'] = np.log(df['log_TSVR'] + 1)
    
    # Power law
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        col_name = f'TSVR_pow_neg{int(alpha*10):02d}'
        df[col_name] = (df['TSVR'] + 1) ** (-alpha)
    
    # Roots
    df['sqrt_TSVR'] = np.sqrt(df['TSVR'])
    df['cbrt_TSVR'] = np.cbrt(df['TSVR'])
    df['TSVR_pow_025'] = df['TSVR'] ** 0.25
    df['TSVR_pow_033'] = df['TSVR'] ** 0.333
    df['TSVR_pow_067'] = df['TSVR'] ** 0.667
    
    # Reciprocal
    df['recip_TSVR'] = 1.0 / (df['TSVR'] + 1)
    df['recip_TSVR_sq'] = 1.0 / ((df['TSVR'] + 1) ** 2)
    
    # Exponential proxies
    df['neg_TSVR'] = -df['TSVR']
    df['neg_TSVR_sq'] = -(df['TSVR'] ** 2)
    df['neg_sqrt_TSVR'] = -np.sqrt(df['TSVR'])
    
    # Trigonometric
    tsvr_max = df['TSVR'].max()
    df['sin_TSVR'] = np.sin(df['TSVR'] / tsvr_max * 2 * np.pi)
    df['cos_TSVR'] = np.cos(df['TSVR'] / tsvr_max * 2 * np.pi)
    
    # Hyperbolic
    tsvr_norm = df['TSVR'] / tsvr_max
    df['tanh_TSVR'] = np.tanh(tsvr_norm)
    df['arctanh_TSVR'] = np.arctanh(tsvr_norm * 0.99)
    df['sinh_TSVR'] = np.sinh(tsvr_norm)
    
    return df


def _build_formula(model_name: str, outcome: str, df: pd.DataFrame) -> str:
    """Build formula for a specific model.

    If 'domain' or 'Factor' column exists in df, includes domain fixed effects
    and domain×time interactions.
    """

    # Map model names to formulas (from kitchen_sink model definitions)
    formulas = {
        'Linear': 'TSVR',
        'Quadratic': 'TSVR + TSVR_sq',
        'Cubic': 'TSVR + TSVR_sq + TSVR_cub',
        'Quartic': 'TSVR + TSVR_sq + TSVR_cub + TSVR_4th',
        'Quadratic_pure': 'TSVR_sq',
        'Cubic_pure': 'TSVR_cub',
        'Log': 'log_TSVR',
        'Log2': 'log2_TSVR',
        'Log10': 'log10_TSVR',
        'LogLog': 'log_log_TSVR',
        'Lin+Log': 'TSVR + log_TSVR',
        'Quad+Log': 'TSVR + TSVR_sq + log_TSVR',
        'Log+LogLog': 'log_TSVR + log_log_TSVR',
        'Lin+Quad+Log': 'TSVR + TSVR_sq + log_TSVR',
        'PowerLaw_01': 'TSVR_pow_neg01',
        'PowerLaw_02': 'TSVR_pow_neg02',
        'PowerLaw_03': 'TSVR_pow_neg03',
        'PowerLaw_04': 'TSVR_pow_neg04',
        'PowerLaw_05': 'TSVR_pow_neg05',
        'PowerLaw_06': 'TSVR_pow_neg06',
        'PowerLaw_07': 'TSVR_pow_neg07',
        'PowerLaw_08': 'TSVR_pow_neg08',
        'PowerLaw_09': 'TSVR_pow_neg09',
        'PowerLaw_10': 'TSVR_pow_neg10',
        'PowerLaw_Log': 'TSVR_pow_neg05 + log_TSVR',
        'PowerLaw_Lin': 'TSVR_pow_neg05 + TSVR',
        'SquareRoot': 'sqrt_TSVR',
        'CubeRoot': 'cbrt_TSVR',
        'FourthRoot': 'TSVR_pow_025',
        'Root_033': 'TSVR_pow_033',
        'Root_067': 'TSVR_pow_067',
        'SquareRoot+Log': 'sqrt_TSVR + log_TSVR',
        'CubeRoot+Log': 'cbrt_TSVR + log_TSVR',
        'SquareRoot+Lin': 'sqrt_TSVR + TSVR',
        'Root_Multi': 'sqrt_TSVR + cbrt_TSVR',
        'Reciprocal': 'recip_TSVR',
        'Recip+Log': 'recip_TSVR + log_TSVR',
        'Recip+Lin': 'recip_TSVR + TSVR',
        'Recip+Quad': 'recip_TSVR + TSVR + TSVR_sq',
        'Recip_sq': 'recip_TSVR_sq',
        'Recip+PowerLaw': 'recip_TSVR + TSVR_pow_neg05',
        'Exponential_proxy': 'neg_TSVR',
        'Exp+Log': 'neg_TSVR + log_TSVR',
        'Exp+Lin': 'neg_TSVR + TSVR',
        'Exp_fast': 'neg_TSVR_sq',
        'Exp_slow': 'neg_sqrt_TSVR',
        'Exp+PowerLaw': 'neg_TSVR + TSVR_pow_neg05',
        'Exp+Recip': 'neg_TSVR + recip_TSVR',
        'Sin': 'sin_TSVR',
        'Cos': 'cos_TSVR',
        'Sin+Cos': 'sin_TSVR + cos_TSVR',
        'Sin+Log': 'sin_TSVR + log_TSVR',
        'Tanh': 'tanh_TSVR',
        'Tanh+Log': 'tanh_TSVR + log_TSVR',
        'Arctanh': 'arctanh_TSVR',
        'Sinh': 'sinh_TSVR',
        'Log+PowerLaw05': 'log_TSVR + TSVR_pow_neg05',
        'Log+SquareRoot': 'log_TSVR + sqrt_TSVR',
        'Log+Recip': 'log_TSVR + recip_TSVR',
        'SquareRoot+PowerLaw': 'sqrt_TSVR + TSVR_pow_neg05',
        'SquareRoot+Recip': 'sqrt_TSVR + recip_TSVR',
        'Recip+PowerLaw05': 'recip_TSVR + TSVR_pow_neg05',
        'Lin+Log+PowerLaw': 'TSVR + log_TSVR + TSVR_pow_neg05',
        'Quad+Log+SquareRoot': 'TSVR + TSVR_sq + log_TSVR + sqrt_TSVR',
        'PowerLaw+Recip+Log': 'TSVR_pow_neg05 + recip_TSVR + log_TSVR',
        'Ultimate': 'TSVR + TSVR_sq + log_TSVR + sqrt_TSVR + TSVR_pow_neg05 + recip_TSVR',
    }
    
    if model_name not in formulas:
        raise ValueError(f"Unknown model: {model_name}")

    time_formula = formulas[model_name]

    # Check if domain/factor/paradigm/congruence column exists (factor-specific analysis)
    domain_col = None
    if 'domain' in df.columns:
        domain_col = 'domain'
    elif 'Factor' in df.columns:
        domain_col = 'Factor'
    elif 'paradigm' in df.columns:
        domain_col = 'paradigm'
    elif 'congruence' in df.columns:
        domain_col = 'congruence'

    if domain_col is not None:
        # Include domain main effects and domain×time interactions
        # Use treatment coding with first level as reference
        domain_term = f"C({domain_col}, Treatment)"

        # Get time terms from formula
        time_terms = time_formula.split(' + ')

        # Build formula with domain effects
        # Main effects: outcome ~ time_terms + domain
        # Interactions: time_terms:domain for each time term
        interaction_terms = [f"{term}:{domain_term}" for term in time_terms]

        full_formula = f"{outcome} ~ {time_formula} + {domain_term}"
        if len(interaction_terms) > 0:
            full_formula += " + " + " + ".join(interaction_terms)

        return full_formula
    else:
        # No domain column, just time effects
        return f"{outcome} ~ {time_formula}"


def compute_model_averaged_random_effects(
    data: pd.DataFrame,
    competitive_models: pd.DataFrame,
    outcome_var: str,
    tsvr_var: str,
    groups_var: str,
    include_random_slopes: bool = False,
    reml: bool = False,
) -> Dict:
    """
    Compute model-averaged random effects (BLUPs) for use in derivative analyses.

    This is essential for downstream RQs that need:
    - ICC decomposition (intercept-slope correlation)
    - Clustering (individual difference phenotypes)

    Parameters
    ----------
    data : DataFrame
        Original data with all variables
    competitive_models : DataFrame
        From identify_competitive_models() with renorm_weight column
    outcome_var : str
        Outcome variable name
    tsvr_var : str
        Time variable name
    groups_var : str
        Grouping variable for random effects
    include_random_slopes : bool
        If True, fit models with random slopes (slower but needed for ICC)
    reml : bool
        Use REML (True) or ML (False)

    Returns
    -------
    dict
        {
            'random_intercepts': DataFrame with UID and model-averaged intercept,
            'random_slopes': DataFrame with UID and model-averaged slope (if requested),
            'models_fitted': list of successfully fitted model names,
            'effective_n_models': float,
        }
    """
    print(f"\n[MODEL-AVERAGED RANDOM EFFECTS] Fitting {len(competitive_models)} models...")

    # Create time transformations
    data_trans = _create_transformations(data, tsvr_var)

    # Storage for random effects from each model
    intercepts_by_model = {}
    slopes_by_model = {}
    fitted_weights = []
    fitted_names = []

    for idx, row in competitive_models.iterrows():
        model_name = row['model_name']
        weight = row['renorm_weight']

        try:
            # Get formula
            formula = _build_formula(model_name, outcome_var, data_trans)

            # Determine random effects structure
            if include_random_slopes:
                # Get the time term from formula
                time_term = _get_primary_time_term(model_name)
                if time_term and time_term in data_trans.columns:
                    re_formula = f"1 + {time_term}"
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
                fitted = model.fit(reml=reml, method='powell')

            # Extract random effects
            re_df = pd.DataFrame(fitted.random_effects).T
            re_df.index.name = groups_var
            re_df = re_df.reset_index()

            # Store intercepts
            if 'Group' in re_df.columns:
                re_df = re_df.rename(columns={'Group': 'Intercept'})
            intercepts_by_model[model_name] = re_df[[groups_var, 'Intercept']].copy()

            # Store slopes if present
            if include_random_slopes and re_df.shape[1] > 2:
                slope_col = [c for c in re_df.columns if c not in [groups_var, 'Intercept']][0]
                slopes_by_model[model_name] = re_df[[groups_var, slope_col]].rename(
                    columns={slope_col: 'Slope'}
                )

            fitted_weights.append(weight)
            fitted_names.append(model_name)
            print(f"  [{len(fitted_names)}/{len(competitive_models)}] {model_name:25s} ✓")

        except Exception as e:
            print(f"  [{idx+1}/{len(competitive_models)}] {model_name:25s} FAILED: {str(e)[:50]}")
            continue

    if len(fitted_names) == 0:
        raise ValueError("All models failed to fit")

    # Renormalize weights
    fitted_weights = np.array(fitted_weights)
    fitted_weights = fitted_weights / fitted_weights.sum()

    # Compute model-averaged intercepts
    uids = intercepts_by_model[fitted_names[0]][groups_var].values
    ma_intercepts = np.zeros(len(uids))

    for name, weight in zip(fitted_names, fitted_weights):
        int_df = intercepts_by_model[name].set_index(groups_var)
        ma_intercepts += weight * int_df.loc[uids, 'Intercept'].values

    result_intercepts = pd.DataFrame({
        groups_var: uids,
        'ma_intercept': ma_intercepts
    })

    # Compute model-averaged slopes if available
    result_slopes = None
    if include_random_slopes and len(slopes_by_model) > 0:
        ma_slopes = np.zeros(len(uids))
        slope_weights = []
        for name, weight in zip(fitted_names, fitted_weights):
            if name in slopes_by_model:
                slope_df = slopes_by_model[name].set_index(groups_var)
                ma_slopes += weight * slope_df.loc[uids, 'Slope'].values
                slope_weights.append(weight)

        if len(slope_weights) > 0:
            # Renormalize for models that had slopes
            ma_slopes = ma_slopes / sum(slope_weights) * sum(slope_weights)
            result_slopes = pd.DataFrame({
                groups_var: uids,
                'ma_slope': ma_slopes
            })

    # Effective N models
    effective_n = np.exp(-np.sum(fitted_weights * np.log(fitted_weights + 1e-10)))

    print(f"\nModel-averaged random effects computed")
    print(f"  Models used: {len(fitted_names)}")
    print(f"  Effective N: {effective_n:.2f}")
    print(f"  Participants: {len(uids)}")
    if result_slopes is not None:
        print(f"  Random slopes: YES (from {len(slopes_by_model)} models)")
    else:
        print(f"  Random slopes: NO")

    return {
        'random_intercepts': result_intercepts,
        'random_slopes': result_slopes,
        'models_fitted': fitted_names,
        'weights': dict(zip(fitted_names, fitted_weights)),
        'effective_n_models': effective_n,
    }


def _get_primary_time_term(model_name: str) -> str:
    """Get the primary time transformation term for random slopes."""
    # Map model names to their primary time predictor
    primary_terms = {
        'Linear': 'TSVR',
        'Quadratic': 'TSVR',
        'Cubic': 'TSVR',
        'Quartic': 'TSVR',
        'Quadratic_pure': 'TSVR_sq',
        'Cubic_pure': 'TSVR_cub',
        'Log': 'log_TSVR',
        'Log2': 'log2_TSVR',
        'Log10': 'log10_TSVR',
        'LogLog': 'log_log_TSVR',
        'Lin+Log': 'TSVR',
        'Quad+Log': 'TSVR',
        'Log+LogLog': 'log_TSVR',
        'Lin+Quad+Log': 'TSVR',
        'PowerLaw_01': 'TSVR_pow_neg01',
        'PowerLaw_02': 'TSVR_pow_neg02',
        'PowerLaw_03': 'TSVR_pow_neg03',
        'PowerLaw_04': 'TSVR_pow_neg04',
        'PowerLaw_05': 'TSVR_pow_neg05',
        'PowerLaw_06': 'TSVR_pow_neg06',
        'PowerLaw_07': 'TSVR_pow_neg07',
        'PowerLaw_08': 'TSVR_pow_neg08',
        'PowerLaw_09': 'TSVR_pow_neg09',
        'PowerLaw_10': 'TSVR_pow_neg10',
        'PowerLaw_Log': 'TSVR_pow_neg05',
        'PowerLaw_Lin': 'TSVR_pow_neg05',
        'SquareRoot': 'sqrt_TSVR',
        'CubeRoot': 'cbrt_TSVR',
        'FourthRoot': 'TSVR_pow_025',
        'Root_033': 'TSVR_pow_033',
        'Root_067': 'TSVR_pow_067',
        'SquareRoot+Log': 'sqrt_TSVR',
        'CubeRoot+Log': 'cbrt_TSVR',
        'SquareRoot+Lin': 'sqrt_TSVR',
        'Root_Multi': 'sqrt_TSVR',
        'Reciprocal': 'recip_TSVR',
        'Recip+Log': 'recip_TSVR',
        'Recip+Lin': 'recip_TSVR',
        'Recip+Quad': 'recip_TSVR',
        'Recip_sq': 'recip_TSVR_sq',
        'Recip+PowerLaw': 'recip_TSVR',
        'Exponential_proxy': 'neg_TSVR',
        'Exp+Log': 'neg_TSVR',
        'Exp+Lin': 'neg_TSVR',
        'Exp_fast': 'neg_TSVR_sq',
        'Exp_slow': 'neg_sqrt_TSVR',
        'Exp+PowerLaw': 'neg_TSVR',
        'Exp+Recip': 'neg_TSVR',
        'Sin': 'sin_TSVR',
        'Cos': 'cos_TSVR',
        'Sin+Cos': 'sin_TSVR',
        'Sin+Log': 'sin_TSVR',
        'Tanh': 'tanh_TSVR',
        'Tanh+Log': 'tanh_TSVR',
        'Arctanh': 'arctanh_TSVR',
        'Sinh': 'sinh_TSVR',
        'Log+PowerLaw05': 'log_TSVR',
        'Log+SquareRoot': 'log_TSVR',
        'Log+Recip': 'log_TSVR',
        'SquareRoot+PowerLaw': 'sqrt_TSVR',
        'SquareRoot+Recip': 'sqrt_TSVR',
        'Recip+PowerLaw05': 'recip_TSVR',
        'Lin+Log+PowerLaw': 'TSVR',
        'Quad+Log+SquareRoot': 'TSVR',
        'PowerLaw+Recip+Log': 'TSVR_pow_neg05',
        'Ultimate': 'TSVR',
    }
    return primary_terms.get(model_name, 'TSVR')


def run_model_averaging_pipeline(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    outcome_var: str,
    tsvr_var: str,
    groups_var: str,
    delta_aic_threshold: float = 7.0,
    include_random_effects: bool = True,
    include_random_slopes: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Complete model averaging pipeline for a ROOT RQ.

    Steps:
    1. Identify competitive models (ΔAIC < threshold)
    2. Compute model-averaged predictions
    3. Compute model-averaged random effects (if requested)
    4. Save outputs to CSV files

    Parameters
    ----------
    data : DataFrame
        Original LMM input data
    comparison : DataFrame
        Model comparison table from kitchen sink
    outcome_var : str
        Outcome variable name
    tsvr_var : str
        Time variable name
    groups_var : str
        Grouping variable
    delta_aic_threshold : float
        ΔAIC threshold for competitive models (default=7.0)
    include_random_effects : bool
        Compute model-averaged random effects
    include_random_slopes : bool
        Include random slopes in random effects
    output_dir : Path, optional
        Directory to save output CSVs

    Returns
    -------
    dict
        Complete results including predictions, random effects, metadata
    """
    print("=" * 70)
    print("MODEL AVERAGING PIPELINE")
    print("=" * 70)
    print(f"Outcome: {outcome_var}")
    print(f"Groups: {groups_var}")
    print(f"ΔAIC threshold: {delta_aic_threshold}")
    print("=" * 70)

    # Step 1: Identify competitive models
    competitive = identify_competitive_models(
        comparison,
        delta_aic_threshold=delta_aic_threshold
    )

    # Step 2: Compute model-averaged predictions
    pred_results = compute_model_averaged_predictions(
        data=data,
        comparison=competitive,
        outcome_var=outcome_var,
        tsvr_var=tsvr_var,
        groups_var=groups_var,
        delta_aic_threshold=delta_aic_threshold * 10,  # Already filtered
        reml=False,
    )

    results = {
        'competitive_models': competitive,
        'predictions': pred_results['averaged_predictions'],
        'prediction_variance': pred_results['prediction_variance'],
        'effective_n_models': pred_results['effective_n_models'],
        'models_used': pred_results['models_used'],
        'weights': pred_results['weights_normalized'],
    }

    # Step 3: Model-averaged random effects
    if include_random_effects:
        re_results = compute_model_averaged_random_effects(
            data=data,
            competitive_models=competitive,
            outcome_var=outcome_var,
            tsvr_var=tsvr_var,
            groups_var=groups_var,
            include_random_slopes=include_random_slopes,
            reml=False,
        )
        results['random_intercepts'] = re_results['random_intercepts']
        results['random_slopes'] = re_results['random_slopes']
        results['re_effective_n'] = re_results['effective_n_models']

    # Step 4: Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save competitive models
        competitive.to_csv(output_dir / 'step05b_competitive_models.csv', index=False)

        # Save predictions with data
        pred_df = data.copy()
        pred_df['ma_prediction'] = results['predictions'].values
        pred_df['ma_pred_variance'] = results['prediction_variance'].values
        pred_df.to_csv(output_dir / 'step05b_model_averaged_predictions.csv', index=False)

        # Save random effects
        if include_random_effects:
            results['random_intercepts'].to_csv(
                output_dir / 'step05b_model_averaged_intercepts.csv', index=False
            )
            if results['random_slopes'] is not None:
                results['random_slopes'].to_csv(
                    output_dir / 'step05b_model_averaged_slopes.csv', index=False
                )

        # Save metadata
        meta = pd.DataFrame([{
            'n_competitive_models': len(competitive),
            'delta_aic_threshold': delta_aic_threshold,
            'effective_n_models': results['effective_n_models'],
            'total_original_weight': competitive['akaike_weight'].sum(),
            'top_model': competitive.iloc[0]['model_name'],
            'top_model_weight': competitive.iloc[0]['renorm_weight'],
        }])
        meta.to_csv(output_dir / 'step05b_model_averaging_metadata.csv', index=False)

        print(f"\nOutputs to {output_dir}")

    print("\n" + "=" * 70)
    print("MODEL AVERAGING COMPLETE")
    print("=" * 70)

    return results
