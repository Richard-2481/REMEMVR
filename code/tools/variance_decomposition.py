"""
Model-Averaged Variance Decomposition for LMM Trajectories

Implements Burnham & Anderson (2002) multi-model inference for variance component
analysis. When functional form uncertainty is high, model averaging provides robust
estimates of ICC, variance components, and random effects.

This tool extends model_selection.py and model_averaging.py to handle variance
decomposition specifically - a capability not addressed by trajectory-focused tools.

Author: REMEMVR Team
Date: 2025-12-09
Version: 1.0.0

DESIGN PHILOSOPHY:
- Reuse model_selection.py for model comparison (don't reinvent the wheel)
- Focus on variance-specific outputs (ICC, random effects, variance components)
- Support stratified analysis (fit separate models per categorical level)
- Provide both model-specific and model-averaged results (transparency)
- Handle convergence failures gracefully (report, don't crash)

USE CASES:
- RQ 5.4.6: Schema congruence variance decomposition (3 stratified LMMs)
- RQ 5.2.6: Domain variance decomposition (2-3 stratified LMMs)
- RQ 5.3.7: Paradigm variance decomposition (3 stratified LMMs)
- Any RQ testing "Is forgetting rate trait-like?" with model uncertainty
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime

# Import kitchen sink for model comparison
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from tools.model_selection import compare_lmm_models_kitchen_sink


# =============================================================================
# MAIN FUNCTION: MODEL-AVERAGED VARIANCE DECOMPOSITION
# =============================================================================

def compute_model_averaged_variance_decomposition(
    data: pd.DataFrame,
    outcome_var: str,              # e.g., 'theta'
    tsvr_var: str,                 # e.g., 'TSVR_hours' (continuous)
    groups_var: str,               # e.g., 'UID'
    stratify_var: str,             # e.g., 'congruence' (categorical)
    stratify_levels: List[str],    # e.g., ['Common', 'Congruent', 'Incongruent']

    # Model selection parameters
    delta_aic_threshold: float = 2.0,  # Only use models with ΔAIC < 2
    min_models: int = 3,                # Minimum models to average (warning if less)
    max_models: int = 10,               # Maximum models to average (cap computational cost)

    # Random effects specification
    re_intercept: bool = True,          # Include random intercepts?
    re_slope: bool = True,              # Include random slopes?

    # Output options
    save_dir: Path = None,              # If provided, save all outputs
    log_file: Path = None,              # If provided, write detailed log
    return_fitted_models: bool = False, # Return model objects? (large memory)

    # Advanced options
    reml: bool = False,                 # Use REML (True) or ML (False)
    handle_convergence_failures: str = 'warn',  # 'warn', 'skip', or 'error'
) -> Dict:
    """
    Compute model-averaged variance decomposition across stratified LMMs.

    This function addresses a critical gap: when functional form uncertainty is high
    (e.g., PowerLaw vs Log vs Recip+Log all competitive), variance component estimates
    (ICC, random effects) depend on the chosen time transformation. Model averaging
    provides robust estimates acknowledging this uncertainty.

    WORKFLOW:
    1. Run kitchen sink model comparison on FULL dataset (unstratified)
    2. Identify competitive models (ΔAIC < threshold)
    3. For EACH stratified level:
       a. Fit all competitive models separately
       b. Extract variance components (var_int, var_slope, cov, var_resid)
       c. Compute ICCs (simple, conditional)
       d. Extract random effects (participant-specific intercepts/slopes)
    4. Model-average variance components using Akaike weights
    5. Model-average random effects using Akaike weights
    6. Return both model-specific AND averaged results

    Parameters
    ----------
    data : DataFrame
        Input data with columns: {outcome_var}, {tsvr_var}, {groups_var}, {stratify_var}
        Must have continuous TSVR (not categorical sessions)
    outcome_var : str
        Name of outcome variable (continuous, e.g., 'theta')
    tsvr_var : str
        Name of continuous TSVR column (e.g., 'TSVR_hours')
    groups_var : str
        Subject identifier for random effects (e.g., 'UID')
    stratify_var : str
        Categorical variable for stratification (e.g., 'congruence', 'domain')
    stratify_levels : list of str
        Levels of stratify_var to analyze separately (e.g., ['Common', 'Congruent', 'Incongruent'])
    delta_aic_threshold : float, default=2.0
        Only average models with ΔAIC < threshold
    min_models : int, default=3
        Minimum models to average (warning if fewer converge)
    max_models : int, default=10
        Maximum models to average (cap computational cost at 10 × n_levels LMMs)
    re_intercept : bool, default=True
        Include random intercepts in all models
    re_slope : bool, default=True
        Include random slopes (on time variable) in all models
        If False, fits random-intercepts-only models (convergence fallback)
    save_dir : Path, optional
        Directory to save outputs (comparison tables, variance tables, random effects CSVs)
    log_file : Path, optional
        Log file path for detailed execution log
    return_fitted_models : bool, default=False
        If True, return fitted MixedLMResults objects (warning: large memory footprint)
    reml : bool, default=False
        Use REML (True) or ML (False). ML required for AIC comparison.
    handle_convergence_failures : str, default='warn'
        How to handle convergence failures:
        - 'warn': Log warning and skip failed model
        - 'skip': Silently skip failed model
        - 'error': Raise exception on first failure

    Returns
    -------
    dict
        {
            'model_comparison': DataFrame from kitchen_sink (model_name, AIC, weights),
            'competitive_models': list of model names used for averaging,
            'stratified_results': dict mapping stratify_level → results dict,
            'averaged_variance_components': DataFrame with model-averaged variance estimates,
            'averaged_ICCs': DataFrame with model-averaged ICC estimates,
            'averaged_random_effects': DataFrame with model-averaged random effects,
            'model_specific_results': dict with all model-specific estimates (transparency),
            'summary_stats': dict with overall statistics,
        }

        Where stratified_results[level] contains:
        {
            'variance_components_by_model': DataFrame (model × component),
            'ICCs_by_model': DataFrame (model × ICC_type),
            'random_effects_by_model': dict mapping model_name → DataFrame,
            'variance_components_averaged': Series with model-averaged values,
            'ICCs_averaged': Series with model-averaged ICCs,
            'random_effects_averaged': DataFrame with model-averaged REs,
            'convergence_status': dict mapping model_name → bool,
            'n_models_converged': int,
        }

    Raises
    ------
    ValueError
        If stratify_var not in data, stratify_levels invalid, or no models converge
    AssertionError
        If TSVR is not continuous or has insufficient variance

    Examples
    --------
    >>> # RQ 5.4.6: Schema congruence variance decomposition
    >>> results = compute_model_averaged_variance_decomposition(
    ...     data=lmm_input,
    ...     outcome_var='theta',
    ...     tsvr_var='TSVR_hours',
    ...     groups_var='UID',
    ...     stratify_var='congruence',
    ...     stratify_levels=['Common', 'Congruent', 'Incongruent'],
    ...     delta_aic_threshold=2.0,
    ...     save_dir=Path('results/ch5/5.4.6/data'),
    ... )
    >>>
    >>> # Access model-averaged ICCs
    >>> print(results['averaged_ICCs'])
    >>>
    >>> # Access random effects for clustering
    >>> random_effects = results['averaged_random_effects']
    >>> # 300 rows: 100 UID × 3 congruence levels
    >>> # Columns: UID, congruence, intercept_avg, slope_avg

    >>> # RQ 5.2.6: Domain variance decomposition (2 levels)
    >>> results = compute_model_averaged_variance_decomposition(
    ...     data=lmm_input,
    ...     outcome_var='theta',
    ...     tsvr_var='TSVR_hours',
    ...     groups_var='UID',
    ...     stratify_var='domain',
    ...     stratify_levels=['What', 'Where'],  # When excluded due to floor effect
    ...     delta_aic_threshold=2.0,
    ...     save_dir=Path('results/ch5/5.2.6/data'),
    ... )

    Notes
    -----
    - Model selection runs ONCE on full dataset (not per stratified level)
      Rationale: Functional form uncertainty is a global property, not level-specific
    - Random effects averaging is DIRECT: avg(RE_model1, RE_model2, ..., weights)
      Alternative approaches (e.g., BLUPs from averaged model) are not supported
    - Convergence failures are expected when ICC_slope ≈ 0 (variance too small to estimate)
      Tool gracefully handles this via handle_convergence_failures parameter
    - Memory footprint: ~100 MB for 10 models × 3 levels × 100 participants
      Use return_fitted_models=False to reduce memory usage
    """

    # Initialize logging
    log_messages = []

    def log(msg: str):
        """Log to console and log_messages list."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {msg}"
        print(log_entry)
        log_messages.append(log_entry)

    log("=" * 80)
    log("Model-Averaged Variance Decomposition")
    log("=" * 80)

    # =========================================================================
    # STEP 1: Validate Inputs
    # =========================================================================

    log("\n[STEP 1] Validating inputs...")

    # Check required columns
    required_cols = [outcome_var, tsvr_var, groups_var, stratify_var]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate stratify_var
    if not pd.api.types.is_object_dtype(data[stratify_var]) and \
       not pd.api.types.is_categorical_dtype(data[stratify_var]):
        warnings.warn(
            f"{stratify_var} is not categorical/object dtype. "
            f"Treating as categorical for stratification."
        )

    # Validate stratify_levels
    available_levels = data[stratify_var].unique()
    invalid_levels = [lvl for lvl in stratify_levels if lvl not in available_levels]
    if invalid_levels:
        raise ValueError(
            f"Invalid stratify_levels: {invalid_levels}. "
            f"Available levels: {list(available_levels)}"
        )

    log(f"  Outcome: {outcome_var}")
    log(f"  TSVR: {tsvr_var}")
    log(f"  Groups: {groups_var} (N={data[groups_var].nunique()})")
    log(f"  Stratify by: {stratify_var}")
    log(f"  Stratify levels: {stratify_levels}")
    log(f"  N observations: {len(data)}")
    log(f"  Random effects: intercept={re_intercept}, slope={re_slope}")

    # =========================================================================
    # STEP 2: Run Kitchen Sink Model Comparison (Full Dataset)
    # =========================================================================

    log("\n[STEP 2] Running kitchen sink model comparison (unstratified)...")
    log("  This identifies competitive functional forms across ALL data")
    log("  (not stratified by congruence/domain - form is global property)")

    # Build random effects formula
    if re_intercept and re_slope:
        re_formula = f"~{tsvr_var}"  # Random intercepts + slopes
    elif re_intercept:
        re_formula = "~1"  # Random intercepts only
    else:
        raise ValueError("At least one of re_intercept or re_slope must be True")

    log(f"  Random effects formula: {re_formula}")

    # Run kitchen sink comparison
    try:
        comparison_results = compare_lmm_models_kitchen_sink(
            data=data,
            outcome_var=outcome_var,
            tsvr_var=tsvr_var,
            groups_var=groups_var,
            re_formula=re_formula,
            reml=reml,
            return_models=False,  # We'll refit stratified models separately
            save_dir=None,  # Don't save intermediate results
            log_file=None,  # Suppress kitchen sink log (we have our own)
        )
    except Exception as e:
        log(f"  ✗ Kitchen sink comparison FAILED: {e}")
        raise

    comparison_df = comparison_results['comparison']
    best_model = comparison_results['best_model']

    log(f"  ✓ Kitchen sink complete: {len(comparison_df)} models converged")
    log(f"  Best model: {best_model['name']} (weight={best_model['weight']:.1%})")

    # =========================================================================
    # STEP 3: Select Competitive Models for Averaging
    # =========================================================================

    log(f"\n[STEP 3] Selecting competitive models (ΔAIC < {delta_aic_threshold})...")

    competitive = comparison_df[comparison_df['delta_AIC'] < delta_aic_threshold].copy()

    log(f"  Competitive models: {len(competitive)}")
    log(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}")

    if len(competitive) == 0:
        raise ValueError(
            f"No models with ΔAIC < {delta_aic_threshold}. "
            f"Try increasing delta_aic_threshold or check model convergence."
        )

    if len(competitive) < min_models:
        warnings.warn(
            f"Only {len(competitive)} competitive models (min={min_models} recommended). "
            f"Results may be unstable."
        )

    # Cap at max_models (computational cost control)
    if len(competitive) > max_models:
        log(f"  Capping at {max_models} models (computational cost control)")
        competitive = competitive.head(max_models)
        log(f"  Adjusted cumulative weight: {competitive['akaike_weight'].sum():.1%}")

    # Renormalize weights (sum to 1.0 across competitive set)
    competitive['weight_renorm'] = (
        competitive['akaike_weight'] / competitive['akaike_weight'].sum()
    )

    competitive_models = competitive['model_name'].tolist()
    log(f"  Models to average: {competitive_models}")

    # =========================================================================
    # STEP 4: Fit Stratified LMMs for Each Level × Model
    # =========================================================================

    log(f"\n[STEP 4] Fitting stratified LMMs...")
    log(f"  Total LMMs to fit: {len(competitive_models)} models × {len(stratify_levels)} levels = {len(competitive_models) * len(stratify_levels)}")

    stratified_results = {}

    for level in stratify_levels:
        log(f"\n  === Stratified Level: {level} ===")

        # Subset data for this level
        level_data = data[data[stratify_var] == level].copy()
        n_obs = len(level_data)
        n_groups = level_data[groups_var].nunique()

        log(f"    N observations: {n_obs}")
        log(f"    N groups: {n_groups}")

        # Initialize storage for this level
        variance_components_list = []
        iccs_list = []
        random_effects_dict = {}
        convergence_status = {}
        fitted_models_dict = {}

        # Fit each competitive model
        for i, row in competitive.iterrows():
            model_name = row['model_name']
            weight = row['weight_renorm']

            log(f"    [{i+1}/{len(competitive)}] Fitting {model_name} (w={weight:.4f})...")

            try:
                # Fit model and extract components
                result = _fit_single_model_and_extract(
                    data=level_data,
                    model_name=model_name,
                    outcome_var=outcome_var,
                    tsvr_var=tsvr_var,
                    groups_var=groups_var,
                    re_intercept=re_intercept,
                    re_slope=re_slope,
                    reml=reml,
                )

                # Store results
                variance_components_list.append(result['variance_components'])
                iccs_list.append(result['ICCs'])
                random_effects_dict[model_name] = result['random_effects']
                convergence_status[model_name] = result['converged']

                if return_fitted_models:
                    fitted_models_dict[model_name] = result['fitted_model']

                log(f"      ✓ Converged: {result['converged']}")
                log(f"        ICC_int={result['ICCs']['ICC_intercept']:.3f}, "
                    f"ICC_slope_simple={result['ICCs']['ICC_slope_simple']:.6f}")

            except Exception as e:
                log(f"      ✗ FAILED: {str(e)[:100]}")

                convergence_status[model_name] = False

                if handle_convergence_failures == 'error':
                    raise
                elif handle_convergence_failures == 'warn':
                    warnings.warn(f"Model {model_name} failed for level {level}: {e}")
                # 'skip' mode: silently continue

        # Check if any models converged
        n_converged = sum(convergence_status.values())
        if n_converged == 0:
            raise ValueError(f"All models failed for level {level}")

        log(f"    Models converged: {n_converged}/{len(competitive)}")

        # Convert lists to DataFrames
        variance_df = pd.DataFrame(variance_components_list)
        iccs_df = pd.DataFrame(iccs_list)

        # =====================================================================
        # STEP 5: Model-Average Variance Components for This Level
        # =====================================================================

        log(f"    Computing model-averaged variance components...")

        # Get weights for converged models only
        converged_mask = variance_df['model_name'].isin(
            [m for m, status in convergence_status.items() if status]
        )
        variance_converged = variance_df[converged_mask].copy()
        iccs_converged = iccs_df[converged_mask].copy()

        # Merge with renormalized weights
        variance_converged = variance_converged.merge(
            competitive[['model_name', 'weight_renorm']],
            on='model_name'
        )
        iccs_converged = iccs_converged.merge(
            competitive[['model_name', 'weight_renorm']],
            on='model_name'
        )

        # Renormalize weights (some models may have failed)
        weight_sum = variance_converged['weight_renorm'].sum()
        variance_converged['weight_final'] = variance_converged['weight_renorm'] / weight_sum
        iccs_converged['weight_final'] = iccs_converged['weight_renorm'] / weight_sum

        # Compute weighted averages
        variance_cols = ['var_intercept', 'var_slope', 'cov_intercept_slope', 'var_residual']
        variance_averaged = {}
        for col in variance_cols:
            variance_averaged[col] = np.average(
                variance_converged[col],
                weights=variance_converged['weight_final']
            )

        icc_cols = ['ICC_intercept', 'ICC_slope_simple', 'ICC_slope_conditional']
        iccs_averaged = {}
        for col in icc_cols:
            iccs_averaged[col] = np.average(
                iccs_converged[col],
                weights=iccs_converged['weight_final']
            )

        log(f"      ICC_int (avg) = {iccs_averaged['ICC_intercept']:.3f}")
        log(f"      ICC_slope_simple (avg) = {iccs_averaged['ICC_slope_simple']:.6f}")

        # =====================================================================
        # STEP 6: Model-Average Random Effects for This Level
        # =====================================================================

        log(f"    Computing model-averaged random effects...")

        # Initialize averaged random effects DataFrame
        # Get UIDs from first converged model
        first_converged_model = [m for m, status in convergence_status.items() if status][0]
        re_template = random_effects_dict[first_converged_model].copy()

        re_averaged = re_template[[groups_var]].copy()
        re_averaged['intercept_avg'] = 0.0
        if re_slope:
            re_averaged['slope_avg'] = 0.0

        # Weighted average across models
        for model_name, re_df in random_effects_dict.items():
            if not convergence_status[model_name]:
                continue  # Skip failed models

            # Get weight for this model
            weight = competitive[competitive['model_name'] == model_name]['weight_renorm'].iloc[0]
            weight = weight / weight_sum  # Renormalize

            # Add weighted contribution
            re_averaged['intercept_avg'] += weight * re_df['intercept']
            if re_slope:
                re_averaged['slope_avg'] += weight * re_df['slope']

        log(f"      Random effects averaged across {n_converged} models")
        log(f"      N participants: {len(re_averaged)}")

        # =====================================================================
        # Store Results for This Level
        # =====================================================================

        stratified_results[level] = {
            'variance_components_by_model': variance_df,
            'ICCs_by_model': iccs_df,
            'random_effects_by_model': random_effects_dict,
            'variance_components_averaged': pd.Series(variance_averaged),
            'ICCs_averaged': pd.Series(iccs_averaged),
            'random_effects_averaged': re_averaged,
            'convergence_status': convergence_status,
            'n_models_converged': n_converged,
        }

        if return_fitted_models:
            stratified_results[level]['fitted_models'] = fitted_models_dict

    # =========================================================================
    # STEP 7: Aggregate Results Across Levels
    # =========================================================================

    log(f"\n[STEP 7] Aggregating results across levels...")

    # Combine averaged variance components into single DataFrame
    variance_summary = []
    iccs_summary = []
    random_effects_combined = []

    for level in stratify_levels:
        level_results = stratified_results[level]

        # Variance components
        var_row = level_results['variance_components_averaged'].to_dict()
        var_row[stratify_var] = level
        variance_summary.append(var_row)

        # ICCs
        icc_row = level_results['ICCs_averaged'].to_dict()
        icc_row[stratify_var] = level
        iccs_summary.append(icc_row)

        # Random effects
        re_df = level_results['random_effects_averaged'].copy()
        re_df[stratify_var] = level
        random_effects_combined.append(re_df)

    averaged_variance_components = pd.DataFrame(variance_summary)
    averaged_ICCs = pd.DataFrame(iccs_summary)
    averaged_random_effects = pd.concat(random_effects_combined, ignore_index=True)

    # Reorder columns
    averaged_variance_components = averaged_variance_components[
        [stratify_var, 'var_intercept', 'var_slope', 'cov_intercept_slope', 'var_residual']
    ]
    averaged_ICCs = averaged_ICCs[
        [stratify_var, 'ICC_intercept', 'ICC_slope_simple', 'ICC_slope_conditional']
    ]

    if re_slope:
        averaged_random_effects = averaged_random_effects[
            [groups_var, stratify_var, 'intercept_avg', 'slope_avg']
        ]
    else:
        averaged_random_effects = averaged_random_effects[
            [groups_var, stratify_var, 'intercept_avg']
        ]

    log(f"  ✓ Averaged variance components: {len(averaged_variance_components)} levels")
    log(f"  ✓ Averaged ICCs: {len(averaged_ICCs)} levels")
    log(f"  ✓ Averaged random effects: {len(averaged_random_effects)} rows "
        f"({data[groups_var].nunique()} UID × {len(stratify_levels)} levels)")

    # =========================================================================
    # STEP 8: Compute Summary Statistics
    # =========================================================================

    log(f"\n[STEP 8] Computing summary statistics...")

    # Effective number of models (Shannon diversity)
    weights_array = competitive['weight_renorm'].values
    effective_n_models = np.exp(-np.sum(weights_array * np.log(weights_array + 1e-10)))

    # Convergence summary
    total_fits = len(competitive_models) * len(stratify_levels)
    total_converged = sum(
        sum(stratified_results[level]['convergence_status'].values())
        for level in stratify_levels
    )

    summary_stats = {
        'n_models_competitive': len(competitive_models),
        'n_stratify_levels': len(stratify_levels),
        'total_lmm_fits': total_fits,
        'total_converged': total_converged,
        'convergence_rate': total_converged / total_fits,
        'effective_n_models': effective_n_models,
        'cumulative_weight': competitive['akaike_weight'].sum(),
        'best_model': best_model['name'],
        'best_model_weight': best_model['weight'],
    }

    log(f"  Effective N models: {effective_n_models:.2f}")
    log(f"  Convergence rate: {summary_stats['convergence_rate']:.1%}")

    # =========================================================================
    # STEP 9: Save Outputs (if requested)
    # =========================================================================

    if save_dir is not None:
        log(f"\n[STEP 9] Saving outputs to {save_dir}...")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Model comparison
        comparison_path = save_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"  ✓ {comparison_path.name}")

        # Competitive models with renormalized weights
        competitive_path = save_dir / "competitive_models.csv"
        competitive.to_csv(competitive_path, index=False, encoding='utf-8')
        log(f"  ✓ {competitive_path.name}")

        # Averaged variance components
        variance_path = save_dir / "variance_components_averaged.csv"
        averaged_variance_components.to_csv(variance_path, index=False, encoding='utf-8')
        log(f"  ✓ {variance_path.name}")

        # Averaged ICCs
        icc_path = save_dir / "ICCs_averaged.csv"
        averaged_ICCs.to_csv(icc_path, index=False, encoding='utf-8')
        log(f"  ✓ {icc_path.name}")

        # Averaged random effects
        re_path = save_dir / "random_effects_averaged.csv"
        averaged_random_effects.to_csv(re_path, index=False, encoding='utf-8')
        log(f"  ✓ {re_path.name}")

        # Model-specific results (transparency)
        for level in stratify_levels:
            level_dir = save_dir / f"level_{level}"
            level_dir.mkdir(exist_ok=True)

            level_results = stratified_results[level]

            # Variance by model
            var_model_path = level_dir / "variance_components_by_model.csv"
            level_results['variance_components_by_model'].to_csv(
                var_model_path, index=False, encoding='utf-8'
            )

            # ICCs by model
            icc_model_path = level_dir / "ICCs_by_model.csv"
            level_results['ICCs_by_model'].to_csv(
                icc_model_path, index=False, encoding='utf-8'
            )

            log(f"  ✓ {level}_* (model-specific results)")

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_messages))
        log(f"  ✓ Log saved to {log_file.name}")

    log("\n" + "=" * 80)
    log("Model-averaged variance decomposition COMPLETE")
    log("=" * 80)

    # =========================================================================
    # Return Results
    # =========================================================================

    results = {
        'model_comparison': comparison_df,
        'competitive_models': competitive_models,
        'stratified_results': stratified_results,
        'averaged_variance_components': averaged_variance_components,
        'averaged_ICCs': averaged_ICCs,
        'averaged_random_effects': averaged_random_effects,
        'summary_stats': summary_stats,
    }

    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _fit_single_model_and_extract(
    data: pd.DataFrame,
    model_name: str,
    outcome_var: str,
    tsvr_var: str,
    groups_var: str,
    re_intercept: bool,
    re_slope: bool,
    reml: bool,
) -> Dict:
    """
    Fit a single LMM model and extract variance components + random effects.

    This is a helper function called by the main function for each model × level.

    Returns
    -------
    dict
        {
            'variance_components': dict with var_intercept, var_slope, cov, var_residual,
            'ICCs': dict with ICC_intercept, ICC_slope_simple, ICC_slope_conditional,
            'random_effects': DataFrame with UID, intercept, slope,
            'converged': bool,
            'fitted_model': MixedLMResults object,
        }
    """

    # Create time transformations
    data_trans = _create_transformations(data.copy(), tsvr_var)

    # Build formula
    formula = _build_formula(model_name, outcome_var)

    # Build random effects formula
    time_var = _get_time_variable_for_model(model_name)
    if re_intercept and re_slope:
        re_formula = f"~{time_var}"
    elif re_intercept:
        re_formula = "~1"
    else:
        raise ValueError("At least one of re_intercept or re_slope must be True")

    # Fit model
    model = smf.mixedlm(
        formula=formula,
        data=data_trans,
        groups=data_trans[groups_var],
        re_formula=re_formula
    )

    fitted = model.fit(reml=reml, method='powell')

    # Extract variance components
    cov_re = fitted.cov_re
    var_intercept = cov_re.iloc[0, 0] if len(cov_re) > 0 else 0.0
    var_slope = cov_re.iloc[1, 1] if len(cov_re) > 1 else 0.0
    cov_int_slope = cov_re.iloc[0, 1] if len(cov_re) > 1 else 0.0
    var_residual = fitted.scale

    # Compute ICCs
    var_total = var_intercept + var_residual
    icc_intercept = var_intercept / var_total if var_total > 0 else 0.0

    var_total_slope = var_slope + var_residual
    icc_slope_simple = var_slope / var_total_slope if var_total_slope > 0 else 0.0

    # Conditional ICC (at end of study)
    # Assumes TSVR max ~ 7 days = 168 hours / 24 = 7 days
    tsvr_max_days = data_trans['TSVR'].max()
    var_conditional = var_intercept + 2 * tsvr_max_days * cov_int_slope + (tsvr_max_days ** 2) * var_slope
    var_total_conditional = var_conditional + var_residual
    icc_slope_conditional = var_conditional / var_total_conditional if var_total_conditional > 0 else 0.0

    # Extract random effects
    random_effects_raw = fitted.random_effects

    re_list = []
    for uid, effects in random_effects_raw.items():
        re_row = {groups_var: uid}
        re_row['intercept'] = effects.iloc[0]
        if re_slope:
            re_row['slope'] = effects.iloc[1] if len(effects) > 1 else 0.0
        re_list.append(re_row)

    random_effects = pd.DataFrame(re_list)

    # Package results
    result = {
        'variance_components': {
            'model_name': model_name,
            'var_intercept': var_intercept,
            'var_slope': var_slope,
            'cov_intercept_slope': cov_int_slope,
            'var_residual': var_residual,
        },
        'ICCs': {
            'model_name': model_name,
            'ICC_intercept': icc_intercept,
            'ICC_slope_simple': icc_slope_simple,
            'ICC_slope_conditional': icc_slope_conditional,
        },
        'random_effects': random_effects,
        'converged': fitted.converged,
        'fitted_model': fitted,
    }

    return result


def _create_transformations(df: pd.DataFrame, tsvr_var: str) -> pd.DataFrame:
    """
    Create all time transformations needed for model suite.

    This duplicates logic from model_selection.py to ensure consistency.
    """

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


def _build_formula(model_name: str, outcome: str) -> str:
    """Build formula for a specific model (no interactions)."""

    # Map model names to formulas (from kitchen_sink)
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

    return f"{outcome} ~ {formulas[model_name]}"


def _get_time_variable_for_model(model_name: str) -> str:
    """
    Determine which time variable to use for random slopes.

    For models with multiple time terms (e.g., 'Recip+Log'), use the FIRST term.
    This matches the approach in model_selection.py.
    """

    # Map to first time variable in formula
    time_var_map = {
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

    if model_name not in time_var_map:
        raise ValueError(f"Unknown model: {model_name}")

    return time_var_map[model_name]
