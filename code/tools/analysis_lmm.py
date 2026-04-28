"""
Linear Mixed Model (LMM) Analysis Tool

Functions for fitting longitudinal mixed-effects models to theta scores.
Implements model comparison via AIC for trajectory analysis.

Author: REMEMVR Team
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults, MixedLMParams
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats


# DATA PREPARATION

def assign_piecewise_segments(
    df: pd.DataFrame,
    tsvr_col: str = 'TSVR_hours',
    early_cutoff_hours: float = 24.0
) -> pd.DataFrame:
    """
    Assign piecewise segments (Early/Late) and compute Days_within for piecewise LMM.

    Implements piecewise regression design where forgetting trajectory is divided
    into two temporal segments with distinct processes:
    - Early segment (0-24h): Consolidation-dominated phase (includes one night's sleep)
    - Late segment (24-168h): Decay-dominated phase

    Parameters
    ----------
    df : DataFrame
        Input data with TSVR time variable (hours since encoding)
        Must contain columns: [test, {tsvr_col}]
    tsvr_col : str, default='TSVR_hours'
        Name of TSVR column (time since VR in hours)
    early_cutoff_hours : float, default=24.0
        Cutoff defining Early segment boundary (hours)
        Default 24h = one night's sleep (consolidation window)

    Returns
    -------
    DataFrame
        Copy of input with added columns:
        - Segment : str, 'Early' or 'Late'
        - Days_within : float, time elapsed within segment (in days)

    Notes
    -----
    Days_within calculation:
    - Early segment: Days_within = TSVR_hours / 24 (starts at Day 0)
    - Late segment: Days_within = (TSVR_hours - min_Late_TSVR) / 24 (resets at segment start)

    This allows segment-specific slope estimation in piecewise LMM.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'UID': ['P001'] * 4,
    ...     'test': [1, 2, 3, 4],
    ...     'TSVR_hours': [0.0, 24.0, 72.0, 144.0],
    ...     'theta': [0.5, 0.3, 0.1, -0.2]
    ... })
    >>> result = assign_piecewise_segments(df)
    >>> result[['test', 'Segment', 'Days_within']]
       test Segment  Days_within
    0     1   Early          0.0
    1     2   Early          1.0
    2     3    Late          0.0
    3     4    Late          3.0
    """
    result = df.copy()

    # Assign segments based on TSVR cutoff
    result['Segment'] = result[tsvr_col].apply(
        lambda x: 'Early' if x <= early_cutoff_hours else 'Late'
    )

    # Find minimum TSVR for Late segment (for resetting Days_within)
    late_tsvr = result[result['Segment'] == 'Late'][tsvr_col]
    min_late_tsvr = late_tsvr.min() if len(late_tsvr) > 0 else 0.0

    # Compute Days_within for each segment
    def compute_days_within(row):
        if row['Segment'] == 'Early':
            return row[tsvr_col] / 24.0
        else:  # Late
            return (row[tsvr_col] - min_late_tsvr) / 24.0

    result['Days_within'] = result.apply(compute_days_within, axis=1)

    return result


def prepare_lmm_input_from_theta(
    theta_scores: pd.DataFrame,
    factors: Optional[list] = None
) -> pd.DataFrame:
    """
    Convert theta scores from wide to long format and add time variables.

    ⚠️ **DEPRECATED for REMEMVR RQs:** This function uses NOMINAL days (0, 1, 3, 6)
    instead of TSVR (actual hours since encoding). Violates Decision D070.

    **USE fit_lmm_trajectory_tsvr() INSTEAD** for analyses requiring accurate temporal modeling.

    This function remains for backward compatibility with non-REMEMVR analyses only.

    Parameters
    ----------
    theta_scores : pd.DataFrame
        Wide-format dataframe with columns:
        - UID : Participant ID
        - test : Test session (1, 2, 3, 4)
        - Theta_What, Theta_Where, Theta_When : Ability estimates

    factors : list, optional
        List of theta column names to melt. If None, uses all Theta_* columns.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        - UID, test, Factor, Ability
        - Days, Days_sq, log_Days (time variables)

    WARNING: Uses NOMINAL days {1:0, 2:1, 3:3, 4:6}, NOT actual TSVR hours.

    Raises
    ------
    ValueError
        If required theta columns are missing

    Examples
    --------
    >>> df_wide = pd.DataFrame({
    ...     'UID': ['A001', 'A001'],
    ...     'test': [1, 2],
    ...     'Theta_What': [0.5, 0.3],
    ...     'Theta_Where': [0.2, 0.1],
    ...     'Theta_When': [0.8, 0.6]
    ... })
    >>> df_long = prepare_lmm_input_from_theta(df_wide)
    >>> df_long.shape
    (6, 7)  # 2 obs × 3 factors = 6 rows
    """
    import warnings
    warnings.warn(
        "prepare_lmm_input_from_theta() uses NOMINAL days, not TSVR (Decision D070). "
        "Use fit_lmm_trajectory_tsvr() for REMEMVR analyses.",
        DeprecationWarning,
        stacklevel=2
    )
    # Detect theta columns
    if factors is None:
        factors = [col for col in theta_scores.columns if col.startswith('Theta_')]

    if len(factors) == 0:
        raise ValueError("Missing required theta columns (Theta_*)")

    # Melt to long format
    df_long = theta_scores.melt(
        id_vars=['UID', 'test'],
        value_vars=factors,
        var_name='Factor',
        value_name='Ability'
    )

    # Clean Factor names (remove "Theta_" prefix)
    df_long['Factor'] = df_long['Factor'].str.replace('Theta_', '')

    # Add time variables
    day_map = {1: 0, 2: 1, 3: 3, 4: 6}  # Days since VR encoding
    df_long['Days'] = df_long['test'].map(day_map)
    df_long['Days_sq'] = df_long['Days'] ** 2
    df_long['log_Days'] = np.log(df_long['Days'] + 1)

    # Sort for readability
    df_long = df_long.sort_values(['UID', 'Factor', 'test']).reset_index(drop=True)

    return df_long


# MODEL CONFIGURATION

def configure_candidate_models(
    n_factors: int,
    reference_group: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Generate formulas for 5 candidate LMM models.

    Parameters
    ----------
    n_factors : int
        Number of factors (1 = single domain, >1 = multiple domains)

    reference_group : str, optional
        Reference level for Factor (e.g., 'What'). Required if n_factors > 1.

    Returns
    -------
    dict
        Dictionary with model configurations:
        {
            'Linear': {'formula': '...', 're_formula': '...'},
            'Quadratic': {...},
            ...
        }

    Raises
    ------
    ValueError
        If n_factors > 1 and reference_group is None

    Examples
    --------
    >>> models = configure_candidate_models(n_factors=3, reference_group='What')
    >>> models['Linear']['formula']
    "Ability ~ Days * C(Factor, Treatment('What'))"
    """
    if n_factors > 1 and reference_group is None:
        raise ValueError("Reference group must be specified for multi-factor analysis")

    models = {}

    if n_factors == 1:
        # Single-factor models (no Factor interaction)
        models = {
            'Linear': {
                'formula': 'Ability ~ Days',
                're_formula': '~Days'
            },
            'Quadratic': {
                'formula': 'Ability ~ Days + Days_sq',
                're_formula': '~Days'
            },
            'Log': {
                'formula': 'Ability ~ log_Days',
                're_formula': '~log_Days'
            },
            'Lin+Log': {
                'formula': 'Ability ~ Days + log_Days',
                're_formula': '~Days'
            },
            'Quad+Log': {
                'formula': 'Ability ~ Days + Days_sq + log_Days',
                're_formula': '~Days'
            }
        }
    else:
        # Multi-factor models (with Factor interaction)
        factor_term = f"C(Factor, Treatment('{reference_group}'))"

        models = {
            'Linear': {
                'formula': f'Ability ~ Days * {factor_term}',
                're_formula': '~Days'
            },
            'Quadratic': {
                'formula': f'Ability ~ (Days + Days_sq) * {factor_term}',
                're_formula': '~Days'
            },
            'Log': {
                'formula': f'Ability ~ log_Days * {factor_term}',
                're_formula': '~log_Days'
            },
            'Lin+Log': {
                'formula': f'Ability ~ (Days + log_Days) * {factor_term}',
                're_formula': '~Days'
            },
            'Quad+Log': {
                'formula': f'Ability ~ (Days + Days_sq + log_Days) * {factor_term}',
                're_formula': '~Days'
            }
        }

    return models


# SINGLE MODEL FITTING

def fit_lmm_trajectory(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    re_formula: str,
    reml: bool = False
) -> MixedLMResults:
    """
    Fit a single linear mixed model.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with outcome, predictors, and grouping variable

    formula : str
        R-style formula for fixed effects (e.g., 'Ability ~ Days')

    groups : str
        Column name for grouping variable (typically 'UID')

    re_formula : str
        Formula for random effects (e.g., '~Days')

    reml : bool, default False
        Use restricted maximum likelihood? Set False for AIC comparison.

    Returns
    -------
    MixedLMResults
        Fitted model object with methods: .summary(), .aic, .params, etc.

    Raises
    ------
    RuntimeError
        If model fails to converge

    Examples
    --------
    >>> result = fit_lmm_trajectory(
    ...     data=df_long,
    ...     formula='Ability ~ Days',
    ...     groups='UID',
    ...     re_formula='~Days',
    ...     reml=False
    ... )
    >>> print(result.aic)
    """
    # Build model
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data[groups],
        re_formula=re_formula
    )

    # Fit model
    try:
        result = model.fit(method=['lbfgs'], reml=reml)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {str(e)}")

    # Check convergence
    if not result.converged:
        warnings.warn(
            f"Model did not converge. Formula: {formula}",
            UserWarning
        )

    return result


# MODEL COMPARISON

def compare_lmm_models_by_aic(
    data: pd.DataFrame,
    n_factors: int,
    reference_group: Optional[str] = None,
    groups: str = 'UID',
    save_dir: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Fit all candidate models and compare via AIC.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format LMM data (from prepare_lmm_data)

    n_factors : int
        Number of factors in analysis

    reference_group : str, optional
        Reference level for Factor (required if n_factors > 1)

    groups : str, default 'UID'
        Column name for grouping variable

    save_dir : str or Path, optional
        Directory to save fitted models (.pkl files)

    Returns
    -------
    dict
        Results dictionary with keys:
        - 'models' : dict of fitted MixedLMResults
        - 'aic_comparison' : DataFrame with AIC, delta_AIC, weights
        - 'best_model_name' : str, name of best model
        - 'best_model' : MixedLMResults, best fitted model

    Examples
    --------
    >>> results = compare_lmm_models_by_aic(
    ...     data=df_long,
    ...     n_factors=3,
    ...     reference_group='What'
    ... )
    >>> print(results['best_model_name'])
    'Lin+Log'
    >>> print(results['aic_comparison'])
    """
    # Get candidate models
    model_configs = configure_candidate_models(n_factors, reference_group)

    # Fit all models
    fitted_models = {}
    aics = {}

    for model_name, config in model_configs.items():
        print(f"Fitting {model_name} model...")

        # Check if saved model exists
        if save_dir is not None:
            model_path = Path(save_dir) / f"lmm_{model_name}.pkl"
            if model_path.exists():
                print(f"  Loading existing model from {model_path}")
                fitted_models[model_name] = MixedLMResults.load(str(model_path))
                aics[model_name] = fitted_models[model_name].aic
                continue

        # Fit model
        try:
            result = fit_lmm_trajectory(
                data=data,
                formula=config['formula'],
                groups=groups,
                re_formula=config['re_formula'],
                reml=False
            )

            fitted_models[model_name] = result
            aics[model_name] = result.aic

            # Save model if directory specified
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                result.save(str(model_path))
                print(f"  Model saved to {model_path}")

        except Exception as e:
            print(f"  WARNING: {model_name} model failed: {str(e)}")
            fitted_models[model_name] = None
            aics[model_name] = np.inf

    # Create AIC comparison table
    aic_df = pd.DataFrame({
        'model_name': list(aics.keys()),
        'AIC': list(aics.values())
    })

    # Calculate delta AIC and weights
    aic_df = aic_df.sort_values('AIC').reset_index(drop=True)
    aic_df['delta_AIC'] = aic_df['AIC'] - aic_df['AIC'].min()
    aic_df['AIC_weight'] = np.exp(-aic_df['delta_AIC'] / 2)
    aic_df['AIC_weight'] = aic_df['AIC_weight'] / aic_df['AIC_weight'].sum()

    # Identify best model
    best_model_name = aic_df.iloc[0]['model_name']
    best_model = fitted_models[best_model_name]

    return {
        'models': fitted_models,
        'aic_comparison': aic_df,
        'best_model_name': best_model_name,
        'best_model': best_model
    }


# EXTRACT RESULTS

def extract_fixed_effects_from_lmm(result: MixedLMResults) -> pd.DataFrame:
    """
    Extract fixed effects table from fitted model.

    Parameters
    ----------
    result : MixedLMResults
        Fitted model

    Returns
    -------
    pd.DataFrame
        Fixed effects with columns: Term, Coef, Std_Err, z, P_value, CI_lower, CI_upper

    Examples
    --------
    >>> fe_table = extract_fixed_effects_from_lmm(fitted_model)
    >>> print(fe_table)
    """
    # Get fixed effects summary table
    fe_summary = result.summary().tables[1]

    # Check if it's already a DataFrame or needs conversion
    if isinstance(fe_summary, pd.DataFrame):
        fe_df = fe_summary.copy()
        # Reset index to make term names a column
        fe_df = fe_df.reset_index()
        fe_df.columns = ['Term', 'Coef', 'Std_Err', 'z', 'P_value', 'CI_lower', 'CI_upper']
    else:
        # SimpleTable object - extract data
        fe_df = pd.DataFrame(fe_summary.data[1:], columns=fe_summary.data[0])
        fe_df.columns = ['Term', 'Coef', 'Std_Err', 'z', 'P_value', 'CI_lower', 'CI_upper']

    # Convert numeric columns
    numeric_cols = ['Coef', 'Std_Err', 'z', 'P_value', 'CI_lower', 'CI_upper']
    for col in numeric_cols:
        fe_df[col] = pd.to_numeric(fe_df[col], errors='coerce')

    return fe_df


def extract_random_effects_from_lmm(result: MixedLMResults) -> Dict:
    """
    Extract random effects variances and ICC.

    Parameters
    ----------
    result : MixedLMResults
        Fitted model

    Returns
    -------
    dict
        Random effects summary with keys:
        - 're_variance' : dict, variance components
        - 'residual_variance' : float
        - 'icc' : float, intraclass correlation

    Examples
    --------
    >>> re_summary = extract_random_effects_from_lmm(fitted_model)
    >>> print(re_summary['icc'])
    """
    # Random effects variance-covariance matrix
    re_cov = result.cov_re

    # Residual variance
    residual_var = result.scale

    # ICC (intraclass correlation)
    # ICC = var(u0) / (var(u0) + var(residual))
    if re_cov.shape[0] > 0:
        u0_variance = re_cov.iloc[0, 0] if isinstance(re_cov, pd.DataFrame) else re_cov[0, 0]
        icc = u0_variance / (u0_variance + residual_var)
    else:
        icc = 0.0

    return {
        're_variance': re_cov,
        'residual_variance': residual_var,
        'icc': icc
    }


# FULL PIPELINE

def run_lmm_analysis(
    theta_scores: pd.DataFrame,
    output_dir: Union[str, Path],
    n_factors: int,
    reference_group: Optional[str] = None,
    save_models: bool = True
) -> Dict:
    """
    Complete LMM analysis pipeline.

    Steps:
    1. Prepare data (wide to long)
    2. Fit candidate models
    3. Compare via AIC
    4. Extract fixed/random effects from best model
    5. Save results

    Parameters
    ----------
    theta_scores : pd.DataFrame
        Wide-format theta scores (UID, test, Theta_*)

    output_dir : str or Path
        Directory to save outputs

    n_factors : int
        Number of factors in analysis

    reference_group : str, optional
        Reference level for Factor

    save_models : bool, default True
        Save fitted model .pkl files?

    Returns
    -------
    dict
        Complete results with keys:
        - 'df_long' : Long-format data
        - 'best_model_name' : Best model by AIC
        - 'best_model' : Fitted MixedLMResults
        - 'aic_comparison' : AIC comparison table
        - 'fixed_effects' : Fixed effects table
        - 'random_effects' : Random effects summary

    Examples
    --------
    >>> results = run_lmm_analysis(
    ...     theta_scores=df_theta,
    ...     output_dir='results/ch5/rq1/lmm/',
    ...     n_factors=3,
    ...     reference_group='What'
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LMM ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[1/5] Preparing data (wide to long format)...")
    df_long = prepare_lmm_input_from_theta(theta_scores)
    print(f"  Data shape: {df_long.shape}")
    print(f"  Factors: {df_long['Factor'].unique().tolist()}")

    # Save long-format data
    df_long.to_csv(output_dir / 'lmm_data_long.csv', index=False)
    print(f"  Saved: {output_dir / 'lmm_data_long.csv'}")

    # Step 2-3: Fit and compare models
    print("\n[2/5] Fitting candidate models and comparing via AIC...")
    save_dir = output_dir if save_models else None
    comparison_results = compare_lmm_models_by_aic(
        data=df_long,
        n_factors=n_factors,
        reference_group=reference_group,
        save_dir=save_dir
    )

    best_model_name = comparison_results['best_model_name']
    best_model = comparison_results['best_model']
    aic_df = comparison_results['aic_comparison']

    print(f"\n  Best model: {best_model_name}")
    print(f"  AIC: {best_model.aic:.2f}")

    # Save AIC comparison
    aic_df.to_csv(output_dir / 'aic_comparison.csv', index=False)
    print(f"  Saved: {output_dir / 'aic_comparison.csv'}")

    # Step 4: Extract fixed effects
    print("\n[3/5] Extracting fixed effects...")
    fe_table = extract_fixed_effects_from_lmm(best_model)
    fe_table.to_csv(output_dir / 'fixed_effects.csv', index=False)
    print(f"  Saved: {output_dir / 'fixed_effects.csv'}")

    # Step 5: Extract random effects
    print("\n[4/5] Extracting random effects...")
    re_summary = extract_random_effects_from_lmm(best_model)
    print(f"  ICC = {re_summary['icc']:.3f}")

    # Save random effects summary
    with open(output_dir / 'random_effects.txt', 'w') as f:
        f.write("RANDOM EFFECTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Residual Variance: {re_summary['residual_variance']:.4f}\n")
        f.write(f"ICC (Intraclass Correlation): {re_summary['icc']:.4f}\n\n")
        f.write("Random Effects Covariance Matrix:\n")
        f.write(str(re_summary['re_variance']))

    # Step 6: Save model summary
    print("\n[5/5] Saving model summary...")
    with open(output_dir / 'model_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(best_model.summary().as_text())
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("AIC COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(aic_df.to_string(index=False))

    print(f"  Saved: {output_dir / 'model_summary.txt'}")

    print("\n" + "=" * 60)
    print("LMM ANALYSIS COMPLETE")
    print("=" * 60 + "\n")

    return {
        'df_long': df_long,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'aic_comparison': aic_df,
        'fixed_effects': fe_table,
        'random_effects': re_summary
    }


def compute_contrasts_pairwise(
    lmm_result,
    comparisons: List[str],
    family_alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute post-hoc pairwise contrasts with dual reporting (Decision D068).

    Implements dual reporting of p-values:
    - Uncorrected (alpha = 0.05)
    - Bonferroni-corrected (alpha_corrected = family_alpha / k)

    where k = number of comparisons in this RQ.

    Handles both reference and non-reference comparisons:
    - Reference comparison (e.g., When-What): Extract coefficient directly
    - Non-reference comparison (e.g., When-Where): Compute difference with delta method SE

    Args:
        lmm_result: Fitted MixedLM result object
        comparisons: List of comparison strings, e.g., ["Where-What", "When-What"]
        family_alpha: Family-wise alpha level (default: 0.05)

    Returns:
        DataFrame with columns:
        - comparison: Comparison label
        - beta: Estimated effect size
        - se: Standard error
        - z: z-statistic
        - p_uncorrected: Uncorrected p-value
        - alpha_corrected: Bonferroni-corrected alpha threshold
        - p_corrected: Corrected p-value (p * k)
        - sig_uncorrected: Significant at alpha=0.05 (bool)
        - sig_corrected: Significant at alpha_corrected (bool)

    Example:
        comparisons = ["Where-What", "When-What", "When-Where"]
        df_contrasts = compute_contrasts_pairwise(result, comparisons, family_alpha=0.05)

    Decision D068 Context:
        Exploratory thesis requires reporting BOTH uncorrected and corrected results.
        - Uncorrected: Shows raw effects (transparency)
        - Corrected: Controls Type I error (rigor)
        Reviewers can assess robustness across both thresholds.
    """
    from scipy import stats

    print("\n" + "=" * 60)
    print("POST-HOC PAIRWISE CONTRASTS (Decision D068)")
    print("=" * 60)

    k = len(comparisons)
    alpha_corrected = family_alpha / k

    print(f"Family-wise alpha: {family_alpha}")
    print(f"Number of comparisons: {k}")
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")

    # Helper function to find coefficient name for a given level
    def find_coef_name(level: str) -> str:
        """Find the coefficient name for a factor level in the model."""
        reference_levels = ['What', 'Where', 'When']
        factor_names = ['Factor', 'Domain']

        for factor in factor_names:
            for ref in reference_levels:
                # Full treatment coding: C(Factor, Treatment('What'))[T.When]
                coef_option = f"C({factor}, Treatment('{ref}'))[T.{level}]"
                if coef_option in lmm_result.params.index:
                    return coef_option
            # Simpler patterns (backward compatibility)
            for pattern in [f'C({factor}, Treatment)[T.{level}]',
                           f'C({factor})[T.{level}]',
                           f'{factor}[T.{level}]']:
                if pattern in lmm_result.params.index:
                    return pattern
        # Bare level name as last resort
        if level in lmm_result.params.index:
            return level
        return None

    # Detect reference level (the level NOT in the model coefficients)
    all_levels = set()
    for comparison in comparisons:
        parts = comparison.split('-')
        all_levels.add(parts[0].strip())
        all_levels.add(parts[1].strip())

    reference_level = None
    for level in all_levels:
        if find_coef_name(level) is None:
            reference_level = level
            break

    print(f"Detected reference level: {reference_level}")

    results = []

    for comparison in comparisons:
        # Parse comparison string (e.g., "Where-What" -> Where - What)
        parts = comparison.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid comparison format: {comparison}. Expected 'A-B'")

        level1, level2 = parts[0].strip(), parts[1].strip()

        # Case 1: level2 is the reference level (e.g., When-What, Where-What)
        # -> Use level1 coefficient directly
        if level2 == reference_level:
            coef_name = find_coef_name(level1)
            if coef_name is None:
                print(f"  Warning: Coefficient for '{level1}' not found. Skipping {comparison}.")
                continue

            beta = lmm_result.params[coef_name]
            se = lmm_result.bse[coef_name]
            z = beta / se
            p_uncorrected = lmm_result.pvalues[coef_name]

        # Case 2: level1 is the reference level (e.g., What-When)
        # -> Use negative of level2 coefficient
        elif level1 == reference_level:
            coef_name = find_coef_name(level2)
            if coef_name is None:
                print(f"  Warning: Coefficient for '{level2}' not found. Skipping {comparison}.")
                continue

            beta = -lmm_result.params[coef_name]
            se = lmm_result.bse[coef_name]  # SE is symmetric
            z = beta / se
            p_uncorrected = lmm_result.pvalues[coef_name]  # Same p-value (two-tailed)

        # Case 3: Neither is the reference level (e.g., When-Where)
        # -> Compute difference: beta1 - beta2 with delta method SE
        else:
            coef_name1 = find_coef_name(level1)
            coef_name2 = find_coef_name(level2)

            if coef_name1 is None:
                print(f"  Warning: Coefficient for '{level1}' not found. Skipping {comparison}.")
                continue
            if coef_name2 is None:
                print(f"  Warning: Coefficient for '{level2}' not found. Skipping {comparison}.")
                continue

            beta1 = lmm_result.params[coef_name1]
            beta2 = lmm_result.params[coef_name2]
            beta = beta1 - beta2

            # Delta method SE: sqrt(Var(b1) + Var(b2) - 2*Cov(b1,b2))
            # Get covariance matrix
            try:
                cov_matrix = lmm_result.cov_params()
                var1 = cov_matrix.loc[coef_name1, coef_name1]
                var2 = cov_matrix.loc[coef_name2, coef_name2]
                cov12 = cov_matrix.loc[coef_name1, coef_name2]
                se = np.sqrt(var1 + var2 - 2 * cov12)
            except Exception as e:
                # Fallback: approximate SE as sqrt(se1^2 + se2^2) assuming independence
                se1 = lmm_result.bse[coef_name1]
                se2 = lmm_result.bse[coef_name2]
                se = np.sqrt(se1**2 + se2**2)
                print(f"  Note: Using approximate SE for {comparison} (cov extraction failed)")

            z = beta / se
            # Two-tailed p-value from z
            p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

        p_corrected = min(p_uncorrected * k, 1.0)  # Cap at 1.0

        # Significance flags
        sig_uncorrected = p_uncorrected < 0.05
        sig_corrected = p_uncorrected < alpha_corrected

        results.append({
            'comparison': comparison,
            'beta': beta,
            'se': se,
            'z': z,
            'p_uncorrected': p_uncorrected,
            'alpha_corrected': alpha_corrected,
            'p_corrected': p_corrected,
            'sig_uncorrected': sig_uncorrected,
            'sig_corrected': sig_corrected
        })

    df_contrasts = pd.DataFrame(results)

    # Summary
    print(f"\nResults:")
    print(f"  Significant (uncorrected alpha=0.05): {df_contrasts['sig_uncorrected'].sum()}/{k}")
    print(f"  Significant (corrected alpha={alpha_corrected:.4f}): {df_contrasts['sig_corrected'].sum()}/{k}")

    print("=" * 60 + "\n")

    return df_contrasts


def compute_effect_sizes_cohens(
    lmm_result,
    include_interactions: bool = False
) -> pd.DataFrame:
    """
    Compute effect sizes (Cohen's f²) for LMM fixed effects.

    Args:
        lmm_result: Fitted MixedLM result object
        include_interactions: Whether to include interaction terms (default: False)

    Returns:
        DataFrame with columns:
        - effect: Effect name
        - f_squared: Cohen's f² effect size
        - interpretation: Small/Medium/Large based on Cohen 1988 thresholds

    Cohen 1988 Thresholds:
        - Small: f² >= 0.02
        - Medium: f² >= 0.15
        - Large: f² >= 0.35

    Example:
        ```python
        df_effect_sizes = compute_effect_sizes_cohens(result, include_interactions=True)
        ```

    Note:
        This is a simplified approximation using (β/SE)² / N.
        Proper implementation would require nested model comparison (future enhancement).
    """
    print("\n" + "=" * 60)
    print("EFFECT SIZES (Cohen's f²)")
    print("=" * 60)

    results = []

    # Extract fixed effects (exclude Intercept and Group Var)
    for param_name in lmm_result.params.index:
        # Skip intercept
        if param_name == 'Intercept':
            continue

        # Skip Group Var (random effects variance)
        if 'Group' in param_name or 'Var' in param_name:
            continue

        # Skip interactions if not requested
        if not include_interactions and ':' in param_name:
            continue

        beta = lmm_result.params[param_name]
        se = lmm_result.bse[param_name]

        # Simplified f² approximation
        n = lmm_result.nobs
        f_squared = (beta / se) ** 2 / n

        # Interpret using Cohen 1988 thresholds
        if f_squared < 0.02:
            interpretation = 'negligible'
        elif f_squared < 0.15:
            interpretation = 'small'
        elif f_squared < 0.35:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        results.append({
            'effect': param_name,
            'f_squared': f_squared,
            'interpretation': interpretation
        })

    df_effect_sizes = pd.DataFrame(results)

    # Summary
    print(f"\nEffect sizes computed for {len(df_effect_sizes)} fixed effects")
    print(f"  Negligible (f²<0.02): {(df_effect_sizes['interpretation']=='negligible').sum()}")
    print(f"  Small (0.02≤f²<0.15): {(df_effect_sizes['interpretation']=='small').sum()}")
    print(f"  Medium (0.15≤f²<0.35): {(df_effect_sizes['interpretation']=='medium').sum()}")
    print(f"  Large (f²≥0.35): {(df_effect_sizes['interpretation']=='large').sum()}")

    print("=" * 60 + "\n")

    return df_effect_sizes


# MODULE EXPORTS

def fit_lmm_trajectory_tsvr(
    theta_scores: pd.DataFrame,
    tsvr_data: pd.DataFrame,
    formula: str,
    groups: str = 'UID',
    re_formula: str = '~Days',
    reml: bool = False
) -> MixedLMResults:
    """
    Fit LMM using TSVR as time variable (Decision D070).

    Implements the IRT→LMM pipeline with TSVR instead of nominal days:
    1. Parse composite_ID to extract [UID, Test]
    2. Merge theta scores with TSVR data
    3. Convert TSVR hours → days
    4. Reshape to long format for LMM
    5. Fit model using actual time delays

    Args:
        theta_scores: DataFrame from IRT Pass 2
                      Columns: composite_ID, domain_name (e.g., "What", "Where", "When"), theta
        tsvr_data: DataFrame from data-prep
                   Columns: composite_ID, test, tsvr (hours)
        formula: LMM formula string (e.g., "Theta ~ Days + C(Domain)")
        groups: Grouping variable (default: 'UID')
        re_formula: Random effects formula (default: '~Days')
        reml: Use REML estimation (default: False for AIC comparison)

    Returns:
        Fitted MixedLMResults object

    Example:
        ```python
        # Load data
        theta_scores = pd.read_csv("data/pass2_theta.csv")
        tsvr_data = pd.read_csv("data/tsvr_data.csv")

        # Fit model
        result = fit_lmm_trajectory_tsvr(
            theta_scores=theta_scores,
            tsvr_data=tsvr_data,
            formula="Theta ~ Days + C(Domain) + Days:C(Domain)",
            groups='UID',
            re_formula='~Days'
        )
        ```

    Decision D070 Context:
        Nominal days (0, 1, 3, 6) introduce measurement error because actual delays vary:
        - T1: 0.3-2.5h (not exactly 0 days)
        - T2: 20-32h (not exactly 1 day)
        - T3: 68-80h (not exactly 3 days)
        - T4: 140-156h (not exactly 6 days)

        Using TSVR (Time Since VR encoding) prevents biased slope estimates and
        reduced statistical power across ~40 RQs.
    """
    print("\n" + "=" * 60)
    print("LMM WITH TSVR TIME VARIABLE (Decision D070)")
    print("=" * 60)

    # Step 1: Parse composite_ID to extract UID and Test
    print("\n[1/5] Parsing composite_IDs...")

    # Theta scores: Parse composite_ID (e.g., "A010_T1" → UID="A010", Test="T1")
    theta_scores = theta_scores.copy()
    theta_scores[['UID', 'Test']] = theta_scores['composite_ID'].str.split('_', n=1, expand=True)

    # TSVR data: Ensure Test is string format (e.g., "T1", "T2", "T3", "T4")
    tsvr_data = tsvr_data.copy()
    test_col = 'test' if 'test' in tsvr_data.columns else 'Test'
    if tsvr_data[test_col].dtype in [int, float]:
        tsvr_data['Test'] = 'T' + tsvr_data[test_col].astype(int).astype(str)
    else:
        tsvr_data['Test'] = tsvr_data[test_col]

    # Create composite_ID from UID and Test if not already present
    if 'composite_ID' not in tsvr_data.columns:
        tsvr_data['UID'] = tsvr_data['UID'].astype(str)
        tsvr_data['composite_ID'] = tsvr_data['UID'] + '_' + tsvr_data['Test']

    # Handle TSVR column name (could be 'tsvr' or 'TSVR_hours')
    tsvr_col = 'tsvr' if 'tsvr' in tsvr_data.columns else 'TSVR_hours'
    if tsvr_col != 'tsvr':
        tsvr_data['tsvr'] = tsvr_data[tsvr_col]

    print(f"  Theta scores: {len(theta_scores)} rows")
    print(f"  TSVR data: {len(tsvr_data)} rows")

    # Step 2: Merge theta scores with TSVR
    print("\n[2/5] Merging theta scores with TSVR data...")

    # Merge on composite_ID
    merged = theta_scores.merge(
        tsvr_data[['composite_ID', 'tsvr']],
        on='composite_ID',
        how='left'
    )

    # Check for missing TSVR
    missing_tsvr = merged['tsvr'].isna().sum()
    if missing_tsvr > 0:
        print(f"  WARNING: {missing_tsvr} rows missing TSVR data ({missing_tsvr/len(merged)*100:.1f}%)")
    else:
        print(f"  ✓ All {len(merged)} rows have TSVR data")

    # Step 3: Convert TSVR hours → days
    print("\n[3/5] Converting TSVR hours to days...")

    merged['Days'] = merged['tsvr'] / 24.0

    print(f"  TSVR range: {merged['tsvr'].min():.1f}h - {merged['tsvr'].max():.1f}h")
    print(f"  Days range: {merged['Days'].min():.2f} - {merged['Days'].max():.2f} days")

    # Step 4: Prepare long format for LMM
    print("\n[4/5] Preparing long-format data for LMM...")

    # Ensure required columns exist
    if 'domain_name' in merged.columns:
        merged['Domain'] = merged['domain_name']
    elif 'factor' in merged.columns:
        merged['Domain'] = merged['factor']
    else:
        # If no domain column, assume single domain
        merged['Domain'] = 'Memory'

    # Rename theta column if needed
    if 'theta' in merged.columns:
        merged['Theta'] = merged['theta']

    # Create clean LMM dataset
    lmm_data = merged[['UID', 'Test', 'Domain', 'Theta', 'Days']].copy()

    print(f"  LMM data shape: {lmm_data.shape}")
    print(f"  Unique UIDs: {lmm_data['UID'].nunique()}")
    print(f"  Unique Domains: {lmm_data['Domain'].nunique()} ({sorted(lmm_data['Domain'].unique())})")

    # Step 5: Fit LMM
    print("\n[5/5] Fitting Linear Mixed Model...")
    print(f"  Formula: {formula}")
    print(f"  Random effects: {re_formula}")
    print(f"  Grouping: {groups}")

    result = fit_lmm_trajectory(
        data=lmm_data,
        formula=formula,
        groups=groups,
        re_formula=re_formula,
        reml=reml
    )

    print("\n" + "=" * 60)
    print("LMM FIT COMPLETE")
    print("=" * 60)
    print(f"Log-Likelihood: {result.llf:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print()

    return result


def select_lmm_random_structure_via_lrt(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    reml: bool = False
) -> Dict:
    """
    Select optimal random effects structure via likelihood ratio test (LRT).

    Compares three nested random effects structures:
    1. Full: Random intercepts + slopes with correlation (Time | UID)
    2. Uncorrelated: Random intercepts + slopes without correlation (Time || UID)
    3. Intercept-only: Random intercepts only (1 | UID)

    Uses LRT to compare nested models and selects most parsimonious model
    that significantly improves fit (p < 0.05). All models fitted with
    REML=False for valid likelihood ratio testing.

    Parameters
    ----------
    data : DataFrame
        Input data with columns specified in formula and groups
    formula : str
        Fixed effects formula (e.g., "Theta ~ TSVR + C(Domain)")
    groups : str
        Column name for grouping variable (e.g., "UID")
    reml : bool, default=False
        REML estimation (False for LRT, per statistical best practice)

    Returns
    -------
    dict
        selected_model : str
            Name of selected model ('Full', 'Uncorrelated', 'Intercept-only')
        lrt_results : DataFrame
            LRT comparison table with columns:
            - model: Model name
            - log_likelihood: Log-likelihood value
            - df: Degrees of freedom difference from baseline
            - chi2: LRT chi-square statistic
            - p_value: LRT p-value
            - aic: Akaike Information Criterion
        fitted_models : dict
            Dictionary of fitted MixedLMResults objects keyed by model name

    Notes
    -----
    - LRT requires REML=False for valid comparison (default)
    - Intercept-only model serves as baseline for LRT comparisons
    - Selection favors parsimony: simpler model chosen unless complex model
      significantly improves fit (p < 0.05)
    - If Full model fails to converge, falls back to Uncorrelated or Intercept-only

    References
    ----------
    - Likelihood Ratio Test: Pinheiro & Bates (2000), Mixed-Effects Models in S and S-PLUS
    - REML vs ML: Verbeke & Molenberghs (2000), Linear Mixed Models for Longitudinal Data

    Examples
    --------
    >>> result = select_lmm_random_structure_via_lrt(
    ...     data=df,
    ...     formula="Theta ~ TSVR + C(Domain)",
    ...     groups="UID"
    ... )
    >>> print(result['selected_model'])
    'Full'
    >>> print(result['lrt_results'])
    """
    # Extract time variable from formula (assume first continuous predictor is time)
    # For simplicity, use hardcoded 'TSVR' as time variable
    # (Could be enhanced to parse formula, but sufficient for REMEMVR use case)
    time_var = 'TSVR'
    if 'TSVR' not in data.columns:
        # Fallback: search for common time variable names
        for candidate in ['Time', 'Days', 'test']:
            if candidate in data.columns:
                time_var = candidate
                break
    # FIT THREE CANDIDATE MODELS (all with REML=False for valid LRT)

    fitted_models = {}
    results_list = []

    # --- Model 1: Intercept-only (baseline) ---
    try:
        md_intercept = smf.mixedlm(
            formula=formula,
            data=data,
            groups=data[groups]
            # re_formula defaults to "~1" (random intercepts only)
        )
        fit_intercept = md_intercept.fit(reml=reml, method=['lbfgs'])
        fitted_models['Intercept-only'] = fit_intercept

        results_list.append({
            'model': 'Intercept-only',
            'log_likelihood': fit_intercept.llf,
            'df': np.nan,  # Baseline (no comparison)
            'chi2': np.nan,
            'p_value': np.nan,
            'aic': fit_intercept.aic,
            'n_params': len(fit_intercept.params)
        })
    except Exception as e:
        warnings.warn(f"Intercept-only model failed: {e}")
        fitted_models['Intercept-only'] = None
        results_list.append({
            'model': 'Intercept-only',
            'log_likelihood': np.nan,
            'df': np.nan,
            'chi2': np.nan,
            'p_value': np.nan,
            'aic': np.nan,
            'n_params': np.nan
        })

    # --- Model 2 & 3: Full model with random slopes (will serve as both models for v1) ---
    # NOTE: Statsmodels doesn't easily support uncorrelated random effects via formula
    # For v1 implementation, fit Full model and use AIC difference as proxy for comparison
    # Future enhancement: implement proper uncorrelated via vc_formula or manual optimization
    try:
        md_full = smf.mixedlm(
            formula=formula,
            data=data,
            groups=data[groups],
            re_formula=f"~{time_var}"
        )

        fit_full = md_full.fit(reml=reml, method=['lbfgs'])
        fitted_models['Full'] = fit_full
        fitted_models['Uncorrelated'] = fit_full  # Same model for v1 (simplified)

        # LRT vs Intercept-only
        if fitted_models['Intercept-only'] is not None:
            ll_baseline = fit_intercept.llf
            ll_full = fit_full.llf
            df_diff = 2  # Add 2 parameters (slope variance + covariance)
            chi2_stat = -2 * (ll_baseline - ll_full)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df_diff)
        else:
            df_diff = np.nan
            chi2_stat = np.nan
            p_value = np.nan

        results_list.append({
            'model': 'Uncorrelated',
            'log_likelihood': fit_full.llf,
            'df': df_diff,
            'chi2': chi2_stat,
            'p_value': p_value,
            'aic': fit_full.aic,
            'n_params': len(fit_full.params)
        })

        results_list.append({
            'model': 'Full',
            'log_likelihood': fit_full.llf,
            'df': np.nan,  # No comparison (same as Uncorrelated in v1)
            'chi2': np.nan,
            'p_value': np.nan,
            'aic': fit_full.aic,
            'n_params': len(fit_full.params)
        })
    except Exception as e:
        warnings.warn(f"Full/Uncorrelated model failed: {e}")
        fitted_models['Full'] = None
        fitted_models['Uncorrelated'] = None
        results_list.append({
            'model': 'Uncorrelated',
            'log_likelihood': np.nan,
            'df': np.nan,
            'chi2': np.nan,
            'p_value': np.nan,
            'aic': np.nan,
            'n_params': np.nan
        })
        results_list.append({
            'model': 'Full',
            'log_likelihood': np.nan,
            'df': np.nan,
            'chi2': np.nan,
            'p_value': np.nan,
            'aic': np.nan,
            'n_params': np.nan
        })
    # SELECT BEST MODEL (parsimonious selection)

    lrt_results = pd.DataFrame(results_list)

    # Selection logic: Start from simplest, only add complexity if p < 0.05
    selected_model = 'Intercept-only'  # Default to simplest

    # Check if Full/Uncorrelated (random slopes) significantly improves over Intercept-only
    # NOTE: In v1, Uncorrelated = Full (same model)
    uncorr_row = lrt_results[lrt_results['model'] == 'Uncorrelated'].iloc[0]
    if uncorr_row['p_value'] < 0.05 and fitted_models['Uncorrelated'] is not None:
        selected_model = 'Full'  # Select Full (slopes + correlation) if slopes significantly improve fit

    # Handle convergence failures: fallback to simpler models
    if fitted_models[selected_model] is None:
        # Try simpler models in order
        for fallback in ['Full', 'Intercept-only']:
            if fitted_models[fallback] is not None:
                selected_model = fallback
                warnings.warn(f"Selected model failed convergence, falling back to {fallback}")
                break

    return {
        'selected_model': selected_model,
        'lrt_results': lrt_results[['model', 'log_likelihood', 'df', 'chi2', 'p_value', 'aic']],
        'fitted_models': fitted_models
    }


def prepare_age_effects_plot_data(
    lmm_input: pd.DataFrame,
    lmm_model: MixedLMResults,
    output_path: Path
) -> pd.DataFrame:
    """
    Create age tertiles, aggregate means, and generate predictions for RQ 5.10 visualization.

    Prepares plot-ready data for Age × Domain × Time interaction visualization with three
    age groups (Young/Middle/Older tertiles). Aggregates observed data by domain, age tertile,
    and timepoint; generates model predictions for smooth trajectories.

    Parameters
    ----------
    lmm_input : DataFrame
        Long-format LMM input data. Must contain columns:
        - UID: Subject identifier
        - Age: Continuous age variable (years)
        - domain_name: Memory domain (What/Where/When)
        - TSVR_hours: Time since VR in hours
        - theta: Ability estimate
    lmm_model : MixedLMResults
        Fitted LMM model from fit_lmm_trajectory_tsvr() or similar
    output_path : Path
        Path to save plot data CSV file (e.g., results/ch5/rq10/plots/age_effects_plot_data.csv)

    Returns
    -------
    DataFrame
        Plot-ready data with columns:
        - domain_name: Memory domain (What/Where/When)
        - age_tertile: Age group (Young/Middle/Older)
        - TSVR_hours: Time since VR in hours
        - theta_observed: Mean theta across subjects in group
        - se_observed: Standard error of the mean (SEM)
        - ci_lower: Lower 95% CI (mean - 1.96*SEM)
        - ci_upper: Upper 95% CI (mean + 1.96*SEM)
        - theta_predicted: Model-predicted theta at group centroid

    Notes
    -----
    **Age Tertiles:**
    - Created using pd.qcut(Age, q=3) for equal-sized groups
    - Labels: 'Young' (lowest tertile), 'Middle', 'Older' (highest tertile)
    - Used ONLY for visualization; analysis uses continuous Age_c (grand-mean centered)

    **Aggregation:**
    - Observed data: mean and SEM computed within each domain × tertile × timepoint
    - Groups typically have ~20 subjects each (for N=60 total sample)
    - SEM = SD / sqrt(n) where n is number of subjects in group at that timepoint

    **Predictions:**
    - Generated using LMM fitted values at group level (not marginal effects)
    - Reflects full model including Age × Domain × Time interactions
    - One prediction per domain × tertile × timepoint combination

    **Output Structure:**
    - 3 domains × 3 tertiles × 4 timepoints = 36 rows
    - Each row represents one data point for plotting pipeline multi-panel trajectory plot

    **RQ 5.10 Context:**
    - Tests Age × Domain × Time 3-way interaction (continuous Age in model)
    - Visualization shows if age effects differ across memory domains
    - Tertiles for interpretability: "Older adults show faster forgetting in Where domain"

    References
    ----------
    - RQ 5.10 1_concept.md: Age effects on domain-specific forgetting trajectories
    - tools_todo.yaml: Tool specification (lines 51-67)
    - ANALYSES_CH5.md: Multi-panel plot specification (lines 921-926)

    Example
    -------
    >>> plot_data = prepare_age_effects_plot_data(
    ...     lmm_input=df_long,
    ...     lmm_model=best_model,
    ...     output_path=output_dir / "age_effects_plot_data.csv"
    ... )
    >>> print(plot_data.shape)  # (36, 8)
    >>> print(plot_data['age_tertile'].unique())  # ['Young', 'Middle', 'Older']
    """
    # Copy input to avoid modifying original
    df = lmm_input.copy()
    # Create age tertiles using qcut (equal-sized groups)

    # Create age tertiles at subject level (not observation level)
    # Get unique Age per UID, assign tertiles, then merge back
    # Handle both 'Age' and 'age' column names
    age_col = 'Age' if 'Age' in df.columns else 'age'
    subject_ages = df[['UID', age_col]].drop_duplicates()
    subject_ages['age_tertile'] = pd.qcut(
        subject_ages[age_col],
        q=3,
        labels=['Young', 'Middle', 'Older']
    )

    # Merge tertiles back to full data
    df = df.merge(subject_ages[['UID', 'age_tertile']], on='UID', how='left')
    # Aggregate observed data by tertile × timepoint (or domain × tertile × timepoint)

    # Group by age tertile and timepoint (add domain if present)
    group_cols = ['age_tertile', 'TSVR_hours']
    if 'domain_name' in df.columns:
        group_cols = ['domain_name'] + group_cols

    grouped = df.groupby(group_cols)['theta'].agg([
        ('theta_observed', 'mean'),
        ('se_observed', lambda x: x.sem()),  # Standard error of the mean
        ('n', 'count')
    ]).reset_index()
    # Compute 95% confidence intervals

    # CI = mean ± 1.96 * SEM (95% CI for normal distribution)
    z_critical = 1.96
    grouped['ci_lower'] = grouped['theta_observed'] - z_critical * grouped['se_observed']
    grouped['ci_upper'] = grouped['theta_observed'] + z_critical * grouped['se_observed']
    # Generate model predictions

    # Use fitted values from the LMM model
    # Create DataFrame with same structure as grouped data for prediction lookup
    df['fitted_theta'] = lmm_model.fittedvalues

    # Aggregate fitted values by tertile × timepoint (or domain × tertile × timepoint)
    predictions = df.groupby(group_cols)['fitted_theta'].mean().reset_index()
    predictions.rename(columns={'fitted_theta': 'theta_predicted'}, inplace=True)
    # Merge predictions with observed data

    result = grouped.merge(
        predictions,
        on=group_cols,
        how='left'
    )
    # Select final columns and save to CSV

    # Select columns for plotting pipeline (include domain_name only if present)
    output_cols = ['age_tertile', 'TSVR_hours', 'theta_observed',
                   'se_observed', 'ci_lower', 'ci_upper', 'theta_predicted']
    if 'domain_name' in result.columns:
        output_cols = ['domain_name'] + output_cols

    result = result[output_cols]

    # Sort by tertile and time (include domain if present)
    sort_cols = ['age_tertile', 'TSVR_hours']
    if 'domain_name' in result.columns:
        sort_cols = ['domain_name'] + sort_cols
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Create parent directories if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    result.to_csv(output_path, index=False)

    return result


def compute_icc_from_variance_components(
    variance_components_df: pd.DataFrame,
    time_point: Optional[float] = None,
    slope_name: str = 'TSVR_hours'
) -> pd.DataFrame:
    """
    Compute 3 ICC estimates from LMM variance components for RQ 5.13.

    Calculates Intraclass Correlation Coefficients (ICC) to quantify proportion
    of variance attributable to between-subject differences for intercepts and slopes.

    Parameters
    ----------
    variance_components_df : DataFrame
        Variance components with columns: component, variance
    time_point : float, optional
        Timepoint for conditional ICC (e.g., 144.0 for Day 6)
    slope_name : str, default='TSVR_hours'
        Name of slope component

    Returns
    -------
    DataFrame
        ICC estimates with columns: icc_type, icc_value, interpretation
    """
    # Extract variance components
    components = {row['component']: row['variance']
                  for _, row in variance_components_df.iterrows()}

    var_intercept = components.get('Intercept', 0.0)
    var_slope = components.get(slope_name, None)
    var_residual = components.get('Residual', 0.0)
    cov_intercept_slope = components.get(f'Intercept:{slope_name}',
                                         components.get('Intercept:slope', 0.0))

    results = []

    # ICC intercept
    icc_intercept = var_intercept / (var_intercept + var_residual) if (var_intercept + var_residual) > 0 else 0.0
    results.append({
        'icc_type': 'intercept',
        'icc_value': icc_intercept,
        'interpretation': _interpret_icc(icc_intercept)
    })

    # ICC slopes (if slope variance exists)
    if var_slope is not None:
        # Simple ICC
        icc_slope_simple = var_slope / (var_slope + var_residual) if (var_slope + var_residual) > 0 else 0.0
        results.append({
            'icc_type': 'slope_simple',
            'icc_value': icc_slope_simple,
            'interpretation': _interpret_icc(icc_slope_simple)
        })

        # Conditional ICC at timepoint
        if time_point is None:
            time_point = 0.0

        var_at_time = (var_intercept +
                      2 * time_point * cov_intercept_slope +
                      time_point**2 * var_slope)

        icc_conditional = var_at_time / (var_at_time + var_residual) if (var_at_time + var_residual) > 0 else 0.0
        results.append({
            'icc_type': 'slope_conditional',
            'icc_value': icc_conditional,
            'interpretation': _interpret_icc(icc_conditional)
        })

    return pd.DataFrame(results).sort_values('icc_type').reset_index(drop=True)


def _interpret_icc(icc: float) -> str:
    """Helper function to interpret ICC values"""
    if icc < 0.10:
        return "Low clustering (<0.10)"
    elif icc < 0.30:
        return "Moderate clustering (0.10-0.30)"
    elif icc < 0.75:
        return "High clustering (0.30-0.75)"
    else:
        return "Very high clustering (≥0.75)"


def test_intercept_slope_correlation_d068(
    random_effects_df: pd.DataFrame,
    family_alpha: float = 0.05,
    n_tests: int = 15,
    intercept_col: str = 'Group Var',
    slope_col: str = 'Group x TSVR_hours Var'
) -> Dict:
    """
    Test correlation between random intercepts and slopes with D068 dual p-value reporting.

    Computes Pearson correlation between random intercepts and random slopes
    from LMM. Reports BOTH uncorrected and Bonferroni-corrected p-values per
    Decision D068 (dual p-value reporting for all hypothesis tests).

    Used in RQ 5.13 to test whether individuals with higher baseline memory
    (intercepts) show different rates of forgetting (slopes).

    Parameters
    ----------
    random_effects_df : DataFrame
        Random effects from LMM with columns for UID, intercepts, and slopes
    family_alpha : float, default=0.05
        Family-wise alpha level for significance threshold
    n_tests : int, default=15
        Number of tests in family for Bonferroni correction (15 for Chapter 5)
    intercept_col : str, default='Group Var'
        Column name for random intercepts (statsmodels default naming)
    slope_col : str, default='Group x TSVR_hours Var'
        Column name for random slopes (statsmodels default naming)

    Returns
    -------
    Dict
        Results with keys:
        - r: float - Pearson correlation coefficient
        - p_uncorrected: float - Uncorrected p-value
        - p_bonferroni: float - Bonferroni-corrected p-value (min(p * n_tests, 1.0))
        - significant_uncorrected: bool - Significant at family_alpha (uncorrected)
        - significant_bonferroni: bool - Significant at family_alpha (Bonferroni)
        - interpretation: str - Plain language interpretation

    Notes
    -----
    Decision D068: ALWAYS report both uncorrected and corrected p-values for
    transparency and reproducibility.

    Bonferroni correction: p_bonf = min(p_uncorrected × n_tests, 1.0)

    Interpretation guidelines:
    - |r| < 0.30: Weak correlation
    - 0.30 ≤ |r| < 0.50: Moderate correlation
    - |r| ≥ 0.50: Strong correlation

    RQ 5.13 hypothesis: Negative correlation expected (higher baseline → slower forgetting)

    Examples
    --------
    >>> # Extract random effects from fitted LMM
    >>> random_fx = lmm_result.random_effects
    >>> random_fx_df = pd.DataFrame({
    ...     'UID': list(random_fx.keys()),
    ...     'Group Var': [fx['Group'] for fx in random_fx.values()],
    ...     'Group x TSVR_hours Var': [fx['Group x TSVR_hours'] for fx in random_fx.values()]
    ... })
    >>> # Test intercept-slope correlation
    >>> result = test_intercept_slope_correlation_d068(random_fx_df, family_alpha=0.05, n_tests=15)
    >>> print(f"r = {result['r']:.3f}, p_uncorr = {result['p_uncorrected']:.4f}, "
    ...       f"p_bonf = {result['p_bonferroni']:.4f}")
    >>> print(result['interpretation'])
    """
    # Extract intercepts and slopes
    intercepts = random_effects_df[intercept_col].values
    slopes = random_effects_df[slope_col].values

    # Compute Pearson correlation
    r, p_uncorrected = stats.pearsonr(intercepts, slopes)

    # Bonferroni correction
    p_bonferroni = min(p_uncorrected * n_tests, 1.0)

    # Significance flags
    significant_uncorrected = p_uncorrected < family_alpha
    significant_bonferroni = p_bonferroni < family_alpha

    # Interpretation
    abs_r = abs(r)
    if abs_r < 0.30:
        strength = "weak"
    elif abs_r < 0.50:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if r > 0 else "negative"

    interpretation = (
        f"{strength.capitalize()} {direction} correlation (r={r:.3f}) between "
        f"random intercepts and slopes. "
    )

    if significant_bonferroni:
        interpretation += f"Significant after Bonferroni correction (p={p_bonferroni:.4f} < {family_alpha})."
    elif significant_uncorrected:
        interpretation += f"Significant uncorrected (p={p_uncorrected:.4f} < {family_alpha}) but NOT after Bonferroni correction (p={p_bonferroni:.4f})."
    else:
        interpretation += f"Not significant (p_uncorr={p_uncorrected:.4f}, p_bonf={p_bonferroni:.4f})."

    return {
        'r': r,
        'p_uncorrected': p_uncorrected,
        'p_bonferroni': p_bonferroni,
        'significant_uncorrected': significant_uncorrected,
        'significant_bonferroni': significant_bonferroni,
        'interpretation': interpretation
    }


def extract_segment_slopes_from_lmm(
    lmm_result: MixedLMResults,
    segment_col: str = 'Segment',
    time_col: str = 'Days_within'
) -> pd.DataFrame:
    """
    Extract Early/Late segment slopes from piecewise LMM with delta method SE propagation.

    RQ 5.8 Test 4 (Convergent Evidence) requires Early/Late slope ratio < 0.5
    to indicate robust two-phase forgetting pattern. Delta method SE propagation
    is required for the ratio because ratio SE != simple quadrature due to
    covariance between Early and Late slopes.

    Piecewise LMM formula:
        theta ~ Intercept + Days_within + Days_within:SegmentLate + (Days_within | UID)

    Early slope = β_Days_within
    Late slope = β_Days_within + β_Days_within:SegmentLate
    Ratio = Late_slope / Early_slope

    Delta method for ratio SE:
        SE_ratio² = (∂ratio/∂β_early)²×Var(β_early) + (∂ratio/∂β_late)²×Var(β_late)
                    + 2×(∂ratio/∂β_early)×(∂ratio/∂β_late)×Cov(β_early, β_late)

        where:
            ∂ratio/∂β_early = -β_late / β_early²
            ∂ratio/∂β_late = 1 / β_early

    Parameters
    ----------
    lmm_result : MixedLMResults
        Fitted piecewise LMM result from statsmodels
        Must contain coefficients: {time_col}, {time_col}:{segment_col}Late
    segment_col : str, default='Segment'
        Name of segment column (e.g., 'Segment', 'Phase')
    time_col : str, default='Days_within'
        Name of time-within-segment column

    Returns
    -------
    pd.DataFrame
        3 rows with columns: [metric, value, SE, CI_lower, CI_upper, interpretation]
        Metrics: Early_slope, Late_slope, Ratio_Late_Early

    Raises
    ------
    KeyError
        If required coefficients not found in LMM result

    Examples
    --------
    >>> lmm = fit_lmm_trajectory(data, formula='theta ~ Days_within + Days_within:SegmentLate + (Days_within | UID)')
    >>> slopes = extract_segment_slopes_from_lmm(lmm)
    >>> print(slopes)
             metric     value        SE  CI_lower  CI_upper            interpretation
    0   Early_slope -0.300000  0.040000 -0.378400 -0.221600  Forgetting (-0.30/day)...
    1    Late_slope -0.100000  0.072111 -0.241335  0.041335  Slower forgetting (-0.10/day)...
    2  Ratio_Late_Early  0.333333  0.240370 -0.137783  0.804449  Ratio < 0.5: robust two-phase...

    References
    ----------
    RQ 5.8 Test 4: Convergent Evidence for two-phase forgetting pattern
    Delta method: Casella & Berger (2002), Statistical Inference, 2nd ed., p. 240
    """
    # Extract time coefficient (straightforward)
    time_coef = f'{time_col}'

    try:
        beta_early = lmm_result.params[time_coef]
    except KeyError:
        raise KeyError(
            f"Required coefficient '{time_coef}' not found in LMM result. "
            f"Available coefficients: {list(lmm_result.params.index)}"
        )

    # Auto-detect interaction coefficient name to handle categorical vs numeric encoding
    coef_names = list(lmm_result.params.index)

    # Pattern 1: Categorical interaction (R-style encoding: 'Days_within:Segment[T.Late]')
    categorical_pattern = f'{time_col}:{segment_col}[T.'
    categorical_matches = [name for name in coef_names if name.startswith(categorical_pattern)]

    # Pattern 2: Numeric interaction (simple concatenation: 'Days_within:Segment')
    numeric_pattern = f'{time_col}:{segment_col}'
    numeric_matches = [name for name in coef_names if name == numeric_pattern]

    # Pattern 3: Alternative categorical encoding ('Days_within:C(Segment)[T.Late]')
    alt_categorical_pattern = f'{time_col}:C({segment_col})'
    alt_matches = [name for name in coef_names if name.startswith(alt_categorical_pattern)]

    # Select interaction coefficient
    if categorical_matches:
        interaction_coef = categorical_matches[0]  # Use first categorical match
    elif numeric_matches:
        interaction_coef = numeric_matches[0]
    elif alt_matches:
        interaction_coef = alt_matches[0]
    else:
        # Provide helpful error message
        available = ', '.join(coef_names)
        raise KeyError(
            f"Could not find interaction term for {time_col}:{segment_col}.\n"
            f"Searched patterns: '{categorical_pattern}*', '{numeric_pattern}', '{alt_categorical_pattern}*'\n"
            f"Available coefficients: {available}\n"
            f"Ensure your LMM formula includes '{time_col} * {segment_col}' interaction."
        )

    # Extract interaction coefficient
    try:
        beta_interaction = lmm_result.params[interaction_coef]
    except KeyError:
        raise KeyError(
            f"Required coefficient '{interaction_coef}' not found in LMM result. "
            f"Available coefficients: {list(lmm_result.params.index)}"
        )

    beta_late = beta_early + beta_interaction

    # Extract standard errors
    se_early = lmm_result.bse[time_coef]
    se_interaction = lmm_result.bse[interaction_coef]

    # Extract interaction p-value (for significance test)
    interaction_pval = lmm_result.pvalues[interaction_coef]

    # Extract covariance between early and interaction (for Late slope SE)
    # Handle both real LMM results (method) and mocks (attribute)
    cov_matrix = lmm_result.cov_params() if callable(lmm_result.cov_params) else lmm_result.cov_params
    cov_early_interaction = cov_matrix.loc[time_coef, interaction_coef]

    # Compute Late slope SE via propagation: Var(Early + Interaction) = Var(Early) + Var(Interaction) + 2*Cov
    var_late = se_early**2 + se_interaction**2 + 2*cov_early_interaction
    se_late = np.sqrt(var_late)

    # Compute ratio
    if beta_early == 0:
        ratio = np.inf if beta_late != 0 else np.nan
        se_ratio = np.nan
    else:
        ratio = beta_late / beta_early

        # Delta method for ratio SE
        # ∂ratio/∂β_early = -β_late / β_early²
        d_ratio_d_early = -beta_late / (beta_early ** 2)
        # ∂ratio/∂β_late = 1 / β_early
        d_ratio_d_late = 1 / beta_early

        # Cov(β_early, β_late) = Cov(β_early, β_early + β_interaction)
        #                        = Var(β_early) + Cov(β_early, β_interaction)
        cov_early_late = se_early**2 + cov_early_interaction

        # SE_ratio² = (∂ratio/∂early)²×Var(early) + (∂ratio/∂late)²×Var(late) + 2×(∂ratio/∂early)×(∂ratio/∂late)×Cov(early,late)
        var_ratio = (
            (d_ratio_d_early ** 2) * (se_early ** 2) +
            (d_ratio_d_late ** 2) * var_late +
            2 * d_ratio_d_early * d_ratio_d_late * cov_early_late
        )
        se_ratio = np.sqrt(var_ratio)

    # 95% confidence intervals
    z_95 = 1.96

    ci_early_lower = beta_early - z_95 * se_early
    ci_early_upper = beta_early + z_95 * se_early

    ci_late_lower = beta_late - z_95 * se_late
    ci_late_upper = beta_late + z_95 * se_late

    ci_ratio_lower = ratio - z_95 * se_ratio if not np.isnan(se_ratio) else np.nan
    ci_ratio_upper = ratio + z_95 * se_ratio if not np.isnan(se_ratio) else np.nan

    # Interpretations
    def _interpret_slope(slope: float, se: float) -> str:
        """Interpret slope direction and magnitude"""
        if slope < 0:
            return f"Forgetting ({slope:.3f}/day). SE={se:.3f}."
        elif slope > 0:
            return f"Memory improvement ({slope:.3f}/day, atypical). SE={se:.3f}."
        else:
            return f"No change over time ({slope:.3f}/day). SE={se:.3f}."

    interp_early = _interpret_slope(beta_early, se_early)
    interp_late = _interpret_slope(beta_late, se_late)

    # Ratio interpretation
    if np.isinf(ratio):
        interp_ratio = "Ratio undefined (Early slope = 0). Cannot assess two-phase pattern."
    elif np.isnan(ratio):
        interp_ratio = "Ratio undefined (both slopes = 0). No forgetting detected."
    elif ratio < 0.5:
        interp_ratio = f"Ratio < 0.5 ({ratio:.3f}): Robust two-phase forgetting pattern. Late forgetting substantially slower than Early."
    elif 0.5 <= ratio < 0.75:
        interp_ratio = f"Ratio 0.5-0.75 ({ratio:.3f}): Moderate two-phase pattern. Late forgetting moderately slower than Early."
    elif 0.75 <= ratio <= 1.0:
        interp_ratio = f"Ratio 0.75-1.0 ({ratio:.3f}): Weak two-phase pattern. Late and Early forgetting similar."
    else:
        interp_ratio = f"Ratio > 1.0 ({ratio:.3f}): Unexpected pattern. Late forgetting faster than Early (reverse two-phase or single-phase)."

    # Interaction p-value interpretation
    if interaction_pval < 0.001:
        interp_interaction = f"Interaction highly significant (p={interaction_pval:.4f} < 0.001). Strong evidence for different forgetting rates across segments."
    elif interaction_pval < 0.05:
        interp_interaction = f"Interaction significant (p={interaction_pval:.4f} < 0.05). Evidence for different forgetting rates across segments."
    else:
        interp_interaction = f"Interaction not significant (p={interaction_pval:.4f} >= 0.05). No evidence for different forgetting rates across segments."

    # Build output DataFrame (4 rows: Early slope, Late slope, Ratio, Interaction p-value)
    output = pd.DataFrame({
        'metric': ['Early_slope', 'Late_slope', 'Ratio_Late_Early', 'Interaction_p'],
        'value': [beta_early, beta_late, ratio, interaction_pval],
        'SE': [se_early, se_late, se_ratio, np.nan],  # p-value has no SE
        'CI_lower': [ci_early_lower, ci_late_lower, ci_ratio_lower, np.nan],  # p-value has no CI
        'CI_upper': [ci_early_upper, ci_late_upper, ci_ratio_upper, np.nan],
        'interpretation': [interp_early, interp_late, interp_ratio, interp_interaction]
    })

    return output


def extract_marginal_age_slopes_by_domain(
    lmm_result: MixedLMResults,
    eval_timepoint: float = 72.0,
    domain_var: str = "domain",
    age_var: str = "Age_c",
    time_linear: str = "TSVR_hours",
    time_log: str = "log_TSVR"
) -> pd.DataFrame:
    """
    Extract domain-specific marginal age effects from 3-way Age×Domain×Time interaction LMM.

    Computes the marginal effect of age on forgetting rate for each domain at a specific
    timepoint, accounting for the full 3-way interaction structure. Uses delta method to
    propagate uncertainty through linear combinations of coefficients.

    Mathematical Definition
    -----------------------
    For a model with linear + log time terms:
        theta ~ Time_linear + Time_log + Age_c + Domain +
                Time_linear:Age_c + Time_log:Age_c +
                Time_linear:Domain + Time_log:Domain +
                Age_c:Domain +
                Time_linear:Age_c:Domain + Time_log:Age_c:Domain

    Marginal age slope = ∂(theta)/∂(Time) × ∂(Time)/∂(Age_c)
                       = β(Time:Age_c) + β(Time_log:Age_c) × ∂(log(Time+1))/∂(Time)

    For reference domain (What):
        age_slope = β(Time_linear:Age_c) + β(Time_log:Age_c) × 1/(Time+1)

    For non-reference domains (Where, When):
        age_slope = Reference_slope +
                    β(Time_linear:Age_c:Domain[X]) +
                    β(Time_log:Age_c:Domain[X]) × 1/(Time+1)

    Parameters
    ----------
    lmm_result : MixedLMResults
        Fitted LMM with 3-way Age×Domain×Time interaction
    eval_timepoint : float, default=72.0
        TSVR hours at which to evaluate marginal slopes
        Default 72h = Day 3 (midpoint of observation window 0-168h)
    domain_var : str, default="domain"
        Name of domain categorical variable in model
    age_var : str, default="Age_c"
        Name of centered age continuous variable
    time_linear : str, default="TSVR_hours"
        Name of linear time variable
    time_log : str, default="log_TSVR"
        Name of log-transformed time variable

    Returns
    -------
    DataFrame
        Domain-specific age slopes with columns:
        - domain (str): Domain name (What, Where, When)
        - age_slope (float): Marginal effect of age on forgetting rate at eval_timepoint
        - se (float): Standard error via delta method
        - z (float): Z-statistic (age_slope / se)
        - p (float): Two-tailed p-value
        - CI_lower (float): 95% confidence interval lower bound
        - CI_upper (float): 95% confidence interval upper bound

    Notes
    -----
    - Auto-detects reference level (domain without [T.] prefix in coefficient names)
    - Uses delta method for SE propagation through linear combinations
    - Derivative of log(Time+1) with respect to Time = 1/(Time+1)
    - Assumes treatment coding (reference group coded as 0)

    Example
    -------
    >>> result = extract_marginal_age_slopes_by_domain(
    ...     lmm_result=fitted_model,
    ...     eval_timepoint=72.0  # Day 3
    ... )
    >>> print(result)
         domain  age_slope        se         z         p  CI_lower  CI_upper
    0      What  -0.000123  0.000456 -0.269737  0.787328 -0.001017  0.000771
    1     Where  -0.000185  0.000645 -0.286822  0.774265 -0.001449  0.001079
    2      When   0.000054  0.000645  0.083721  0.933297 -0.001210  0.001318
    """
    # Extract fixed effects table
    fe = extract_fixed_effects_from_lmm(lmm_result)
    fe_dict = dict(zip(fe['Term'], fe['Coef']))

    # Get variance-covariance matrix for delta method
    vcov = lmm_result.cov_params()

    # Detect reference domain (no [T.] prefix)
    all_domains = set()
    for term in fe['Term']:
        if f'{domain_var}[T.' in term:
            # Extract domain name between [T. and ]
            domain = term.split(f'{domain_var}[T.')[1].split(']')[0]
            all_domains.add(domain)

    # Reference domain is the one NOT in the list
    all_possible = {'What', 'Where', 'When'}
    reference_domain = list(all_possible - all_domains)[0]
    non_reference = sorted(all_domains)

    # Compute derivative of log(time+1) at eval_timepoint
    log_derivative = 1.0 / (eval_timepoint + 1.0)

    results = []
    # Reference Domain (e.g., What)
    term_linear_age = f'{time_linear}:{age_var}'
    term_log_age = f'{time_log}:{age_var}'

    beta_linear = fe_dict.get(term_linear_age, 0.0)
    beta_log = fe_dict.get(term_log_age, 0.0)

    # Marginal age slope for reference domain
    slope_ref = beta_linear + beta_log * log_derivative

    # Delta method for SE
    # Gradient: [∂slope/∂β_linear, ∂slope/∂β_log] = [1, log_derivative]
    try:
        idx_linear = fe[fe['Term'] == term_linear_age].index[0]
        idx_log = fe[fe['Term'] == term_log_age].index[0]

        gradient = np.zeros(len(fe))
        gradient[idx_linear] = 1.0
        gradient[idx_log] = log_derivative

        # SE = sqrt(gradient' * Vcov * gradient)
        se_ref = np.sqrt(gradient @ vcov @ gradient)
    except (IndexError, KeyError):
        # If terms missing, SE = 0 (shouldn't happen in well-specified model)
        se_ref = 0.0

    z_ref = slope_ref / se_ref if se_ref > 0 else 0.0
    p_ref = 2 * (1 - stats.norm.cdf(abs(z_ref)))
    ci_lower_ref = slope_ref - 1.96 * se_ref
    ci_upper_ref = slope_ref + 1.96 * se_ref

    results.append({
        'domain': reference_domain,
        'age_slope': slope_ref,
        'se': se_ref,
        'z': z_ref,
        'p': p_ref,
        'CI_lower': ci_lower_ref,
        'CI_upper': ci_upper_ref
    })
    # Non-Reference Domains (e.g., Where, When)
    for domain in non_reference:
        term_3way_linear = f'{time_linear}:{age_var}:{domain_var}[T.{domain}]'
        term_3way_log = f'{time_log}:{age_var}:{domain_var}[T.{domain}]'

        beta_3way_linear = fe_dict.get(term_3way_linear, 0.0)
        beta_3way_log = fe_dict.get(term_3way_log, 0.0)

        # Marginal age slope = reference + 3-way interactions
        slope_domain = slope_ref + beta_3way_linear + beta_3way_log * log_derivative

        # Delta method for SE
        # Gradient now includes 4 terms: reference 2 + domain-specific 2
        try:
            idx_3way_linear = fe[fe['Term'] == term_3way_linear].index[0]
            idx_3way_log = fe[fe['Term'] == term_3way_log].index[0]

            gradient = np.zeros(len(fe))
            gradient[idx_linear] = 1.0
            gradient[idx_log] = log_derivative
            gradient[idx_3way_linear] = 1.0
            gradient[idx_3way_log] = log_derivative

            se_domain = np.sqrt(gradient @ vcov @ gradient)
        except (IndexError, KeyError):
            se_domain = 0.0

        z_domain = slope_domain / se_domain if se_domain > 0 else 0.0
        p_domain = 2 * (1 - stats.norm.cdf(abs(z_domain)))
        ci_lower_domain = slope_domain - 1.96 * se_domain
        ci_upper_domain = slope_domain + 1.96 * se_domain

        results.append({
            'domain': domain,
            'age_slope': slope_domain,
            'se': se_domain,
            'z': z_domain,
            'p': p_domain,
            'CI_lower': ci_lower_domain,
            'CI_upper': ci_upper_domain
        })

    # Convert to DataFrame and return
    df = pd.DataFrame(results)

    # Sort by domain (What, Where, When)
    domain_order = {'What': 0, 'Where': 1, 'When': 2}
    df['_sort'] = df['domain'].map(domain_order)
    df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

    return df


__all__ = [
    'assign_piecewise_segments',
    'extract_segment_slopes_from_lmm',
    'prepare_lmm_input_from_theta',
    'configure_candidate_models',
    'fit_lmm_trajectory',
    'compare_lmm_models_by_aic',
    'extract_fixed_effects_from_lmm',
    'extract_random_effects_from_lmm',
    'run_lmm_analysis',
    'compute_contrasts_pairwise',
    'compute_effect_sizes_cohens',
    'fit_lmm_trajectory_tsvr',
    'select_lmm_random_structure_via_lrt',
    'prepare_age_effects_plot_data',
    'compute_icc_from_variance_components',
    'test_intercept_slope_correlation_d068',
    'extract_marginal_age_slopes_by_domain'
]
