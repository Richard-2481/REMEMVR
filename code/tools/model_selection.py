"""
LMM Model Selection Tool - Kitchen Sink Approach

Universal model comparison tool for longitudinal mixed-effects models.
Tests comprehensive suite of time transformations (70+ models) to identify
best-fitting functional form for forgetting trajectories.

Author: REMEMVR Team
Date: 2025-12-08
Version: 1.0.0

CRITICAL REQUIREMENTS:
1. ALWAYS uses continuous TSVR (never nominal days/sessions)
2. Works with 0/1/2-way interactions (no 3-way - insufficient data)
3. Random slopes + intercepts (RQ specifies which time variable)
4. Comprehensive model suite (~70 models: polynomial, logarithmic, power-law,
   root, reciprocal, exponential, trigonometric, hyperbolic, hybrids)
5. Standardized outputs for RQ workflow integration

Design Philosophy:
- Zero assumptions about data structure
- Test EVERY mathematically plausible time transformation
- Proceed with warning if <10 models converge (don't error)
- RQ specifies interaction level (not exhaustive cross-product)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import traceback
from datetime import datetime


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compare_lmm_models_kitchen_sink(
    data: pd.DataFrame,
    outcome_var: str,              # e.g., 'theta', 'theta_confidence', 'HCE_rate'
    tsvr_var: str,                 # MUST be continuous TSVR (e.g., 'TSVR_hours')
    groups_var: str,               # e.g., 'UID'

    # Interaction factors (categorical or continuous)
    factor1_var: str = None,       # e.g., 'domain' (categorical)
    factor1_type: str = 'categorical',  # 'categorical' or 'continuous'
    factor1_reference: str = None, # Reference level if categorical

    factor2_var: str = None,       # e.g., 'paradigm' (categorical)
    factor2_type: str = 'categorical',
    factor2_reference: str = None,

    # Random effects specification
    re_formula: str = '~TSVR',     # Random slope variable (will be expanded with transformations)

    # Model fitting
    reml: bool = False,            # ML (False) for model comparison

    # Output options
    return_models: bool = False,   # Return fitted model objects?
    save_dir: Path = None,         # If provided, save outputs to directory
    log_file: Path = None,         # If provided, write detailed log

    # Advanced options
    min_converged_models: int = 10,  # Minimum successful fits to proceed (warning if less)
    aic_tolerance: float = 0.001,    # Tolerance for Akaike weight validation
) -> dict:
    """
    Compare LMM trajectory models using kitchen-sink approach (70+ models).

    Tests comprehensive suite of time transformations to identify best functional
    form for forgetting curves. Handles 0/1/2-way interactions automatically.

    Parameters
    ----------
    data : DataFrame
        Input data with columns: {outcome_var}, {tsvr_var}, {groups_var}, {factors}
        TSVR MUST be continuous (not categorical session indicators)
    outcome_var : str
        Name of outcome variable (continuous, e.g., theta scores)
    tsvr_var : str
        Name of TSVR column (continuous time variable in hours)
        Expected range: 0-200 hours for 7-day study
    groups_var : str
        Subject identifier for random effects (e.g., 'UID')
    factor1_var : str, optional
        First grouping variable for interactions (e.g., 'domain')
    factor1_type : str, default='categorical'
        'categorical' (uses Treatment coding) or 'continuous' (mean-centered)
    factor1_reference : str, optional
        Reference level for categorical factor1 (required if factor1_type='categorical')
    factor2_var : str, optional
        Second grouping variable for 2-way interactions (e.g., 'paradigm')
    factor2_type : str, default='categorical'
        'categorical' or 'continuous'
    factor2_reference : str, optional
        Reference level for categorical factor2
    re_formula : str, default='~TSVR'
        Random effects formula base. Tool will expand to match time transforms.
        Examples:
        - '~TSVR' → random intercept + slope on TSVR
        - '~log_TSVR' → random intercept + slope on log(TSVR)
        - '~1' → random intercepts only (no slopes)
    reml : bool, default=False
        Use REML (True) or ML (False). ML required for AIC comparison.
    return_models : bool, default=False
        If True, return fitted MixedLMResults objects (may have pickling issues)
    save_dir : Path, optional
        Directory to save comparison CSV and summary text files
    log_file : Path, optional
        Log file path for detailed execution log
    min_converged_models : int, default=10
        Minimum successful fits to proceed. Warning (not error) if less.
    aic_tolerance : float, default=0.001
        Tolerance for Akaike weight sum validation (should equal 1.0 ± tolerance)

    Returns
    -------
    dict
        {
            'comparison': DataFrame with columns [model_name, AIC, delta_AIC,
                          akaike_weight, cumulative_weight, BIC, log_likelihood,
                          n_params, converged],
            'best_model': dict with best model summary,
            'log_model_info': dict with Log model benchmark info,
            'top_10': DataFrame with top 10 models,
            'failed_models': list of model names that didn't converge,
            'transformations': dict of all time transformations created,
            'fitted_models': dict of MixedLMResults (if return_models=True),
            'summary_stats': dict with overall statistics,
        }

    Raises
    ------
    AssertionError
        If TSVR is not continuous, has insufficient variance, or is out of range
    ValueError
        If factor specifications are invalid

    Examples
    --------
    >>> # 0-way interaction (simple trajectory)
    >>> results = compare_lmm_models_kitchen_sink(
    ...     data=lmm_input,
    ...     outcome_var='theta',
    ...     tsvr_var='TSVR_hours',
    ...     groups_var='UID',
    ...     re_formula='~log_TSVR',
    ... )
    >>> print(results['best_model'])

    >>> # 1-way interaction (domain × time)
    >>> results = compare_lmm_models_kitchen_sink(
    ...     data=lmm_input,
    ...     outcome_var='theta',
    ...     tsvr_var='TSVR_hours',
    ...     groups_var='UID',
    ...     factor1_var='domain',
    ...     factor1_type='categorical',
    ...     factor1_reference='What',
    ...     re_formula='~log_TSVR',
    ... )

    >>> # 2-way interaction (domain × paradigm × time)
    >>> results = compare_lmm_models_kitchen_sink(
    ...     data=lmm_input,
    ...     outcome_var='theta',
    ...     tsvr_var='TSVR_hours',
    ...     groups_var='UID',
    ...     factor1_var='domain',
    ...     factor1_type='categorical',
    ...     factor1_reference='What',
    ...     factor2_var='paradigm',
    ...     factor2_type='categorical',
    ...     factor2_reference='free_recall',
    ...     re_formula='~1',  # Random intercepts only (convergence)
    ... )
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
    log("LMM Model Selection: Kitchen Sink Suite")
    log("=" * 80)

    # =========================================================================
    # STEP 1: Validate Inputs
    # =========================================================================

    log("[VALIDATION] Checking inputs...")

    # Copy data to avoid modifying original
    df = data.copy()

    # Check required columns exist
    required_cols = [outcome_var, tsvr_var, groups_var]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate TSVR is continuous
    if not pd.api.types.is_numeric_dtype(df[tsvr_var]):
        raise AssertionError(
            f"{tsvr_var} must be numeric (continuous time variable). "
            f"Found dtype: {df[tsvr_var].dtype}"
        )

    # Check TSVR has sufficient variance (not just session indicators)
    tsvr_unique = df[tsvr_var].nunique()
    if tsvr_unique < 10:
        raise AssertionError(
            f"{tsvr_var} has only {tsvr_unique} unique values - appears categorical. "
            f"Use continuous TSVR (hours since encoding), not session indicators (T1/T2/T3/T4)."
        )

    # Check TSVR range (expected 0-200 hours for 7-day study)
    tsvr_min, tsvr_max = df[tsvr_var].min(), df[tsvr_var].max()
    if tsvr_max > 300:
        warnings.warn(
            f"{tsvr_var} max = {tsvr_max:.1f} hours (>300h = 12.5 days). "
            f"Is this correct for your study design?"
        )

    log(f"  TSVR variable: {tsvr_var}")
    log(f"  TSVR range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")
    log(f"  TSVR unique values: {tsvr_unique}")
    log(f"  Outcome: {outcome_var}")
    log(f"  Groups: {groups_var} (N={df[groups_var].nunique()})")
    log(f"  N observations: {len(df)}")

    # Validate factors
    factors = []

    if factor1_var is not None:
        if factor1_var not in df.columns:
            raise ValueError(f"factor1_var '{factor1_var}' not in data")

        if factor1_type == 'categorical':
            if factor1_reference is None:
                raise ValueError(f"factor1_reference required for categorical {factor1_var}")
            if factor1_reference not in df[factor1_var].unique():
                raise ValueError(
                    f"factor1_reference '{factor1_reference}' not in {factor1_var} values: "
                    f"{df[factor1_var].unique()}"
                )

        factors.append({
            'var': factor1_var,
            'type': factor1_type,
            'reference': factor1_reference
        })
        log(f"  Factor 1: {factor1_var} ({factor1_type}, ref={factor1_reference})")

    if factor2_var is not None:
        if factor2_var not in df.columns:
            raise ValueError(f"factor2_var '{factor2_var}' not in data")

        if factor2_type == 'categorical':
            if factor2_reference is None:
                raise ValueError(f"factor2_reference required for categorical {factor2_var}")
            if factor2_reference not in df[factor2_var].unique():
                raise ValueError(
                    f"factor2_reference '{factor2_reference}' not in {factor2_var} values: "
                    f"{df[factor2_var].unique()}"
                )

        factors.append({
            'var': factor2_var,
            'type': factor2_type,
            'reference': factor2_reference
        })
        log(f"  Factor 2: {factor2_var} ({factor2_type}, ref={factor2_reference})")

    if len(factors) == 0:
        log("  Interaction: None (simple trajectory)")
    elif len(factors) == 1:
        log("  Interaction: 1-way (time × factor1)")
    elif len(factors) == 2:
        log("  Interaction: 2-way (time × factor1 × factor2)")

    log(f"  Random effects: {re_formula}")
    log(f"  REML: {reml}")

    # =========================================================================
    # STEP 2: Create Time Transformations
    # =========================================================================

    log("[TRANSFORM] Creating time transformations...")

    # Convert TSVR to days
    df['TSVR'] = df[tsvr_var] / 24.0  # Internal standard: TSVR in days

    transformations = {}

    # 1. POLYNOMIAL FAMILY
    log("  Polynomial family...")
    df['TSVR_sq'] = df['TSVR'] ** 2
    df['TSVR_cub'] = df['TSVR'] ** 3
    df['TSVR_4th'] = df['TSVR'] ** 4
    transformations.update({
        'TSVR_sq': 'TSVR^2',
        'TSVR_cub': 'TSVR^3',
        'TSVR_4th': 'TSVR^4',
    })

    # 2. LOGARITHMIC FAMILY
    log("  Logarithmic family...")
    df['log_TSVR'] = np.log(df['TSVR'] + 1)         # +1 for TSVR=0
    df['log2_TSVR'] = np.log2(df['TSVR'] + 1)
    df['log10_TSVR'] = np.log10(df['TSVR'] + 1)
    df['log_log_TSVR'] = np.log(df['log_TSVR'] + 1)
    transformations.update({
        'log_TSVR': 'log(TSVR+1)',
        'log2_TSVR': 'log2(TSVR+1)',
        'log10_TSVR': 'log10(TSVR+1)',
        'log_log_TSVR': 'log(log(TSVR+1)+1)',
    })

    # 3. POWER LAW FAMILY
    log("  Power law family...")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        col_name = f'TSVR_pow_neg{int(alpha*10):02d}'
        df[col_name] = (df['TSVR'] + 1) ** (-alpha)
        transformations[col_name] = f'(TSVR+1)^(-{alpha})'

    # 4. ROOT FAMILY
    log("  Root family...")
    df['sqrt_TSVR'] = np.sqrt(df['TSVR'])
    df['cbrt_TSVR'] = np.cbrt(df['TSVR'])
    df['TSVR_pow_025'] = df['TSVR'] ** 0.25
    df['TSVR_pow_033'] = df['TSVR'] ** 0.333
    df['TSVR_pow_067'] = df['TSVR'] ** 0.667
    transformations.update({
        'sqrt_TSVR': 'sqrt(TSVR)',
        'cbrt_TSVR': 'cbrt(TSVR)',
        'TSVR_pow_025': 'TSVR^0.25',
        'TSVR_pow_033': 'TSVR^(1/3)',
        'TSVR_pow_067': 'TSVR^(2/3)',
    })

    # 5. RECIPROCAL FAMILY
    log("  Reciprocal family...")
    df['recip_TSVR'] = 1.0 / (df['TSVR'] + 1)
    df['recip_TSVR_sq'] = 1.0 / ((df['TSVR'] + 1) ** 2)
    transformations.update({
        'recip_TSVR': '1/(TSVR+1)',
        'recip_TSVR_sq': '1/(TSVR+1)^2',
    })

    # 6. EXPONENTIAL FAMILY (linear proxies)
    log("  Exponential family...")
    df['neg_TSVR'] = -df['TSVR']
    df['neg_TSVR_sq'] = -(df['TSVR'] ** 2)
    df['neg_sqrt_TSVR'] = -np.sqrt(df['TSVR'])
    transformations.update({
        'neg_TSVR': '-TSVR',
        'neg_TSVR_sq': '-TSVR^2',
        'neg_sqrt_TSVR': '-sqrt(TSVR)',
    })

    # 7. TRIGONOMETRIC FAMILY
    log("  Trigonometric family...")
    tsvr_max = df['TSVR'].max()
    df['sin_TSVR'] = np.sin(df['TSVR'] / tsvr_max * 2 * np.pi)
    df['cos_TSVR'] = np.cos(df['TSVR'] / tsvr_max * 2 * np.pi)
    transformations.update({
        'sin_TSVR': 'sin(TSVR normalized)',
        'cos_TSVR': 'cos(TSVR normalized)',
    })

    # 8. HYPERBOLIC FAMILY
    log("  Hyperbolic family...")
    tsvr_norm = df['TSVR'] / tsvr_max
    df['tanh_TSVR'] = np.tanh(tsvr_norm)
    df['arctanh_TSVR'] = np.arctanh(tsvr_norm * 0.99)  # Keep in valid range
    df['sinh_TSVR'] = np.sinh(tsvr_norm)
    transformations.update({
        'tanh_TSVR': 'tanh(TSVR/max)',
        'arctanh_TSVR': 'arctanh(TSVR/max*0.99)',
        'sinh_TSVR': 'sinh(TSVR/max)',
    })

    # Mean-center continuous factors
    for factor in factors:
        if factor['type'] == 'continuous':
            centered_name = f"{factor['var']}_c"
            df[centered_name] = df[factor['var']] - df[factor['var']].mean()
            log(f"  Mean-centered {factor['var']} → {centered_name}")
            factor['var_formula'] = centered_name  # Use centered version in formula
        else:
            factor['var_formula'] = factor['var']  # Use original for categorical

    log(f"  Total transformations created: {len(transformations)}")

    # =========================================================================
    # STEP 3: Define Model Suite
    # =========================================================================

    log("[CONFIG] Defining model suite...")

    models = {}

    # 1. POLYNOMIAL FAMILY (6 models)
    models.update({
        'Linear':           'TSVR',
        'Quadratic':        'TSVR + TSVR_sq',
        'Cubic':            'TSVR + TSVR_sq + TSVR_cub',
        'Quartic':          'TSVR + TSVR_sq + TSVR_cub + TSVR_4th',
        'Quadratic_pure':   'TSVR_sq',
        'Cubic_pure':       'TSVR_cub',
    })

    # 2. LOGARITHMIC FAMILY (8 models)
    models.update({
        'Log':              'log_TSVR',
        'Log2':             'log2_TSVR',
        'Log10':            'log10_TSVR',
        'LogLog':           'log_log_TSVR',
        'Lin+Log':          'TSVR + log_TSVR',
        'Quad+Log':         'TSVR + TSVR_sq + log_TSVR',
        'Log+LogLog':       'log_TSVR + log_log_TSVR',
        'Lin+Quad+Log':     'TSVR + TSVR_sq + log_TSVR',
    })

    # 3. POWER LAW FAMILY (12 models)
    models.update({
        'PowerLaw_01':      'TSVR_pow_neg01',
        'PowerLaw_02':      'TSVR_pow_neg02',
        'PowerLaw_03':      'TSVR_pow_neg03',
        'PowerLaw_04':      'TSVR_pow_neg04',
        'PowerLaw_05':      'TSVR_pow_neg05',
        'PowerLaw_06':      'TSVR_pow_neg06',
        'PowerLaw_07':      'TSVR_pow_neg07',
        'PowerLaw_08':      'TSVR_pow_neg08',
        'PowerLaw_09':      'TSVR_pow_neg09',
        'PowerLaw_10':      'TSVR_pow_neg10',
        'PowerLaw_Log':     'TSVR_pow_neg05 + log_TSVR',
        'PowerLaw_Lin':     'TSVR_pow_neg05 + TSVR',
    })

    # 4. ROOT FAMILY (9 models)
    models.update({
        'SquareRoot':       'sqrt_TSVR',
        'CubeRoot':         'cbrt_TSVR',
        'FourthRoot':       'TSVR_pow_025',
        'Root_033':         'TSVR_pow_033',
        'Root_067':         'TSVR_pow_067',
        'SquareRoot+Log':   'sqrt_TSVR + log_TSVR',
        'CubeRoot+Log':     'cbrt_TSVR + log_TSVR',
        'SquareRoot+Lin':   'sqrt_TSVR + TSVR',
        'Root_Multi':       'sqrt_TSVR + cbrt_TSVR',
    })

    # 5. RECIPROCAL FAMILY (6 models)
    models.update({
        'Reciprocal':       'recip_TSVR',
        'Recip+Log':        'recip_TSVR + log_TSVR',
        'Recip+Lin':        'recip_TSVR + TSVR',
        'Recip+Quad':       'recip_TSVR + TSVR + TSVR_sq',
        'Recip_sq':         'recip_TSVR_sq',
        'Recip+PowerLaw':   'recip_TSVR + TSVR_pow_neg05',
    })

    # 6. EXPONENTIAL FAMILY (7 models)
    models.update({
        'Exponential_proxy': 'neg_TSVR',
        'Exp+Log':          'neg_TSVR + log_TSVR',
        'Exp+Lin':          'neg_TSVR + TSVR',
        'Exp_fast':         'neg_TSVR_sq',
        'Exp_slow':         'neg_sqrt_TSVR',
        'Exp+PowerLaw':     'neg_TSVR + TSVR_pow_neg05',
        'Exp+Recip':        'neg_TSVR + recip_TSVR',
    })

    # 7. TRIGONOMETRIC FAMILY (4 models)
    models.update({
        'Sin':              'sin_TSVR',
        'Cos':              'cos_TSVR',
        'Sin+Cos':          'sin_TSVR + cos_TSVR',
        'Sin+Log':          'sin_TSVR + log_TSVR',
    })

    # 8. HYPERBOLIC FAMILY (4 models)
    models.update({
        'Tanh':             'tanh_TSVR',
        'Tanh+Log':         'tanh_TSVR + log_TSVR',
        'Arctanh':          'arctanh_TSVR',
        'Sinh':             'sinh_TSVR',
    })

    # 9. KITCHEN SINK HYBRIDS (10 models)
    models.update({
        'Log+PowerLaw05':       'log_TSVR + TSVR_pow_neg05',
        'Log+SquareRoot':       'log_TSVR + sqrt_TSVR',
        'Log+Recip':            'log_TSVR + recip_TSVR',
        'SquareRoot+PowerLaw':  'sqrt_TSVR + TSVR_pow_neg05',
        'SquareRoot+Recip':     'sqrt_TSVR + recip_TSVR',
        'Recip+PowerLaw05':     'recip_TSVR + TSVR_pow_neg05',
        'Lin+Log+PowerLaw':     'TSVR + log_TSVR + TSVR_pow_neg05',
        'Quad+Log+SquareRoot':  'TSVR + TSVR_sq + log_TSVR + sqrt_TSVR',
        'PowerLaw+Recip+Log':   'TSVR_pow_neg05 + recip_TSVR + log_TSVR',
        'Ultimate':             'TSVR + TSVR_sq + log_TSVR + sqrt_TSVR + TSVR_pow_neg05 + recip_TSVR',
    })

    log(f"  Total models defined: {len(models)}")

    # =========================================================================
    # STEP 4: Build Formulas with Interactions
    # =========================================================================

    log("[FORMULA] Building model formulas...")

    def build_formula(time_expr: str, outcome: str, factors: List[dict]) -> str:
        """
        Build interaction formula dynamically.

        Examples:
        - 0-way: "theta ~ log_TSVR"
        - 1-way: "theta ~ log_TSVR * C(domain, Treatment('What'))"
        - 2-way: "theta ~ log_TSVR * C(domain, Treatment('What')) * C(paradigm, Treatment('free_recall'))"
        """
        if len(factors) == 0:
            # No interactions
            return f"{outcome} ~ {time_expr}"

        # Build factor expressions
        factor_exprs = []
        for factor in factors:
            if factor['type'] == 'categorical':
                # Use Treatment coding with explicit reference
                factor_expr = f"C({factor['var']}, Treatment('{factor['reference']}'))"
            else:
                # Continuous (already mean-centered)
                factor_expr = factor['var_formula']
            factor_exprs.append(factor_expr)

        # Join with * (full factorial)
        all_terms = [time_expr] + factor_exprs
        formula = f"{outcome} ~ {' * '.join(all_terms)}"

        return formula

    formulas = {}
    for model_name, time_expr in models.items():
        formula = build_formula(time_expr, outcome_var, factors)
        formulas[model_name] = formula

    log(f"  Example formula (Linear): {formulas['Linear']}")
    log(f"  Example formula (Log): {formulas['Log']}")

    # =========================================================================
    # STEP 5: Fit All Models
    # =========================================================================

    log("[ANALYSIS] Fitting all models...")
    log(f"  This may take 5-10 minutes for {len(models)} models...")

    fitted_models = {}
    model_stats = []
    failed_models = []

    for i, (model_name, formula) in enumerate(formulas.items(), 1):
        log(f"  [{i}/{len(formulas)}] Fitting {model_name}...")

        try:
            # Fit mixed model
            model = smf.mixedlm(
                formula=formula,
                data=df,
                groups=df[groups_var],
                re_formula=re_formula
            )

            result = model.fit(reml=reml)

            fitted_models[model_name] = result

            model_stats.append({
                'model_name': model_name,
                'AIC': result.aic,
                'BIC': result.bic,
                'log_likelihood': result.llf,
                'n_params': len(result.params),
                'converged': result.converged,
            })

            log(f"    ✓ AIC={result.aic:.2f}, converged={result.converged}")

        except Exception as e:
            log(f"    ✗ FAILED: {str(e)[:100]}")
            failed_models.append(model_name)

            # Still record failure
            model_stats.append({
                'model_name': model_name,
                'AIC': np.inf,
                'BIC': np.inf,
                'log_likelihood': -np.inf,
                'n_params': np.nan,
                'converged': False,
            })

    log(f"  Successful fits: {len(fitted_models)}/{len(models)}")
    log(f"  Failed fits: {len(failed_models)}")

    # Check if enough models converged
    if len(fitted_models) < min_converged_models:
        warnings.warn(
            f"Only {len(fitted_models)} models converged (minimum {min_converged_models} recommended). "
            f"Proceeding with available models, but results may be unreliable."
        )

    # =========================================================================
    # STEP 6: Compute AIC Comparison Metrics
    # =========================================================================

    log("[COMPUTE] Computing AIC comparison metrics...")

    comparison_df = pd.DataFrame(model_stats)

    # Remove failed models
    comparison_df = comparison_df[comparison_df['AIC'] != np.inf].copy()

    if len(comparison_df) == 0:
        raise RuntimeError("All models failed to converge. Cannot proceed with model selection.")

    # Sort by AIC
    comparison_df = comparison_df.sort_values('AIC').reset_index(drop=True)

    # Compute delta AIC
    aic_min = comparison_df['AIC'].min()
    comparison_df['delta_AIC'] = comparison_df['AIC'] - aic_min

    # Compute Akaike weights
    comparison_df['akaike_weight'] = np.exp(-0.5 * comparison_df['delta_AIC'])
    weight_sum = comparison_df['akaike_weight'].sum()
    comparison_df['akaike_weight'] = comparison_df['akaike_weight'] / weight_sum

    # Compute cumulative weights
    comparison_df['cumulative_weight'] = comparison_df['akaike_weight'].cumsum()

    log(f"  Minimum AIC: {aic_min:.2f}")
    log(f"  delta_AIC range: [{comparison_df['delta_AIC'].min():.2f}, {comparison_df['delta_AIC'].max():.2f}]")
    log(f"  Akaike weight sum: {comparison_df['akaike_weight'].sum():.6f} (should be 1.0)")

    # =========================================================================
    # STEP 7: Identify Best Model
    # =========================================================================

    log("[RESULTS] Identifying best model...")

    best_model_name = comparison_df.iloc[0]['model_name']
    best_model_aic = comparison_df.iloc[0]['AIC']
    best_model_weight = comparison_df.iloc[0]['akaike_weight']

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

    best_model_info = {
        'name': best_model_name,
        'AIC': best_model_aic,
        'weight': best_model_weight,
        'weight_pct': best_model_weight * 100,
        'uncertainty': uncertainty,
        'interpretation': interpretation,
        'rank': 1,
    }

    log(f"  Best model: {best_model_name}")
    log(f"  AIC: {best_model_aic:.2f}")
    log(f"  Akaike weight: {best_model_weight:.4f} ({best_model_weight*100:.1f}%)")
    log(f"  Uncertainty: {uncertainty} ({interpretation})")

    # Find Log model for benchmark
    log_model_row = comparison_df[comparison_df['model_name'] == 'Log']
    if len(log_model_row) > 0:
        log_rank = log_model_row.index[0] + 1
        log_aic = log_model_row.iloc[0]['AIC']
        log_weight = log_model_row.iloc[0]['akaike_weight']
        log_delta = log_model_row.iloc[0]['delta_AIC']

        log_model_info = {
            'rank': log_rank,
            'AIC': log_aic,
            'delta_AIC': log_delta,
            'weight': log_weight,
            'weight_pct': log_weight * 100,
        }

        log(f"  Log model (benchmark): Rank #{log_rank}, AIC={log_aic:.2f}, "
            f"Δ={log_delta:.2f}, weight={log_weight:.4f}")
    else:
        log_model_info = {'error': 'Log model failed to converge'}
        log("  Log model: FAILED TO CONVERGE")

    # =========================================================================
    # STEP 8: Validate AIC Comparison
    # =========================================================================

    log("[VALIDATION] Validating AIC comparison metrics...")

    # Check weights sum to 1.0
    weight_sum = comparison_df['akaike_weight'].sum()
    if not (1.0 - aic_tolerance <= weight_sum <= 1.0 + aic_tolerance):
        warnings.warn(f"Akaike weights sum = {weight_sum:.6f} (expected 1.0 ± {aic_tolerance})")
    else:
        log(f"  ✓ Akaike weights sum to 1.0 ({weight_sum:.6f})")

    # Check all weights in (0, 1)
    if not ((comparison_df['akaike_weight'] > 0).all() and (comparison_df['akaike_weight'] < 1).all()):
        warnings.warn("Some Akaike weights outside (0, 1) range")
    else:
        log("  ✓ All Akaike weights in (0, 1)")

    # Check delta_AIC correct
    if comparison_df.iloc[0]['delta_AIC'] != 0.0:
        warnings.warn(f"Best model delta_AIC = {comparison_df.iloc[0]['delta_AIC']} (expected 0)")
    else:
        log("  ✓ Best model has delta_AIC = 0")

    # Check cumulative_weight monotonic
    if not (comparison_df['cumulative_weight'].diff().dropna() >= 0).all():
        warnings.warn("cumulative_weight not monotonic increasing")
    else:
        log("  ✓ cumulative_weight monotonic increasing")

    # Check cumulative_weight ends at 1.0
    cum_weight_final = comparison_df.iloc[-1]['cumulative_weight']
    if not (1.0 - aic_tolerance <= cum_weight_final <= 1.0 + aic_tolerance):
        warnings.warn(f"cumulative_weight final = {cum_weight_final:.6f} (expected 1.0)")
    else:
        log(f"  ✓ cumulative_weight ends at 1.0 ({cum_weight_final:.6f})")

    # =========================================================================
    # STEP 9: Prepare Outputs
    # =========================================================================

    log("[OUTPUT] Preparing output dictionary...")

    # Top 10 models
    top_10 = comparison_df.head(10)[['model_name', 'AIC', 'delta_AIC', 'akaike_weight', 'cumulative_weight']]

    # Summary statistics
    summary_stats = {
        'n_models_tested': len(models),
        'n_models_converged': len(comparison_df),
        'n_models_failed': len(failed_models),
        'best_model': best_model_name,
        'best_aic': best_model_aic,
        'aic_range': (comparison_df['AIC'].min(), comparison_df['AIC'].max()),
        'n_competitive_models': len(comparison_df[comparison_df['delta_AIC'] < 2]),  # ΔAIC < 2
    }

    log(f"  Competitive models (ΔAIC < 2): {summary_stats['n_competitive_models']}")

    # =========================================================================
    # STEP 10: Save Outputs (if requested)
    # =========================================================================

    if save_dir is not None:
        log(f"[SAVE] Saving outputs to {save_dir}...")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        comparison_path = save_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"  ✓ {comparison_path.name}")

        # Save best model summary
        summary_path = save_dir / "best_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BEST MODEL SUMMARY - Kitchen Sink Model Selection\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"AIC: {best_model_aic:.2f}\n")
            f.write(f"Akaike Weight: {best_model_weight:.4f} ({best_model_weight*100:.1f}%)\n")
            f.write(f"Uncertainty: {uncertainty}\n")
            f.write(f"Interpretation: {interpretation}\n\n")

            if 'error' not in log_model_info:
                f.write(f"Log Model (Benchmark):\n")
                f.write(f"  Rank: #{log_model_info['rank']}\n")
                f.write(f"  AIC: {log_model_info['AIC']:.2f}\n")
                f.write(f"  ΔAIC: {log_model_info['delta_AIC']:.2f}\n")
                f.write(f"  Weight: {log_model_info['weight']:.4f} ({log_model_info['weight_pct']:.1f}%)\n\n")

            f.write("=" * 80 + "\n")
            f.write("Top 10 Models:\n")
            f.write("=" * 80 + "\n\n")
            f.write(top_10.to_string(index=False))
            f.write("\n")

        log(f"  ✓ {summary_path.name}")

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_messages))
        log(f"  ✓ Log saved to {log_file.name}")

    log("=" * 80)
    log("Model selection complete!")
    log("=" * 80)

    # =========================================================================
    # Return Results
    # =========================================================================

    results = {
        'comparison': comparison_df,
        'best_model': best_model_info,
        'log_model_info': log_model_info,
        'top_10': top_10,
        'failed_models': failed_models,
        'transformations': transformations,
        'summary_stats': summary_stats,
    }

    if return_models:
        results['fitted_models'] = fitted_models

    return results
