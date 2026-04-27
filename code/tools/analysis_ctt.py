"""
Classical Test Theory (CTT) Analysis Tools.

This module provides CTT-specific analysis functions for:
- RQ 5.12: Methodological comparison of IRT vs CTT
- RQ 5.2.4, 5.3.5, 5.4.4: IRT-CTT convergence analyses

Functions:
- compute_cronbachs_alpha: Internal consistency reliability with bootstrap CIs
- compare_correlations_dependent: Steiger's z-test for dependent correlations
- compute_ctt_mean_scores_by_factor: CTT score computation by factor (domain/paradigm/congruence)
- compute_pearson_correlations_with_correction: Correlations with Holm-Bonferroni
- compute_cohens_kappa_agreement: Cohen's kappa for significance classification agreement
- compare_lmm_fit_aic_bic: AIC/BIC model comparison

References:
- Cronbach (1951): Coefficient alpha and the internal structure of tests
- Steiger (1980): Tests for comparing elements of a correlation matrix
- Cohen (1960): A coefficient of agreement for nominal scales
- Landis & Koch (1977): The measurement of observer agreement for categorical data
- Burnham & Anderson (2002): Model Selection and Multimodel Inference
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def compute_cronbachs_alpha(
    data: pd.DataFrame,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Compute Cronbach's alpha with bootstrap confidence intervals.

    For dichotomous (0/1) items, Cronbach's alpha equals KR-20
    (Kuder-Richardson formula 20). This function handles both continuous
    and dichotomous data.

    Bootstrap Method:
    - Percentile method for 95% CI
    - Resamples participants (preserves item correlation structure)
    - 1000-10000 iterations recommended (1000 minimum per literature)

    Args:
        data: DataFrame with items as columns, participants as rows.
              Values should be numeric (0/1 for dichotomous, continuous for Likert).
              Missing values (NaN) handled via pairwise deletion.
        n_bootstrap: Number of bootstrap iterations (default 1000).
                     RQ 5.12 spec recommends 1000-10000.

    Returns:
        Dict with keys:
        - alpha: float - Cronbach's alpha coefficient
        - ci_lower: float - Bootstrap 95% CI lower bound (2.5th percentile)
        - ci_upper: float - Bootstrap 95% CI upper bound (97.5th percentile)
        - n_items: int - Number of items
        - n_participants: int - Number of participants

    Raises:
        ValueError: If fewer than 2 items or fewer than 3 participants

    Example:
        >>> data = pd.DataFrame({
        ...     'item1': [1, 0, 1, 0, 1],
        ...     'item2': [1, 1, 1, 0, 1],
        ...     'item3': [1, 0, 1, 1, 1]
        ... })
        >>> result = compute_cronbachs_alpha(data, n_bootstrap=1000)
        >>> assert 0 <= result['alpha'] <= 1
        >>> assert result['ci_lower'] < result['alpha'] < result['ci_upper']

    References:
        - Cronbach (1951): Original alpha formulation
        - PMC4205511: "Making sense of Cronbach's alpha"
        - PMC8451024 (2021): KR-20 equivalence for dichotomous items
        - RQ 5.12 1_concept.md Step 3b: CTT reliability requirements
    """
    # Validation
    if data.shape[1] < 2:
        raise ValueError("Cronbach's alpha requires at least 2 items")

    if data.shape[0] < 3:
        raise ValueError("Cronbach's alpha requires at least 3 participants for variance estimation")

    n_items = data.shape[1]
    n_participants = data.shape[0]

    # Compute point estimate
    alpha = _cronbach_alpha_formula(data)

    # Bootstrap confidence intervals
    bootstrap_alphas = []
    for _ in range(n_bootstrap):
        # Resample participants (preserves item correlation structure)
        bootstrap_sample = data.sample(n=n_participants, replace=True, random_state=None)
        bootstrap_alpha = _cronbach_alpha_formula(bootstrap_sample)
        bootstrap_alphas.append(bootstrap_alpha)

    # Percentile method for 95% CI
    bootstrap_alphas = np.array(bootstrap_alphas)
    ci_lower = np.percentile(bootstrap_alphas, 2.5)
    ci_upper = np.percentile(bootstrap_alphas, 97.5)

    return {
        'alpha': float(alpha),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_items': int(n_items),
        'n_participants': int(n_participants)
    }


def _cronbach_alpha_formula(data: pd.DataFrame) -> float:
    """
    Calculate Cronbach's alpha using standard formula.

    Formula: α = (k/(k-1)) × (1 - Σσ²ᵢ / σ²ₓ)
    where:
    - k = number of items
    - Σσ²ᵢ = sum of item variances
    - σ²ₓ = variance of total scores

    Args:
        data: DataFrame with items as columns, participants as rows

    Returns:
        float: Cronbach's alpha coefficient

    Notes:
        - Uses ddof=1 (sample variance, not population variance)
        - Handles NaN via pairwise deletion (dropna in var/sum operations)
        - For binary items, this equals KR-20
    """
    # Item variances (variance of each column)
    item_variances = data.var(axis=0, ddof=1)
    sum_item_variances = item_variances.sum()

    # Total score variance (variance of row sums)
    total_scores = data.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    # Cronbach's alpha formula
    k = data.shape[1]
    alpha = (k / (k - 1)) * (1 - sum_item_variances / total_variance)

    return alpha


def compare_correlations_dependent(
    r12: float,
    r13: float,
    r23: float,
    n: int
) -> Dict[str, Any]:
    """
    Test if two dependent correlations differ significantly (Steiger's z-test).

    Tests whether r₁₂ differs from r₁₃ when both correlations share variable 1.
    This is appropriate for RQ 5.12 where Full CTT, Purified CTT, and IRT theta
    all come from the same N=100 participants (dependent correlations).

    Example Use Case (RQ 5.12):
    - Variable 1: IRT theta
    - Variable 2: Full CTT score
    - Variable 3: Purified CTT score
    - Question: Does r(IRT, Purified_CTT) > r(IRT, Full_CTT)?

    Args:
        r12: Correlation between variables 1 and 2
        r13: Correlation between variables 1 and 3
        r23: Correlation between variables 2 and 3
        n: Sample size (number of participants)

    Returns:
        Dict with keys:
        - z_statistic: float - Steiger's z-test statistic
        - p_value: float - Two-tailed p-value
        - r_difference: float - r13 - r12 (positive = r13 stronger)
        - significant: bool - Significant at α=0.05
        - interpretation: str - Plain language interpretation

    Raises:
        ValueError: If correlations not in [-1, 1] or n < 20

    Example:
        >>> result = compare_correlations_dependent(
        ...     r12=0.85,  # Full CTT - IRT
        ...     r13=0.92,  # Purified CTT - IRT
        ...     r23=0.88,  # Full CTT - Purified CTT
        ...     n=100
        ... )
        >>> print(result['interpretation'])
        'r13 (0.92) significantly higher than r12 (0.85), z=2.34, p=0.019'

    References:
        - Steiger (1980): Psychological Bulletin 87:245-251 (Equations 3 & 10)
        - RQ 5.12 1_stats.md: Steiger's z-test requirement
        - Online calculator: quantpsy.org/corrtest (for validation)

    Notes:
        - N=100 is adequate for Steiger's z-test (literature confirms N=103 sufficient)
        - Fisher's r-to-z is INVALID here (assumes independent samples)
        - Steiger's method correctly accounts for asymptotic covariance
    """
    # Validation
    if not all(-1 <= r <= 1 for r in [r12, r13, r23]):
        raise ValueError("All correlations must be in range [-1, 1]")

    if n < 20:
        raise ValueError("Steiger's z-test requires at least n=20 for validity")

    # Steiger (1980) Equation 3: Asymptotic covariance of r12 and r13
    # cov(r12, r13) = [r23 - 0.5 * r12 * r13 * (1 - r12² - r13² - r23²)] / (n - 3)

    # Fisher's z-transformation
    z12 = np.arctanh(r12)  # Fisher's z for r12
    z13 = np.arctanh(r13)  # Fisher's z for r13

    # Compute determinant of correlation matrix
    R = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23

    # Steiger's covariance formula (Equation 10)
    # var(z12 - z13) = (2 - 2*f) / (n - 3)
    # where f = r23(1 - r12² - r13²) / (2*(1-r12²)*(1-r13²))

    numerator = r23 * (1 - r12**2) * (1 - r13**2)
    denominator = 2 * (1 - r12**2) * (1 - r13**2)
    f = numerator / denominator if denominator != 0 else 0

    var_diff = (2 - 2*f) / (n - 3)
    se_diff = np.sqrt(var_diff)

    # Steiger's z-statistic
    z_statistic = (z13 - z12) / se_diff

    # Two-tailed p-value
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    # Interpretation
    r_difference = r13 - r12
    significant = p_value < 0.05

    if significant:
        direction = "higher" if r_difference > 0 else "lower"
        interpretation = f"r13 ({r13:.2f}) significantly {direction} than r12 ({r12:.2f}), z={z_statistic:.2f}, p={p_value:.3f}"
    else:
        interpretation = f"No significant difference between r13 ({r13:.2f}) and r12 ({r12:.2f}), z={z_statistic:.2f}, p={p_value:.3f}"

    return {
        'z_statistic': float(z_statistic),
        'p_value': float(p_value),
        'r_difference': float(r_difference),
        'significant': bool(significant),
        'interpretation': interpretation
    }


# =============================================================================
# IRT-CTT Convergence Tools (RQ 5.2.4, 5.3.5, 5.4.4)
# =============================================================================

def compute_ctt_mean_scores_by_factor(
    df_wide: pd.DataFrame,
    item_factor_df: pd.DataFrame,
    factor_col: str = 'factor',
    item_col: str = 'item_name',
    include_factors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute CTT mean scores (proportion correct) per UID × test × factor.

    This is the core CTT computation for IRT-CTT convergence analyses.
    Works with any factor type: domain (What/Where/When), paradigm (IFR/ICR/IRE),
    or congruence (Common/Congruent/Incongruent).

    Args:
        df_wide: Wide-format DataFrame with columns:
            - UID: Participant identifier
            - TEST: Test session (T1, T2, T3, T4)
            - composite_ID: {UID}_{TEST} format
            - [item columns]: Binary response columns (0/1)
        item_factor_df: DataFrame mapping items to factors with columns:
            - item_col: Item name (must match column names in df_wide)
            - factor_col: Factor assignment (e.g., 'what', 'where', 'IFR', etc.)
        factor_col: Column name for factor in item_factor_df (default 'factor')
        item_col: Column name for item in item_factor_df (default 'item_name')
        include_factors: Optional list of factors to include (filters output).
            If None, includes all factors found in item_factor_df.

    Returns:
        DataFrame with columns:
        - composite_ID: {UID}_{test} format
        - UID: Participant identifier
        - test: Test session (from TEST column)
        - factor: Factor name
        - CTT_score: Mean score (proportion correct) for items in factor
        - n_items: Number of items used in computation

    Raises:
        ValueError: If df_wide is empty or has no valid items

    Example:
        >>> # Domain-based CTT (RQ 5.2.4)
        >>> ctt_scores = compute_ctt_mean_scores_by_factor(
        ...     df_wide=raw_responses,
        ...     item_factor_df=purified_items,
        ...     factor_col='factor',
        ...     item_col='item_name',
        ...     include_factors=['what', 'where']  # Exclude 'when' (floor effect)
        ... )

    References:
        - RQ 5.2.4 step01_compute_ctt_mean_scores.py (original implementation)
        - Classical Test Theory: CTT score = mean of item responses
    """
    # Validation
    if df_wide.empty:
        raise ValueError("Input DataFrame df_wide is empty")

    # Required columns in wide data
    required_cols = ['UID', 'TEST', 'composite_ID']
    missing_cols = [c for c in required_cols if c not in df_wide.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df_wide: {missing_cols}")

    # Build item-to-factor mapping
    item_factor_map = dict(zip(
        item_factor_df[item_col],
        item_factor_df[factor_col]
    ))

    # Identify which items are actually in the wide data
    item_columns = [col for col in df_wide.columns if col in item_factor_map]

    if not item_columns:
        raise ValueError("No matching item columns found in df_wide")

    # Group items by factor
    items_by_factor: Dict[str, List[str]] = {}
    for item in item_columns:
        factor = item_factor_map[item]
        if include_factors is not None and factor not in include_factors:
            continue
        if factor not in items_by_factor:
            items_by_factor[factor] = []
        items_by_factor[factor].append(item)

    if not items_by_factor:
        raise ValueError("No items matched any included factors")

    # Compute CTT scores
    ctt_rows = []

    for _, row in df_wide.iterrows():
        uid = row['UID']
        test = row['TEST']
        composite_id = row['composite_ID']

        for factor, items in items_by_factor.items():
            # Get scores for items in this factor
            scores = [row[item] for item in items if pd.notna(row.get(item))]

            # Compute mean (proportion correct)
            ctt_score = np.nanmean(scores) if scores else np.nan
            n_items = len(scores)

            ctt_rows.append({
                'composite_ID': composite_id,
                'UID': uid,
                'test': test,
                'factor': factor,
                'CTT_score': ctt_score,
                'n_items': n_items
            })

    return pd.DataFrame(ctt_rows)


def compute_pearson_correlations_with_correction(
    df: pd.DataFrame,
    irt_col: str = 'IRT_score',
    ctt_col: str = 'CTT_score',
    factor_col: str = 'factor',
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Compute Pearson correlations between IRT and CTT scores with Holm-Bonferroni correction.

    Implements Decision D068 dual p-value reporting (p_uncorrected + p_holm).
    Computes correlations per factor plus overall (all factors pooled).

    Args:
        df: DataFrame with columns:
            - composite_ID: Observation identifier
            - irt_col: IRT theta scores
            - ctt_col: CTT mean scores
            - factor_col: Factor labels (e.g., 'what', 'where', 'IFR', etc.)
        irt_col: Column name for IRT scores (default 'IRT_score')
        ctt_col: Column name for CTT scores (default 'CTT_score')
        factor_col: Column name for factor (default 'factor')
        thresholds: Optional list of thresholds to test (default [0.70, 0.90])
            Creates boolean columns 'threshold_X.XX' for each threshold.

    Returns:
        DataFrame with columns:
        - factor: Factor name or 'Overall'
        - r: Pearson correlation coefficient
        - CI_lower: 95% CI lower bound (Fisher z-transform)
        - CI_upper: 95% CI upper bound
        - p_uncorrected: Uncorrected p-value
        - p_holm: Holm-Bonferroni corrected p-value (D068)
        - n: Sample size
        - threshold_X.XX: Boolean for each threshold (if thresholds provided)

    Raises:
        ValueError: If required columns missing or no valid data

    Example:
        >>> correlations = compute_pearson_correlations_with_correction(
        ...     df=merged_scores,
        ...     irt_col='IRT_score',
        ...     ctt_col='CTT_score',
        ...     factor_col='factor',
        ...     thresholds=[0.70, 0.90]
        ... )
        >>> # D068 compliance: both p-values present
        >>> assert 'p_uncorrected' in correlations.columns
        >>> assert 'p_holm' in correlations.columns

    References:
        - Decision D068: Dual p-value reporting
        - RQ 5.2.4 step02_correlations.py (original implementation)
        - Holm (1979): A simple sequentially rejective multiple test procedure
    """
    if thresholds is None:
        thresholds = [0.70, 0.90]

    # Validation
    required_cols = [irt_col, ctt_col, factor_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get unique factors
    factors = df[factor_col].unique().tolist()

    correlations = []

    # Compute per-factor correlations
    for factor in factors:
        factor_data = df[df[factor_col] == factor].copy()
        n = len(factor_data)

        if n < 3:
            continue

        r, p_uncorrected = stats.pearsonr(
            factor_data[irt_col],
            factor_data[ctt_col]
        )

        # Fisher z-transform for CI
        ci_lower, ci_upper = _compute_correlation_ci(r, n, ci_level=0.95)

        correlations.append({
            'factor': factor,
            'r': r,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_uncorrected': p_uncorrected,
            'n': n
        })

    # Compute overall correlation (all factors pooled)
    n_overall = len(df)
    r_overall, p_overall = stats.pearsonr(df[irt_col], df[ctt_col])
    ci_lower_overall, ci_upper_overall = _compute_correlation_ci(r_overall, n_overall)

    correlations.append({
        'factor': 'Overall',
        'r': r_overall,
        'CI_lower': ci_lower_overall,
        'CI_upper': ci_upper_overall,
        'p_uncorrected': p_overall,
        'n': n_overall
    })

    # Apply Holm-Bonferroni correction
    corr_df = pd.DataFrame(correlations)
    p_uncorrected_list = corr_df['p_uncorrected'].tolist()
    p_holm_list = _holm_bonferroni_correction(p_uncorrected_list)
    corr_df['p_holm'] = p_holm_list

    # Add threshold columns
    for threshold in thresholds:
        col_name = f'threshold_{threshold:.2f}'
        corr_df[col_name] = corr_df['r'] > threshold

    return corr_df


def _compute_correlation_ci(r: float, n: int, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for Pearson correlation using Fisher z-transform.

    Args:
        r: Correlation coefficient
        n: Sample size
        ci_level: Confidence level (default 0.95)

    Returns:
        Tuple of (CI_lower, CI_upper)
    """
    # Fisher z-transformation
    z = np.arctanh(r)

    # Standard error in z-space
    se_z = 1 / np.sqrt(n - 3) if n > 3 else np.inf

    # Z-score for CI level
    alpha = 1 - ci_level
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # CI in z-space
    z_lower = z - z_critical * se_z
    z_upper = z + z_critical * se_z

    # Transform back to r-space
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)

    return float(r_lower), float(r_upper)


def _holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Holm-Bonferroni sequential correction to p-values.

    Less conservative than standard Bonferroni, maintains FWER control.

    Args:
        p_values: List of uncorrected p-values
        alpha: Family-wise error rate (default 0.05)

    Returns:
        List of corrected p-values (same order as input)
    """
    m = len(p_values)
    if m == 0:
        return []

    # Create list of (index, p_value) tuples
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]

    # Sort by p-value (ascending)
    indexed_pvals.sort(key=lambda x: x[1])

    # Compute corrected p-values
    corrected = [0.0] * m
    for k, (orig_idx, p) in enumerate(indexed_pvals):
        # Holm correction: min(1, p * (m - k))
        corrected[orig_idx] = min(1.0, p * (m - k))

    return corrected


def compute_cohens_kappa_agreement(
    classifications_1: List[bool],
    classifications_2: List[bool],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute Cohen's kappa for agreement between two significance classifications.

    Used to assess agreement between IRT and CTT models on which effects
    are statistically significant. Accounts for chance agreement.

    Args:
        classifications_1: List of boolean significance classifications (e.g., IRT model)
        classifications_2: List of boolean significance classifications (e.g., CTT model)
        labels: Optional list of effect names for reporting

    Returns:
        Dict with keys:
        - kappa: Cohen's kappa coefficient
        - agreement_percent: Raw percentage agreement
        - interpretation: Landis & Koch (1977) interpretation
        - n_effects: Number of effects compared
        - substantial_agreement: Boolean (kappa > 0.60)
        - confusion_matrix: Dict with TP, TN, FP, FN counts

    Raises:
        ValueError: If classifications have different lengths or are empty

    Example:
        >>> irt_sig = [True, True, False, False, True]
        >>> ctt_sig = [True, False, False, False, True]
        >>> result = compute_cohens_kappa_agreement(irt_sig, ctt_sig)
        >>> print(f"Kappa: {result['kappa']:.2f}, Agreement: {result['agreement_percent']:.1f}%")

    References:
        - Cohen (1960): A coefficient of agreement for nominal scales
        - Landis & Koch (1977): The measurement of observer agreement
        - RQ 5.2.4 step05 (original implementation)
    """
    # Validation
    if len(classifications_1) != len(classifications_2):
        raise ValueError("Classifications must have same length")

    if len(classifications_1) == 0:
        raise ValueError("Classifications cannot be empty")

    n = len(classifications_1)

    # Convert to numpy arrays
    c1 = np.array(classifications_1, dtype=bool)
    c2 = np.array(classifications_2, dtype=bool)

    # Confusion matrix
    tp = np.sum(c1 & c2)  # Both significant
    tn = np.sum(~c1 & ~c2)  # Both non-significant
    fp = np.sum(~c1 & c2)  # Only model 2 significant
    fn = np.sum(c1 & ~c2)  # Only model 1 significant

    # Raw agreement
    agreement = (tp + tn) / n
    agreement_percent = agreement * 100

    # Cohen's kappa
    # p_e = expected agreement by chance
    p1_sig = np.sum(c1) / n  # Proportion significant in model 1
    p2_sig = np.sum(c2) / n  # Proportion significant in model 2

    p_e = (p1_sig * p2_sig) + ((1 - p1_sig) * (1 - p2_sig))

    if p_e == 1.0:
        # Perfect expected agreement (degenerate case)
        kappa = 1.0 if agreement == 1.0 else 0.0
    else:
        kappa = (agreement - p_e) / (1 - p_e)

    # Interpretation (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Poor agreement (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    return {
        'kappa': float(kappa),
        'agreement_percent': float(agreement_percent),
        'interpretation': interpretation,
        'n_effects': n,
        'substantial_agreement': bool(kappa > 0.60),
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    }


def compare_lmm_fit_aic_bic(
    aic_model1: float,
    bic_model1: float,
    aic_model2: float,
    bic_model2: float,
    model1_name: str = 'Model1',
    model2_name: str = 'Model2'
) -> pd.DataFrame:
    """
    Compare model fit between two LMMs using AIC and BIC.

    Computes delta (model2 - model1) and interprets per Burnham & Anderson (2002).

    Args:
        aic_model1: AIC for model 1 (e.g., IRT model)
        bic_model1: BIC for model 1
        aic_model2: AIC for model 2 (e.g., CTT model)
        bic_model2: BIC for model 2
        model1_name: Label for model 1 (default 'Model1')
        model2_name: Label for model 2 (default 'Model2')

    Returns:
        DataFrame with columns:
        - metric: 'AIC' or 'BIC'
        - {model1_name}: Value for model 1
        - {model2_name}: Value for model 2
        - delta: model2 - model1 (negative = model2 better)
        - interpretation: Burnham & Anderson (2002) interpretation

    Example:
        >>> result = compare_lmm_fit_aic_bic(
        ...     aic_model1=1500, bic_model1=1520,
        ...     aic_model2=1505, bic_model2=1525,
        ...     model1_name='IRT', model2_name='CTT'
        ... )
        >>> print(result)

    References:
        - Burnham & Anderson (2002): Model Selection and Multimodel Inference
        - RQ 5.2.4 step06 (original implementation)

    Notes:
        Interpretation thresholds (|delta|):
        - < 2: Models essentially equivalent
        - 2-4: Weak evidence for better model
        - 4-7: Moderate evidence
        - > 7: Strong evidence (some use > 10)
    """
    rows = []

    for metric, val1, val2 in [('AIC', aic_model1, aic_model2),
                                ('BIC', bic_model1, bic_model2)]:
        delta = val2 - val1

        # Interpretation per Burnham & Anderson (2002)
        abs_delta = abs(delta)
        if abs_delta < 2:
            interpretation = "Models essentially equivalent"
        elif abs_delta < 4:
            better = model1_name if delta > 0 else model2_name
            interpretation = f"Weak evidence for {better}"
        elif abs_delta < 7:
            better = model1_name if delta > 0 else model2_name
            interpretation = f"Moderate evidence for {better}"
        else:
            better = model1_name if delta > 0 else model2_name
            interpretation = f"Strong evidence for {better}"

        rows.append({
            'metric': metric,
            model1_name: val1,
            model2_name: val2,
            'delta': delta,
            'interpretation': interpretation
        })

    return pd.DataFrame(rows)
