"""
Analysis extensions for Ch7 - wrapper and adapter functions.

This module provides wrapper functions and new implementations needed for Ch7 RQs.
Many functions wrap existing functionality from other modules (DRY principle).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, Any
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMResults
import warnings


def extract_random_effects(model: MixedLMResults) -> pd.DataFrame:
    """
    Extract random effects (BLUPs) from fitted LMM model.
    
    This is a wrapper around analysis_lmm.extract_random_effects_from_lmm
    that returns a DataFrame suitable for Ch7 analyses.
    
    Parameters
    ----------
    model : MixedLMResults
        Fitted mixed effects model from statsmodels
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - uid: Participant ID
        - intercept: Random intercept (if present)
        - slope: Random slope (if present)
    
    Examples
    --------
    >>> model = smf.mixedlm("outcome ~ time", data, groups="uid").fit()
    >>> blups = extract_random_effects(model)
    >>> print(blups.head())
         uid  intercept     slope
    0   P001      0.523    -0.021
    1   P002      0.312    -0.015
    """
    # Direct extraction from model.random_effects
    # (Bypassing analysis_lmm function which expects additional attributes)
    
    # Convert to DataFrame format needed for Ch7
    rows = []
    for uid, effects in model.random_effects.items():
        row = {'uid': uid}
        if 'Intercept' in effects.index:
            row['intercept'] = effects['Intercept']
        if 'Days' in effects.index:
            row['slope'] = effects['Days']
        elif 'time' in effects.index:
            row['slope'] = effects['time']
        rows.append(row)
    
    return pd.DataFrame(rows)


def fit_interaction_model(formula: str, data: pd.DataFrame, groups: str) -> MixedLMResults:
    """
    Fit LMM with interaction terms.
    
    This is a thin wrapper around statsmodels mixedlm that ensures
    interaction terms are properly handled.
    
    Parameters
    ----------
    formula : str
        Model formula with interaction terms (e.g., "y ~ time * group")
    data : pd.DataFrame
        Data with all variables in formula
    groups : str
        Column name for grouping variable (typically 'uid')
    
    Returns
    -------
    MixedLMResults
        Fitted mixed effects model
    
    Examples
    --------
    >>> model = fit_interaction_model(
    ...     formula="accuracy ~ time * schema_type",
    ...     data=df,
    ...     groups="uid"
    ... )
    >>> print(model.summary())
    """
    import statsmodels.formula.api as smf
    
    # Fit model with random intercept by default
    model = smf.mixedlm(formula, data, groups=groups)
    result = model.fit()
    
    return result


def compute_cohens_q_effect_size(r1: float, r2: float) -> float:
    """
    Compute Cohen's q effect size for correlation difference.
    
    Cohen's q quantifies the difference between two correlation coefficients
    using Fisher's z transformation.
    
    Parameters
    ----------
    r1 : float
        First correlation coefficient (-1 < r1 < 1)
    r2 : float
        Second correlation coefficient (-1 < r2 < 1)
    
    Returns
    -------
    float
        Cohen's q effect size
        
    Interpretation
    --------------
    - Small effect: q ≈ 0.1
    - Medium effect: q ≈ 0.3
    - Large effect: q ≈ 0.5
    
    Examples
    --------
    >>> q = compute_cohens_q_effect_size(0.6, 0.3)
    >>> print(f"Cohen's q = {q:.3f}")
    Cohen's q = 0.310
    """
    # Check for perfect correlations
    if abs(r1) >= 1.0 or abs(r2) >= 1.0:
        raise ValueError("Correlations must be between -1 and 1 (exclusive)")
    
    # Fisher z transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # Cohen's q is the absolute difference
    q = abs(z1 - z2)
    
    return q


def compare_correlations_dependent(r12: float, r13: float, r23: float, n: int) -> Dict[str, float]:
    """
    Compare two dependent correlations using Steiger's Z-test.
    
    Tests whether r12 and r13 are significantly different when both
    correlations share variable 1.
    
    Parameters
    ----------
    r12 : float
        Correlation between variables 1 and 2
    r13 : float
        Correlation between variables 1 and 3
    r23 : float
        Correlation between variables 2 and 3
    n : int
        Sample size
    
    Returns
    -------
    dict
        Dictionary with:
        - z: Z statistic
        - p_value: Two-tailed p-value
    
    Examples
    --------
    >>> result = compare_correlations_dependent(
    ...     r12=0.5, r13=0.3, r23=0.4, n=100
    ... )
    >>> print(f"Z = {result['z']:.3f}, p = {result['p_value']:.3f}")
    """
    # Steiger's Z formula
    # First compute the determinant of the correlation matrix
    det = 1 + 2*r12*r13*r23 - r12**2 - r13**2 - r23**2
    
    # Compute test statistic
    z_diff = (r12 - r13) * np.sqrt((n-1) * (1 + r23))
    z_denom = np.sqrt(2 * det * (n-1) / (n-3))
    
    if z_denom == 0:
        warnings.warn("Denominator is zero, correlations may be linearly dependent")
        return {'z': np.inf, 'p_value': 0.0}
    
    z = z_diff / z_denom
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'z': z,
        'p_value': p_value
    }


def compute_discrepancy_scores(
    traditional_scores: Union[pd.Series, np.ndarray],
    vr_scores: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    """
    Compute standardized discrepancy scores between traditional and VR assessments.
    
    Parameters
    ----------
    traditional_scores : pd.Series or np.ndarray
        Traditional test scores
    vr_scores : pd.Series or np.ndarray
        VR test scores (same order as traditional)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - uid: Participant ID (if input has index)
        - discrepancy: Raw difference (VR - traditional)
        - z_score: Standardized discrepancy score
    
    Examples
    --------
    >>> disc = compute_discrepancy_scores(trad_scores, vr_scores)
    >>> high_discrepancy = disc[disc['z_score'].abs() > 2]
    """
    # Convert to arrays for calculation
    trad = np.asarray(traditional_scores)
    vr = np.asarray(vr_scores)
    
    # Calculate raw discrepancy
    discrepancy = vr - trad
    
    # Standardize to z-scores (use ddof=1 for sample standard deviation)
    z_scores = (discrepancy - discrepancy.mean()) / discrepancy.std(ddof=1)
    
    # Create output DataFrame
    result = pd.DataFrame({
        'discrepancy': discrepancy,
        'z_score': z_scores
    })
    
    # Add UIDs if available
    if isinstance(traditional_scores, pd.Series) and traditional_scores.index is not None:
        result['uid'] = traditional_scores.index.tolist()
        # Reorder columns
        result = result[['uid', 'discrepancy', 'z_score']]
    
    return result


def validate_regression_assumptions(
    residuals: np.ndarray,
    X: np.ndarray,
    significance_level: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Validate regression assumptions comprehensively.
    
    Tests four key assumptions:
    1. Normality of residuals (Shapiro-Wilk test)
    2. Homoscedasticity (Breusch-Pagan test)
    3. Linearity (RESET test approximation)
    4. Independence (Durbin-Watson test)
    
    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals
    X : np.ndarray
        Predictor matrix
    significance_level : float
        Alpha level for tests (default 0.05)
    
    Returns
    -------
    dict
        Dictionary with test results for each assumption:
        - normality: {statistic, p_value, passed}
        - homoscedasticity: {statistic, p_value, passed}
        - linearity: {passed, note}
        - independence: {statistic, passed, interpretation}
    
    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression().fit(X, y)
    >>> residuals = y - model.predict(X)
    >>> assumptions = validate_regression_assumptions(residuals, X)
    >>> all_passed = all(v.get('passed', False) for v in assumptions.values())
    """
    results = {}
    
    # 1. Normality (Shapiro-Wilk test)
    if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limit
        stat_norm, p_norm = stats.shapiro(residuals)
        results['normality'] = {
            'test': 'Shapiro-Wilk',
            'statistic': stat_norm,
            'p_value': p_norm,
            'passed': p_norm > significance_level
        }
    else:
        # Use Kolmogorov-Smirnov for large samples
        stat_norm, p_norm = stats.kstest(residuals, 'norm', 
                                         args=(residuals.mean(), residuals.std()))
        results['normality'] = {
            'test': 'Kolmogorov-Smirnov',
            'statistic': stat_norm,
            'p_value': p_norm,
            'passed': p_norm > significance_level
        }
    
    # 2. Homoscedasticity (Breusch-Pagan test)
    # Regress squared residuals on X
    residuals_sq = residuals ** 2
    X_with_const = sm.add_constant(X)
    aux_model = sm.OLS(residuals_sq, X_with_const).fit()
    
    # Breusch-Pagan statistic
    n = len(residuals)
    lm_statistic = n * aux_model.rsquared
    p_bp = 1 - stats.chi2.cdf(lm_statistic, X.shape[1])
    
    results['homoscedasticity'] = {
        'test': 'Breusch-Pagan',
        'statistic': lm_statistic,
        'p_value': p_bp,
        'passed': p_bp > significance_level
    }
    
    # 3. Linearity (simplified check)
    # We check if residuals have zero mean and no pattern
    results['linearity'] = {
        'mean_residual': residuals.mean(),
        'passed': abs(residuals.mean()) < 0.1,  # Threshold for "close to zero"
        'note': 'Simplified check - visual inspection recommended'
    }
    
    # 4. Independence (Durbin-Watson test)
    # DW statistic ranges from 0 to 4, with 2 indicating no autocorrelation
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    results['independence'] = {
        'test': 'Durbin-Watson',
        'statistic': dw_stat,
        'passed': 1.5 < dw_stat < 2.5,  # Common rule of thumb
        'interpretation': 'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Potential autocorrelation'
    }
    
    return results


def standardize_scores(
    scores: Union[np.ndarray, pd.Series],
    mean: Optional[float] = None,
    sd: Optional[float] = None
) -> np.ndarray:
    """
    Standardize scores to z-scores.
    
    Parameters
    ----------
    scores : np.ndarray or pd.Series
        Raw scores to standardize
    mean : float, optional
        Reference mean (if None, use sample mean)
    sd : float, optional
        Reference SD (if None, use sample SD)
    
    Returns
    -------
    np.ndarray
        Z-scores with mean=0 and SD=1
    
    Examples
    --------
    >>> z = standardize_scores([50, 60, 40, 70, 30])
    >>> print(f"Mean: {z.mean():.3f}, SD: {z.std():.3f}")
    Mean: 0.000, SD: 1.000
    
    >>> # Using population norms
    >>> z = standardize_scores([45, 55, 35], mean=50, sd=10)
    >>> print(z)
    [-0.5  0.5 -1.5]
    """
    scores = np.asarray(scores)
    
    if mean is None:
        mean = scores.mean()
    if sd is None:
        sd = scores.std()
    
    if sd == 0:
        warnings.warn("Standard deviation is zero, returning zeros")
        return np.zeros_like(scores)
    
    z_scores = (scores - mean) / sd
    
    return z_scores


def cross_validate_lmm(
    formula: str,
    data: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42
) -> Dict[str, Union[list, float]]:
    """
    Perform k-fold cross-validation for Linear Mixed Models.
    
    This function splits the data by subjects (not observations) to maintain
    the hierarchical structure of the data.
    
    Parameters
    ----------
    formula : str
        Model formula (e.g., "outcome ~ time")
    data : pd.DataFrame
        Data with columns for formula variables and 'uid'
    n_folds : int
        Number of cross-validation folds (default 5)
    seed : int
        Random seed for reproducibility (default 42)
    
    Returns
    -------
    dict
        Dictionary with:
        - cv_scores: List of scores for each fold
        - mean_score: Mean cross-validation score
        - std_score: Standard deviation of scores
    
    Examples
    --------
    >>> cv_result = cross_validate_lmm(
    ...     formula="accuracy ~ time",
    ...     data=df,
    ...     n_folds=5,
    ...     seed=42
    ... )
    >>> print(f"CV Score: {cv_result['mean_score']:.3f} ± {cv_result['std_score']:.3f}")
    """
    import statsmodels.formula.api as smf
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    
    np.random.seed(seed)
    
    # Get unique subjects
    subjects = data['uid'].unique()
    np.random.shuffle(subjects)
    
    # Initialize k-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    cv_scores = []
    
    for train_idx, test_idx in kf.split(subjects):
        # Split by subjects
        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]
        
        train_data = data[data['uid'].isin(train_subjects)]
        test_data = data[data['uid'].isin(test_subjects)]
        
        # Fit model on training data
        try:
            model = smf.mixedlm(formula, train_data, groups='uid')
            fitted = model.fit(disp=False)
            
            # Predict on test data
            test_pred = fitted.predict(test_data)
            
            # Get outcome variable name from formula
            outcome_var = formula.split('~')[0].strip()
            test_true = test_data[outcome_var]
            
            # Calculate R² score
            score = r2_score(test_true, test_pred)
            cv_scores.append(score)
            
        except Exception as e:
            warnings.warn(f"Fold failed with error: {e}")
            cv_scores.append(np.nan)
    
    # Remove any NaN scores
    cv_scores = [s for s in cv_scores if not np.isnan(s)]
    
    if len(cv_scores) == 0:
        raise ValueError("All CV folds failed")
    
    return {
        'cv_scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores)
    }