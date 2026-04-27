"""
Regression analysis tools for Ch7 RQs
Provides comprehensive regression functionality with diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


def fit_multiple_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit multiple linear regression with comprehensive output.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Predictor variables
    y : array-like, shape (n_samples,)
        Outcome variable
    feature_names : list of str, optional
        Names for features (if X is array)
    
    Returns
    -------
    dict
        coefficients: Dict mapping feature names to coefficients
        pvalues: Dict mapping feature names to p-values
        rsquared: R-squared value
        rsquared_adj: Adjusted R-squared
        residuals: Array of residuals
        fvalue: F-statistic for overall model
        f_pvalue: P-value for F-test
        std_errors: Standard errors of coefficients
        conf_int: 95% confidence intervals
    """
    # Convert to arrays if needed
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X = X.values
    
    if isinstance(y, pd.Series):
        y = y.values
    
    if feature_names is None:
        feature_names = [f'x{i+1}' for i in range(X.shape[1])]
    
    # Add intercept
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    # Create coefficient dictionary
    coef_dict = {'intercept': model.params[0]}
    pval_dict = {'intercept': model.pvalues[0]}
    stderr_dict = {'intercept': model.bse[0]}
    
    for i, name in enumerate(feature_names):
        coef_dict[name] = model.params[i+1]
        pval_dict[name] = model.pvalues[i+1]
        stderr_dict[name] = model.bse[i+1]
    
    # Get confidence intervals
    conf_int = model.conf_int()
    ci_dict = {}
    ci_dict['intercept'] = (conf_int[0, 0], conf_int[0, 1])
    for i, name in enumerate(feature_names):
        ci_dict[name] = (conf_int[i+1, 0], conf_int[i+1, 1])
    
    return {
        'coefficients': coef_dict,
        'pvalues': pval_dict,
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'residuals': model.resid,
        'fvalue': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'std_errors': stderr_dict,
        'conf_int': ci_dict,
        'aic': model.aic,
        'bic': model.bic,
        'model': model  # Keep for diagnostics
    }


def fit_hierarchical_regression(
    X_blocks: List[np.ndarray],
    y: Union[np.ndarray, pd.Series],
    block_names: List[str]
) -> Dict[str, Any]:
    """
    Fit hierarchical/block regression with incremental R² tests.
    
    Parameters
    ----------
    X_blocks : list of arrays
        List of predictor blocks, each shape (n_samples, n_features_block)
    y : array-like
        Outcome variable
    block_names : list of str
        Names for each block
    
    Returns
    -------
    dict
        models: List of fitted models for each step
        delta_r2: Dict of R² change for each block
        f_tests: Dict of F-tests for R² change
        cumulative_r2: R² at each step
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    models = []
    delta_r2 = {}
    f_tests = {}
    cumulative_r2 = []
    
    # Build predictors cumulatively
    X_cumulative = None
    prev_r2 = 0
    prev_k = 0  # Number of predictors in previous model
    
    for i, (X_block, name) in enumerate(zip(X_blocks, block_names)):
        # Add new block to cumulative predictors
        if X_cumulative is None:
            X_cumulative = X_block
        else:
            X_cumulative = np.column_stack([X_cumulative, X_block])
        
        # Fit model with cumulative predictors
        X_with_const = sm.add_constant(X_cumulative)
        model = sm.OLS(y, X_with_const).fit()
        models.append(model)
        
        # Calculate R² change
        curr_r2 = model.rsquared
        cumulative_r2.append(curr_r2)
        
        if i == 0:
            # First block - compare to null model
            delta_r2[name] = curr_r2
            # F-test for first block
            f_stat = model.fvalue
            f_pval = model.f_pvalue
        else:
            # Subsequent blocks - incremental test
            delta_r2[name] = curr_r2 - prev_r2
            
            # F-test for R² change
            n = len(y)
            k_new = X_block.shape[1]  # Number of new predictors
            k_full = X_cumulative.shape[1]  # Total predictors
            
            # F = [(R²_full - R²_reduced) / (k_full - k_reduced)] / [(1 - R²_full) / (n - k_full - 1)]
            if curr_r2 > prev_r2:
                f_stat = ((curr_r2 - prev_r2) / k_new) / ((1 - curr_r2) / (n - k_full - 1))
                f_pval = 1 - stats.f.cdf(f_stat, k_new, n - k_full - 1)
            else:
                f_stat = 0
                f_pval = 1.0
        
        f_tests[name] = {
            'f_statistic': f_stat,
            'p_value': f_pval,
            'df_num': X_block.shape[1] if i > 0 else X_cumulative.shape[1],
            'df_denom': len(y) - X_cumulative.shape[1] - 1
        }
        
        prev_r2 = curr_r2
        prev_k = X_cumulative.shape[1]
    
    return {
        'models': models,
        'delta_r2': delta_r2,
        'f_tests': f_tests,
        'cumulative_r2': cumulative_r2,
        'block_names': block_names
    }


def compute_regression_diagnostics(
    model: sm.regression.linear_model.RegressionResults,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive regression diagnostics.
    
    Parameters
    ----------
    model : statsmodels RegressionResults
        Fitted regression model
    X : array-like
        Predictor variables (without intercept)
    y : array-like
        Outcome variable
    
    Returns
    -------
    dict
        vif: Variance Inflation Factors for each predictor
        cooks_d: Cook's distance for each observation
        leverage: Leverage values
        studentized_residuals: Studentized residuals
        durbin_watson: Durbin-Watson statistic
    """
    # Ensure arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # VIF calculation
    vif = []
    X_with_const = sm.add_constant(X)
    for i in range(1, X_with_const.shape[1]):  # Skip intercept
        vif.append(variance_inflation_factor(X_with_const, i))
    
    # Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    # Leverage
    leverage = influence.hat_matrix_diag
    
    # Studentized residuals
    studentized_residuals = influence.resid_studentized_external
    
    # Durbin-Watson
    dw = durbin_watson(model.resid)
    
    # Additional diagnostics
    # Condition number (multicollinearity)
    condition_number = np.linalg.cond(X_with_const)
    
    # Breusch-Pagan test for heteroscedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    
    return {
        'vif': vif,
        'cooks_d': cooks_d,
        'leverage': leverage,
        'studentized_residuals': studentized_residuals,
        'durbin_watson': dw,
        'condition_number': condition_number,
        'breusch_pagan': {
            'lm_statistic': bp_test[0],
            'p_value': bp_test[1]
        }
    }


def cross_validate_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_folds: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for regression.
    
    Parameters
    ----------
    X : array-like
        Predictor variables
    y : array-like
        Outcome variable
    n_folds : int
        Number of CV folds
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        cv_scores: R² for each fold
        mean_r2: Mean R² across folds
        std_r2: Standard deviation of R²
        fold_predictions: Predictions for each fold
        fold_indices: Train/test indices for each fold
    """
    # Convert to arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    cv_scores = []
    fold_predictions = {}
    fold_indices = {}
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Predict and score
        y_pred = model.predict(X_test_const)
        
        # Calculate R² on test set
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        cv_scores.append(r2)
        fold_predictions[f'fold_{i+1}'] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'test_indices': test_idx
        }
        fold_indices[f'fold_{i+1}'] = {
            'train': train_idx,
            'test': test_idx
        }
    
    return {
        'cv_scores': cv_scores,
        'mean_r2': np.mean(cv_scores),
        'std_r2': np.std(cv_scores),
        'fold_predictions': fold_predictions,
        'fold_indices': fold_indices,
        'confidence_interval': (
            np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(n_folds),
            np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(n_folds)
        )
    }


def bootstrap_regression_ci(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Bootstrap confidence intervals for regression coefficients.
    
    Parameters
    ----------
    X : array-like
        Predictor variables
    y : array-like
        Outcome variable
    n_bootstrap : int
        Number of bootstrap iterations
    seed : int
        Random seed
    alpha : float
        Significance level (default 0.05 for 95% CI)
    
    Returns
    -------
    dict
        ci_lower: Lower confidence bounds
        ci_upper: Upper confidence bounds
        boot_samples: Bootstrap coefficient samples
        point_estimate: Point estimates from original data
    """
    # Convert to arrays
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f'x{i+1}' for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y = y.values
    
    np.random.seed(seed)
    n = len(y)
    n_coef = X.shape[1] + 1  # Including intercept
    
    # Fit original model for point estimate
    X_const = sm.add_constant(X)
    original_model = sm.OLS(y, X_const).fit()
    point_estimate = original_model.params
    
    # Bootstrap
    boot_samples = np.zeros((n_bootstrap, n_coef))
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Fit model
        X_boot_const = sm.add_constant(X_boot)
        try:
            model = sm.OLS(y_boot, X_boot_const).fit()
            boot_samples[i, :] = model.params
        except:
            # If singular, use previous sample
            if i > 0:
                boot_samples[i, :] = boot_samples[i-1, :]
            else:
                boot_samples[i, :] = point_estimate
    
    # Calculate percentile CIs
    ci_lower = np.percentile(boot_samples, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(boot_samples, (1 - alpha/2) * 100, axis=0)
    
    # Standard percentile method (BCa requires additional complexity)
    
    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'boot_samples': boot_samples,
        'point_estimate': point_estimate,
        'feature_names': ['intercept'] + feature_names,
        'ci_width': ci_upper - ci_lower,
        'se_boot': np.std(boot_samples, axis=0)
    }


def compute_cohens_f2(r2_full: float, r2_reduced: float) -> float:
    """
    Compute Cohen's f² effect size for regression model comparison.
    
    Cohen's f² = (R²_full - R²_reduced) / (1 - R²_full)
    
    Interpretation:
    - f² ≥ 0.02: Small effect
    - f² ≥ 0.15: Medium effect  
    - f² ≥ 0.35: Large effect
    
    Parameters
    ----------
    r2_full : float
        R-squared of full model
    r2_reduced : float
        R-squared of reduced model
    
    Returns
    -------
    float
        Cohen's f² effect size
    """
    if r2_full <= r2_reduced:
        return 0.0
    
    if r2_full >= 1.0:
        return np.inf
    
    f2 = (r2_full - r2_reduced) / (1 - r2_full)
    return f2


def compute_post_hoc_power(
    n: int,
    k_predictors: int,
    r2: float,
    alpha: float = 0.05
) -> float:
    """
    Compute post-hoc power for multiple regression.
    
    Uses Cohen's f² and non-central F distribution.
    
    Parameters
    ----------
    n : int
        Sample size
    k_predictors : int
        Number of predictors
    r2 : float
        Observed R-squared
    alpha : float
        Significance level
    
    Returns
    -------
    float
        Statistical power (0-1)
    """
    from scipy.stats import ncf
    
    # Convert R² to f²
    f2 = r2 / (1 - r2) if r2 < 1 else np.inf
    
    # Degrees of freedom
    df_num = k_predictors
    df_denom = n - k_predictors - 1
    
    if df_denom <= 0:
        return 0.0
    
    # Non-centrality parameter
    lambda_nc = f2 * (df_num + df_denom + 1)
    
    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
    
    # Power is probability of exceeding critical value under alternative hypothesis
    power = 1 - ncf.cdf(f_crit, df_num, df_denom, lambda_nc)
    
    return power


def variance_decomposition(
    model: sm.regression.linear_model.RegressionResults,
    measurement_error: float = 0.0
) -> Dict[str, float]:
    """
    Decompose variance into components.
    
    Separates total variance into:
    - True variance (explained by model)
    - Measurement error variance
    - Residual variance (unexplained)
    
    Parameters
    ----------
    model : statsmodels RegressionResults
        Fitted model
    measurement_error : float
        Known measurement error variance (proportion)
    
    Returns
    -------
    dict
        true_variance: Proportion of variance that is true signal
        error_variance: Proportion due to measurement error
        residual_variance: Unexplained variance
        icc: Intraclass correlation coefficient
    """
    r2 = model.rsquared
    
    # Adjust for measurement error if known
    if measurement_error > 0:
        # Reliability correction
        # true_r2 = observed_r2 / reliability
        reliability = 1 - measurement_error
        true_r2 = min(r2 / reliability, 1.0) if reliability > 0 else r2
    else:
        true_r2 = r2
        reliability = 1.0
    
    # Variance components
    true_variance = true_r2
    error_variance = measurement_error
    residual_variance = 1 - true_variance - error_variance
    
    # Ensure non-negative
    residual_variance = max(0, residual_variance)
    
    # ICC (for hierarchical data)
    # Here we use a simple approximation
    icc = true_variance / (true_variance + residual_variance) if (true_variance + residual_variance) > 0 else 0
    
    return {
        'true_variance': true_variance,
        'error_variance': error_variance,
        'residual_variance': residual_variance,
        'icc': icc,
        'reliability': reliability,
        'observed_r2': r2,
        'corrected_r2': true_r2
    }