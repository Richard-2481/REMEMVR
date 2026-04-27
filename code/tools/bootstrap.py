"""Bootstrap resampling methods for confidence intervals and hypothesis testing."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Optional, Dict, Union, Tuple
import warnings


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    method: str = 'pearson',
    seed: Optional[int] = None
) -> Dict:
    """
    Calculate bootstrap confidence intervals for correlation coefficients.
    
    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Correlation method ('pearson' or 'spearman')
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict containing:
        - r: Original correlation coefficient
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - se: Standard error
        - bootstrap_samples: Array of bootstrap correlations
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate original correlation
    if method == 'pearson':
        r_orig = np.corrcoef(x, y)[0, 1]
    elif method == 'spearman':
        r_orig = stats.spearmanr(x, y)[0]
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Bootstrap resampling
    n = len(x)
    bootstrap_samples = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement (paired bootstrap)
        idx = np.random.choice(n, n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        
        # Calculate correlation for bootstrap sample
        if method == 'pearson':
            bootstrap_samples[i] = np.corrcoef(x_boot, y_boot)[0, 1]
        elif method == 'spearman':
            bootstrap_samples[i] = stats.spearmanr(x_boot, y_boot)[0]
    
    # Calculate confidence intervals (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    # Calculate standard error
    se = np.std(bootstrap_samples)
    
    return {
        'r': r_orig,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bootstrap_samples': bootstrap_samples
    }


def bootstrap_mean_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    method: str = 'percentile',
    seed: Optional[int] = None
) -> Dict:
    """
    Calculate bootstrap confidence intervals for the mean.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    method : str
        CI method ('percentile' or 'bca')
    seed : int, optional
        Random seed
    
    Returns
    -------
    Dict containing:
        - mean: Original mean
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - se: Standard error
        - bootstrap_samples: Array of bootstrap means
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean_orig = np.mean(data)
    n = len(data)
    
    # Bootstrap resampling
    bootstrap_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        bootstrap_samples[i] = np.mean(sample)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    
    if method == 'percentile':
        ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    elif method == 'bca':
        # Bias-corrected and accelerated (BCa) method
        # Calculate bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_samples < mean_orig))
        
        # Calculate acceleration using jackknife
        jackknife_means = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_means[i] = np.mean(jack_sample)
        
        mean_jackknife = np.mean(jackknife_means)
        a = np.sum((mean_jackknife - jackknife_means) ** 3) / (
            6 * np.sum((mean_jackknife - jackknife_means) ** 2) ** 1.5
        )
        
        # Adjust percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1alpha = stats.norm.ppf(1 - alpha / 2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))
        
        ci_lower = np.percentile(bootstrap_samples, 100 * p_lower)
        ci_upper = np.percentile(bootstrap_samples, 100 * p_upper)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    se = np.std(bootstrap_samples)
    
    return {
        'mean': mean_orig,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bootstrap_samples': bootstrap_samples
    }


def bootstrap_median_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict:
    """
    Calculate bootstrap confidence intervals for the median.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    seed : int, optional
        Random seed
    
    Returns
    -------
    Dict containing:
        - median: Original median
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - se: Standard error
        - bootstrap_samples: Array of bootstrap medians
    """
    if seed is not None:
        np.random.seed(seed)
    
    median_orig = np.median(data)
    n = len(data)
    
    # Bootstrap resampling
    bootstrap_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        bootstrap_samples[i] = np.median(sample)
    
    # Calculate confidence intervals (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    se = np.std(bootstrap_samples)
    
    return {
        'median': median_orig,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bootstrap_samples': bootstrap_samples
    }


def bootstrap_statistic(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict:
    """
    General bootstrap for any statistic function.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (can be 1D or 2D for paired/multivariate data)
    statistic : Callable
        Function that computes the statistic of interest
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    seed : int, optional
        Random seed
    
    Returns
    -------
    Dict containing:
        - statistic: Original statistic value
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - se: Standard error
        - bootstrap_samples: Array of bootstrap statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate original statistic
    stat_orig = statistic(data)
    
    # Determine sample size
    if data.ndim == 1:
        n = len(data)
    else:
        n = data.shape[0]
    
    # Bootstrap resampling
    bootstrap_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        if data.ndim == 1:
            sample = data[idx]
        else:
            sample = data[idx, :]
        
        bootstrap_samples[i] = statistic(sample)
    
    # Calculate confidence intervals (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    se = np.std(bootstrap_samples)
    
    return {
        'statistic': stat_orig,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bootstrap_samples': bootstrap_samples
    }