"""
Statistical analysis tools with D068 dual p-value reporting
Provides comprehensive statistical tests with corrected and uncorrected p-values
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')


def one_way_anova_d068(
    groups: Optional[List[np.ndarray]] = None,
    data: Optional[pd.DataFrame] = None,
    dv: Optional[str] = None,
    between: Optional[str] = None,
    correction: str = 'bonferroni',
    n_comparisons: Optional[int] = None,
    post_hoc: Optional[str] = None
) -> Dict[str, Any]:
    """
    One-way ANOVA with D068 dual p-value reporting.
    
    Parameters
    ----------
    groups : list of arrays, optional
        List of group data arrays
    data : pd.DataFrame, optional
        DataFrame with data
    dv : str, optional
        Dependent variable column name
    between : str, optional
        Grouping variable column name
    correction : str
        Multiple comparison correction method ('bonferroni', 'fdr', 'none')
    n_comparisons : int, optional
        Number of comparisons for correction
    post_hoc : str, optional
        Post-hoc test type ('tukey', 'bonferroni')
    
    Returns
    -------
    dict
        F: F-statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        eta_squared: Effect size
        df_between: Between-group degrees of freedom
        df_within: Within-group degrees of freedom
        post_hoc: Post-hoc test results (if requested)
    """
    # Prepare data
    if data is not None and dv is not None and between is not None:
        # DataFrame input
        grouped = data.groupby(between)[dv].apply(list)
        groups = [np.array(g) for g in grouped.values]
        group_labels = grouped.index.tolist()
    else:
        # List of arrays input
        if groups is None:
            raise ValueError("Either provide groups or data+dv+between")
        group_labels = [f'Group_{i+1}' for i in range(len(groups))]
    
    # Perform ANOVA
    f_stat, p_uncorr = stats.f_oneway(*groups)
    
    # Calculate degrees of freedom
    k = len(groups)  # Number of groups
    n = sum(len(g) for g in groups)  # Total sample size
    df_between = k - 1
    df_within = n - k
    
    # Calculate eta squared (effect size)
    # η² = SS_between / SS_total
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Apply correction
    if n_comparisons is None:
        # For ANOVA, default is number of pairwise comparisons
        n_comparisons = k * (k - 1) // 2
    
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    result = {
        'F': f_stat,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'eta_squared': eta_squared,
        'df_between': df_between,
        'df_within': df_within,
        'omega_squared': calculate_omega_squared(f_stat, df_between, df_within, n),
        'group_means': {group_labels[i]: np.mean(g) for i, g in enumerate(groups)},
        'group_stds': {group_labels[i]: np.std(g, ddof=1) for i, g in enumerate(groups)}
    }
    
    # Perform post-hoc tests if requested
    if post_hoc is not None:
        if post_hoc.lower() == 'tukey':
            # Prepare data for Tukey HSD
            all_data = np.concatenate(groups)
            all_groups = np.repeat(group_labels, [len(g) for g in groups])
            
            tukey_result = pairwise_tukeyhsd(all_data, all_groups)
            
            result['post_hoc'] = {
                'method': 'Tukey HSD',
                'pairwise_comparisons': tukey_result.summary().as_html() if hasattr(tukey_result.summary(), 'as_html') else str(tukey_result.summary())
            }
        elif post_hoc.lower() == 'bonferroni':
            # Manual Bonferroni post-hoc
            pairwise = []
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                    p_adj = min(p_val * n_comparisons, 1.0)
                    pairwise.append({
                        'group1': group_labels[i],
                        'group2': group_labels[j],
                        't': t_stat,
                        'p_uncorrected': p_val,
                        'p_bonferroni': p_adj
                    })
            
            result['post_hoc'] = {
                'method': 'Bonferroni',
                'pairwise_comparisons': pairwise
            }
    
    return result


def chi_square_test_d068(
    contingency_table: Union[np.ndarray, pd.DataFrame],
    correction: str = 'bonferroni',
    n_comparisons: Optional[int] = None
) -> Dict[str, Any]:
    """
    Chi-square test with D068 dual p-value reporting.
    
    Parameters
    ----------
    contingency_table : array-like
        Contingency table
    correction : str
        Multiple comparison correction method
    n_comparisons : int, optional
        Number of comparisons for correction
    
    Returns
    -------
    dict
        chi2: Chi-square statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        cramers_v: Cramér's V effect size
        df: Degrees of freedom
    """
    if isinstance(contingency_table, pd.DataFrame):
        table = contingency_table.values
    else:
        table = contingency_table
    
    # Perform chi-square test
    chi2, p_uncorr, dof, expected = stats.chi2_contingency(table)
    
    # Calculate Cramér's V
    n = table.sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
    
    # Apply correction
    if n_comparisons is None:
        n_comparisons = 1  # Single test
    
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    return {
        'chi2': chi2,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'cramers_v': cramers_v,
        'df': dof,
        'expected_frequencies': expected,
        'observed_frequencies': table,
        'phi_coefficient': np.sqrt(chi2 / n) if min_dim == 1 else None
    }


def compute_cramers_v(chi2: float, n: int, k: int) -> float:
    """
    Compute Cramér's V effect size for contingency tables.
    
    Cramér's V = sqrt(chi² / (n * (k-1)))
    where k = min(rows-1, cols-1)
    
    Parameters
    ----------
    chi2 : float
        Chi-square statistic
    n : int
        Total sample size
    k : int
        Minimum dimension (min of rows or columns)
    
    Returns
    -------
    float
        Cramér's V coefficient
    """
    if n == 0 or k <= 1:
        return 0.0
    
    return np.sqrt(chi2 / (n * (k - 1)))


def t_test_d068(
    group1: np.ndarray,
    group2: np.ndarray,
    paired: bool = False,
    correction: str = 'none',
    n_comparisons: int = 1
) -> Dict[str, Any]:
    """
    T-test with D068 dual p-value reporting.
    
    Parameters
    ----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    paired : bool
        Whether to perform paired t-test
    correction : str
        Multiple comparison correction method
    n_comparisons : int
        Number of comparisons for correction
    
    Returns
    -------
    dict
        t: T-statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        cohens_d: Cohen's d effect size
        df: Degrees of freedom
        ci_lower: Lower confidence interval
        ci_upper: Upper confidence interval
    """
    # Ensure arrays
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    # Perform t-test
    if paired:
        t_stat, p_uncorr = stats.ttest_rel(group1, group2)
        df = len(group1) - 1
        
        # Cohen's d for paired samples
        diff = group2 - group1
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    else:
        t_stat, p_uncorr = stats.ttest_ind(group1, group2)
        df = len(group1) + len(group2) - 2
        
        # Cohen's d for independent samples
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                              (len(group2) - 1) * np.var(group2, ddof=1)) / df)
        cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval for mean difference
    if paired:
        diff = group2 - group1
        mean_diff = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    else:
        mean_diff = np.mean(group2) - np.mean(group1)
        se = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
    
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    
    # Apply correction
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    return {
        't': t_stat,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'cohens_d': cohens_d,
        'df': df,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_diff': mean_diff,
        'hedges_g': cohens_d * (1 - 3/(4*(len(group1) + len(group2)) - 9))  # Bias correction
    }


def kruskal_wallis_d068(
    groups: List[np.ndarray],
    correction: str = 'bonferroni',
    n_comparisons: Optional[int] = None
) -> Dict[str, Any]:
    """
    Kruskal-Wallis H test with D068 dual p-value reporting.
    
    Parameters
    ----------
    groups : list of arrays
        List of group data arrays
    correction : str
        Multiple comparison correction method
    n_comparisons : int, optional
        Number of comparisons
    
    Returns
    -------
    dict
        H: H-statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        eta_squared: Eta-squared effect size
    """
    # Perform Kruskal-Wallis test
    h_stat, p_uncorr = stats.kruskal(*groups)
    
    # Calculate eta-squared
    # η² = (H - k + 1) / (n - k)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eta_squared = (h_stat - k + 1) / (n - k) if n > k else 0
    eta_squared = max(0, eta_squared)  # Ensure non-negative
    
    # Apply correction
    if n_comparisons is None:
        n_comparisons = k * (k - 1) // 2
    
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    return {
        'H': h_stat,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'eta_squared': eta_squared,
        'df': k - 1,
        'n': n,
        'median_ranks': {f'Group_{i+1}': np.median(stats.rankdata(np.concatenate(groups))[sum(len(g) for g in groups[:i]):sum(len(g) for g in groups[:i+1])]) 
                        for i in range(k)}
    }


def mann_whitney_d068(
    group1: np.ndarray,
    group2: np.ndarray,
    correction: str = 'none',
    n_comparisons: int = 1
) -> Dict[str, Any]:
    """
    Mann-Whitney U test with D068 dual p-value reporting.
    
    Parameters
    ----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    correction : str
        Multiple comparison correction method
    n_comparisons : int
        Number of comparisons
    
    Returns
    -------
    dict
        U: U-statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        rank_biserial: Rank-biserial correlation
    """
    # Ensure arrays
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    # Perform Mann-Whitney test
    u_stat, p_uncorr = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Calculate rank-biserial correlation
    # r = 1 - (2U / (n1 * n2))
    n1, n2 = len(group1), len(group2)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    
    # Apply correction
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    return {
        'U': u_stat,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'rank_biserial': rank_biserial,
        'n1': n1,
        'n2': n2,
        'median1': np.median(group1),
        'median2': np.median(group2)
    }


def friedman_test_d068(
    measurements: List[np.ndarray],
    correction: str = 'bonferroni',
    n_comparisons: Optional[int] = None
) -> Dict[str, Any]:
    """
    Friedman test for repeated measures with D068 dual p-value reporting.
    
    Parameters
    ----------
    measurements : list of arrays
        List of measurement arrays (each same length)
    correction : str
        Multiple comparison correction method
    n_comparisons : int, optional
        Number of comparisons
    
    Returns
    -------
    dict
        chi2: Chi-square statistic
        p_uncorrected: Uncorrected p-value
        p_corrected: Corrected p-value
        kendall_w: Kendall's W coefficient of concordance
    """
    # Ensure all measurements have same length
    lengths = [len(m) for m in measurements]
    if len(set(lengths)) > 1:
        raise ValueError("All measurements must have the same length")
    
    # Perform Friedman test
    chi2_stat, p_uncorr = stats.friedmanchisquare(*measurements)
    
    # Calculate Kendall's W (coefficient of concordance)
    # W = chi² / (k(n-1))
    k = len(measurements)  # Number of conditions
    n = len(measurements[0])  # Number of subjects
    kendall_w = chi2_stat / (n * (k - 1)) if n * (k - 1) > 0 else 0
    
    # Apply correction
    if n_comparisons is None:
        n_comparisons = k * (k - 1) // 2
    
    p_corr = apply_correction(p_uncorr, correction, n_comparisons)
    
    return {
        'chi2': chi2_stat,
        'p_uncorrected': p_uncorr,
        'p_corrected': p_corr,
        'kendall_w': kendall_w,
        'df': k - 1,
        'n_subjects': n,
        'n_conditions': k,
        'mean_ranks': {f'Condition_{i+1}': np.mean(stats.rankdata(np.column_stack(measurements), axis=1)[:, i])
                      for i in range(k)}
    }


def compute_effect_sizes(
    group1: np.ndarray,
    group2: np.ndarray,
    test_type: str = 't-test'
) -> Dict[str, float]:
    """
    Compute various effect sizes for group comparisons.
    
    Parameters
    ----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    test_type : str
        Type of test performed
    
    Returns
    -------
    dict
        cohens_d: Cohen's d
        hedges_g: Hedges' g (bias-corrected Cohen's d)
        glass_delta: Glass's delta (using group1 SD)
        r: Correlation coefficient equivalent
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
    
    # Hedges' g (bias correction)
    correction_factor = 1 - 3/(4*(n1 + n2) - 9)
    hedges_g = cohens_d * correction_factor
    
    # Glass's delta (using control group SD)
    glass_delta = (mean2 - mean1) / np.sqrt(var1) if var1 > 0 else 0
    
    # Convert to correlation coefficient
    r = cohens_d / np.sqrt(cohens_d**2 + 4) if cohens_d != 0 else 0
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'glass_delta': glass_delta,
        'r': r,
        'r_squared': r**2
    }


def apply_correction(
    p_value: float,
    method: str,
    n_comparisons: int
) -> float:
    """
    Apply multiple comparison correction to p-value.
    
    Parameters
    ----------
    p_value : float
        Uncorrected p-value
    method : str
        Correction method ('bonferroni', 'fdr', 'none')
    n_comparisons : int
        Number of comparisons
    
    Returns
    -------
    float
        Corrected p-value
    """
    if method.lower() == 'none':
        return p_value
    
    if method.lower() == 'bonferroni':
        return min(p_value * n_comparisons, 1.0)
    
    if method.lower() == 'fdr' or method.lower() == 'bh':
        # For single p-value, FDR is same as uncorrected
        # In practice, would need all p-values for proper FDR
        return p_value
    
    return p_value


def calculate_omega_squared(F: float, df_between: int, df_within: int, n: int) -> float:
    """
    Calculate omega-squared effect size for ANOVA.
    
    ω² = (F - 1) / (F + (df_within + 1) / df_between)
    
    Parameters
    ----------
    F : float
        F-statistic
    df_between : int
        Between-group degrees of freedom
    df_within : int
        Within-group degrees of freedom
    n : int
        Total sample size
    
    Returns
    -------
    float
        Omega-squared effect size
    """
    if F <= 1:
        return 0.0
    
    omega_sq = (F - 1) / (F + (df_within + 1) / df_between)
    return max(0, omega_sq)  # Ensure non-negative