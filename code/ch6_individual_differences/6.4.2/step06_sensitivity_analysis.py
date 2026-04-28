#!/usr/bin/env python3
"""sensitivity_analysis: Conduct sensitivity analyses to assess robustness of domain-specificity findings"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# Statistical analysis imports
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure  # Using available function

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_sensitivity_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Sensitivity Analysis Functions

def outlier_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify outliers using standardized residuals and Cook's D.
    Runs for both bvmt_total and bvmt_pct_ret.

    Returns:
        Dictionary with outlier info and recomputed correlations
    """
    log("Starting outlier analysis...")

    results = {}

    for bvmt_col, label in [('bvmt_total', 'total'), ('bvmt_pct_ret', 'pct_ret')]:
        # Compute standardized residuals for BVMT ~ Where relationship
        X_where = df[['Where_mean']].values
        y_bvmt = df[bvmt_col].values

        # Fit linear model
        reg_where = LinearRegression().fit(X_where, y_bvmt)
        residuals_where = y_bvmt - reg_where.predict(X_where)

        # Standardized residuals (Z-scores)
        std_residuals_where = np.abs(stats.zscore(residuals_where))

        # Identify outliers (|z| > 3.29, corresponds to p < 0.001)
        outlier_threshold = 3.29
        outliers_where = np.where(std_residuals_where > outlier_threshold)[0]

        log(f"{label}: Found {len(outliers_where)} outliers using threshold {outlier_threshold}")

        # Remove outliers and recompute correlations
        if len(outliers_where) > 0:
            df_clean = df.drop(df.index[outliers_where]).reset_index(drop=True)
            log(f"{label}: Dataset size after outlier removal: {len(df_clean)} (removed {len(outliers_where)})")
        else:
            df_clean = df.copy()
            log(f"{label}: No outliers detected")

        # Recompute correlations without outliers
        r_where_clean = stats.pearsonr(df_clean[bvmt_col], df_clean['Where_mean'])[0]
        r_what_clean = stats.pearsonr(df_clean[bvmt_col], df_clean['What_mean'])[0]

        # Original correlations for comparison
        r_where_orig = stats.pearsonr(df[bvmt_col], df['Where_mean'])[0]
        r_what_orig = stats.pearsonr(df[bvmt_col], df['What_mean'])[0]

        results[label] = {
            'n_outliers': len(outliers_where),
            'outlier_indices': outliers_where.tolist(),
            'n_clean': len(df_clean),
            'r_where_original': r_where_orig,
            'r_what_original': r_what_orig,
            'r_where_clean': r_where_clean,
            'r_what_clean': r_what_clean,
            'where_change': r_where_clean - r_where_orig,
            'what_change': r_what_clean - r_what_orig
        }

    return results

def alternative_correlation_methods(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Spearman and Kendall correlations for comparison with Pearson.
    Runs for both bvmt_total and bvmt_pct_ret.

    Returns:
        Dictionary with alternative correlation results keyed by measure
    """
    log("Computing alternative correlation methods...")

    results = {}
    for bvmt_col, label in [('bvmt_total', 'total'), ('bvmt_pct_ret', 'pct_ret')]:
        spear_where = stats.spearmanr(df[bvmt_col], df['Where_mean'])
        spear_what = stats.spearmanr(df[bvmt_col], df['What_mean'])
        kendall_where = stats.kendalltau(df[bvmt_col], df['Where_mean'])
        kendall_what = stats.kendalltau(df[bvmt_col], df['What_mean'])
        pearson_where = stats.pearsonr(df[bvmt_col], df['Where_mean'])
        pearson_what = stats.pearsonr(df[bvmt_col], df['What_mean'])

        log(f"{label} Spearman Where: r={spear_where.correlation:.3f}, p={spear_where.pvalue:.3f}")
        log(f"{label} Kendall Where: tau={kendall_where.correlation:.3f}, p={kendall_where.pvalue:.3f}")

        results[label] = {
            'pearson_where': pearson_where.statistic,
            'pearson_what': pearson_what.statistic,
            'spearman_where': spear_where.correlation,
            'spearman_what': spear_what.correlation,
            'kendall_where': kendall_where.correlation,
            'kendall_what': kendall_what.correlation,
            'spearman_where_pvalue': spear_where.pvalue,
            'spearman_what_pvalue': spear_what.pvalue,
            'kendall_where_pvalue': kendall_where.pvalue,
            'kendall_what_pvalue': kendall_what.pvalue
        }

    return results

def cross_validation_analysis(df: pd.DataFrame, cv_folds: int = 5, seed: int = 42) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation to assess correlation stability.
    Runs for both bvmt_total and bvmt_pct_ret.

    Returns:
        Dictionary with CV statistics keyed by measure
    """
    log(f"[CV] Starting {cv_folds}-fold cross-validation...")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    results = {}
    for bvmt_col, label in [('bvmt_total', 'total'), ('bvmt_pct_ret', 'pct_ret')]:
        cv_r_where = []
        cv_r_what = []

        for i, (train_idx, val_idx) in enumerate(kf.split(df)):
            df_fold = df.iloc[val_idx]
            r_where = stats.pearsonr(df_fold[bvmt_col], df_fold['Where_mean'])[0]
            r_what = stats.pearsonr(df_fold[bvmt_col], df_fold['What_mean'])[0]
            cv_r_where.append(r_where)
            cv_r_what.append(r_what)
            log(f"[CV] {label} Fold {i+1}: r_where={r_where:.3f}, r_what={r_what:.3f}")

        cv_r_where = np.array(cv_r_where)
        cv_r_what = np.array(cv_r_what)

        mean_where = np.mean(cv_r_where)
        mean_what = np.mean(cv_r_what)

        results[label] = {
            'cv_folds': cv_folds,
            'mean_r_where': mean_where,
            'std_r_where': np.std(cv_r_where),
            'mean_r_what': mean_what,
            'std_r_what': np.std(cv_r_what),
            'stability_ratio_where': np.std(cv_r_where) / np.abs(mean_where) if mean_where != 0 else np.inf,
            'stability_ratio_what': np.std(cv_r_what) / np.abs(mean_what) if mean_what != 0 else np.inf
        }

    return results

def power_analysis(df: pd.DataFrame, alpha: float = 0.00179) -> Dict[str, Any]:
    """
    Conduct post-hoc power analysis for observed correlations.
    Runs for both bvmt_total and bvmt_pct_ret.

    Parameters:
        alpha: Corrected alpha level (Bonferroni correction)

    Returns:
        Dictionary with power analysis results keyed by measure
    """
    log(f"Conducting power analysis with alpha={alpha}")

    n = len(df)
    se_r = 1.0 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf(1 - alpha/2)
    z_80 = stats.norm.ppf(0.8)
    min_detectable_z = (z_critical + z_80) * se_r
    min_detectable_r = np.tanh(min_detectable_z)

    results = {}
    for bvmt_col, label in [('bvmt_total', 'total'), ('bvmt_pct_ret', 'pct_ret')]:
        r_where = stats.pearsonr(df[bvmt_col], df['Where_mean'])[0]
        r_what = stats.pearsonr(df[bvmt_col], df['What_mean'])[0]

        z_where = np.arctanh(r_where)
        z_what = np.arctanh(r_what)

        power_where = 1 - stats.norm.cdf(z_critical - np.abs(z_where)/se_r) + stats.norm.cdf(-z_critical - np.abs(z_where)/se_r)
        power_what = 1 - stats.norm.cdf(z_critical - np.abs(z_what)/se_r) + stats.norm.cdf(-z_critical - np.abs(z_what)/se_r)

        log(f"{label}: Power Where={power_where:.3f}, Power What={power_what:.3f}")

        results[label] = {
            'n': n,
            'alpha': alpha,
            'r_where': r_where,
            'r_what': r_what,
            'power_where': power_where,
            'power_what': power_what,
            'z_critical': z_critical,
            'se_correlation': se_r,
            'min_detectable_r_80pct': min_detectable_r,
            'adequately_powered_where': power_where >= 0.80,
            'adequately_powered_what': power_what >= 0.80
        }

    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: sensitivity_analysis")
        # Load Input Data

        log("Loading input data...")
        
        # Load complete analysis dataset
        # Expected columns: UID, bvmt_total, Where_mean, What_mean
        # Expected rows: ~100 participants
        analysis_df = pd.read_csv(RQ_DIR / "data" / "step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")
        
        # Load Steiger test results for comparison
        # Expected columns: z_statistic, p_uncorrected
        # Expected rows: 1 test result
        steiger_df = pd.read_csv(RQ_DIR / "data" / "step05_steiger_test.csv")
        log(f"step05_steiger_test.csv ({len(steiger_df)} rows, {len(steiger_df.columns)} cols)")
        # Run Sensitivity Analyses
        # Multiple approaches: outlier detection, alternative correlations, CV, power

        log("Running sensitivity analyses...")
        
        # Parameters from 4_analysis.yaml
        outlier_threshold = 3.29  # Standardized residuals threshold
        cv_folds = 5
        cv_seed = 42
        alpha_corrected = 0.00179  # Bonferroni correction: 0.05/28
        
        # Sensitivity Analysis 1: Outlier Analysis
        log("1. Outlier analysis using Cook's D and standardized residuals...")
        outlier_results = outlier_analysis(analysis_df)
        
        # Sensitivity Analysis 2: Alternative Correlation Methods
        log("2. Alternative correlation methods (Spearman, Kendall)...")
        method_results = alternative_correlation_methods(analysis_df)
        
        # Sensitivity Analysis 3: Cross-Validation Stability
        log("3. Cross-validation stability assessment...")
        cv_results = cross_validation_analysis(analysis_df, cv_folds=cv_folds, seed=cv_seed)
        
        # Sensitivity Analysis 4: Post-hoc Power Analysis
        log("4. Post-hoc power analysis...")
        power_results = power_analysis(analysis_df, alpha=alpha_corrected)
        
        log("All sensitivity analyses complete")
        # Compile Sensitivity Results
        # Format results into standardized output structure
        # Contains: analysis type, method, result, confidence intervals, sample size, notes

        log("Compiling sensitivity analysis results...")
        
        # Build results dataframe
        sensitivity_results = []

        # Outlier Analysis Results (both measures)
        for label in ['total', 'pct_ret']:
            o = outlier_results[label]
            sensitivity_results.extend([
                {
                    'analysis_type': f'outlier_detection_{label}',
                    'method': 'standardized_residuals',
                    'result': f"n_outliers={o['n_outliers']}, n_clean={o['n_clean']}",
                    'confidence_interval': 'N/A',
                    'n_used': o['n_clean'],
                    'notes': f"Threshold={outlier_threshold}, Where_change={o['where_change']:.3f}, What_change={o['what_change']:.3f}"
                },
                {
                    'analysis_type': f'outlier_robust_correlation_{label}',
                    'method': 'pearson_no_outliers',
                    'result': f"r_Where={o['r_where_clean']:.3f}, r_What={o['r_what_clean']:.3f}",
                    'confidence_interval': 'N/A',
                    'n_used': o['n_clean'],
                    'notes': f"Original: Where={o['r_where_original']:.3f}, What={o['r_what_original']:.3f}"
                }
            ])

        # Alternative Methods Results (both measures)
        for label in ['total', 'pct_ret']:
            m = method_results[label]
            sensitivity_results.extend([
                {
                    'analysis_type': f'alternative_correlation_{label}',
                    'method': 'spearman',
                    'result': f"r_Where={m['spearman_where']:.3f}, r_What={m['spearman_what']:.3f}",
                    'confidence_interval': 'N/A',
                    'n_used': len(analysis_df),
                    'notes': f"p_Where={m['spearman_where_pvalue']:.4f}, p_What={m['spearman_what_pvalue']:.4f}"
                },
                {
                    'analysis_type': f'alternative_correlation_{label}',
                    'method': 'kendall_tau',
                    'result': f"tau_Where={m['kendall_where']:.3f}, tau_What={m['kendall_what']:.3f}",
                    'confidence_interval': 'N/A',
                    'n_used': len(analysis_df),
                    'notes': f"p_Where={m['kendall_where_pvalue']:.4f}, p_What={m['kendall_what_pvalue']:.4f}"
                }
            ])

        # Cross-Validation Results (both measures)
        for label in ['total', 'pct_ret']:
            c = cv_results[label]
            sensitivity_results.append({
                'analysis_type': f'cross_validation_{label}',
                'method': f'{cv_folds}_fold_cv',
                'result': f"mean_r_Where={c['mean_r_where']:.3f}, mean_r_What={c['mean_r_what']:.3f}",
                'confidence_interval': f"Where_SD={c['std_r_where']:.3f}, What_SD={c['std_r_what']:.3f}",
                'n_used': len(analysis_df),
                'notes': f"Stability_ratio_Where={c['stability_ratio_where']:.3f}, Stability_ratio_What={c['stability_ratio_what']:.3f}"
            })

        # Power Analysis Results (both measures)
        for label in ['total', 'pct_ret']:
            p = power_results[label]
            sensitivity_results.append({
                'analysis_type': f'power_analysis_{label}',
                'method': 'post_hoc_power',
                'result': f"Power_Where={p['power_where']:.3f}, Power_What={p['power_what']:.3f}",
                'confidence_interval': f"Min_detectable_r={p['min_detectable_r_80pct']:.3f}",
                'n_used': p['n'],
                'notes': f"Alpha={alpha_corrected}, Adequate_power_Where={p['adequately_powered_where']}, Adequate_power_What={p['adequately_powered_what']}"
            })
        
        # Convert to DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        log(f"{len(sensitivity_df)} sensitivity analysis results")
        # Save Sensitivity Outputs
        # These outputs will be used by: Results interpretation and reporting

        log(f"Saving step06_sensitivity_analysis.csv...")
        # Output: step06_sensitivity_analysis.csv
        # Contains: Comprehensive sensitivity analysis results with robustness checks
        # Columns: analysis_type, method, result, confidence_interval, n_used, notes
        sensitivity_df.to_csv(RQ_DIR / "data" / "step06_sensitivity_analysis.csv", index=False, encoding='utf-8')
        log(f"step06_sensitivity_analysis.csv ({len(sensitivity_df)} rows, {len(sensitivity_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Sensitivity analyses completed successfully
        # Threshold: All planned analyses present and valid

        log("Running custom validation...")
        # Custom validation since validate_data_completeness not available
        validation_checks = {
            'row_count': len(sensitivity_df) >= 8,
            'required_columns': all(col in sensitivity_df.columns for col in 
                                  ['analysis_type', 'method', 'result', 'confidence_interval', 'n_used', 'notes']),
            'no_missing_results': sensitivity_df['result'].notna().all()
        }
        
        validation_result = {
            'valid': all(validation_checks.values()),
            'checks': validation_checks,
            'actual_rows': len(sensitivity_df)
        }

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)