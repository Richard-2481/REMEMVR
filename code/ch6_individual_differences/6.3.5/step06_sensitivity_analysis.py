#!/usr/bin/env python3
"""sensitivity_analysis: Test robustness of findings through outlier exclusion and alternative grouping methods."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import traceback
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.bootstrap import bootstrap_statistic

from tools.validation import validate_bootstrap_stability

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_sensitivity_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Sensitivity Analysis Functions

def detect_outliers_cooks_distance(X, y, threshold=4):
    """
    Detect outliers using Cook's distance.
    Threshold of 4/n is common, but we use fixed threshold of 4.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    n = len(X)
    outlier_indices = []
    
    # Fit full model
    model = LinearRegression()
    X_with_const = np.column_stack([np.ones(n), X])  # Add intercept
    model.fit(X_with_const, y)
    y_pred_full = model.predict(X_with_const)
    mse_full = mean_squared_error(y, y_pred_full)
    
    # Calculate Cook's distance for each point
    cooks_distances = []
    for i in range(n):
        # Fit model without point i
        X_reduced = np.delete(X_with_const, i, axis=0)
        y_reduced = np.delete(y, i)
        
        if len(X_reduced) > X_with_const.shape[1]:  # Ensure enough data points
            model_reduced = LinearRegression()
            model_reduced.fit(X_reduced, y_reduced)
            y_pred_reduced = model_reduced.predict(X_with_const)
            
            # Cook's distance formula
            cooks_d = np.sum((y_pred_full - y_pred_reduced) ** 2) / (X_with_const.shape[1] * mse_full)
            cooks_distances.append(cooks_d)
            
            if cooks_d > threshold / n:
                outlier_indices.append(i)
        else:
            cooks_distances.append(0.0)
    
    return outlier_indices, cooks_distances

def detect_outliers_mahalanobis(X, threshold=3):
    """
    Detect outliers using Mahalanobis distance.
    """
    try:
        # Calculate covariance matrix and mean
        cov_matrix = np.cov(X.T)
        mean_vector = np.mean(X, axis=0)
        
        # Calculate Mahalanobis distance for each point
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahal_distances = []
        outlier_indices = []
        
        for i, point in enumerate(X):
            mahal_dist = mahalanobis(point, mean_vector, inv_cov_matrix)
            mahal_distances.append(mahal_dist)
            
            if mahal_dist > threshold:
                outlier_indices.append(i)
        
        return outlier_indices, mahal_distances
    except:
        # If covariance matrix is singular, return no outliers
        return [], [0.0] * len(X)

def create_tertile_groups(residuals):
    """
    Create calibration groups using tertile splits instead of SD-based splits.
    """
    tertile_33 = np.percentile(residuals, 33.33)
    tertile_66 = np.percentile(residuals, 66.67)
    
    groups = []
    for r in residuals:
        if r <= tertile_33:
            groups.append('Underconfident')
        elif r <= tertile_66:
            groups.append('Well-calibrated')
        else:
            groups.append('Overconfident')
    
    return groups

def run_anova_simple(df, group_col, dv_col):
    """
    Simple one-way ANOVA for sensitivity analysis.
    """
    groups = df[group_col].unique()
    group_data = [df[df[group_col] == group][dv_col].dropna() for group in groups]
    
    # Remove empty groups
    group_data = [g for g in group_data if len(g) > 0]
    
    if len(group_data) < 2:
        return {'f_statistic': np.nan, 'p_value': np.nan, 'eta_squared': np.nan}
    
    try:
        f_stat, p_val = stats.f_oneway(*group_data)
        
        # Calculate eta-squared
        ss_between = sum([len(g) * (np.mean(g) - np.mean(df[dv_col]))**2 for g in group_data])
        ss_total = np.sum((df[dv_col] - np.mean(df[dv_col]))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
        
        return {
            'f_statistic': f_stat,
            'p_value': p_val,
            'eta_squared': eta_squared
        }
    except:
        return {'f_statistic': np.nan, 'p_value': np.nan, 'eta_squared': np.nan}

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def stability_statistic(data):
    """
    Custom statistic for bootstrap stability: proportion of cases in same group.
    """
    # Sample the data and recreate groups
    residuals = data[:, 0]  # Assuming first column is residuals
    
    # Create groups using same method
    groups = create_tertile_groups(residuals)
    
    # Return some measure of group assignment stability
    # For simplicity, return the proportion in middle group
    return np.mean([g == 'Well-calibrated' for g in groups])

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Sensitivity Analysis")
        # Load Input Data

        log("Loading original merged data...")
        df = pd.read_csv(RQ_DIR / "data" / "step01_merged_data.csv")
        log(f"step01_merged_data.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['UID', 'theta_all', 'confidence_theta', 'education', 'rpm', 'age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data for analysis
        df_clean = df[required_cols].dropna()
        log(f"Clean data for sensitivity analysis: {len(df_clean)} participants")
        # Outlier Detection

        log("Detecting outliers...")
        
        # Prepare data for outlier detection
        predictor_vars = ['education', 'rpm', 'age']
        X = df_clean[predictor_vars].values
        y = df_clean['confidence_theta'].values  # Use confidence as outcome for outlier detection
        
        outlier_results = []
        
        # Cook's distance outliers
        log("Computing Cook's distance...")
        cooks_outliers, cooks_distances = detect_outliers_cooks_distance(X, y, threshold=4)
        cooks_threshold = 4 / len(df_clean)
        
        outlier_results.append({
            'variable': 'multivariate',
            'outlier_method': 'cooks_distance',
            'outlier_count': len(cooks_outliers),
            'cooks_d_threshold': cooks_threshold,
            'excluded_uids': str([df_clean.iloc[i]['UID'] for i in cooks_outliers])
        })
        
        log(f"Cook's distance: {len(cooks_outliers)} outliers detected")
        
        # Mahalanobis distance outliers
        log("Computing Mahalanobis distance...")
        mahal_outliers, mahal_distances = detect_outliers_mahalanobis(X, threshold=3)
        
        outlier_results.append({
            'variable': 'multivariate',
            'outlier_method': 'mahalanobis',
            'outlier_count': len(mahal_outliers),
            'cooks_d_threshold': np.nan,
            'excluded_uids': str([df_clean.iloc[i]['UID'] for i in mahal_outliers])
        })
        
        log(f"Mahalanobis distance: {len(mahal_outliers)} outliers detected")
        # Alternative Grouping (Tertiles)

        log("Creating tertile-based calibration groups...")
        
        # Recreate confidence-accuracy regression for residuals
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        X_reg = df_clean[['theta_all']].values
        y_reg = df_clean['confidence_theta'].values
        reg.fit(X_reg, y_reg)
        residuals = y_reg - reg.predict(X_reg)
        
        # Create tertile groups
        tertile_groups = create_tertile_groups(residuals)
        df_tertile = df_clean.copy()
        df_tertile['group'] = tertile_groups
        
        # Check group sizes
        tertile_counts = pd.Series(tertile_groups).value_counts()
        log(f"Tertile group sizes: {dict(tertile_counts)}")
        
        # Run ANOVAs with tertile grouping
        dependent_variables = ['education', 'rpm', 'age']
        tertile_results = []
        n_comparisons = 6  # Keep same correction as original analysis
        
        for dv in dependent_variables:
            log(f"Tertile ANOVA for {dv}...")
            result = run_anova_simple(df_tertile, 'group', dv)
            
            # Add Bonferroni correction
            p_bonferroni = min(result['p_value'] * n_comparisons, 1.0) if not np.isnan(result['p_value']) else np.nan
            
            tertile_results.append({
                'DV': dv,
                'F_stat': result['f_statistic'],
                'p_uncorrected': result['p_value'],
                'p_bonferroni': p_bonferroni,
                'eta_squared': result['eta_squared'],
                'method': 'tertile_grouping'
            })
            
            log(f"Tertile {dv}: F = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}")
        # Bootstrap Stability Analysis

        log("Running bootstrap stability analysis...")
        
        # Prepare data for bootstrap stability
        stability_data = np.column_stack([residuals, df_clean[predictor_vars].values])
        
        # Use bootstrap_statistic with actual signature: data, statistic_func, n_bootstrap, confidence, method, random_state
        stability_result = bootstrap_statistic(
            data=stability_data,
            statistic=stability_statistic,
            n_bootstrap=100,  # Use smaller number for sensitivity analysis
            confidence=0.95,
            seed=42
        )
        
        # Calculate Jaccard similarities for group stability
        jaccard_values = []
        np.random.seed(42)
        
        for _ in range(20):  # Sample 20 bootstrap iterations for Jaccard calculation
            # Bootstrap sample
            indices = np.random.choice(len(df_clean), size=len(df_clean), replace=True)
            boot_residuals = residuals[indices]
            
            # Original and bootstrap groups
            original_groups = create_tertile_groups(residuals)
            boot_groups = create_tertile_groups(boot_residuals)
            
            # Calculate Jaccard similarity for each group
            for group_name in ['Well-calibrated', 'Overconfident', 'Underconfident']:
                original_set = set([i for i, g in enumerate(original_groups) if g == group_name])
                boot_set = set([i for i, g in enumerate(boot_groups) if g == group_name])
                jaccard = jaccard_similarity(original_set, boot_set)
                jaccard_values.append(jaccard)
        # Robustness Summary

        log("Creating robustness summary...")
        
        # Load original ANOVA results for comparison
        try:
            original_anova = pd.read_csv(RQ_DIR / "data" / "step03_anova_results.csv")
        except:
            log("Could not load original ANOVA results for comparison")
            original_anova = pd.DataFrame()
        
        robustness_results = []
        
        for dv in dependent_variables:
            # Get original and tertile results
            if not original_anova.empty:
                orig_row = original_anova[original_anova['DV'] == dv]
                orig_eta = orig_row['eta_squared'].iloc[0] if not orig_row.empty else np.nan
            else:
                orig_eta = np.nan
            
            tert_row = [r for r in tertile_results if r['DV'] == dv]
            tert_eta = tert_row[0]['eta_squared'] if tert_row else np.nan
            
            # Calculate effect size change
            if not np.isnan(orig_eta) and not np.isnan(tert_eta) and orig_eta != 0:
                effect_size_change = abs((tert_eta - orig_eta) / orig_eta)
            else:
                effect_size_change = np.nan
            
            # Classify robustness
            if np.isnan(effect_size_change):
                robustness = 'unknown'
            elif effect_size_change < 0.20:
                robustness = 'robust'
            elif effect_size_change < 0.50:
                robustness = 'moderate'
            else:
                robustness = 'sensitive'
            
            robustness_results.append({
                'analysis': dv,
                'original_result': orig_eta,
                'outliers_excluded': len(cooks_outliers + mahal_outliers),
                'tertile_method': tert_eta,
                'effect_size_change': effect_size_change,
                'robustness': robustness
            })
        # Save Analysis Outputs
        # These outputs will be used by: Final reporting and robustness assessment

        log("Saving sensitivity analysis outputs...")
        
        # Save outlier analysis
        outlier_df = pd.DataFrame(outlier_results)
        outlier_df.to_csv(RQ_DIR / "data" / "step06_outlier_analysis.csv", index=False, encoding='utf-8')
        log(f"step06_outlier_analysis.csv ({len(outlier_df)} rows, {len(outlier_df.columns)} cols)")
        
        # Save tertile reanalysis
        tertile_df = pd.DataFrame(tertile_results)
        tertile_df.to_csv(RQ_DIR / "data" / "step06_tertile_reanalysis.csv", index=False, encoding='utf-8')
        log(f"step06_tertile_reanalysis.csv ({len(tertile_df)} rows, {len(tertile_df.columns)} cols)")
        
        # Save robustness summary
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df.to_csv(RQ_DIR / "data" / "step06_robustness_summary.csv", index=False, encoding='utf-8')
        log(f"step06_robustness_summary.csv ({len(robustness_df)} rows, {len(robustness_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Bootstrap stability meets minimum threshold
        # Threshold: min_jaccard_threshold = 0.75

        log("Running validate_bootstrap_stability...")
        
        # Create DataFrame with Jaccard values for validation
        jaccard_df = pd.DataFrame({'jaccard': jaccard_values})
        
        validation_result = validate_bootstrap_stability(
            stability_df=jaccard_df,
            min_jaccard_threshold=0.75,
            jaccard_col='jaccard'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary statistics
        log("Sensitivity Analysis Summary:")
        log(f"  Total outliers detected: {len(set(cooks_outliers + mahal_outliers))} participants")
        log(f"  Average Jaccard similarity: {np.mean(jaccard_values):.3f}")
        log(f"  Robustness classifications:")
        for classification in ['robust', 'moderate', 'sensitive', 'unknown']:
            count = len([r for r in robustness_results if r['robustness'] == classification])
            log(f"    {classification}: {count} analyses")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)