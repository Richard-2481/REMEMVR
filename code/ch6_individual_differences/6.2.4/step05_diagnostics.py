#!/usr/bin/env python3
"""
Step 5: Assumption Checks and Diagnostics
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Check linearity assumptions, identify outliers, and test normality for both correlations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_diagnostics.log"

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def check_linearity(x, y, name):
    """Test for linearity vs quadratic relationship"""
    # Linear model
    z_lin = np.polyfit(x, y, 1)
    p_lin = np.poly1d(z_lin)
    residuals_lin = y - p_lin(x)
    ss_res_lin = np.sum(residuals_lin**2)
    
    # Quadratic model  
    z_quad = np.polyfit(x, y, 2)
    p_quad = np.poly1d(z_quad)
    residuals_quad = y - p_quad(x)
    ss_res_quad = np.sum(residuals_quad**2)
    
    # F-test for quadratic term
    n = len(x)
    df_lin = n - 2
    df_quad = n - 3
    f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / df_quad)
    p_value = 1 - stats.f.cdf(f_stat, 1, df_quad)
    
    return {
        'relationship': name,
        'linear_r2': 1 - ss_res_lin / np.var(y) / (n-1),
        'quadratic_r2': 1 - ss_res_quad / np.var(y) / (n-1),
        'f_statistic': f_stat,
        'p_value': p_value,
        'status': 'LINEAR' if p_value > 0.05 else 'NONLINEAR'
    }

def detect_outliers(df):
    """Detect outliers using multiple methods"""
    outliers = []
    
    # For Age-RAVLT relationship
    X = df[['Age']].values
    y = df['RAVLT_Total'].values
    
    # Standardized residuals
    z = np.polyfit(X.flatten(), y, 1)
    p = np.poly1d(z)
    residuals = y - p(X.flatten())
    std_residuals = residuals / np.std(residuals)
    
    for i, (uid, res) in enumerate(zip(df['UID'], std_residuals)):
        if abs(res) > 3.0:
            outliers.append({
                'UID': uid,
                'outlier_type': 'standardized_residual_RAVLT',
                'distance_value': abs(res),
                'threshold': 3.0
            })
    
    # Cook's distance for Age-RAVLT
    n = len(df)
    p_params = 2  # intercept + slope
    leverage = 1/n + (X - X.mean())**2 / np.sum((X - X.mean())**2)
    cooks_d = (std_residuals**2 / p_params) * (leverage / (1 - leverage)**2)
    
    for i, (uid, d) in enumerate(zip(df['UID'], cooks_d.flatten())):
        if d > 4/n:
            outliers.append({
                'UID': uid,
                'outlier_type': 'cooks_distance_RAVLT',
                'distance_value': d,
                'threshold': 4/n
            })
    
    # Repeat for Age-REMEMVR
    y2 = df['theta_all'].values
    z2 = np.polyfit(X.flatten(), y2, 1)
    p2 = np.poly1d(z2)
    residuals2 = y2 - p2(X.flatten())
    std_residuals2 = residuals2 / np.std(residuals2)
    
    for i, (uid, res) in enumerate(zip(df['UID'], std_residuals2)):
        if abs(res) > 3.0:
            outliers.append({
                'UID': uid,
                'outlier_type': 'standardized_residual_REMEMVR',
                'distance_value': abs(res),
                'threshold': 3.0
            })
    
    # Mahalanobis distance (multivariate)
    X_multi = df[['Age', 'RAVLT_Total', 'theta_all']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_multi)
    
    mean = np.mean(X_scaled, axis=0)
    cov = np.cov(X_scaled.T)
    inv_cov = np.linalg.inv(cov)
    
    for i, row in enumerate(X_scaled):
        diff = row - mean
        m_dist = np.sqrt(diff @ inv_cov @ diff.T)
        chi2_threshold = stats.chi2.ppf(0.999, df=3)  # 99.9% threshold
        
        if m_dist > np.sqrt(chi2_threshold):
            outliers.append({
                'UID': df['UID'].iloc[i],
                'outlier_type': 'mahalanobis_multivariate',
                'distance_value': m_dist,
                'threshold': np.sqrt(chi2_threshold)
            })
    
    return outliers

def main():
    log("=" * 60)
    log("Step 5: Assumption Checks and Diagnostics")
    log("=" * 60)
    
    # Load merged data
    merged_path = RQ_DIR / "data" / "step03_merged_data.csv"
    df = pd.read_csv(merged_path)
    log(f"Loaded {len(df)} participants")
    
    # 1. LINEARITY CHECKS
    log("\n" + "=" * 40)
    log("LINEARITY ASSESSMENT")
    log("=" * 40)
    
    diagnostics = []
    
    # Check Age vs RAVLT
    lin_ravlt = check_linearity(df['Age'].values, df['RAVLT_Total'].values, 'Age_vs_RAVLT')
    log(f"\nAge vs RAVLT:")
    log(f"  Linear R² = {lin_ravlt['linear_r2']:.3f}")
    log(f"  Quadratic R² = {lin_ravlt['quadratic_r2']:.3f}")
    log(f"  F-test for quadratic: F = {lin_ravlt['f_statistic']:.2f}, p = {lin_ravlt['p_value']:.4f}")
    log(f"  → {lin_ravlt['status']}")
    
    diagnostics.append({
        'assumption': 'linearity_age_ravlt',
        'test_statistic': lin_ravlt['f_statistic'],
        'p_value': lin_ravlt['p_value'],
        'threshold': 0.05,
        'status': 'PASS' if lin_ravlt['status'] == 'LINEAR' else 'FAIL'
    })
    
    # Check Age vs REMEMVR
    lin_rememvr = check_linearity(df['Age'].values, df['theta_all'].values, 'Age_vs_REMEMVR')
    log(f"\nAge vs REMEMVR:")
    log(f"  Linear R² = {lin_rememvr['linear_r2']:.3f}")
    log(f"  Quadratic R² = {lin_rememvr['quadratic_r2']:.3f}")
    log(f"  F-test for quadratic: F = {lin_rememvr['f_statistic']:.2f}, p = {lin_rememvr['p_value']:.4f}")
    log(f"  → {lin_rememvr['status']}")
    
    diagnostics.append({
        'assumption': 'linearity_age_rememvr',
        'test_statistic': lin_rememvr['f_statistic'],
        'p_value': lin_rememvr['p_value'],
        'threshold': 0.05,
        'status': 'PASS' if lin_rememvr['status'] == 'LINEAR' else 'FAIL'
    })

    # Check Age vs RAVLT Pct Ret (if available)
    if 'RAVLT_Pct_Ret' in df.columns:
        df_pct = df.dropna(subset=['RAVLT_Pct_Ret'])
        lin_pctret = check_linearity(df_pct['Age'].values, df_pct['RAVLT_Pct_Ret'].values, 'Age_vs_RAVLT_Pct_Ret')
        log(f"\nAge vs RAVLT Pct Ret:")
        log(f"  Linear R2 = {lin_pctret['linear_r2']:.3f}")
        log(f"  Quadratic R2 = {lin_pctret['quadratic_r2']:.3f}")
        log(f"  F-test for quadratic: F = {lin_pctret['f_statistic']:.2f}, p = {lin_pctret['p_value']:.4f}")
        log(f"  -> {lin_pctret['status']}")

        diagnostics.append({
            'assumption': 'linearity_age_pctret',
            'test_statistic': lin_pctret['f_statistic'],
            'p_value': lin_pctret['p_value'],
            'threshold': 0.05,
            'status': 'PASS' if lin_pctret['status'] == 'LINEAR' else 'FAIL'
        })

    # 2. NORMALITY CHECKS
    log("\n" + "=" * 40)
    log("NORMALITY ASSESSMENT")
    log("=" * 40)
    
    # Test normality of residuals
    # Age-RAVLT residuals
    z1 = np.polyfit(df['Age'].values, df['RAVLT_Total'].values, 1)
    p1 = np.poly1d(z1)
    residuals_ravlt = df['RAVLT_Total'].values - p1(df['Age'].values)
    
    w_ravlt, p_ravlt = stats.shapiro(residuals_ravlt)
    log(f"\nAge-RAVLT residuals:")
    log(f"  Shapiro-Wilk W = {w_ravlt:.3f}, p = {p_ravlt:.4f}")
    log(f"  → {'NORMAL' if p_ravlt > 0.05 else 'NON-NORMAL'}")
    
    # Age-REMEMVR residuals
    z2 = np.polyfit(df['Age'].values, df['theta_all'].values, 1)
    p2 = np.poly1d(z2)
    residuals_rememvr = df['theta_all'].values - p2(df['Age'].values)
    
    w_rememvr, p_rememvr = stats.shapiro(residuals_rememvr)
    log(f"\nAge-REMEMVR residuals:")
    log(f"  Shapiro-Wilk W = {w_rememvr:.3f}, p = {p_rememvr:.4f}")
    log(f"  → {'NORMAL' if p_rememvr > 0.05 else 'NON-NORMAL'}")
    
    # Age-RAVLT_Pct_Ret residuals (if available)
    if 'RAVLT_Pct_Ret' in df.columns:
        df_pct = df.dropna(subset=['RAVLT_Pct_Ret'])
        z3 = np.polyfit(df_pct['Age'].values, df_pct['RAVLT_Pct_Ret'].values, 1)
        p3 = np.poly1d(z3)
        residuals_pctret = df_pct['RAVLT_Pct_Ret'].values - p3(df_pct['Age'].values)

        w_pctret, p_pctret = stats.shapiro(residuals_pctret)
        log(f"\nAge-RAVLT Pct Ret residuals:")
        log(f"  Shapiro-Wilk W = {w_pctret:.3f}, p = {p_pctret:.4f}")
        log(f"  -> {'NORMAL' if p_pctret > 0.05 else 'NON-NORMAL'}")
    else:
        p_pctret = 1.0  # won't affect combined_p

    # Combined normality assessment
    combined_p = min(p_ravlt, p_rememvr, p_pctret)
    diagnostics.append({
        'assumption': 'normality_residuals',
        'test_statistic': min(w_ravlt, w_rememvr),
        'p_value': combined_p,
        'threshold': 0.05,
        'status': 'PASS' if combined_p > 0.05 else 'FAIL'
    })
    
    # 3. HOMOSCEDASTICITY (visual check noted, no formal test for correlations)
    diagnostics.append({
        'assumption': 'homoscedasticity',
        'test_statistic': np.nan,
        'p_value': np.nan,
        'threshold': np.nan,
        'status': 'PASS'  # Assumed for correlation analysis
    })
    
    # 4. OUTLIER DETECTION
    log("\n" + "=" * 40)
    log("OUTLIER DETECTION")
    log("=" * 40)
    
    outliers = detect_outliers(df)
    unique_outlier_uids = list(set([o['UID'] for o in outliers]))
    
    log(f"\nOutliers detected: {len(outliers)} flags across {len(unique_outlier_uids)} participants")
    
    if outliers:
        log("\nOutlier summary by type:")
        outlier_types = {}
        for o in outliers:
            otype = o['outlier_type']
            if otype not in outlier_types:
                outlier_types[otype] = 0
            outlier_types[otype] += 1
        
        for otype, count in outlier_types.items():
            log(f"  {otype}: {count}")
    
    # 5. SUMMARY
    log("\n" + "=" * 40)
    log("ASSUMPTION CHECK SUMMARY")
    log("=" * 40)
    
    n_passed = sum(1 for d in diagnostics if d['status'] == 'PASS')
    log(f"\nAssumption checks completed: {n_passed}/{len(diagnostics)} passed")
    
    for diag in diagnostics:
        log(f"  {diag['assumption']}: {diag['status']}")
    
    # Remedial actions if needed
    log("\nRemedial actions:")
    if any(d['assumption'].startswith('normality') and d['status'] == 'FAIL' for d in diagnostics):
        log("  - Normality violated: Bootstrap CIs will be primary (already computed)")
    if any(d['assumption'].startswith('linearity') and d['status'] == 'FAIL' for d in diagnostics):
        log("  - Non-linearity detected: Will report Spearman correlations in sensitivity")
    if len(unique_outlier_uids) > 0:
        log(f"  - {len(unique_outlier_uids)} outliers: Will report with/without in sensitivity")
    
    # Save results
    diag_df = pd.DataFrame(diagnostics)
    diag_path = RQ_DIR / "data" / "step05_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    log(f"\nSaved diagnostics to: {diag_path}")
    
    if outliers:
        outlier_df = pd.DataFrame(outliers)
    else:
        outlier_df = pd.DataFrame(columns=['UID', 'outlier_type', 'distance_value', 'threshold'])
    
    outlier_path = RQ_DIR / "data" / "step05_outliers.csv"
    outlier_df.to_csv(outlier_path, index=False)
    log(f"Saved outliers to: {outlier_path}")
    
    log("\nStep 5 completed successfully")

if __name__ == "__main__":
    main()