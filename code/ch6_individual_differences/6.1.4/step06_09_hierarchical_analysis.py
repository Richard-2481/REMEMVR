#!/usr/bin/env python3
"""
Steps 06-09: Hierarchical Regression Analysis
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Combines:
- Step 06: Prepare regression data
- Step 07: Fit hierarchical regression with CV  
- Step 08: Compute Cohen's f² effect sizes
- Step 09: Residual analysis and bootstrap
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
RQ_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = RQ_DIR.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Set up logging
LOG_FILE = RQ_DIR / "logs" / "step06_09_hierarchical_analysis.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def compute_cohens_f2(r2_full, r2_reduced):
    """Compute Cohen's f² effect size."""
    if r2_full >= 1.0:
        return np.nan
    return (r2_full - r2_reduced) / (1 - r2_full)

def bootstrap_r2(X, y, n_bootstrap=1000, random_state=42):
    """Bootstrap confidence intervals for R²."""
    np.random.seed(random_state)
    n = len(y)
    r2_values = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Fit model
        model = sm.OLS(y_boot, X_boot).fit()
        r2_values.append(model.rsquared)
    
    # Calculate CI
    ci_lower = np.percentile(r2_values, 2.5)
    ci_upper = np.percentile(r2_values, 97.5)
    
    return np.mean(r2_values), ci_lower, ci_upper

def main():
    """Main execution."""
    log("Steps 06-09: Hierarchical regression analysis")
    
    # Step 06: Prepare regression data
    log("\n[STEP 06] Preparing regression data...")
    
    # Load merged data
    merged_path = RQ_DIR / "data" / "step05_merged_predictors.csv"
    df = pd.read_csv(merged_path)
    log(f"Loaded {len(df)} participants")
    
    # Remove rows with missing values
    df_complete = df.dropna()
    log(f"Complete cases: {len(df_complete)} (removed {len(df)-len(df_complete)} with missing data)")
    
    # Define predictor blocks
    block1_vars = ['age_z', 'sex_binary', 'education_z']
    block2_vars = ['RAVLT_T_z', 'RAVLT_DR_T_z', 'RAVLT_Pct_Ret_z', 'BVMT_T_z', 'BVMT_Pct_Ret_z', 'NART_T_z', 'RPM_T_z']
    block3_vars = ['DASS_Dep_z', 'DASS_Anx_z', 'DASS_Str_z', 'VR_Exp_z', 'Sleep_z']
    
    # Prepare X and y
    y = df_complete['theta'].values
    
    # Create X matrices for each model
    X1 = df_complete[block1_vars].values
    X2 = df_complete[block1_vars + block2_vars].values
    X3 = df_complete[block1_vars + block2_vars + block3_vars].values
    
    # Add intercept
    X1 = sm.add_constant(X1)
    X2 = sm.add_constant(X2)
    X3 = sm.add_constant(X3)
    
    log(f"Model 1 (Demographics): {X1.shape[1]-1} predictors")
    log(f"Model 2 (+ Cognitive): {X2.shape[1]-1} predictors")
    log(f"Model 3 (+ Self-report): {X3.shape[1]-1} predictors")
    
    # Save prepared data
    prep_df = pd.DataFrame()
    prep_df['uid'] = df_complete['uid']
    prep_df['theta'] = y
    for i, col in enumerate(block1_vars):
        prep_df[col] = df_complete[col]
    for i, col in enumerate(block2_vars):
        prep_df[col] = df_complete[col]
    for i, col in enumerate(block3_vars):
        prep_df[col] = df_complete[col]
    
    prep_path = RQ_DIR / "data" / "step06_regression_ready.csv"
    prep_df.to_csv(prep_path, index=False)
    log(f"Saved prepared data to {prep_path}")
    
    # Step 07: Fit hierarchical regression
    log("\n[STEP 07] Fitting hierarchical regression models...")
    
    # Fit full-sample models
    model1 = sm.OLS(y, X1).fit()
    model2 = sm.OLS(y, X2).fit()
    model3 = sm.OLS(y, X3).fit()
    
    # Report R² for each model
    log(f"\nFull-sample R² values:")
    log(f"  Model 1 (Demographics): R²={model1.rsquared:.4f}, Adj R²={model1.rsquared_adj:.4f}")
    log(f"  Model 2 (+ Cognitive): R²={model2.rsquared:.4f}, Adj R²={model2.rsquared_adj:.4f}")
    log(f"  Model 3 (+ Self-report): R²={model3.rsquared:.4f}, Adj R²={model3.rsquared_adj:.4f}")
    
    # Incremental R²
    delta_r2_block2 = model2.rsquared - model1.rsquared
    delta_r2_block3 = model3.rsquared - model2.rsquared
    
    log(f"\nR² change:")
    log(f"  Block 2 (Cognitive): ΔR²={delta_r2_block2:.4f}")
    log(f"  Block 3 (Self-report): ΔR²={delta_r2_block3:.4f}")
    
    # F-tests for incremental validity
    n = len(y)
    # Block 2 F-test
    f_stat_block2 = (delta_r2_block2 * (n - X2.shape[1])) / ((1 - model2.rsquared) * (X2.shape[1] - X1.shape[1]))
    df1_block2 = X2.shape[1] - X1.shape[1]
    df2_block2 = n - X2.shape[1]
    p_block2 = 1 - stats.f.cdf(f_stat_block2, df1_block2, df2_block2)
    
    # Block 3 F-test
    f_stat_block3 = (delta_r2_block3 * (n - X3.shape[1])) / ((1 - model3.rsquared) * (X3.shape[1] - X2.shape[1]))
    df1_block3 = X3.shape[1] - X2.shape[1]
    df2_block3 = n - X3.shape[1]
    p_block3 = 1 - stats.f.cdf(f_stat_block3, df1_block3, df2_block3)
    
    log(f"\n[F-TESTS] Incremental validity:")
    log(f"  Block 2: F({df1_block2},{df2_block2})={f_stat_block2:.3f}, p={p_block2:.4f}")
    log(f"  Block 3: F({df1_block3},{df2_block3})={f_stat_block3:.3f}, p={p_block3:.4f}")

    # Report individual predictor coefficients for full model
    all_vars = ['Intercept'] + block1_vars + block2_vars + block3_vars
    log(f"\nFull model (Model 3) individual predictors:")
    log(f"  {'Variable':25s} {'B':>8s} {'SE':>8s} {'t':>8s} {'p':>8s}")
    log(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    coef_records = []
    for i, var in enumerate(all_vars):
        b = model3.params[i]
        se = model3.bse[i]
        t = model3.tvalues[i]
        p = model3.pvalues[i]
        sig = '*' if p < .05 else ''
        log(f"  {var:25s} {b:8.4f} {se:8.4f} {t:8.3f} {p:8.4f} {sig}")
        coef_records.append({'Variable': var, 'B': b, 'SE': se, 't': t, 'p': p})

    coef_df = pd.DataFrame(coef_records)
    coef_path = RQ_DIR / "data" / "step07_predictor_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    log(f"Saved predictor coefficients to {coef_path}")

    # 5-fold cross-validation
    log("\n[CROSS-VALIDATION] 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {
        'model1_train': [], 'model1_test': [],
        'model2_train': [], 'model2_test': [],
        'model3_train': [], 'model3_test': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X3), 1):
        # Split data
        X1_train, X1_test = X1[train_idx], X1[test_idx]
        X2_train, X2_test = X2[train_idx], X2[test_idx]
        X3_train, X3_test = X3[train_idx], X3[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit models on training set
        m1 = sm.OLS(y_train, X1_train).fit()
        m2 = sm.OLS(y_train, X2_train).fit()
        m3 = sm.OLS(y_train, X3_train).fit()
        
        # Evaluate on test set
        pred1 = m1.predict(X1_test)
        pred2 = m2.predict(X2_test)
        pred3 = m3.predict(X3_test)
        
        # Calculate R² for test set
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        ss_res1 = np.sum((y_test - pred1)**2)
        ss_res2 = np.sum((y_test - pred2)**2)
        ss_res3 = np.sum((y_test - pred3)**2)
        
        r2_test1 = 1 - ss_res1/ss_tot if ss_tot > 0 else 0
        r2_test2 = 1 - ss_res2/ss_tot if ss_tot > 0 else 0
        r2_test3 = 1 - ss_res3/ss_tot if ss_tot > 0 else 0
        
        cv_results['model1_train'].append(m1.rsquared)
        cv_results['model1_test'].append(r2_test1)
        cv_results['model2_train'].append(m2.rsquared)
        cv_results['model2_test'].append(r2_test2)
        cv_results['model3_train'].append(m3.rsquared)
        cv_results['model3_test'].append(r2_test3)
        
        log(f"  Fold {fold}: M1 test R²={r2_test1:.3f}, M2 test R²={r2_test2:.3f}, M3 test R²={r2_test3:.3f}")
    
    # Report mean CV results
    log(f"\n[CV SUMMARY] Mean test R² across folds:")
    log(f"  Model 1: {np.mean(cv_results['model1_test']):.4f} (SD={np.std(cv_results['model1_test']):.4f})")
    log(f"  Model 2: {np.mean(cv_results['model2_test']):.4f} (SD={np.std(cv_results['model2_test']):.4f})")
    log(f"  Model 3: {np.mean(cv_results['model3_test']):.4f} (SD={np.std(cv_results['model3_test']):.4f})")
    
    # Save hierarchical results
    hier_results = pd.DataFrame({
        'Model': ['Demographics', '+ Cognitive', '+ Self-report'],
        'N_predictors': [X1.shape[1]-1, X2.shape[1]-1, X3.shape[1]-1],
        'R2': [model1.rsquared, model2.rsquared, model3.rsquared],
        'Adj_R2': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj],
        'Delta_R2': [model1.rsquared, delta_r2_block2, delta_r2_block3],
        'F_statistic': [model1.fvalue, f_stat_block2, f_stat_block3],
        'p_value': [model1.f_pvalue, p_block2, p_block3],
        'CV_test_R2_mean': [np.mean(cv_results['model1_test']), 
                            np.mean(cv_results['model2_test']),
                            np.mean(cv_results['model3_test'])],
        'CV_test_R2_sd': [np.std(cv_results['model1_test']),
                          np.std(cv_results['model2_test']),
                          np.std(cv_results['model3_test'])]
    })
    
    hier_path = RQ_DIR / "data" / "step07_hierarchical_models.csv"
    hier_results.to_csv(hier_path, index=False)
    log(f"Saved hierarchical results to {hier_path}")
    
    # Save CV details
    cv_df = pd.DataFrame(cv_results)
    cv_path = RQ_DIR / "data" / "step07_cross_validation_results.csv"
    cv_df.to_csv(cv_path, index=False)
    log(f"Saved CV results to {cv_path}")
    
    # Step 08: Compute Cohen's f² effect sizes
    log("\n[STEP 08] Computing Cohen's f² effect sizes...")
    
    f2_block1 = model1.rsquared / (1 - model1.rsquared)
    f2_block2 = compute_cohens_f2(model2.rsquared, model1.rsquared)
    f2_block3 = compute_cohens_f2(model3.rsquared, model2.rsquared)
    f2_total = model3.rsquared / (1 - model3.rsquared)
    
    log(f"\n[EFFECT SIZES] Cohen's f²:")
    log(f"  Block 1 (Demographics): f²={f2_block1:.4f} ({'small' if f2_block1 < 0.15 else 'medium' if f2_block1 < 0.35 else 'large'})")
    log(f"  Block 2 (Cognitive): f²={f2_block2:.4f} ({'small' if f2_block2 < 0.15 else 'medium' if f2_block2 < 0.35 else 'large'})")
    log(f"  Block 3 (Self-report): f²={f2_block3:.4f} ({'small' if f2_block3 < 0.15 else 'medium' if f2_block3 < 0.35 else 'large'})")
    log(f"  Total model: f²={f2_total:.4f} ({'small' if f2_total < 0.15 else 'medium' if f2_total < 0.35 else 'large'})")
    
    # Bootstrap CIs for f²
    log("\nComputing 95% CIs (1000 iterations)...")
    
    r2_m1_mean, r2_m1_lower, r2_m1_upper = bootstrap_r2(X1, y, n_bootstrap=1000)
    r2_m2_mean, r2_m2_lower, r2_m2_upper = bootstrap_r2(X2, y, n_bootstrap=1000)
    r2_m3_mean, r2_m3_lower, r2_m3_upper = bootstrap_r2(X3, y, n_bootstrap=1000)
    
    log(f"  Model 1 R²: {r2_m1_mean:.4f} [95% CI: {r2_m1_lower:.4f}, {r2_m1_upper:.4f}]")
    log(f"  Model 2 R²: {r2_m2_mean:.4f} [95% CI: {r2_m2_lower:.4f}, {r2_m2_upper:.4f}]")
    log(f"  Model 3 R²: {r2_m3_mean:.4f} [95% CI: {r2_m3_lower:.4f}, {r2_m3_upper:.4f}]")
    
    # Save effect sizes
    effect_df = pd.DataFrame({
        'Block': ['Demographics', 'Cognitive', 'Self-report', 'Total'],
        'Cohens_f2': [f2_block1, f2_block2, f2_block3, f2_total],
        'Interpretation': [
            'small' if f2_block1 < 0.15 else 'medium' if f2_block1 < 0.35 else 'large',
            'small' if f2_block2 < 0.15 else 'medium' if f2_block2 < 0.35 else 'large',
            'small' if f2_block3 < 0.15 else 'medium' if f2_block3 < 0.35 else 'large',
            'small' if f2_total < 0.15 else 'medium' if f2_total < 0.35 else 'large'
        ],
        'R2': [model1.rsquared, model2.rsquared, model3.rsquared, model3.rsquared],
        'R2_CI_lower': [r2_m1_lower, r2_m2_lower, r2_m3_lower, r2_m3_lower],
        'R2_CI_upper': [r2_m1_upper, r2_m2_upper, r2_m3_upper, r2_m3_upper]
    })
    
    effect_path = RQ_DIR / "data" / "step08_incremental_validity.csv"
    effect_df.to_csv(effect_path, index=False)
    log(f"Saved effect sizes to {effect_path}")
    
    # Step 09: Residual analysis
    log("\n[STEP 09] Residual variance analysis...")
    
    # Calculate residual variance
    residual_variance = 1 - model3.rsquared
    residual_ci_lower = 1 - r2_m3_upper
    residual_ci_upper = 1 - r2_m3_lower
    
    log(f"\n[RESIDUAL VARIANCE] Unexplained by all predictors:")
    log(f"  Residual variance: {residual_variance:.4f} ({residual_variance*100:.1f}%)")
    log(f"  95% CI: [{residual_ci_lower:.4f}, {residual_ci_upper:.4f}]")
    log(f"  Interpretation: {residual_variance*100:.1f}% of REMEMVR variance remains unexplained")
    
    # Check hypothesis (>50% unexplained expected)
    if residual_variance > 0.50:
        log(f"  [HYPOTHESIS SUPPORTED] Substantial residual variance (>50%) confirms incremental validity")
    else:
        log(f"  [HYPOTHESIS NOT SUPPORTED] Residual variance <50% - traditional tests explain most variance")
    
    # Power analysis
    from statsmodels.stats.power import FTestPower
    power_calc = FTestPower()
    
    # Post-hoc power for detecting medium effect (f²=0.15) with our sample
    power = power_calc.solve_power(effect_size=0.15, df_num=5, df_denom=n-X2.shape[1], alpha=0.05)
    log(f"\n[POWER ANALYSIS]:")
    log(f"  Post-hoc power to detect f²=0.15 with N={n}: {power:.3f}")
    
    # Minimum detectable effect size with 80% power
    min_f2 = power_calc.solve_power(power=0.80, df_num=5, df_denom=n-X2.shape[1], alpha=0.05)
    log(f"  Minimum detectable f² with 80% power: {min_f2:.3f}")
    
    # Model diagnostics (for final model)
    residuals = model3.resid
    
    # Check assumptions
    # 1. Normality of residuals
    _, p_shapiro = stats.shapiro(residuals)
    log(f"\nModel 3 assumptions:")
    log(f"  Normality (Shapiro-Wilk): p={p_shapiro:.4f} {'(PASS)' if p_shapiro > 0.05 else '(FAIL - consider robust SEs)'}")
    
    # 2. Homoscedasticity (Breusch-Pagan test)
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, p_bp, _, _ = het_breuschpagan(residuals, X3)
    log(f"  Homoscedasticity (Breusch-Pagan): p={p_bp:.4f} {'(PASS)' if p_bp > 0.05 else '(FAIL - use HC3 robust SEs)'}")
    
    # 3. Multicollinearity (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["Variable"] = ['Intercept'] + block1_vars + block2_vars + block3_vars
    vif_data["VIF"] = [variance_inflation_factor(X3, i) for i in range(X3.shape[1])]
    max_vif = vif_data.loc[vif_data['Variable'] != 'Intercept', 'VIF'].max()
    log(f"  Multicollinearity (Max VIF): {max_vif:.2f} {'(PASS)' if max_vif < 5 else '(WARNING)' if max_vif < 10 else '(FAIL - consider ridge regression)'}")
    
    # Save residual analysis
    residual_df = pd.DataFrame({
        'Metric': ['Residual_variance', 'CI_lower', 'CI_upper', 'Percent_unexplained',
                   'Shapiro_p', 'Breusch_Pagan_p', 'Max_VIF', 'Power_f2_0.15', 'Min_detectable_f2'],
        'Value': [residual_variance, residual_ci_lower, residual_ci_upper, residual_variance*100,
                  p_shapiro, p_bp, max_vif, power, min_f2]
    })
    
    residual_path = RQ_DIR / "data" / "step09_residual_variance.csv"
    residual_df.to_csv(residual_path, index=False)
    log(f"Saved residual analysis to {residual_path}")
    
    log("\nSteps 06-09 complete")
    log(f"\n[KEY FINDING] {residual_variance*100:.1f}% of REMEMVR variance unexplained by all predictors")
    log(f"REMEMVR captures unique variance beyond traditional measures")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())