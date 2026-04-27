#!/usr/bin/env python3
"""
Step 02: Fit Domain-Specific Regression Models
RQ: ch7/7.1.3
Purpose: Fit separate multiple regression models for each memory domain
Output: Regression results for What, Where, and When domains
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step02_fit_models.log"

# Input file
INPUT_FILE = RQ_DIR / "data" / "step01_merged_dataset.csv"

# Output files
OUTPUT_WHAT = RQ_DIR / "data" / "step02_what_model_results.csv"
OUTPUT_WHERE = RQ_DIR / "data" / "step02_where_model_results.csv"
OUTPUT_WHEN = RQ_DIR / "data" / "step02_when_model_results.csv"
OUTPUT_DIAGNOSTICS = RQ_DIR / "data" / "step02_model_diagnostics.csv"
OUTPUT_BOOTSTRAP = RQ_DIR / "data" / "step02_bootstrap_coefficients.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def fit_domain_model(data, domain_name, predictors=['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']):
    """Fit regression model for a specific domain."""
    
    log(f"\n[FITTING] {domain_name} domain model...")
    
    # Select domain data
    domain_data = data[data['domain'] == domain_name].copy()
    log(f"[INFO] {domain_name} domain: {len(domain_data)} observations")
    
    # Prepare data
    X = domain_data[predictors].copy()
    y = domain_data['theta_mean'].copy()
    
    # Standardize predictors for comparable beta coefficients
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=predictors,
        index=X.index
    )
    
    # Add constant
    X_scaled = sm.add_constant(X_scaled)
    
    # Fit OLS model
    model = sm.OLS(y, X_scaled).fit()
    
    log(f"[INFO] {domain_name} R²: {model.rsquared:.4f}")
    log(f"[INFO] {domain_name} Adj R²: {model.rsquared_adj:.4f}")
    log(f"[INFO] {domain_name} F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # Extract results
    results = []
    for i, predictor in enumerate(['intercept'] + predictors):
        results.append({
            'predictor': predictor,
            'beta': model.params[i],
            'se': model.bse[i],
            'ci_lower': model.conf_int().iloc[i, 0],
            'ci_upper': model.conf_int().iloc[i, 1],
            'p_value': model.pvalues[i],
            't_statistic': model.tvalues[i]
        })
    
    # Calculate VIF for multicollinearity check
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # VIF for each predictor (excluding intercept)
    X_for_vif = X_scaled.iloc[:, 1:]  # Exclude intercept
    for i, predictor in enumerate(predictors):
        vif = variance_inflation_factor(X_for_vif.values, i)
        results[i+1]['vif'] = vif
    results[0]['vif'] = np.nan  # No VIF for intercept
    
    # Check for influential points using Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    max_cooks = cooks_d.max()
    n_influential = (cooks_d > 4/len(domain_data)).sum()
    
    log(f"[INFO] {domain_name} Max Cook's D: {max_cooks:.4f}")
    log(f"[INFO] {domain_name} Influential points (Cook's D > 4/n): {n_influential}")
    
    # Store max Cook's D for each predictor row
    for result in results:
        result['cooks_d_max'] = max_cooks
    
    # Model diagnostics
    diagnostics = {
        'domain': domain_name,
        'n': len(domain_data),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'durbin_watson': sm.stats.stattools.durbin_watson(model.resid),
        'max_cooks_d': max_cooks,
        'n_influential': n_influential
    }
    
    # Assumption checks
    residuals = model.resid
    
    # Normality test (Shapiro-Wilk)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    diagnostics['shapiro_w'] = shapiro_stat
    diagnostics['shapiro_p'] = shapiro_p
    diagnostics['normality_violated'] = shapiro_p < 0.05
    
    # Heteroscedasticity test (Breusch-Pagan)
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(residuals, X_scaled)
    diagnostics['breusch_pagan_stat'] = bp_test[0]
    diagnostics['breusch_pagan_p'] = bp_test[1]
    diagnostics['heteroscedasticity_violated'] = bp_test[1] < 0.05
    
    log(f"[INFO] {domain_name} Normality: Shapiro-Wilk p={shapiro_p:.4f} ({'violated' if shapiro_p < 0.05 else 'OK'})")
    log(f"[INFO] {domain_name} Homoscedasticity: Breusch-Pagan p={bp_test[1]:.4f} ({'violated' if bp_test[1] < 0.05 else 'OK'})")
    
    return pd.DataFrame(results), diagnostics, model, X_scaled, y

def bootstrap_coefficients(X, y, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Bootstrap confidence intervals for regression coefficients."""
    
    np.random.seed(random_state)
    n_samples = len(y)
    n_predictors = X.shape[1]
    
    bootstrap_coefs = np.zeros((n_bootstrap, n_predictors))
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]
        
        # Fit model
        try:
            model_boot = sm.OLS(y_boot, X_boot).fit()
            bootstrap_coefs[i] = model_boot.params
        except:
            bootstrap_coefs[i] = np.nan
    
    # Calculate percentile CIs
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.nanpercentile(bootstrap_coefs, lower_percentile, axis=0)
    ci_upper = np.nanpercentile(bootstrap_coefs, upper_percentile, axis=0)
    
    return bootstrap_coefs, ci_lower, ci_upper

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 02: Fit Domain-Specific Regression Models")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        
        # =========================================================================
        # STEP 1: Load merged dataset
        # =========================================================================
        log("\n[STEP 1] Loading merged dataset...")
        
        data = pd.read_csv(INPUT_FILE)
        log(f"[INFO] Loaded data: {data.shape}")
        log(f"[INFO] Domains: {data['domain'].value_counts().to_dict()}")
        
        # =========================================================================
        # STEP 2: Fit regression models for each domain
        # =========================================================================
        log("\n[STEP 2] Fitting domain-specific regression models...")
        
        all_diagnostics = []
        all_bootstrap_results = []
        
        # Fit models for each domain
        domains = ['What', 'Where', 'When']
        models = {}
        
        for domain in domains:
            # Fit model
            results_df, diagnostics, model, X_scaled, y = fit_domain_model(data, domain)
            
            # Save model results
            output_file = {
                'What': OUTPUT_WHAT,
                'Where': OUTPUT_WHERE,
                'When': OUTPUT_WHEN
            }[domain]
            
            results_df.to_csv(output_file, index=False)
            log(f"[OUTPUT] {domain} model results saved to: {output_file}")
            
            # Store diagnostics
            all_diagnostics.append(diagnostics)
            models[domain] = (model, X_scaled, y)
            
        # Save diagnostics
        diagnostics_df = pd.DataFrame(all_diagnostics)
        diagnostics_df.to_csv(OUTPUT_DIAGNOSTICS, index=False)
        log(f"[OUTPUT] Model diagnostics saved to: {OUTPUT_DIAGNOSTICS}")
        
        # =========================================================================
        # STEP 3: Bootstrap confidence intervals
        # =========================================================================
        log("\n[STEP 3] Computing bootstrap confidence intervals...")
        
        for domain in domains:
            log(f"\n[BOOTSTRAP] {domain} domain (1000 iterations)...")
            
            model, X_scaled, y = models[domain]
            
            # Bootstrap
            boot_coefs, ci_lower, ci_upper = bootstrap_coefficients(
                X_scaled, y, n_bootstrap=1000, confidence_level=0.95, random_state=42
            )
            
            # Create results dataframe
            predictors = ['intercept', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
            for i, predictor in enumerate(predictors):
                all_bootstrap_results.append({
                    'domain': domain,
                    'predictor': predictor,
                    'bootstrap_mean': np.nanmean(boot_coefs[:, i]),
                    'bootstrap_se': np.nanstd(boot_coefs[:, i]),
                    'bootstrap_ci_lower': ci_lower[i],
                    'bootstrap_ci_upper': ci_upper[i],
                    'model_beta': model.params[i],
                    'model_se': model.bse[i]
                })
                
            log(f"[INFO] {domain} bootstrap complete")
            
        # Save bootstrap results
        bootstrap_df = pd.DataFrame(all_bootstrap_results)
        bootstrap_df.to_csv(OUTPUT_BOOTSTRAP, index=False)
        log(f"[OUTPUT] Bootstrap results saved to: {OUTPUT_BOOTSTRAP}")
        
        # =========================================================================
        # STEP 4: Summary of key findings
        # =========================================================================
        log("\n[STEP 4] Summary of findings...")
        
        log("\n[SUMMARY] Model R² values:")
        for _, row in diagnostics_df.iterrows():
            log(f"  {row['domain']}: R²={row['r_squared']:.4f}, Adj R²={row['adj_r_squared']:.4f}")
            
        log("\n[SUMMARY] Key beta coefficients:")
        for domain in domains:
            results_file = {
                'What': OUTPUT_WHAT,
                'Where': OUTPUT_WHERE,
                'When': OUTPUT_WHEN
            }[domain]
            
            results = pd.read_csv(results_file)
            log(f"\n  {domain} domain:")
            for _, row in results[results['predictor'] != 'intercept'].iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
                log(f"    {row['predictor']}: β={row['beta']:.3f} (p={row['p_value']:.3f}) {sig}")
                
        # Check hypothesis predictions
        log("\n[HYPOTHESIS CHECK] Domain-specific predictions:")
        
        # Load results for comparison
        what_results = pd.read_csv(OUTPUT_WHAT)
        where_results = pd.read_csv(OUTPUT_WHERE)
        when_results = pd.read_csv(OUTPUT_WHEN)
        
        # RAVLT should predict What > Where
        ravlt_what = what_results[what_results['predictor'] == 'RAVLT_T']['beta'].values[0]
        ravlt_where = where_results[where_results['predictor'] == 'RAVLT_T']['beta'].values[0]
        log(f"  RAVLT_T: What beta={ravlt_what:.3f} vs Where beta={ravlt_where:.3f} {'supported' if ravlt_what > ravlt_where else 'not supported'}")

        # RAVLT_Pct_Ret should also predict What > Where
        ravlt_pct_what = what_results[what_results['predictor'] == 'RAVLT_Pct_Ret_T']['beta'].values[0]
        ravlt_pct_where = where_results[where_results['predictor'] == 'RAVLT_Pct_Ret_T']['beta'].values[0]
        log(f"  RAVLT_Pct_Ret_T: What beta={ravlt_pct_what:.3f} vs Where beta={ravlt_pct_where:.3f} {'supported' if ravlt_pct_what > ravlt_pct_where else 'not supported'}")

        # BVMT should predict Where > What
        bvmt_what = what_results[what_results['predictor'] == 'BVMT_T']['beta'].values[0]
        bvmt_where = where_results[where_results['predictor'] == 'BVMT_T']['beta'].values[0]
        log(f"  BVMT_T: Where beta={bvmt_where:.3f} vs What beta={bvmt_what:.3f} {'supported' if bvmt_where > bvmt_what else 'not supported'}")

        # BVMT_Pct_Ret should also predict Where > What
        bvmt_pct_what = what_results[what_results['predictor'] == 'BVMT_Pct_Ret_T']['beta'].values[0]
        bvmt_pct_where = where_results[where_results['predictor'] == 'BVMT_Pct_Ret_T']['beta'].values[0]
        log(f"  BVMT_Pct_Ret_T: Where beta={bvmt_pct_where:.3f} vs What beta={bvmt_pct_what:.3f} {'supported' if bvmt_pct_where > bvmt_pct_what else 'not supported'}")
        
        # When should have lowest R²
        r2_what = diagnostics_df[diagnostics_df['domain'] == 'What']['r_squared'].values[0]
        r2_where = diagnostics_df[diagnostics_df['domain'] == 'Where']['r_squared'].values[0]
        r2_when = diagnostics_df[diagnostics_df['domain'] == 'When']['r_squared'].values[0]
        log(f"  R² When ({r2_when:.3f}) < What ({r2_what:.3f}) & Where ({r2_where:.3f})? {'✓' if r2_when < min(r2_what, r2_where) else '✗'}")
        
        log("\n[COMPLETE] Step 02 completed successfully")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        sys.exit(1)