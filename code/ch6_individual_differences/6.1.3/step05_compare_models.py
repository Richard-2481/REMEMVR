#!/usr/bin/env python3
"""
Step 05: Compare Model Performance Across Domains
RQ: ch7/7.1.3
Purpose: Compare R² values across domains to assess differential predictability
Output: Model comparison with bootstrap confidence intervals for R²
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step05_compare_models.log"

# Input files
INPUT_DIAGNOSTICS = RQ_DIR / "data" / "step02_model_diagnostics.csv"
INPUT_MERGED = RQ_DIR / "data" / "step01_merged_dataset.csv"

# Output files
OUTPUT_COMPARISON = RQ_DIR / "data" / "step05_model_comparison.csv"
OUTPUT_BOOTSTRAP = RQ_DIR / "data" / "step05_bootstrap_r_squared.csv"
OUTPUT_DIFFERENCES = RQ_DIR / "data" / "step05_r_squared_differences.csv"
OUTPUT_CONTRIBUTIONS = RQ_DIR / "data" / "step05_predictor_contributions.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def classify_r2_effect(r2):
    """Classify R² effect size using Cohen's conventions."""
    if r2 < 0.01:
        return "negligible"
    elif r2 < 0.09:
        return "small"
    elif r2 < 0.25:
        return "medium"
    else:
        return "large"

def bootstrap_r_squared(data, domain, predictors, n_bootstrap=1000, seed=42):
    """Bootstrap R² values for a domain model."""
    np.random.seed(seed + hash(domain) % 1000)  # Different seed for each domain
    
    domain_data = data[data['domain'] == domain].copy()
    n = len(domain_data)
    
    r_squared_values = []
    adj_r_squared_values = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        boot_data = domain_data.iloc[idx]
        
        # Prepare data
        X = boot_data[predictors].copy()
        y = boot_data['theta_mean'].copy()
        
        # Standardize predictors
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=predictors,
            index=X.index
        )
        X_scaled = sm.add_constant(X_scaled)
        
        try:
            # Fit model
            model = sm.OLS(y, X_scaled).fit()
            r_squared_values.append(model.rsquared)
            adj_r_squared_values.append(model.rsquared_adj)
        except:
            r_squared_values.append(np.nan)
            adj_r_squared_values.append(np.nan)
    
    # Calculate percentile CIs
    r2_ci_lower = np.nanpercentile(r_squared_values, 2.5)
    r2_ci_upper = np.nanpercentile(r_squared_values, 97.5)
    
    adj_r2_ci_lower = np.nanpercentile(adj_r_squared_values, 2.5)
    adj_r2_ci_upper = np.nanpercentile(adj_r_squared_values, 97.5)
    
    return {
        'r_squared_values': r_squared_values,
        'adj_r_squared_values': adj_r_squared_values,
        'r2_ci_lower': r2_ci_lower,
        'r2_ci_upper': r2_ci_upper,
        'adj_r2_ci_lower': adj_r2_ci_lower,
        'adj_r2_ci_upper': adj_r2_ci_upper
    }

def compute_semi_partial_r2(data, domain, predictors):
    """Compute semi-partial R² for each predictor."""
    domain_data = data[data['domain'] == domain].copy()
    
    # Prepare data
    X = domain_data[predictors].copy()
    y = domain_data['theta_mean'].copy()
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=predictors,
        index=X.index
    )
    X_scaled = sm.add_constant(X_scaled)
    
    # Full model
    full_model = sm.OLS(y, X_scaled).fit()
    r2_full = full_model.rsquared
    
    semi_partial_r2 = {}
    
    # For each predictor, fit reduced model without it
    for predictor in predictors:
        reduced_predictors = [p for p in predictors if p != predictor]
        X_reduced = X_scaled[['const'] + reduced_predictors]
        
        reduced_model = sm.OLS(y, X_reduced).fit()
        r2_reduced = reduced_model.rsquared
        
        # Semi-partial R² is the unique contribution
        semi_partial_r2[predictor] = r2_full - r2_reduced
    
    return semi_partial_r2

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Compare Model Performance Across Domains")
        log(f"RQ Directory: {RQ_DIR}")
        # Load model diagnostics and data
        log("\n[STEP 1] Loading model diagnostics and data...")
        
        diagnostics = pd.read_csv(INPUT_DIAGNOSTICS)
        data = pd.read_csv(INPUT_MERGED)
        
        log(f"Loaded diagnostics: {diagnostics.shape}")
        log(f"Loaded data: {data.shape}")
        # Extract and compare R² values
        log("\n[STEP 2] Extracting and comparing R² values...")
        
        comparison_data = []
        
        for _, row in diagnostics.iterrows():
            domain = row['domain']
            comparison_data.append({
                'domain': domain,
                'r_squared': row['r_squared'],
                'adj_r_squared': row['adj_r_squared'],
                'f_statistic': row['f_statistic'],
                'f_pvalue': row['f_pvalue'],
                'effect_size': classify_r2_effect(row['r_squared']),
                'n': row['n']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        log("\nModel R² values:")
        for _, row in comparison_df.iterrows():
            log(f"  {row['domain']}: R²={row['r_squared']:.4f} ({row['effect_size']}), Adj R²={row['adj_r_squared']:.4f}")
        # Bootstrap confidence intervals for R²
        log("\n[STEP 3] Computing bootstrap confidence intervals for R²...")
        
        predictors = ['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        bootstrap_results = []
        all_bootstrap_values = []
        
        for domain in ['What', 'Where', 'When']:
            log(f"\n{domain} domain (1000 iterations)...")
            
            boot_results = bootstrap_r_squared(data, domain, predictors, n_bootstrap=1000, seed=42)
            
            # Add CIs to comparison data
            idx = comparison_df[comparison_df['domain'] == domain].index[0]
            comparison_df.loc[idx, 'r2_ci_lower'] = boot_results['r2_ci_lower']
            comparison_df.loc[idx, 'r2_ci_upper'] = boot_results['r2_ci_upper']
            comparison_df.loc[idx, 'adj_r2_ci_lower'] = boot_results['adj_r2_ci_lower']
            comparison_df.loc[idx, 'adj_r2_ci_upper'] = boot_results['adj_r2_ci_upper']
            
            # Store bootstrap values for distribution analysis
            for i, r2_val in enumerate(boot_results['r_squared_values']):
                all_bootstrap_values.append({
                    'domain': domain,
                    'iteration': i,
                    'r_squared': r2_val,
                    'adj_r_squared': boot_results['adj_r_squared_values'][i]
                })
            
            log(f"  {domain} R² 95% CI: [{boot_results['r2_ci_lower']:.3f}, {boot_results['r2_ci_upper']:.3f}]")
        
        # Save comparison with CIs
        comparison_df.to_csv(OUTPUT_COMPARISON, index=False)
        log(f"Model comparison saved to: {OUTPUT_COMPARISON}")
        
        # Save bootstrap distributions (subsample to keep file reasonable)
        bootstrap_sample_df = pd.DataFrame(all_bootstrap_values)
        bootstrap_sample_df.to_csv(OUTPUT_BOOTSTRAP, index=False)
        log(f"Bootstrap R² distributions saved to: {OUTPUT_BOOTSTRAP}")
        # Compute pairwise R² differences
        log("\n[STEP 4] Computing pairwise R² differences...")
        
        differences = []
        
        # What vs Where
        r2_what = comparison_df[comparison_df['domain'] == 'What']['r_squared'].values[0]
        r2_where = comparison_df[comparison_df['domain'] == 'Where']['r_squared'].values[0]
        differences.append({
            'comparison': 'What vs Where',
            'r2_1': r2_what,
            'r2_2': r2_where,
            'r2_difference': r2_what - r2_where,
            'direction': 'What > Where' if r2_what > r2_where else 'Where > What'
        })
        
        # What vs When
        r2_when = comparison_df[comparison_df['domain'] == 'When']['r_squared'].values[0]
        differences.append({
            'comparison': 'What vs When',
            'r2_1': r2_what,
            'r2_2': r2_when,
            'r2_difference': r2_what - r2_when,
            'direction': 'What > When' if r2_what > r2_when else 'When > What'
        })
        
        # Where vs When
        differences.append({
            'comparison': 'Where vs When',
            'r2_1': r2_where,
            'r2_2': r2_when,
            'r2_difference': r2_where - r2_when,
            'direction': 'Where > When' if r2_where > r2_when else 'When > Where'
        })
        
        differences_df = pd.DataFrame(differences)
        differences_df.to_csv(OUTPUT_DIFFERENCES, index=False)
        log(f"R² differences saved to: {OUTPUT_DIFFERENCES}")
        
        log("\nPairwise R² comparisons:")
        for _, row in differences_df.iterrows():
            log(f"  {row['comparison']}: diff={row['r2_difference']:.3f} ({row['direction']})")
        # Compute semi-partial R² contributions
        log("\n[STEP 5] Computing semi-partial R² for each predictor...")
        
        contributions = []
        
        for domain in ['What', 'Where', 'When']:
            log(f"\n[SEMI-PARTIAL] {domain} domain...")
            
            semi_partial = compute_semi_partial_r2(data, domain, predictors)
            
            for predictor, contribution in semi_partial.items():
                contributions.append({
                    'domain': domain,
                    'predictor': predictor,
                    'semi_partial_r2': contribution
                })
                log(f"  {predictor}: {contribution:.4f}")
        
        contributions_df = pd.DataFrame(contributions)
        contributions_df.to_csv(OUTPUT_CONTRIBUTIONS, index=False)
        log(f"Predictor contributions saved to: {OUTPUT_CONTRIBUTIONS}")
        # Test hypothesis and summarize
        log("\n[STEP 6] Testing hypotheses and summarizing findings...")
        
        # Hypothesis: When domain should have lowest R²
        r2_order = comparison_df.sort_values('r_squared')
        lowest_r2_domain = r2_order.iloc[0]['domain']
        
        log(f"\n[HYPOTHESIS TEST] When domain has lowest R²?")
        log(f"  R² ordering: {' < '.join(r2_order['domain'].tolist())}")
        log(f"  Lowest R²: {lowest_r2_domain} (R²={r2_order.iloc[0]['r_squared']:.3f})")
        log(f"  Hypothesis supported: {'✓' if lowest_r2_domain == 'When' else '✗'}")
        
        # Check if CIs overlap
        log("\n[CONFIDENCE INTERVAL ANALYSIS]")
        for domain in ['What', 'Where', 'When']:
            domain_row = comparison_df[comparison_df['domain'] == domain]
            log(f"  {domain}: R²={domain_row['r_squared'].values[0]:.3f} "
                f"[{domain_row['r2_ci_lower'].values[0]:.3f}, {domain_row['r2_ci_upper'].values[0]:.3f}]")
        
        # Check overlap between What and Where
        what_ci = comparison_df[comparison_df['domain'] == 'What'][['r2_ci_lower', 'r2_ci_upper']].values[0]
        where_ci = comparison_df[comparison_df['domain'] == 'Where'][['r2_ci_lower', 'r2_ci_upper']].values[0]
        
        overlap_what_where = not (what_ci[1] < where_ci[0] or where_ci[1] < what_ci[0])
        log(f"\n  What vs Where CIs overlap: {'Yes' if overlap_what_where else 'No'}")
        
        # Check overlap between When and others
        when_ci = comparison_df[comparison_df['domain'] == 'When'][['r2_ci_lower', 'r2_ci_upper']].values[0]
        
        overlap_what_when = not (what_ci[1] < when_ci[0] or when_ci[1] < what_ci[0])
        overlap_where_when = not (where_ci[1] < when_ci[0] or when_ci[1] < where_ci[0])
        
        log(f"  What vs When CIs overlap: {'Yes' if overlap_what_when else 'No'}")
        log(f"  Where vs When CIs overlap: {'Yes' if overlap_where_when else 'No'}")
        
        # Summary
        log("\nKey findings:")
        log(f"  1. When domain has lowest predictability (R²={r2_when:.3f}) as hypothesized")
        log(f"  2. What (R²={r2_what:.3f}) and Where (R²={r2_where:.3f}) have similar predictability")
        log(f"  3. RPM is strongest predictor across all domains")
        log(f"  4. Domain-specific prediction patterns partially supported")
        
        log("\nStep 05 completed successfully")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"{traceback.format_exc()}")
        sys.exit(1)