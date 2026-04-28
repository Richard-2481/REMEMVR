#!/usr/bin/env python3
"""
Step 08: Effect Size Analysis for RQ 7.3.1
Compute comprehensive effect size measures with bootstrap confidence intervals
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_effect_sizes.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def compute_cohens_f2(r2: float) -> float:
    """Compute Cohen's f² from R²."""
    if r2 >= 1.0:
        return np.inf
    return r2 / (1 - r2)

def bootstrap_effect_sizes(X, y, n_bootstrap=1000, random_state=42):
    """Bootstrap effect sizes with confidence intervals."""
    np.random.seed(random_state)
    n_samples = len(y)
    
    # Storage for bootstrap results
    bootstrap_r2 = []
    bootstrap_f2 = []
    bootstrap_sr2 = {col: [] for col in X.columns}
    
    log(f"Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            log(f"Progress: {i}/{n_bootstrap} iterations")
        
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        # Fit full model
        try:
            X_boot_const = sm.add_constant(X_boot)
            model = sm.OLS(y_boot, X_boot_const).fit()
            r2 = model.rsquared
            bootstrap_r2.append(r2)
            bootstrap_f2.append(compute_cohens_f2(r2))
            
            # Semi-partial correlations for each predictor
            for col in X.columns:
                X_reduced = X_boot.drop(columns=[col])
                X_reduced_const = sm.add_constant(X_reduced)
                reduced_model = sm.OLS(y_boot, X_reduced_const).fit()
                sr2 = r2 - reduced_model.rsquared
                bootstrap_sr2[col].append(sr2)
                
        except:
            # If model fails, append NaN
            bootstrap_r2.append(np.nan)
            bootstrap_f2.append(np.nan)
            for col in X.columns:
                bootstrap_sr2[col].append(np.nan)
    
    # Calculate confidence intervals (percentile method)
    ci_results = {}
    ci_results['r2'] = (np.nanpercentile(bootstrap_r2, 2.5), 
                        np.nanpercentile(bootstrap_r2, 97.5))
    ci_results['f2'] = (np.nanpercentile(bootstrap_f2, 2.5), 
                        np.nanpercentile(bootstrap_f2, 97.5))
    
    for col in X.columns:
        ci_results[f'sr2_{col}'] = (np.nanpercentile(bootstrap_sr2[col], 2.5),
                                     np.nanpercentile(bootstrap_sr2[col], 97.5))
    
    return ci_results

try:
    log("Step 08: Effect Size Analysis")
    log("Purpose: Compute comprehensive effect size measures")
    
    # Load data
    log("Loading analysis dataset...")
    df = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")
    log(f"Dataset: {len(df)} rows")
    
    # Load hierarchical models for comparison
    log("Loading hierarchical regression results...")
    hier_df = pd.read_csv(RQ_DIR / "data" / "step05_hierarchical_models.csv")
    log(f"Hierarchical models: {len(hier_df)} models")
    
    # Load individual predictors for semi-partial correlations
    log("Loading individual predictor results...")
    pred_df = pd.read_csv(RQ_DIR / "data" / "step06_individual_predictors.csv")
    log(f"Individual predictors: {len(pred_df)} predictors")
    
    # Prepare data
    predictors = ['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
    X = df[predictors]
    y = df['confidence_theta']
    
    # Extract primary effect sizes
    log("Calculating primary effect sizes...")
    
    # From hierarchical models
    cognitive_model = hier_df[hier_df['model'] == 'Cognitive'].iloc[0]
    r2_full = cognitive_model['R_squared']
    f2_full = cognitive_model['cohens_f2']
    
    demographics_model = hier_df[hier_df['model'] == 'Demographics'].iloc[0]
    r2_demo = demographics_model['R_squared']
    f2_demo = demographics_model['cohens_f2']
    
    # Incremental effect sizes
    delta_r2 = r2_full - r2_demo
    f2_incremental = delta_r2 / (1 - r2_full)
    
    log(f"Overall model R² = {r2_full:.4f}")
    log(f"Overall model f² = {f2_full:.4f}")
    log(f"Demographics R² = {r2_demo:.4f}")
    log(f"Incremental R² = {delta_r2:.4f}")
    log(f"Incremental f² = {f2_incremental:.4f}")
    
    # Bootstrap confidence intervals
    ci_results = bootstrap_effect_sizes(X, y, n_bootstrap=1000, random_state=42)
    
    log("Complete - calculating confidence intervals...")
    log(f"[CI] R² 95% CI: [{ci_results['r2'][0]:.4f}, {ci_results['r2'][1]:.4f}]")
    log(f"[CI] f² 95% CI: [{ci_results['f2'][0]:.4f}, {ci_results['f2'][1]:.4f}]")
    
    # Create effect size summary
    effect_results = []
    
    # Overall model effect
    effect_results.append({
        'predictor': 'Overall_Model',
        'sr2': r2_full,
        'cohens_f2': f2_full,
        'ci_lower': ci_results['r2'][0],
        'ci_upper': ci_results['r2'][1],
        'importance_rank': 0
    })
    
    # Individual cognitive predictors
    cognitive_predictors = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
    for i, pred in enumerate(cognitive_predictors):
        pred_data = pred_df[pred_df['predictor'] == pred].iloc[0]
        sr2 = pred_data['sr2']
        
        # Get bootstrap CI for this predictor
        ci_lower = ci_results[f'sr2_{pred}'][0]
        ci_upper = ci_results[f'sr2_{pred}'][1]
        
        # Individual Cohen's f² (sr² / (1 - R²))
        f2_individual = sr2 / (1 - r2_full)
        
        effect_results.append({
            'predictor': pred,
            'sr2': sr2,
            'cohens_f2': f2_individual,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'importance_rank': i + 1
        })
        
        log(f"{pred}: sr²={sr2:.4f}, f²={f2_individual:.4f}, "
            f"CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Sort by effect size and update ranks
    effect_df = pd.DataFrame(effect_results)
    effect_df_sorted = effect_df[effect_df['predictor'] != 'Overall_Model'].sort_values(
        'sr2', ascending=False
    ).reset_index(drop=True)
    
    for idx, row in effect_df_sorted.iterrows():
        effect_df.loc[effect_df['predictor'] == row['predictor'], 'importance_rank'] = idx + 1
    
    # Interpret effect sizes
    log("Effect size interpretation (Cohen's conventions):")
    log("  f² < 0.02: negligible")
    log("  f² 0.02-0.15: small")
    log("  f² 0.15-0.35: medium")
    log("  f² > 0.35: large")
    
    for _, row in effect_df.iterrows():
        f2 = row['cohens_f2']
        if f2 < 0.02:
            interpretation = "negligible"
        elif f2 < 0.15:
            interpretation = "small"
        elif f2 < 0.35:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        if row['predictor'] != 'Overall_Model':
            log(f"{row['predictor']}: f²={f2:.4f} ({interpretation})")
    
    # Save results
    output_path = RQ_DIR / "data" / "step08_effect_sizes.csv"
    effect_df.to_csv(output_path, index=False)
    log(f"Effect sizes: {output_path}")
    
    # Validation
    log("Checking effect size validity...")
    sum_sr2 = effect_df[effect_df['predictor'] != 'Overall_Model']['sr2'].sum()
    
    if sum_sr2 <= r2_full:
        log(f"PASS: Sum of sr² ({sum_sr2:.4f}) ≤ total R² ({r2_full:.4f})")
    else:
        log(f"Sum of sr² ({sum_sr2:.4f}) > total R² ({r2_full:.4f})")
    
    log("Step 08 complete")
    
except Exception as e:
    log(f"Critical error in effect size analysis: {str(e)}")
    import traceback
    log(f"{traceback.format_exc()}")
    raise
