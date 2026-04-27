#!/usr/bin/env python3
"""
Step 09: Individual Predictors Analysis
Detailed analysis of individual cognitive predictors
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "step09_individual_predictors.log"

def log(msg):
    """Write to log file and print"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    log("[START] Step 09: Individual Predictors Analysis")
    
    # Load data
    log("[LOAD] Loading regression results and analysis data...")
    reg_df = pd.read_csv(DATA_DIR / "step04_regression_results.csv")
    data_df = pd.read_csv(DATA_DIR / "step03_analysis_dataset.csv")
    
    # Focus on cognitive predictors
    cognitive_predictors = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
    
    # Analyze each cognitive predictor
    log("[ANALYSIS] Analyzing individual cognitive predictors...")
    predictor_results = []
    
    for predictor in cognitive_predictors:
        # Get regression results
        pred_row = reg_df[reg_df['predictor'] == predictor].iloc[0]
        
        # Simple correlation with outcome
        corr, p_corr = stats.pearsonr(data_df[predictor], data_df['calibration_quality'])
        
        # Partial correlation (controlling for demographics)
        from sklearn.linear_model import LinearRegression
        
        # Residualize predictor
        X_demo = data_df[['age', 'sex', 'education']].values
        y_pred = data_df[predictor].values
        model_pred = LinearRegression().fit(X_demo, y_pred)
        resid_pred = y_pred - model_pred.predict(X_demo)
        
        # Residualize outcome
        y_out = data_df['calibration_quality'].values
        model_out = LinearRegression().fit(X_demo, y_out)
        resid_out = y_out - model_out.predict(X_demo)
        
        # Partial correlation
        partial_corr, p_partial = stats.pearsonr(resid_pred, resid_out)
        
        predictor_results.append({
            'predictor': predictor,
            'beta': pred_row['beta'],
            'se': pred_row['se'],
            'p_uncorrected': pred_row['p_uncorrected'],
            'p_bonferroni': pred_row['p_bonferroni'],
            'sr_squared': pred_row['sr_squared'],
            'simple_r': corr,
            'simple_p': p_corr,
            'partial_r': partial_corr,
            'partial_p': p_partial,
            'vif': pred_row['vif']
        })
    
    results_df = pd.DataFrame(predictor_results)
    
    # Identify strongest predictor
    log("[ANALYSIS] Identifying strongest predictor...")
    strongest = results_df.loc[results_df['sr_squared'].abs().idxmax()]
    log(f"[INFO] Strongest predictor: {strongest['predictor']} (sr² = {strongest['sr_squared']:.4f})")
    
    # Test hypothesis: RPM > RAVLT, BVMT (learning totals)
    rpm_sr2 = results_df[results_df['predictor'] == 'RPM_T']['sr_squared'].iloc[0]
    ravlt_sr2 = results_df[results_df['predictor'] == 'RAVLT_T']['sr_squared'].iloc[0]
    bvmt_sr2 = results_df[results_df['predictor'] == 'BVMT_T']['sr_squared'].iloc[0]
    ravlt_pct_sr2 = results_df[results_df['predictor'] == 'RAVLT_Pct_Ret_T']['sr_squared'].iloc[0]
    bvmt_pct_sr2 = results_df[results_df['predictor'] == 'BVMT_Pct_Ret_T']['sr_squared'].iloc[0]

    hypothesis_supported = (rpm_sr2 > ravlt_sr2) and (rpm_sr2 > bvmt_sr2)
    log(f"[HYPOTHESIS] RPM > RAVLT & BVMT (learning): {hypothesis_supported}")
    log(f"[INFO] RPM sr² = {rpm_sr2:.4f}, RAVLT sr² = {ravlt_sr2:.4f}, BVMT sr² = {bvmt_sr2:.4f}")
    log(f"[INFO] RAVLT_Pct_Ret sr² = {ravlt_pct_sr2:.4f}, BVMT_Pct_Ret sr² = {bvmt_pct_sr2:.4f}")
    
    # Save results
    log("[SAVE] Saving individual predictor results...")
    results_df.to_csv(DATA_DIR / "step09_individual_predictors.csv", index=False)
    log(f"[SAVED] step09_individual_predictors.csv ({len(results_df)} cognitive predictors)")
    
    # Summary
    summary_df = pd.DataFrame([{
        'strongest_predictor': strongest['predictor'],
        'strongest_sr2': strongest['sr_squared'],
        'hypothesis_supported': hypothesis_supported,
        'n_significant_uncorrected': len(results_df[results_df['p_uncorrected'] < 0.05]),
        'n_significant_corrected': len(results_df[results_df['p_bonferroni'] < 0.05])
    }])
    
    summary_df.to_csv(DATA_DIR / "step09_predictor_summary.csv", index=False)
    log(f"[SAVED] step09_predictor_summary.csv")
    
    log("[SUCCESS] Step 09 complete")