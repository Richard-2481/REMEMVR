#!/usr/bin/env python3
"""
Step 03: Extract and Compare Beta Coefficients
RQ: ch7/7.1.3
Purpose: Extract beta coefficients from all models and create comparison matrix for cross-domain analysis
Output: Beta coefficient matrix and cross-domain comparisons
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step03_extract_coefficients.log"

# Input files
INPUT_WHAT = RQ_DIR / "data" / "step02_what_model_results.csv"
INPUT_WHERE = RQ_DIR / "data" / "step02_where_model_results.csv"
INPUT_WHEN = RQ_DIR / "data" / "step02_when_model_results.csv"

# Output files
OUTPUT_MATRIX = RQ_DIR / "data" / "step03_beta_coefficient_matrix.csv"
OUTPUT_COMPARISONS = RQ_DIR / "data" / "step03_cross_domain_comparisons.csv"
OUTPUT_EFFECT_SIZES = RQ_DIR / "data" / "step03_effect_sizes.csv"
OUTPUT_HEATMAP = RQ_DIR / "data" / "step03_heatmap_plot_data.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def classify_effect_size(beta):
    """Classify effect size based on Cohen's conventions for standardized betas."""
    abs_beta = abs(beta)
    if abs_beta < 0.1:
        return "negligible"
    elif abs_beta < 0.3:
        return "small"
    elif abs_beta < 0.5:
        return "medium"
    else:
        return "large"

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 03: Extract and Compare Beta Coefficients")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        
        # =========================================================================
        # STEP 1: Load model results
        # =========================================================================
        log("\n[STEP 1] Loading model results for all domains...")
        
        what_results = pd.read_csv(INPUT_WHAT)
        where_results = pd.read_csv(INPUT_WHERE)
        when_results = pd.read_csv(INPUT_WHEN)
        
        log(f"[INFO] Loaded What model: {len(what_results)} coefficients")
        log(f"[INFO] Loaded Where model: {len(where_results)} coefficients")
        log(f"[INFO] Loaded When model: {len(when_results)} coefficients")
        
        # =========================================================================
        # STEP 2: Create beta coefficient matrix
        # =========================================================================
        log("\n[STEP 2] Creating beta coefficient matrix...")
        
        # Extract coefficients (excluding intercept)
        predictors = ['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        
        matrix_data = []
        
        for domain, results_df in [('What', what_results), 
                                   ('Where', where_results), 
                                   ('When', when_results)]:
            row = {'domain': domain}
            
            # Add intercept
            intercept_row = results_df[results_df['predictor'] == 'intercept']
            row['intercept_beta'] = intercept_row['beta'].values[0]
            
            # Add predictor betas
            for predictor in predictors:
                pred_row = results_df[results_df['predictor'] == predictor]
                row[f'{predictor}_beta'] = pred_row['beta'].values[0]
                
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        matrix_df.to_csv(OUTPUT_MATRIX, index=False)
        log(f"[OUTPUT] Beta coefficient matrix saved to: {OUTPUT_MATRIX}")
        
        # Display matrix
        log("\n[INFO] Beta Coefficient Matrix:")
        log(f"{'Domain':<10} {'RAVLT':<10} {'RAVLT_Pct':<10} {'BVMT':<10} {'BVMT_Pct':<10} {'RPM':<10}")
        log("-" * 60)
        for _, row in matrix_df.iterrows():
            log(f"{row['domain']:<10} {row['RAVLT_T_beta']:>9.3f} {row['RAVLT_Pct_Ret_T_beta']:>9.3f} {row['BVMT_T_beta']:>9.3f} {row['BVMT_Pct_Ret_T_beta']:>9.3f} {row['RPM_T_beta']:>9.3f}")
        
        # =========================================================================
        # STEP 3: Calculate cross-domain comparisons
        # =========================================================================
        log("\n[STEP 3] Calculating cross-domain comparisons...")
        
        comparisons = []
        
        # Key hypothesis comparisons
        # 1. RAVLT_What - RAVLT_Where (expected positive)
        ravlt_what = matrix_df[matrix_df['domain'] == 'What']['RAVLT_T_beta'].values[0]
        ravlt_where = matrix_df[matrix_df['domain'] == 'Where']['RAVLT_T_beta'].values[0]
        comparisons.append({
            'comparison': 'RAVLT_T_What - RAVLT_T_Where',
            'beta_difference': ravlt_what - ravlt_where,
            'hypothesis': 'positive',
            'observed': 'positive' if ravlt_what > ravlt_where else 'negative'
        })

        # 1b. RAVLT_Pct_Ret_What - RAVLT_Pct_Ret_Where (expected positive)
        ravlt_pct_what = matrix_df[matrix_df['domain'] == 'What']['RAVLT_Pct_Ret_T_beta'].values[0]
        ravlt_pct_where = matrix_df[matrix_df['domain'] == 'Where']['RAVLT_Pct_Ret_T_beta'].values[0]
        comparisons.append({
            'comparison': 'RAVLT_Pct_Ret_T_What - RAVLT_Pct_Ret_T_Where',
            'beta_difference': ravlt_pct_what - ravlt_pct_where,
            'hypothesis': 'positive',
            'observed': 'positive' if ravlt_pct_what > ravlt_pct_where else 'negative'
        })

        # 2. BVMT_Where - BVMT_What (expected positive)
        bvmt_what = matrix_df[matrix_df['domain'] == 'What']['BVMT_T_beta'].values[0]
        bvmt_where = matrix_df[matrix_df['domain'] == 'Where']['BVMT_T_beta'].values[0]
        comparisons.append({
            'comparison': 'BVMT_T_Where - BVMT_T_What',
            'beta_difference': bvmt_where - bvmt_what,
            'hypothesis': 'positive',
            'observed': 'positive' if bvmt_where > bvmt_what else 'negative'
        })

        # 2b. BVMT_Pct_Ret_Where - BVMT_Pct_Ret_What (expected positive)
        bvmt_pct_what = matrix_df[matrix_df['domain'] == 'What']['BVMT_Pct_Ret_T_beta'].values[0]
        bvmt_pct_where = matrix_df[matrix_df['domain'] == 'Where']['BVMT_Pct_Ret_T_beta'].values[0]
        comparisons.append({
            'comparison': 'BVMT_Pct_Ret_T_Where - BVMT_Pct_Ret_T_What',
            'beta_difference': bvmt_pct_where - bvmt_pct_what,
            'hypothesis': 'positive',
            'observed': 'positive' if bvmt_pct_where > bvmt_pct_what else 'negative'
        })

        # 3. RPM consistency check (should be similar across domains)
        rpm_what = matrix_df[matrix_df['domain'] == 'What']['RPM_T_beta'].values[0]
        rpm_where = matrix_df[matrix_df['domain'] == 'Where']['RPM_T_beta'].values[0]
        rpm_when = matrix_df[matrix_df['domain'] == 'When']['RPM_T_beta'].values[0]

        rpm_range = max(rpm_what, rpm_where, rpm_when) - min(rpm_what, rpm_where, rpm_when)
        comparisons.append({
            'comparison': 'RPM_range',
            'beta_difference': rpm_range,
            'hypothesis': 'small (<0.1)',
            'observed': 'small' if rpm_range < 0.1 else 'large'
        })
        
        comparisons_df = pd.DataFrame(comparisons)
        comparisons_df.to_csv(OUTPUT_COMPARISONS, index=False)
        log(f"[OUTPUT] Cross-domain comparisons saved to: {OUTPUT_COMPARISONS}")
        
        # Log hypothesis tests
        log("\n[HYPOTHESIS TESTS] Cross-domain predictions:")
        for _, comp in comparisons_df.iterrows():
            match = "✓" if comp['hypothesis'] == comp['observed'] or (comp['hypothesis'] == 'small (<0.1)' and comp['observed'] == 'small') else "✗"
            log(f"  {comp['comparison']}: diff={comp['beta_difference']:.3f} (expected {comp['hypothesis']}, observed {comp['observed']}) {match}")
        
        # =========================================================================
        # STEP 4: Classify effect sizes
        # =========================================================================
        log("\n[STEP 4] Classifying effect sizes using Cohen's conventions...")
        
        effect_sizes = []
        
        for domain, results_df in [('What', what_results), 
                                   ('Where', where_results), 
                                   ('When', when_results)]:
            for predictor in predictors:
                pred_row = results_df[results_df['predictor'] == predictor]
                beta = pred_row['beta'].values[0]
                p_value = pred_row['p_value'].values[0]
                
                effect_sizes.append({
                    'domain': domain,
                    'predictor': predictor,
                    'beta': beta,
                    'p_value': p_value,
                    'effect_size_magnitude': classify_effect_size(beta),
                    'significant': p_value < 0.05
                })
        
        effect_sizes_df = pd.DataFrame(effect_sizes)
        effect_sizes_df.to_csv(OUTPUT_EFFECT_SIZES, index=False)
        log(f"[OUTPUT] Effect sizes saved to: {OUTPUT_EFFECT_SIZES}")
        
        # Summary of effect sizes
        log("\n[SUMMARY] Effect Size Classification:")
        for domain in ['What', 'Where', 'When']:
            domain_effects = effect_sizes_df[effect_sizes_df['domain'] == domain]
            log(f"\n  {domain} domain:")
            for _, row in domain_effects.iterrows():
                sig_marker = "*" if row['significant'] else ""
                log(f"    {row['predictor']}: {row['effect_size_magnitude']} (β={row['beta']:.3f}){sig_marker}")
        
        # =========================================================================
        # STEP 5: Prepare heatmap data for visualization
        # =========================================================================
        log("\n[STEP 5] Preparing heatmap data for visualization...")
        
        heatmap_data = []
        
        for domain in ['What', 'Where', 'When']:
            results_df = {'What': what_results, 'Where': where_results, 'When': when_results}[domain]
            
            for predictor in predictors:
                pred_row = results_df[results_df['predictor'] == predictor]
                
                heatmap_data.append({
                    'domain': domain,
                    'predictor': predictor,
                    'beta': pred_row['beta'].values[0],
                    'p_value': pred_row['p_value'].values[0],
                    'significance': '***' if pred_row['p_value'].values[0] < 0.001 else
                                  '**' if pred_row['p_value'].values[0] < 0.01 else
                                  '*' if pred_row['p_value'].values[0] < 0.05 else 'ns'
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.to_csv(OUTPUT_HEATMAP, index=False)
        log(f"[OUTPUT] Heatmap plot data saved to: {OUTPUT_HEATMAP}")
        
        # =========================================================================
        # STEP 6: Summary statistics
        # =========================================================================
        log("\n[STEP 6] Computing summary statistics...")
        
        # Find strongest predictor for each domain
        log("\n[SUMMARY] Strongest predictor by domain:")
        for domain in ['What', 'Where', 'When']:
            domain_effects = effect_sizes_df[effect_sizes_df['domain'] == domain]
            strongest = domain_effects.loc[domain_effects['beta'].abs().idxmax()]
            log(f"  {domain}: {strongest['predictor']} (β={strongest['beta']:.3f}, p={strongest['p_value']:.3f})")
        
        # Find strongest domain for each predictor
        log("\n[SUMMARY] Best predicted domain by test:")
        for predictor in predictors:
            pred_effects = effect_sizes_df[effect_sizes_df['predictor'] == predictor]
            strongest = pred_effects.loc[pred_effects['beta'].abs().idxmax()]
            log(f"  {predictor}: {strongest['domain']} (β={strongest['beta']:.3f}, p={strongest['p_value']:.3f})")
        
        log("\n[COMPLETE] Step 03 completed successfully")
        log(f"[SUMMARY] Extracted 15 beta coefficients (5 predictors x 3 domains)")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        sys.exit(1)