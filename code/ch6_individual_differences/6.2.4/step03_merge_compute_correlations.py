#!/usr/bin/env python3
"""
Step 3: Merge Datasets and Compute Correlations
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Merge REMEMVR and RAVLT datasets and compute age correlations with bootstrap CIs
This is the KEY ANALYSIS testing whether RAVLT shows age decline while REMEMVR doesn't
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_compute_correlations.log"

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def bootstrap_correlation(x, y, n_bootstrap=1000, confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for correlation"""
    np.random.seed(seed)
    n = len(x)
    correlations = []
    
    for _ in range(n_bootstrap):
        # Participant-level resampling with replacement
        indices = np.random.choice(n, n, replace=True)
        x_boot = x.iloc[indices] if hasattr(x, 'iloc') else x[indices]
        y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        r, _ = stats.pearsonr(x_boot, y_boot)
        correlations.append(r)
    
    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = np.percentile(correlations, alpha/2 * 100)
    ci_upper = np.percentile(correlations, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper, correlations

def main():
    log("=" * 60)
    log("Step 3: Merge and Compute Correlations")
    log("KEY ANALYSIS: Testing VR Scaffolding Hypothesis")
    log("=" * 60)
    
    # Load datasets
    rememvr_path = RQ_DIR / "data" / "step01_rememvr_theta_data.csv"
    ravlt_path = RQ_DIR / "data" / "step02_ravlt_age_data.csv"
    
    log(f"Loading REMEMVR data from: {rememvr_path}")
    df_rememvr = pd.read_csv(rememvr_path)
    log(f"  Loaded {len(df_rememvr)} participants")
    
    log(f"Loading RAVLT/Age data from: {ravlt_path}")
    df_ravlt = pd.read_csv(ravlt_path)
    log(f"  Loaded {len(df_ravlt)} participants")
    
    # Ensure UIDs are strings for merging
    df_rememvr['UID'] = df_rememvr['UID'].astype(str)
    df_ravlt['UID'] = df_ravlt['UID'].astype(str)
    
    # Merge datasets on UID
    log("\nMerging datasets on UID (inner join)")
    df_merged = pd.merge(df_rememvr, df_ravlt, on='UID', how='inner')
    log(f"Final dataset: {len(df_merged)} participants with complete data")
    
    if len(df_merged) != 100:
        log(f"WARNING: Expected 100 participants, got {len(df_merged)}")
    
    # Compute correlations
    log("\n" + "=" * 40)
    log("COMPUTING AGE CORRELATIONS")
    log("=" * 40)
    
    correlations = []
    
    # 1. Age-RAVLT correlation (expected: negative, traditional decline)
    r_age_ravlt, p_age_ravlt = stats.pearsonr(df_merged['Age'], df_merged['RAVLT_Total'])
    ci_lower_ravlt, ci_upper_ravlt, boot_ravlt = bootstrap_correlation(
        df_merged['Age'], df_merged['RAVLT_Total'], n_bootstrap=1000, seed=42
    )
    
    log(f"\nAge-RAVLT Correlation:")
    log(f"  r = {r_age_ravlt:.3f}, p = {p_age_ravlt:.4f}")
    log(f"  95% CI: [{ci_lower_ravlt:.3f}, {ci_upper_ravlt:.3f}]")
    
    # Interpretation
    if r_age_ravlt < -0.30 and p_age_ravlt < 0.05:
        log(f"  → EXPECTED PATTERN: Significant age-related decline in RAVLT")
    else:
        log(f"  → UNEXPECTED: RAVLT should show r < -0.30, got r = {r_age_ravlt:.3f}")
    
    correlations.append({
        'variable_pair': 'Age_RAVLT',
        'r': r_age_ravlt,
        'p_uncorrected': p_age_ravlt,
        'p_bonferroni': min(p_age_ravlt * 2, 1.0),  # Bonferroni for 2 tests
        'ci_lower': ci_lower_ravlt,
        'ci_upper': ci_upper_ravlt,
        'n_bootstrap': 1000,
        'interpretation': 'Significant decline' if p_age_ravlt < 0.05 else 'Non-significant'
    })
    
    # 2. Age-REMEMVR correlation (expected: near zero, age-invariance)
    r_age_rememvr, p_age_rememvr = stats.pearsonr(df_merged['Age'], df_merged['theta_all'])
    ci_lower_rememvr, ci_upper_rememvr, boot_rememvr = bootstrap_correlation(
        df_merged['Age'], df_merged['theta_all'], n_bootstrap=1000, seed=42
    )
    
    log(f"\nAge-REMEMVR Correlation:")
    log(f"  r = {r_age_rememvr:.3f}, p = {p_age_rememvr:.4f}")
    log(f"  95% CI: [{ci_lower_rememvr:.3f}, {ci_upper_rememvr:.3f}]")
    
    # Interpretation
    if abs(r_age_rememvr) < 0.20 and p_age_rememvr > 0.10:
        log(f"  → EXPECTED PATTERN: Age-invariance in REMEMVR (VR scaffolding)")
    else:
        log(f"  → Pattern: {'Age-invariant' if p_age_rememvr > 0.05 else 'Significant correlation'}")
    
    correlations.append({
        'variable_pair': 'Age_REMEMVR',
        'r': r_age_rememvr,
        'p_uncorrected': p_age_rememvr,
        'p_bonferroni': min(p_age_rememvr * 2, 1.0),
        'ci_lower': ci_lower_rememvr,
        'ci_upper': ci_upper_rememvr,
        'n_bootstrap': 1000,
        'interpretation': 'Age-invariant' if p_age_rememvr > 0.05 else 'Significant correlation'
    })
    
    # 3. Age-RAVLT_Pct_Ret correlation (percent retention measure)
    if 'RAVLT_Pct_Ret' in df_merged.columns:
        df_pct = df_merged.dropna(subset=['RAVLT_Pct_Ret'])
        n_pct = len(df_pct)
        r_age_pctret, p_age_pctret = stats.pearsonr(df_pct['Age'], df_pct['RAVLT_Pct_Ret'])
        ci_lower_pctret, ci_upper_pctret, boot_pctret = bootstrap_correlation(
            df_pct['Age'], df_pct['RAVLT_Pct_Ret'], n_bootstrap=1000, seed=42
        )

        log(f"\nAge-RAVLT Percent Retention Correlation:")
        log(f"  r = {r_age_pctret:.3f}, p = {p_age_pctret:.4f} (N = {n_pct})")
        log(f"  95% CI: [{ci_lower_pctret:.3f}, {ci_upper_pctret:.3f}]")

        correlations.append({
            'variable_pair': 'Age_RAVLT_Pct_Ret',
            'r': r_age_pctret,
            'p_uncorrected': p_age_pctret,
            'p_bonferroni': min(p_age_pctret * 3, 1.0),  # Bonferroni for 3 tests
            'ci_lower': ci_lower_pctret,
            'ci_upper': ci_upper_pctret,
            'n_bootstrap': 1000,
            'interpretation': 'Significant decline' if p_age_pctret < 0.05 else 'Non-significant'
        })

        # Update Bonferroni for original 2 tests to 3 tests
        for corr in correlations:
            if corr['variable_pair'] in ('Age_RAVLT', 'Age_REMEMVR'):
                corr['p_bonferroni'] = min(corr['p_uncorrected'] * 3, 1.0)
    else:
        log("\nWARNING: RAVLT_Pct_Ret column not found in merged data")

    # Also compute correlation between RAVLT and REMEMVR (needed for Steiger's test)
    r_ravlt_rememvr, p_ravlt_rememvr = stats.pearsonr(
        df_merged['RAVLT_Total'], df_merged['theta_all']
    )
    log(f"\nRAVLT-REMEMVR Correlation (for Steiger's test):")
    log(f"  r = {r_ravlt_rememvr:.3f}, p = {p_ravlt_rememvr:.4f}")
    
    # KEY HYPOTHESIS TEST SUMMARY
    log("\n" + "=" * 40)
    log("HYPOTHESIS TEST SUMMARY")
    log("=" * 40)
    
    hypothesis_supported = False
    if r_age_ravlt < -0.20 and abs(r_age_rememvr) < abs(r_age_ravlt):
        log("✓ Primary pattern observed: RAVLT shows more age decline than REMEMVR")
        log(f"  - RAVLT age correlation: r = {r_age_ravlt:.3f}")
        log(f"  - REMEMVR age correlation: r = {r_age_rememvr:.3f}")
        log(f"  - Difference: {abs(r_age_ravlt) - abs(r_age_rememvr):.3f}")
        hypothesis_supported = True
    else:
        log("✗ Unexpected pattern: Need formal Steiger's test for conclusion")
    
    # Decision D068: Report dual p-values
    log("\nDecision D068 Compliance (Dual p-values):")
    for corr in correlations:
        log(f"  {corr['variable_pair']}: p_uncorrected={corr['p_uncorrected']:.4f}, "
            f"p_bonferroni={corr['p_bonferroni']:.4f}")
    
    # Save outputs
    df_corr = pd.DataFrame(correlations)
    corr_output_path = RQ_DIR / "data" / "step03_correlations.csv"
    df_corr.to_csv(corr_output_path, index=False)
    log(f"\nSaved correlations to: {corr_output_path}")
    
    merged_output_path = RQ_DIR / "data" / "step03_merged_data.csv"
    df_merged.to_csv(merged_output_path, index=False)
    log(f"Saved merged data to: {merged_output_path}")
    
    # Add extra info for Steiger's test
    extra_dict = {
        'r_ravlt_rememvr': r_ravlt_rememvr,
        'p_ravlt_rememvr': p_ravlt_rememvr,
        'n_participants': len(df_merged)
    }
    if 'RAVLT_Pct_Ret' in df_merged.columns:
        # Also store Pct_Ret correlations for potential Steiger's test
        extra_dict['r_age_pctret'] = r_age_pctret
        extra_dict['p_age_pctret'] = p_age_pctret
        r_pctret_rememvr, p_pctret_rememvr = stats.pearsonr(
            df_pct['RAVLT_Pct_Ret'], df_pct['theta_all']
        )
        extra_dict['r_pctret_rememvr'] = r_pctret_rememvr
        extra_dict['n_pctret'] = n_pct
    extra_info = pd.DataFrame([extra_dict])
    extra_path = RQ_DIR / "data" / "step03_extra_correlations.csv"
    extra_info.to_csv(extra_path, index=False)
    
    log("\nStep 3 completed successfully")
    log("Ready for Step 4: Steiger's Z-test for formal comparison")

if __name__ == "__main__":
    main()