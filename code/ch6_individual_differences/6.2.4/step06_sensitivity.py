#!/usr/bin/env python3
"""
Step 6: Sensitivity Analyses
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Test robustness through outlier exclusion, non-parametric methods, and age stratification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_sensitivity.log"

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def bootstrap_spearman(x, y, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for Spearman correlation"""
    np.random.seed(seed)
    n = len(x)
    correlations = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        x_boot = x[indices] if isinstance(x, np.ndarray) else x.iloc[indices]
        y_boot = y[indices] if isinstance(y, np.ndarray) else y.iloc[indices]
        rs, _ = stats.spearmanr(x_boot, y_boot)
        correlations.append(rs)
    
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)
    
    return ci_lower, ci_upper

def winsorize_data(x, limit=0.05):
    """Winsorize data at specified percentile"""
    lower = np.percentile(x, limit * 100)
    upper = np.percentile(x, (1 - limit) * 100)
    return np.clip(x, lower, upper)

def main():
    log("=" * 60)
    log("Step 6: Sensitivity Analyses")
    log("=" * 60)
    
    # Load data
    merged_path = RQ_DIR / "data" / "step03_merged_data.csv"
    outlier_path = RQ_DIR / "data" / "step05_outliers.csv"
    
    df = pd.read_csv(merged_path)
    df_outliers = pd.read_csv(outlier_path)
    
    log(f"Loaded {len(df)} participants")
    log(f"Loaded {len(df_outliers)} outlier flags")
    
    # 1. OUTLIER EXCLUSION ANALYSIS
    log("\n" + "=" * 40)
    log("OUTLIER EXCLUSION ANALYSIS")
    log("=" * 40)
    
    outlier_uids = df_outliers['UID'].unique()
    df_clean = df[~df['UID'].isin(outlier_uids)]
    n_clean = len(df_clean)
    
    log(f"Excluding {len(outlier_uids)} outliers, N = {n_clean}")
    
    outlier_results = []
    
    if n_clean > 50:  # Need reasonable sample for correlation
        # Recompute correlations without outliers
        r_age_ravlt_clean, p_age_ravlt_clean = stats.pearsonr(
            df_clean['Age'], df_clean['RAVLT_Total']
        )
        r_age_rememvr_clean, p_age_rememvr_clean = stats.pearsonr(
            df_clean['Age'], df_clean['theta_all']
        )
        
        # Bootstrap CIs
        ci_lower_ravlt, ci_upper_ravlt = bootstrap_spearman(
            df_clean['Age'].values, df_clean['RAVLT_Total'].values
        )[:2]
        ci_lower_rememvr, ci_upper_rememvr = bootstrap_spearman(
            df_clean['Age'].values, df_clean['theta_all'].values
        )[:2]
        
        log(f"\nWithout outliers:")
        log(f"  Age-RAVLT: r = {r_age_ravlt_clean:.3f}, p = {p_age_ravlt_clean:.4f}")
        log(f"             CI: [{ci_lower_ravlt:.3f}, {ci_upper_ravlt:.3f}]")
        log(f"  Age-REMEMVR: r = {r_age_rememvr_clean:.3f}, p = {p_age_rememvr_clean:.4f}")
        log(f"               CI: [{ci_lower_rememvr:.3f}, {ci_upper_rememvr:.3f}]")
        
        outlier_results.append({
            'variable_pair': 'Age_RAVLT',
            'r': r_age_ravlt_clean,
            'ci_lower': ci_lower_ravlt,
            'ci_upper': ci_upper_ravlt,
            'n_participants': n_clean,
            'method': 'outliers_excluded'
        })
        
        outlier_results.append({
            'variable_pair': 'Age_REMEMVR',
            'r': r_age_rememvr_clean,
            'ci_lower': ci_lower_rememvr,
            'ci_upper': ci_upper_rememvr,
            'n_participants': n_clean,
            'method': 'outliers_excluded'
        })
    else:
        log(f"Too few participants after exclusion (N={n_clean})")
        outlier_results.append({
            'variable_pair': 'insufficient_data',
            'r': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_participants': n_clean,
            'method': 'outliers_excluded'
        })
    
    # 2. NON-PARAMETRIC CORRELATIONS
    log("\n" + "=" * 40)
    log("SPEARMAN RANK CORRELATIONS")
    log("=" * 40)
    
    spearman_results = []
    
    # Spearman correlations on full dataset
    rs_age_ravlt, p_age_ravlt_s = stats.spearmanr(df['Age'], df['RAVLT_Total'])
    rs_age_rememvr, p_age_rememvr_s = stats.spearmanr(df['Age'], df['theta_all'])

    # Bootstrap CIs for Spearman
    ci_lower_ravlt_s, ci_upper_ravlt_s = bootstrap_spearman(
        df['Age'].values, df['RAVLT_Total'].values, seed=42
    )
    ci_lower_rememvr_s, ci_upper_rememvr_s = bootstrap_spearman(
        df['Age'].values, df['theta_all'].values, seed=42
    )

    log(f"Spearman correlations (full sample):")
    log(f"  Age-RAVLT: rs = {rs_age_ravlt:.3f}, p = {p_age_ravlt_s:.4f}")
    log(f"             CI: [{ci_lower_ravlt_s:.3f}, {ci_upper_ravlt_s:.3f}]")
    log(f"  Age-REMEMVR: rs = {rs_age_rememvr:.3f}, p = {p_age_rememvr_s:.4f}")
    log(f"               CI: [{ci_lower_rememvr_s:.3f}, {ci_upper_rememvr_s:.3f}]")

    spearman_results.append({
        'variable_pair': 'Age_RAVLT',
        'rs': rs_age_ravlt,
        'ci_lower': ci_lower_ravlt_s,
        'ci_upper': ci_upper_ravlt_s,
        'n_participants': len(df),
        'method': 'spearman_rank'
    })

    spearman_results.append({
        'variable_pair': 'Age_REMEMVR',
        'rs': rs_age_rememvr,
        'ci_lower': ci_lower_rememvr_s,
        'ci_upper': ci_upper_rememvr_s,
        'n_participants': len(df),
        'method': 'spearman_rank'
    })

    # Spearman for RAVLT Pct Ret (if available)
    if 'RAVLT_Pct_Ret' in df.columns:
        df_pct = df.dropna(subset=['RAVLT_Pct_Ret'])
        rs_age_pctret, p_age_pctret_s = stats.spearmanr(df_pct['Age'], df_pct['RAVLT_Pct_Ret'])
        ci_lower_pctret_s, ci_upper_pctret_s = bootstrap_spearman(
            df_pct['Age'].values, df_pct['RAVLT_Pct_Ret'].values, seed=42
        )
        log(f"  Age-RAVLT_Pct_Ret: rs = {rs_age_pctret:.3f}, p = {p_age_pctret_s:.4f}")
        log(f"                     CI: [{ci_lower_pctret_s:.3f}, {ci_upper_pctret_s:.3f}]")

        spearman_results.append({
            'variable_pair': 'Age_RAVLT_Pct_Ret',
            'rs': rs_age_pctret,
            'ci_lower': ci_lower_pctret_s,
            'ci_upper': ci_upper_pctret_s,
            'n_participants': len(df_pct),
            'method': 'spearman_rank'
        })
    
    # 3. AGE STRATIFICATION
    log("\n" + "=" * 40)
    log("AGE-STRATIFIED ANALYSIS")
    log("=" * 40)
    
    age_median = df['Age'].median()
    df_younger = df[df['Age'] <= age_median]
    df_older = df[df['Age'] > age_median]
    
    log(f"Median age: {age_median:.1f} years")
    log(f"Younger adults: N = {len(df_younger)}, age range {df_younger['Age'].min():.0f}-{df_younger['Age'].max():.0f}")
    log(f"Older adults: N = {len(df_older)}, age range {df_older['Age'].min():.0f}-{df_older['Age'].max():.0f}")
    
    age_group_results = []
    
    # Younger adults
    r_young_ravlt, p_young_ravlt = stats.pearsonr(df_younger['Age'], df_younger['RAVLT_Total'])
    r_young_rememvr, p_young_rememvr = stats.pearsonr(df_younger['Age'], df_younger['theta_all'])
    
    ci_young_ravlt = bootstrap_spearman(
        df_younger['Age'].values, df_younger['RAVLT_Total'].values
    )
    ci_young_rememvr = bootstrap_spearman(
        df_younger['Age'].values, df_younger['theta_all'].values
    )
    
    log(f"\nYounger adults (≤{age_median:.0f} years):")
    log(f"  Age-RAVLT: r = {r_young_ravlt:.3f}, p = {p_young_ravlt:.4f}")
    log(f"  Age-REMEMVR: r = {r_young_rememvr:.3f}, p = {p_young_rememvr:.4f}")
    
    age_group_results.append({
        'age_group': 'younger_adults',
        'variable_pair': 'Age_RAVLT',
        'r': r_young_ravlt,
        'ci_lower': ci_young_ravlt[0],
        'ci_upper': ci_young_ravlt[1],
        'n_participants': len(df_younger),
        'median_split': age_median
    })
    
    age_group_results.append({
        'age_group': 'younger_adults',
        'variable_pair': 'Age_REMEMVR',
        'r': r_young_rememvr,
        'ci_lower': ci_young_rememvr[0],
        'ci_upper': ci_young_rememvr[1],
        'n_participants': len(df_younger),
        'median_split': age_median
    })
    
    # Older adults
    r_old_ravlt, p_old_ravlt = stats.pearsonr(df_older['Age'], df_older['RAVLT_Total'])
    r_old_rememvr, p_old_rememvr = stats.pearsonr(df_older['Age'], df_older['theta_all'])
    
    ci_old_ravlt = bootstrap_spearman(
        df_older['Age'].values, df_older['RAVLT_Total'].values
    )
    ci_old_rememvr = bootstrap_spearman(
        df_older['Age'].values, df_older['theta_all'].values
    )
    
    log(f"\nOlder adults (>{age_median:.0f} years):")
    log(f"  Age-RAVLT: r = {r_old_ravlt:.3f}, p = {p_old_ravlt:.4f}")
    log(f"  Age-REMEMVR: r = {r_old_rememvr:.3f}, p = {p_old_rememvr:.4f}")
    
    age_group_results.append({
        'age_group': 'older_adults',
        'variable_pair': 'Age_RAVLT',
        'r': r_old_ravlt,
        'ci_lower': ci_old_ravlt[0],
        'ci_upper': ci_old_ravlt[1],
        'n_participants': len(df_older),
        'median_split': age_median
    })
    
    age_group_results.append({
        'age_group': 'older_adults',
        'variable_pair': 'Age_REMEMVR',
        'r': r_old_rememvr,
        'ci_lower': ci_old_rememvr[0],
        'ci_upper': ci_old_rememvr[1],
        'n_participants': len(df_older),
        'median_split': age_median
    })
    
    # 4. WINSORIZED CORRELATIONS
    log("\n" + "=" * 40)
    log("WINSORIZED CORRELATIONS (5% trim)")
    log("=" * 40)
    
    # Winsorize at 5% on each end
    age_wins = winsorize_data(df['Age'].values, 0.05)
    ravlt_wins = winsorize_data(df['RAVLT_Total'].values, 0.05)
    rememvr_wins = winsorize_data(df['theta_all'].values, 0.05)
    
    r_wins_ravlt, p_wins_ravlt = stats.pearsonr(age_wins, ravlt_wins)
    r_wins_rememvr, p_wins_rememvr = stats.pearsonr(age_wins, rememvr_wins)
    
    log(f"Winsorized correlations:")
    log(f"  Age-RAVLT: r = {r_wins_ravlt:.3f}, p = {p_wins_ravlt:.4f}")
    log(f"  Age-REMEMVR: r = {r_wins_rememvr:.3f}, p = {p_wins_rememvr:.4f}")
    
    # SENSITIVITY SUMMARY
    log("\n" + "=" * 40)
    log("SENSITIVITY ANALYSIS SUMMARY")
    log("=" * 40)
    
    # Check consistency across methods
    methods_supporting_hypothesis = 0
    
    # Original showed RAVLT decline > REMEMVR decline
    original_pattern = True  # From Step 3: RAVLT r=-0.292, REMEMVR r=-0.193
    
    # Check outlier exclusion
    if len(outlier_results) > 1 and not np.isnan(outlier_results[0]['r']):
        if abs(outlier_results[0]['r']) > abs(outlier_results[1]['r']):
            methods_supporting_hypothesis += 1
            log("✓ Outlier exclusion: Pattern maintained")
    
    # Check Spearman
    if abs(rs_age_ravlt) > abs(rs_age_rememvr):
        methods_supporting_hypothesis += 1
        log("✓ Spearman: Pattern maintained")
    
    # Check Winsorized
    if abs(r_wins_ravlt) > abs(r_wins_rememvr):
        methods_supporting_hypothesis += 1
        log("✓ Winsorized: Pattern maintained")
    
    log(f"\nSensitivity analyses: {methods_supporting_hypothesis}/3 methods support main conclusion")
    
    if methods_supporting_hypothesis >= 2:
        log("→ ROBUST: Main finding consistent across sensitivity analyses")
    else:
        log("→ MIXED: Some sensitivity to analytical choices")
    
    # Save results
    outlier_df = pd.DataFrame(outlier_results)
    spearman_df = pd.DataFrame(spearman_results)
    age_group_df = pd.DataFrame(age_group_results)
    
    outlier_df.to_csv(RQ_DIR / "data" / "step06_sensitivity_outliers.csv", index=False)
    spearman_df.to_csv(RQ_DIR / "data" / "step06_sensitivity_spearman.csv", index=False)
    age_group_df.to_csv(RQ_DIR / "data" / "step06_sensitivity_age_groups.csv", index=False)
    
    log(f"\nSaved sensitivity results to data folder")
    log("\nStep 6 completed successfully")

if __name__ == "__main__":
    main()