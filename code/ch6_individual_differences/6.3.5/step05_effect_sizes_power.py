#!/usr/bin/env python3
"""effect_sizes_power: Compute effect sizes and post-hoc power analysis for group comparisons and correlations"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestPower

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_cohens_f2

from tools.validation import validate_effect_sizes

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_effect_sizes_power.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Effect Size and Power Functions

def compute_cohens_d_pairwise(group1_data, group2_data):
    """
    Compute Cohen's d for two groups with pooled standard deviation.
    """
    n1, n2 = len(group1_data), len(group2_data)
    mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
    var1, var2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_sd
    
    # Confidence interval using Hedges' g formula (approximate)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
    ci_lower = cohens_d - 1.96 * se_d
    ci_upper = cohens_d + 1.96 * se_d
    
    return cohens_d, ci_lower, ci_upper

def classify_cohens_d(d):
    """Classify Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'

def eta_squared_to_cohens_f(eta_squared):
    """Convert eta-squared to Cohen's f for power analysis."""
    return np.sqrt(eta_squared / (1 - eta_squared))

def compute_achieved_power_anova(f_stat, df_between, df_within, alpha=0.05):
    """Compute achieved power for ANOVA F-test."""
    try:
        # Use F-distribution to compute power
        critical_f = stats.f.ppf(1 - alpha, df_between, df_within)
        power = 1 - stats.f.cdf(critical_f, df_between, df_within, f_stat * df_between)
        return max(0.0, min(1.0, power))  # Bound between 0 and 1
    except:
        return np.nan

def compute_achieved_power_correlation(r, n, alpha=0.05):
    """Compute achieved power for correlation test."""
    try:
        # Convert correlation to t-statistic
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        critical_t = stats.t.ppf(1 - alpha/2, n - 2)  # Two-tailed
        power = 1 - (stats.t.cdf(critical_t, n - 2) - stats.t.cdf(-critical_t, n - 2))
        return max(0.0, min(1.0, power))
    except:
        return np.nan

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Effect Sizes and Power Analysis")
        # Load Input Data

        log("Loading ANOVA results...")
        anova_df = pd.read_csv(RQ_DIR / "data" / "step03_anova_results.csv")
        log(f"step03_anova_results.csv ({len(anova_df)} rows, {len(anova_df.columns)} cols)")
        
        log("Loading correlation results...")
        corr_df = pd.read_csv(RQ_DIR / "data" / "step04_correlations.csv")
        log(f"step04_correlations.csv ({len(corr_df)} rows, {len(corr_df.columns)} cols)")
        
        log("Loading raw group data...")
        groups_df = pd.read_csv(RQ_DIR / "data" / "step02_calibration_groups.csv")
        log(f"step02_calibration_groups.csv ({len(groups_df)} rows, {len(groups_df.columns)} cols)")
        # Compute Pairwise Effect Sizes (Cohen's d)

        log("Computing pairwise Cohen's d effect sizes...")
        
        dependent_variables = ['education', 'rpm', 'age']
        group_names = groups_df['group'].unique()
        group_pairs = []
        
        # Generate all pairwise combinations
        from itertools import combinations
        for g1, g2 in combinations(group_names, 2):
            group_pairs.append((g1, g2))
        
        effect_size_results = []
        
        for dv in dependent_variables:
            for group1, group2 in group_pairs:
                log(f"Computing Cohen's d: {group1} vs {group2} on {dv}...")
                
                # Get data for each group
                group1_data = groups_df[groups_df['group'] == group1][dv].dropna()
                group2_data = groups_df[groups_df['group'] == group2][dv].dropna()
                
                if len(group1_data) < 3 or len(group2_data) < 3:
                    log(f"Insufficient data for {group1} vs {group2} on {dv}")
                    continue
                
                # Compute Cohen's d
                cohens_d, ci_lower, ci_upper = compute_cohens_d_pairwise(group1_data, group2_data)
                classification = classify_cohens_d(cohens_d)
                
                effect_size_results.append({
                    'comparison': f"{group1}_vs_{group2}",
                    'dv': dv,
                    'cohens_d': cohens_d,
                    'd_ci_lower': ci_lower,
                    'd_ci_upper': ci_upper,
                    'classification': classification,
                    'n1': len(group1_data),
                    'n2': len(group2_data)
                })
                
                log(f"{group1} vs {group2} on {dv}: d = {cohens_d:.3f} ({classification})")
        # Compute Cohen's f² and Power Analysis

        log("Computing Cohen's f² and power analysis...")
        
        power_results = []
        alpha_corrected = 0.0083  # Bonferroni correction: 0.05/6
        
        # ANOVA effect sizes and power
        for _, row in anova_df.iterrows():
            dv = row['DV']
            eta_squared = row['eta_squared']
            f_stat = row['F_stat']
            
            if not np.isnan(eta_squared) and eta_squared > 0:
                # Convert eta² to Cohen's f using custom function (since tools function expects R²)
                cohens_f = eta_squared_to_cohens_f(eta_squared)
                
                # Compute achieved power (approximate)
                df_between = 2  # 3 groups - 1
                df_within = len(groups_df) - 3  # N - k
                achieved_power = compute_achieved_power_anova(f_stat, df_between, df_within, alpha_corrected)
                
                power_results.append({
                    'test_type': 'ANOVA',
                    'effect': dv,
                    'alpha_corrected': alpha_corrected,
                    'achieved_power': achieved_power,
                    'effect_size': cohens_f,
                    'n': len(groups_df)
                })
                
                log(f"ANOVA {dv}: f = {cohens_f:.3f}, power = {achieved_power:.3f}")
        
        # Correlation effect sizes and power
        for _, row in corr_df.iterrows():
            variable_pair = row['variable_pair']
            r = row['r']
            
            if not np.isnan(r):
                # For correlations, effect size IS the correlation coefficient
                # Compute achieved power
                n = len(groups_df)  # Sample size
                achieved_power = compute_achieved_power_correlation(r, n, alpha_corrected)
                
                power_results.append({
                    'test_type': 'Correlation',
                    'effect': variable_pair,
                    'alpha_corrected': alpha_corrected,
                    'achieved_power': achieved_power,
                    'effect_size': abs(r),  # Use absolute value for effect size
                    'n': n
                })
                
                log(f"Correlation {variable_pair}: r = {r:.3f}, power = {achieved_power:.3f}")
        # Save Analysis Outputs
        # These outputs will be used by: Final reporting and interpretation

        log("Saving effect sizes and power analysis...")
        
        # Save effect sizes (Cohen's d)
        if effect_size_results:
            effect_sizes_df = pd.DataFrame(effect_size_results)
        else:
            # Create empty DataFrame with expected columns if no results
            effect_sizes_df = pd.DataFrame(columns=['comparison', 'dv', 'cohens_d', 'd_ci_lower', 'd_ci_upper', 'classification', 'n1', 'n2'])
        
        effect_sizes_df.to_csv(RQ_DIR / "data" / "step05_effect_sizes.csv", index=False, encoding='utf-8')
        log(f"step05_effect_sizes.csv ({len(effect_sizes_df)} rows, {len(effect_sizes_df.columns)} cols)")
        
        # Save power analysis
        if power_results:
            power_df = pd.DataFrame(power_results)
        else:
            # Create empty DataFrame with expected columns if no results
            power_df = pd.DataFrame(columns=['test_type', 'effect', 'alpha_corrected', 'achieved_power', 'effect_size', 'n'])
        
        power_df.to_csv(RQ_DIR / "data" / "step05_power_analysis.csv", index=False, encoding='utf-8')
        log(f"step05_power_analysis.csv ({len(power_df)} rows, {len(power_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Effect size ranges and classifications
        # Threshold: Check that effect sizes are within reasonable bounds

        log("Running validate_effect_sizes...")
        
        # For validation, create a temporary DataFrame with f² column
        # Convert Cohen's d to approximate f² for validation
        if not effect_sizes_df.empty:
            effect_sizes_validation = effect_sizes_df.copy()
            effect_sizes_validation['cohens_f2'] = (effect_sizes_validation['cohens_d'] ** 2) / 4  # Approximate conversion
        else:
            effect_sizes_validation = pd.DataFrame({'cohens_f2': []})
        
        validation_result = validate_effect_sizes(
            effect_sizes_df=effect_sizes_validation,
            f2_column='cohens_f2'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary statistics
        log("Effect Size and Power Summary:")
        
        if not effect_sizes_df.empty:
            log("  Cohen's d Effect Sizes:")
            for classification in ['negligible', 'small', 'medium', 'large']:
                count = len(effect_sizes_df[effect_sizes_df['classification'] == classification])
                log(f"    {classification}: {count} comparisons")
        
        if not power_df.empty:
            log("  Achieved Power:")
            underpowered = len(power_df[power_df['achieved_power'] < 0.80])
            log(f"    Well-powered (>80%): {len(power_df) - underpowered} tests")
            log(f"    Underpowered (<80%): {underpowered} tests")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)