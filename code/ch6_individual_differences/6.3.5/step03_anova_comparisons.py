#!/usr/bin/env python3
"""anova_comparisons: Compare calibration groups on cognitive reserve indicators using one-way ANOVA"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_stats import one_way_anova_d068

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_anova_comparisons.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom ANOVA Function (handles signature mismatch from gcode_lessons.md)

def run_anova_d068_wrapper(df, group_col, dv_col, n_comparisons=6):
    """
    Wrapper for one_way_anova_d068 using actual signature.
    Based on gcode_lessons.md: Always use actual function signatures.
    """
    # Create groups list from DataFrame
    groups = []
    group_names = df[group_col].unique()
    for group_name in group_names:
        group_data = df[df[group_col] == group_name][dv_col].values
        groups.append(group_data)
    
    # Use actual function signature: groups, data, dv, between, correction, n_comparisons, post_hoc
    result = one_way_anova_d068(
        groups=groups,
        data=df,  # Full dataset for post-hoc
        dv=dv_col,
        between=group_col,
        correction='bonferroni',
        n_comparisons=n_comparisons,
        post_hoc='tukey'  # Changed from True to 'tukey' string
    )
    return result

def check_anova_assumptions(df, group_col, dv_col):
    """Check normality and homoscedasticity assumptions."""
    assumptions = []
    
    # Normality test per group (Shapiro-Wilk)
    groups = df[group_col].unique()
    normality_pvals = []
    
    for group in groups:
        group_data = df[df[group_col] == group][dv_col].dropna()
        if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
            stat, p_val = stats.shapiro(group_data)
            normality_pvals.append(p_val)
    
    # Overall normality test result
    min_normality_p = min(normality_pvals) if normality_pvals else 0.0
    assumptions.append({
        'DV': dv_col,
        'test': 'shapiro_wilk',
        'statistic': 'min_group_p',
        'p_value': min_normality_p,
        'assumption': 'normality',
        'status': 'pass' if min_normality_p > 0.05 else 'violation'
    })
    
    # Levene's test for homoscedasticity
    group_data = [df[df[group_col] == group][dv_col].dropna() for group in groups]
    levene_stat, levene_p = stats.levene(*group_data)
    
    assumptions.append({
        'DV': dv_col,
        'test': 'levene',
        'statistic': levene_stat,
        'p_value': levene_p,
        'assumption': 'homoscedasticity',
        'status': 'pass' if levene_p > 0.05 else 'violation'
    })
    
    return assumptions

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: ANOVA Comparisons")
        # Load Input Data

        log("Loading calibration groups data...")
        df = pd.read_csv(RQ_DIR / "data" / "step02_calibration_groups.csv")
        log(f"step02_calibration_groups.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['group', 'education', 'rpm', 'age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check group sizes
        group_counts = df['group'].value_counts()
        log(f"Group sizes: {dict(group_counts)}")
        for group, count in group_counts.items():
            if count < 15:
                log(f"Group {group} has only {count} participants (< 15 minimum)")
        # Run ANOVAs for Each Dependent Variable

        dependent_variables = ['education', 'rpm', 'age']
        n_comparisons = 6  # 3 ANOVAs + 3 correlations (from step04) = 6 total tests
        
        anova_results = []
        assumption_results = []
        posthoc_results = []

        for dv in dependent_variables:
            log(f"Running ANOVA for {dv}...")
            
            # Check assumptions first
            log(f"Checking assumptions for {dv}...")
            assumptions = check_anova_assumptions(df, 'group', dv)
            assumption_results.extend(assumptions)
            
            # Run ANOVA with wrapper
            result = run_anova_d068_wrapper(df, 'group', dv, n_comparisons)
            
            # Calculate effect size (eta-squared)
            if 'f_statistic' in result and 'df_between' in result and 'df_within' in result:
                f_stat = result['f_statistic']
                df_between = result['df_between']
                df_within = result['df_within']
                eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within)
            else:
                eta_squared = np.nan
            
            # Store ANOVA results
            anova_results.append({
                'DV': dv,
                'F_stat': result.get('f_statistic', np.nan),
                'p_uncorrected': result.get('p_uncorrected', np.nan),
                'p_bonferroni': result.get('p_corrected', np.nan),
                'eta_squared': eta_squared,
                'eta_ci_lower': np.nan,  # Would need bootstrap for CI
                'eta_ci_upper': np.nan
            })
            
            # Store post-hoc results if available
            if 'post_hoc_results' in result and result['post_hoc_results'] is not None:
                posthoc = result['post_hoc_results']
                if isinstance(posthoc, dict):
                    for comparison, comp_result in posthoc.items():
                        posthoc_results.append({
                            'DV': dv,
                            'comparison': comparison,
                            'mean_diff': comp_result.get('mean_diff', np.nan),
                            'se': comp_result.get('se', np.nan),
                            't_stat': comp_result.get('t_stat', np.nan),
                            'p_uncorrected': comp_result.get('p_uncorrected', np.nan),
                            'p_tukey': comp_result.get('p_tukey', np.nan),
                            'sig_uncorrected': comp_result.get('p_uncorrected', 1.0) < 0.05,
                            'sig_tukey': comp_result.get('p_tukey', 1.0) < 0.05
                        })
            
            # Log result safely
        f_val = result.get('f_statistic', 'N/A')
        p_val = result.get('p_uncorrected', 'N/A')
        if isinstance(f_val, (int, float)):
            log(f"ANOVA for {dv}: F = {f_val:.3f}, p = {p_val:.3f}")
        else:
            log(f"ANOVA for {dv}: F = {f_val}, p = {p_val}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 05 (effect sizes) and final reporting

        log("Saving ANOVA results...")
        
        # Save main ANOVA results
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(RQ_DIR / "data" / "step03_anova_results.csv", index=False, encoding='utf-8')
        log(f"step03_anova_results.csv ({len(anova_df)} rows, {len(anova_df.columns)} cols)")
        
        # Save assumption check results
        assumptions_df = pd.DataFrame(assumption_results)
        assumptions_df.to_csv(RQ_DIR / "data" / "step03_assumption_checks.csv", index=False, encoding='utf-8')
        log(f"step03_assumption_checks.csv ({len(assumptions_df)} rows, {len(assumptions_df.columns)} cols)")
        
        # Save post-hoc results
        if posthoc_results:
            posthoc_df = pd.DataFrame(posthoc_results)
        else:
            # Create empty DataFrame with expected columns if no post-hoc results
            posthoc_df = pd.DataFrame(columns=['DV', 'comparison', 'mean_diff', 'se', 't_stat', 'p_uncorrected', 'p_tukey', 'sig_uncorrected', 'sig_tukey'])
        
        posthoc_df.to_csv(RQ_DIR / "data" / "step03_posthoc_comparisons.csv", index=False, encoding='utf-8')
        log(f"step03_posthoc_comparisons.csv ({len(posthoc_df)} rows, {len(posthoc_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Dual p-value reporting compliance (Decision D068)
        # Threshold: alpha_bonferroni = 0.05/6 = 0.0083

        log("Running validate_hypothesis_test_dual_pvalues...")
        
        # Prepare data for validation (add 'term' column as expected by validator)
        anova_df_for_validation = anova_df.copy()
        anova_df_for_validation['term'] = anova_df_for_validation['DV']  # Compatibility fix from gcode_lessons.md
        
        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=anova_df_for_validation,
            required_terms=['education', 'rpm', 'age'],
            alpha_bonferroni=0.0083  # 0.05/6 comparisons
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary statistics
        log("ANOVA Results Summary:")
        for _, row in anova_df.iterrows():
            log(f"  {row['DV']}: F = {row['F_stat']:.3f}, p_uncorr = {row['p_uncorrected']:.3f}, p_bonf = {row['p_bonferroni']:.3f}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)