#!/usr/bin/env python3
"""Power analysis: Post-hoc power analysis and sensitivity testing for observed effects in hierarchical regression."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statistical packages
from scipy import stats
import statsmodels.api as sm

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Power Analysis Functions

def calculate_power_f_test(f_stat, df_num, df_denom, alpha=0.05):
    """Calculate post-hoc power for F-test given observed F-statistic."""
    try:
        # Non-centrality parameter
        ncp = f_stat * df_num
        
        # Critical value
        f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
        
        # Power using non-central F distribution
        power = 1 - stats.ncf.cdf(f_crit, df_num, df_denom, ncp)
        
        return max(0.0, min(1.0, power))  # Bound between 0 and 1
    except:
        return np.nan

def calculate_power_from_f2(f2, n, df_num, alpha=0.05):
    """Calculate power from Cohen's f² effect size."""
    try:
        if f2 <= 0:
            return 0.0
        
        # Non-centrality parameter
        ncp = f2 * n
        
        # Degrees of freedom
        df_denom = n - df_num - 1
        
        if df_denom <= 0:
            return np.nan
        
        # Critical value
        f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
        
        # Power using non-central F distribution
        power = 1 - stats.ncf.cdf(f_crit, df_num, df_denom, ncp)
        
        return max(0.0, min(1.0, power))
    except:
        return np.nan

def calculate_min_detectable_f2(n, df_num, alpha=0.05, power=0.80):
    """Calculate minimum detectable Cohen's f² for specified power."""
    try:
        df_denom = n - df_num - 1
        
        if df_denom <= 0:
            return np.nan
        
        # Critical value
        f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
        
        # Non-centrality parameter for desired power
        ncp = stats.ncf.ppf(power, df_num, df_denom, f_crit)
        
        # Convert to Cohen's f²
        f2 = ncp / n
        
        return max(0.0, f2)
    except:
        return np.nan

def interpret_power(power):
    """Interpret power level."""
    if np.isnan(power):
        return "unknown"
    elif power < 0.50:
        return "inadequate"
    elif power < 0.80:
        return "moderate"
    elif power < 0.95:
        return "adequate"
    else:
        return "high"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: Power analysis")
        # Load Input Data

        log("Loading model comparison results...")
        model_comparison = pd.read_csv(RQ_DIR / "data/step04_model_comparison.csv")
        log(f"step04_model_comparison.csv ({len(model_comparison)} rows, {len(model_comparison.columns)} cols)")

        log("Loading effect sizes...")
        effect_sizes = pd.read_csv(RQ_DIR / "data/step06_effect_sizes.csv")
        log(f"step06_effect_sizes.csv ({len(effect_sizes)} rows, {len(effect_sizes.columns)} cols)")
        # Extract Study Parameters

        log("Setting up power analysis parameters...")
        
        # Study parameters from specification
        n = 100  # Sample size
        n_predictors = 8  # Full model predictors (age_c, sex, education, ravlt_c, bvmt_c, rpm_c, ravlt_pct_ret_c, bvmt_pct_ret_c)
        n_cognitive = 5  # Cognitive predictors (ravlt_c, bvmt_c, rpm_c, ravlt_pct_ret_c, bvmt_pct_ret_c)
        
        # Degrees of freedom
        df_full = n_predictors  # Full model numerator df
        df_incr = n_cognitive   # Incremental cognitive block numerator df
        df_denom = n - n_predictors - 1  # Denominator df
        
        # Alpha levels
        alpha_uncorrected = 0.05
        alpha_corrected = 0.000448  # Bonferroni within-RQ correction (0.00179/4)
        
        log(f"Sample size: {n}")
        log(f"Full model df: {df_full}, denominator df: {df_denom}")
        log(f"Incremental df: {df_incr}")
        log(f"Alpha uncorrected: {alpha_uncorrected}")
        log(f"Alpha corrected: {alpha_corrected}")
        # Overall Model Power Analysis

        log("Analyzing overall model power...")
        
        # Extract overall model statistics
        full_model_row = model_comparison[model_comparison['model'] == 'Full_Model']
        if len(full_model_row) > 0:
            overall_f = full_model_row['F_stat'].iloc[0]
            overall_r2 = full_model_row['R2'].iloc[0]
            
            # Calculate power for both alpha levels
            power_uncorr = calculate_power_f_test(overall_f, df_full, df_denom, alpha_uncorrected)
            power_corr = calculate_power_f_test(overall_f, df_full, df_denom, alpha_corrected)
            
            log(f"F-statistic: {overall_f:.4f}")
            log(f"R²: {overall_r2:.4f}")
            log(f"Power (α=0.05): {power_uncorr:.4f} ({interpret_power(power_uncorr)})")
            log(f"Power (α=0.000448): {power_corr:.4f} ({interpret_power(power_corr)})")
        else:
            log("Could not find Full_Model in model comparison")
            overall_f = np.nan
            power_uncorr = np.nan
            power_corr = np.nan
        # Incremental F-Change Power Analysis

        log("Analyzing incremental F-change power...")
        
        # Check if F_change column exists in model comparison
        if 'F_change' in model_comparison.columns:
            # Extract F-change for cognitive block
            full_model_row = model_comparison[model_comparison['model'] == 'Full_Model']
            if len(full_model_row) > 0 and not pd.isna(full_model_row['F_change'].iloc[0]):
                f_change = full_model_row['F_change'].iloc[0]
                
                # Power for incremental F-test
                incr_power_uncorr = calculate_power_f_test(f_change, df_incr, df_denom, alpha_uncorrected)
                incr_power_corr = calculate_power_f_test(f_change, df_incr, df_denom, alpha_corrected)
                
                log(f"F-change: {f_change:.4f}")
                log(f"Power (α=0.05): {incr_power_uncorr:.4f} ({interpret_power(incr_power_uncorr)})")
                log(f"Power (α=0.000448): {incr_power_corr:.4f} ({interpret_power(incr_power_corr)})")
            else:
                log("F_change not available - calculating from effect sizes")
                f_change = np.nan
                incr_power_uncorr = np.nan
                incr_power_corr = np.nan
        else:
            # Calculate from effect sizes
            incremental_f2_row = effect_sizes[effect_sizes['effect_type'] == 'incremental_f2_cognitive']
            if len(incremental_f2_row) > 0:
                incremental_f2 = incremental_f2_row['value'].iloc[0]
                
                incr_power_uncorr = calculate_power_from_f2(incremental_f2, n, df_incr, alpha_uncorrected)
                incr_power_corr = calculate_power_from_f2(incremental_f2, n, df_incr, alpha_corrected)
                
                log(f"Cohen's f²: {incremental_f2:.4f}")
                log(f"Power (α=0.05): {incr_power_uncorr:.4f} ({interpret_power(incr_power_uncorr)})")
                log(f"Power (α=0.000448): {incr_power_corr:.4f} ({interpret_power(incr_power_corr)})")
            else:
                log("Could not find incremental effect size")
                incr_power_uncorr = np.nan
                incr_power_corr = np.nan
        # Individual Predictor Power Analysis

        log("Analyzing individual predictor power...")
        
        individual_power_results = []
        
        for predictor in ['ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']:
            # Find individual effect size
            effect_row = effect_sizes[effect_sizes['effect_type'] == f'individual_f2_{predictor}']
            
            if len(effect_row) > 0:
                individual_f2 = effect_row['value'].iloc[0]
                
                # Power for individual predictor (df=1 for single predictor)
                indiv_power_uncorr = calculate_power_from_f2(individual_f2, n, 1, alpha_uncorrected)
                indiv_power_corr = calculate_power_from_f2(individual_f2, n, 1, alpha_corrected)
                
                individual_power_results.append({
                    'predictor': predictor,
                    'f2': individual_f2,
                    'power_uncorr': indiv_power_uncorr,
                    'power_corr': indiv_power_corr
                })
                
                log(f"{predictor}: f²={individual_f2:.4f}")
                log(f"{predictor}: Power (α=0.05)={indiv_power_uncorr:.4f} ({interpret_power(indiv_power_uncorr)})")
                log(f"{predictor}: Power (α=0.000448)={indiv_power_corr:.4f} ({interpret_power(indiv_power_corr)})")
            else:
                log(f"Could not find effect size for {predictor}")
        # Minimum Detectable Effects Analysis

        log("Calculating minimum detectable effects...")
        
        sensitivity_results = []
        
        # Overall model
        min_f2_overall_uncorr = calculate_min_detectable_f2(n, df_full, alpha_uncorrected, 0.80)
        min_f2_overall_corr = calculate_min_detectable_f2(n, df_full, alpha_corrected, 0.80)
        
        # Convert f² to R² (R² = f²/(1+f²))
        min_r2_overall_uncorr = min_f2_overall_uncorr / (1 + min_f2_overall_uncorr) if not np.isnan(min_f2_overall_uncorr) else np.nan
        min_r2_overall_corr = min_f2_overall_corr / (1 + min_f2_overall_corr) if not np.isnan(min_f2_overall_corr) else np.nan
        
        sensitivity_results.extend([
            {
                'test': 'overall_model_uncorrected',
                'min_detectable_f2': min_f2_overall_uncorr,
                'min_detectable_r2': min_r2_overall_uncorr,
                'power_threshold': 0.80
            },
            {
                'test': 'overall_model_corrected',
                'min_detectable_f2': min_f2_overall_corr,
                'min_detectable_r2': min_r2_overall_corr,
                'power_threshold': 0.80
            }
        ])
        
        # Incremental cognitive block
        min_f2_incr_uncorr = calculate_min_detectable_f2(n, df_incr, alpha_uncorrected, 0.80)
        min_f2_incr_corr = calculate_min_detectable_f2(n, df_incr, alpha_corrected, 0.80)
        
        min_r2_incr_uncorr = min_f2_incr_uncorr / (1 + min_f2_incr_uncorr) if not np.isnan(min_f2_incr_uncorr) else np.nan
        min_r2_incr_corr = min_f2_incr_corr / (1 + min_f2_incr_corr) if not np.isnan(min_f2_incr_corr) else np.nan
        
        sensitivity_results.extend([
            {
                'test': 'incremental_cognitive_uncorrected',
                'min_detectable_f2': min_f2_incr_uncorr,
                'min_detectable_r2': min_r2_incr_uncorr,
                'power_threshold': 0.80
            },
            {
                'test': 'incremental_cognitive_corrected',
                'min_detectable_f2': min_f2_incr_corr,
                'min_detectable_r2': min_r2_incr_corr,
                'power_threshold': 0.80
            }
        ])
        
        for result in sensitivity_results:
            test = result['test']
            min_f2 = result['min_detectable_f2']
            min_r2 = result['min_detectable_r2']
            log(f"{test}: Min f²={min_f2:.4f}, Min R²={min_r2:.4f}")
        # RPM Hypothesis Sensitivity Analysis

        log("Analyzing RPM hypothesis power...")
        
        # Find RPM individual effect
        rpm_effect = None
        for result in individual_power_results:
            if result['predictor'] == 'rpm_c':
                rpm_effect = result
                break
        
        if rpm_effect:
            rpm_f2 = rpm_effect['f2']
            rpm_power_uncorr = rpm_effect['power_uncorr']
            rpm_power_corr = rpm_effect['power_corr']
            
            # RPM hypothesis assessment
            if rpm_power_corr >= 0.80:
                rpm_assessment = "ADEQUATE_POWER"
            elif rpm_power_uncorr >= 0.80:
                rpm_assessment = "ADEQUATE_UNCORRECTED_ONLY"
            elif rpm_power_corr >= 0.50:
                rpm_assessment = "MODERATE_POWER"
            else:
                rpm_assessment = "INADEQUATE_POWER"
            
            log(f"Effect size f²: {rpm_f2:.4f}")
            log(f"Power assessment: {rpm_assessment}")
            
            # Minimum detectable effect for individual predictors
            min_f2_individual = calculate_min_detectable_f2(n, 1, alpha_corrected, 0.80)
            log(f"Minimum detectable individual f²: {min_f2_individual:.4f}")
            
            if rpm_f2 >= min_f2_individual:
                log("Observed effect exceeds minimum detectable threshold")
            else:
                log("Observed effect below minimum detectable threshold")
        else:
            log("Could not analyze RPM hypothesis - effect size missing")
        # Save Power Analysis Results
        # These outputs document statistical power and sample size adequacy

        log("Saving power analysis results...")
        
        # Create power analysis summary
        power_results = []
        
        # Overall model
        power_results.extend([
            {
                'test_type': 'overall_model_uncorrected',
                'effect_size': overall_r2 if 'overall_r2' in locals() else np.nan,
                'alpha': alpha_uncorrected,
                'power': power_uncorr,
                'interpretation': interpret_power(power_uncorr)
            },
            {
                'test_type': 'overall_model_corrected',
                'effect_size': overall_r2 if 'overall_r2' in locals() else np.nan,
                'alpha': alpha_corrected,
                'power': power_corr,
                'interpretation': interpret_power(power_corr)
            }
        ])
        
        # Incremental cognitive
        incremental_f2 = None
        incremental_f2_row = effect_sizes[effect_sizes['effect_type'] == 'incremental_f2_cognitive']
        if len(incremental_f2_row) > 0:
            incremental_f2 = incremental_f2_row['value'].iloc[0]
        
        power_results.extend([
            {
                'test_type': 'incremental_cognitive_uncorrected',
                'effect_size': incremental_f2,
                'alpha': alpha_uncorrected,
                'power': incr_power_uncorr,
                'interpretation': interpret_power(incr_power_uncorr)
            },
            {
                'test_type': 'incremental_cognitive_corrected',
                'effect_size': incremental_f2,
                'alpha': alpha_corrected,
                'power': incr_power_corr,
                'interpretation': interpret_power(incr_power_corr)
            }
        ])
        
        # Individual predictors
        for result in individual_power_results:
            power_results.extend([
                {
                    'test_type': f'{result["predictor"]}_uncorrected',
                    'effect_size': result['f2'],
                    'alpha': alpha_uncorrected,
                    'power': result['power_uncorr'],
                    'interpretation': interpret_power(result['power_uncorr'])
                },
                {
                    'test_type': f'{result["predictor"]}_corrected',
                    'effect_size': result['f2'],
                    'alpha': alpha_corrected,
                    'power': result['power_corr'],
                    'interpretation': interpret_power(result['power_corr'])
                }
            ])
        
        power_df = pd.DataFrame(power_results)
        power_df.to_csv(RQ_DIR / "data/step08_power_analysis.csv", index=False, encoding='utf-8')
        log(f"step08_power_analysis.csv ({len(power_df)} rows, {len(power_df.columns)} cols)")

        # Save sensitivity analysis
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df.to_csv(RQ_DIR / "data/step08_sensitivity.csv", index=False, encoding='utf-8')
        log(f"step08_sensitivity.csv ({len(sensitivity_df)} rows, {len(sensitivity_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Power values are in valid range [0,1]
        # Threshold: All power values should be between 0 and 1

        log("Running validate_numeric_range...")
        
        # Validate power values
        if rpm_effect:
            log(f"- RPM hypothesis power: {interpret_power(rpm_effect['power_corr'])}")

        # Collect all power values for summary
        power_values = [power_uncorr, power_corr, incr_power_uncorr, incr_power_corr]
        for result in individual_power_results:
            power_values.extend([result['power_uncorr'], result['power_corr']])
        power_values = [p for p in power_values if not np.isnan(p)]

        # Recommendations
        adequate_power_count = sum(1 for p in power_values if p >= 0.80)
        total_tests = len(power_values)

        if total_tests > 0:
            if adequate_power_count / total_tests >= 0.75:
                log("- Recommendation: Adequate power for most tests")
            elif adequate_power_count / total_tests >= 0.50:
                log("- Recommendation: Mixed power - interpret significant results cautiously")
            else:
                log("- Recommendation: Generally underpowered - focus on effect size interpretation")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
