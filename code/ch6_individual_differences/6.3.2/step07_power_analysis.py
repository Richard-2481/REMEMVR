#!/usr/bin/env python3
"""Power Analysis: Post-hoc power analysis and effect size interpretation for the hierarchical"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2
LOG_FILE = RQ_DIR / "logs" / "step07_power_analysis.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom Power Analysis Functions (Due to Parameter Name Mismatch)

def compute_post_hoc_power_custom(n, k_predictors, r2, alpha=0.05):
    """
    Custom post-hoc power calculation using non-central F distribution.
    
    Parameters adjusted to match expected signature (k not k_predictors in tools).
    """
    
    # Cohen's f² effect size
    f2 = r2 / (1 - r2) if r2 < 1.0 else np.inf
    
    # Degrees of freedom
    df1 = k_predictors  # Numerator df (number of predictors)
    df2 = n - k_predictors - 1  # Denominator df
    
    # Non-centrality parameter
    ncp = f2 * n
    
    # Critical F value at alpha level
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Power calculation using non-central F distribution
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    
    # Handle edge cases
    if r2 <= 0 or n <= k_predictors + 1:
        power = 0.0
    elif power < 0:
        power = 0.0
    elif power > 1:
        power = 1.0
    
    return power

def interpret_effect_size(f2):
    """
    Interpret Cohen's f² effect size using conventional benchmarks.
    """
    if f2 < 0.02:
        return "negligible"
    elif f2 < 0.15:
        return "small"
    elif f2 < 0.35:
        return "medium" 
    else:
        return "large"

def assess_practical_significance(sr_squared, predictor_name):
    """
    Assess practical significance based on semi-partial correlation squared.
    
    Considers domain-specific expectations for cognitive predictors.
    """
    # Convert sr² to percentage of variance explained
    variance_explained_pct = sr_squared * 100
    
    # Domain-specific thresholds for cognitive predictors
    if "RAVLT" in predictor_name or "BVMT" in predictor_name or "RPM" in predictor_name:
        # Cognitive tests - more lenient thresholds
        if variance_explained_pct >= 5:
            return "high"
        elif variance_explained_pct >= 2:
            return "moderate"
        elif variance_explained_pct >= 0.5:
            return "low"
        else:
            return "negligible"
    else:
        # Demographics - stricter thresholds
        if variance_explained_pct >= 10:
            return "high"
        elif variance_explained_pct >= 5:
            return "moderate"
        elif variance_explained_pct >= 1:
            return "low"
        else:
            return "negligible"

def validate_numeric_range_custom(df, range_checks):
    """
    Custom numeric range validation for power analysis results.
    """
    validation_results = {}
    
    for column, (min_val, max_val) in range_checks.items():
        if column not in df.columns:
            validation_results[column] = {
                'valid': False,
                'message': f"Column '{column}' not found in data"
            }
            continue
        
        values = df[column].dropna()
        if len(values) == 0:
            validation_results[column] = {
                'valid': False,
                'message': f"No valid values in column '{column}'"
            }
            continue
        
        # Check range
        values_in_range = (values >= min_val) & (values <= max_val)
        n_valid = values_in_range.sum()
        n_total = len(values)
        
        proportion_valid = n_valid / n_total
        is_valid = proportion_valid >= 0.95  # 95% of values must be in range
        
        validation_results[column] = {
            'valid': is_valid,
            'message': f"{n_valid}/{n_total} values in range [{min_val}, {max_val}]",
            'min_observed': values.min(),
            'max_observed': values.max(),
            'proportion_valid': proportion_valid
        }
    
    return validation_results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Power Analysis")
        # Load Input Data

        log("Loading cross-validation and regression results...")
        
        # Load cross-validation results
        cv_results = pd.read_csv(RQ_DIR / "data" / "step06_cross_validation.csv")
        log(f"step06_cross_validation.csv ({len(cv_results)} rows)")
        
        # Load regression results
        regression_results = pd.read_csv(RQ_DIR / "data" / "step04_regression_results.csv")
        log(f"step04_regression_results.csv ({len(regression_results)} rows)")
        
        # Extract key parameters
        mean_test_r2 = cv_results['test_r2'].mean()
        n_observations = cv_results['n_test'].mean() + cv_results['n_train'].mean()  # Approximate total N
        k_predictors = len(regression_results)
        
        log(f"Mean test R² from CV: {mean_test_r2:.4f}")
        log(f"Approximate sample size: {n_observations:.0f}")
        log(f"Number of predictors: {k_predictors}")
        # Compute Post-hoc Power Analysis

        log("Computing post-hoc power analysis...")
        
        bonferroni_alpha = 0.000597  # 0.05/28 RQs/3 cognitive tests
        
        # Power for overall model
        overall_power = compute_post_hoc_power_custom(
            n=int(n_observations),
            k_predictors=k_predictors,
            r2=mean_test_r2,
            alpha=bonferroni_alpha
        )
        
        # Cohen's f² for overall model
        overall_f2 = mean_test_r2 / (1 - mean_test_r2) if mean_test_r2 < 1.0 else np.inf
        f2_interpretation = interpret_effect_size(overall_f2)
        
        log(f"Overall model power: {overall_power:.3f}")
        log(f"Overall model f²: {overall_f2:.3f} ({f2_interpretation})")
        
        # Create power analysis results
        power_analysis = pd.DataFrame([{
            'analysis': 'Overall_Model',
            'n': int(n_observations),
            'k': k_predictors,
            'r2': mean_test_r2,
            'alpha': bonferroni_alpha,
            'power': overall_power,
            'f2': overall_f2,
            'interpretation': f"Power={overall_power:.2f}, Effect={f2_interpretation}"
        }])
        
        # Power for incremental R² (cognitive predictors beyond demographics)
        # Approximate by assuming demographics explain ~30% of what the full model explains
        demographics_r2 = mean_test_r2 * 0.7  # Conservative estimate
        incremental_r2 = mean_test_r2 - demographics_r2
        k_incremental = 5  # RAVLT, BVMT, RPM, RAVLT_Pct_Ret, BVMT_Pct_Ret
        
        if incremental_r2 > 0:
            incremental_power = compute_post_hoc_power_custom(
                n=int(n_observations),
                k_predictors=k_incremental,
                r2=incremental_r2,
                alpha=bonferroni_alpha
            )
            
            incremental_f2 = incremental_r2 / (1 - mean_test_r2) if mean_test_r2 < 1.0 else np.inf
            incremental_interpretation = interpret_effect_size(incremental_f2)
            
            power_analysis = pd.concat([power_analysis, pd.DataFrame([{
                'analysis': 'Incremental_Cognitive',
                'n': int(n_observations),
                'k': k_incremental,
                'r2': incremental_r2,
                'alpha': bonferroni_alpha,
                'power': incremental_power,
                'f2': incremental_f2,
                'interpretation': f"Power={incremental_power:.2f}, Effect={incremental_interpretation}"
            }])], ignore_index=True)
            
            log(f"Incremental (cognitive) power: {incremental_power:.3f}")
            log(f"Incremental f²: {incremental_f2:.3f} ({incremental_interpretation})")

        log("Power analysis complete")
        # Effect Size Interpretation

        log("Interpreting individual predictor effect sizes...")
        
        effect_sizes = []
        
        for _, row in regression_results.iterrows():
            predictor = row['predictor']
            sr_squared = row['sr_squared']
            
            # Convert sr² to f² (approximate)
            # f² ≈ sr² / (1 - R²_total) for individual predictors
            predictor_f2 = sr_squared / (1 - mean_test_r2) if mean_test_r2 < 1.0 else sr_squared
            
            # Cohen's convention
            cohens_convention = interpret_effect_size(predictor_f2)
            
            # Practical significance
            practical_sig = assess_practical_significance(sr_squared, predictor)
            
            effect_sizes.append({
                'predictor': predictor,
                'sr_squared': sr_squared,
                'f2': predictor_f2,
                'cohens_convention': cohens_convention,
                'practical_significance': practical_sig
            })
            
            log(f"{predictor}: sr²={sr_squared:.4f}, f²={predictor_f2:.4f} ({cohens_convention}), practical={practical_sig}")
        
        effect_sizes_df = pd.DataFrame(effect_sizes)
        
        log("Effect size interpretation complete")
        # Optional Comparison with RQ 7.1.1

        log("Looking for RQ 7.1.1 comparison data...")
        
        comparison_pattern = str(PROJECT_ROOT / "results" / "ch7" / "7.1.1" / "data" / "*regression*.csv")
        comparison_files = glob.glob(comparison_pattern)
        
        if comparison_files:
            try:
                comparison_file = comparison_files[0]
                log(f"Found comparison file: {comparison_file}")
                
                comparison_df = pd.read_csv(comparison_file)
                if 'r2' in comparison_df.columns or 'rsquared' in comparison_df.columns:
                    r2_col = 'r2' if 'r2' in comparison_df.columns else 'rsquared'
                    comparison_r2 = comparison_df[r2_col].iloc[0] if len(comparison_df) > 0 else np.nan
                    
                    power_analysis = pd.concat([power_analysis, pd.DataFrame([{
                        'analysis': 'RQ_7.1.1_Comparison',
                        'n': np.nan,  # Not available from comparison
                        'k': np.nan,
                        'r2': comparison_r2,
                        'alpha': bonferroni_alpha,
                        'power': np.nan,  # Cannot compute without n, k
                        'f2': comparison_r2 / (1 - comparison_r2) if not pd.isna(comparison_r2) and comparison_r2 < 1 else np.nan,
                        'interpretation': f"Comparison R²={comparison_r2:.3f}" if not pd.isna(comparison_r2) else "No valid comparison"
                    }])], ignore_index=True)
                    
                    log(f"RQ 7.1.1 R²: {comparison_r2:.3f} vs current: {mean_test_r2:.3f}")
                else:
                    log("No R² data found in comparison file")
                    
            except Exception as e:
                log(f"Could not process comparison file: {e}")
        else:
            log("No RQ 7.1.1 comparison data available")
        # Save Power Analysis Outputs
        # These outputs provide final interpretation of statistical power and effect sizes

        log("Saving power analysis results...")
        
        # Output: step07_power_analysis.csv
        # Contains: Post-hoc power analysis and effect size interpretation
        power_analysis.to_csv(RQ_DIR / "data" / "step07_power_analysis.csv", index=False, encoding='utf-8')
        log(f"step07_power_analysis.csv ({len(power_analysis)} rows, {len(power_analysis.columns)} cols)")
        
        # Output: step07_effect_sizes.csv
        # Contains: Effect size interpretations with practical significance
        effect_sizes_df.to_csv(RQ_DIR / "data" / "step07_effect_sizes.csv", index=False, encoding='utf-8')
        log(f"step07_effect_sizes.csv ({len(effect_sizes_df)} rows, {len(effect_sizes_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Power values [0,1], f² values [0,10], R² values [0,1]
        # Criteria: All values within expected ranges for power analysis

        log("Running numeric range validation...")
        
        range_checks = {
            'power': [0, 1],
            'f2': [0, 10],
            'r2': [0, 1]
        }
        
        validation_result = validate_numeric_range_custom(power_analysis, range_checks)

        # Report validation results
        all_valid = all(result['valid'] for result in validation_result.values())
        log(f"All metrics in expected ranges: {all_valid}")
        
        for metric, result in validation_result.items():
            if result['valid']:
                log(f"{metric}: PASS - {result['message']}")
            else:
                log(f"{metric}: FAIL - {result['message']}")
        # Final Power and Effect Size Summary

        log("Final power and effect size assessment...")
        
        # Overall model assessment
        if overall_power >= 0.8:
            power_conclusion = "adequate"
        elif overall_power >= 0.6:
            power_conclusion = "marginal"
        else:
            power_conclusion = "inadequate"
        
        # Effect size assessment
        strong_effects = effect_sizes_df['cohens_convention'].isin(['medium', 'large']).sum()
        total_predictors = len(effect_sizes_df)
        
        log(f"Overall model power: {overall_power:.3f} ({power_conclusion} for α = {bonferroni_alpha})")
        log(f"Overall effect size: f² = {overall_f2:.3f} ({f2_interpretation})")
        log(f"Strong individual effects: {strong_effects}/{total_predictors} predictors")
        
        # Research implications
        if power_conclusion == "adequate" and f2_interpretation in ['medium', 'large']:
            log("Study has adequate power to detect meaningful effects")
        elif power_conclusion == "adequate" and f2_interpretation == 'small':
            log("Study has adequate power but effects are small - consider practical significance")
        elif power_conclusion == "marginal":
            log("Study has marginal power - interpret non-significant results cautiously")
        else:
            log("Study may be underpowered - consider larger sample size for future research")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)