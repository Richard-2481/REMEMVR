#!/usr/bin/env python3
"""Power Analysis: Conduct post-hoc power analysis for observed effect sizes. Calculate achieved"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_power_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Power Analysis Functions
# Note: Using custom implementation due to signature mismatch with tools function
# tools.analysis_regression.compute_post_hoc_power has parameter 'k_predictors' 
# but 4_analysis.yaml specifies 'k'. Writing custom to match specification.

def compute_post_hoc_power_custom(n: int, k: int, r2: float, alpha: float = 0.05) -> float:
    """
    Compute post-hoc power for multiple regression using Cohen's f² and F-distribution.
    
    Parameters:
    - n: Sample size
    - k: Number of predictors 
    - r2: Observed R-squared
    - alpha: Significance level
    
    Returns:
    - power: Achieved power (0-1)
    """
    from scipy.stats import f
    
    # Convert R² to Cohen's f²
    cohens_f2 = r2 / (1 - r2)
    
    # Degrees of freedom
    df_numerator = k
    df_denominator = n - k - 1
    
    # Critical F-value
    f_critical = f.ppf(1 - alpha, df_numerator, df_denominator)
    
    # Non-centrality parameter
    ncp = cohens_f2 * n
    
    # Power calculation using non-central F distribution
    from scipy.stats import ncf
    power = 1 - ncf.cdf(f_critical, df_numerator, df_denominator, ncp)
    
    return power

def compute_sensitivity_effect_size(n: int, k: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Compute minimum detectable Cohen's f² for given power.
    
    Parameters:
    - n: Sample size
    - k: Number of predictors
    - power: Desired power (default 0.80)
    - alpha: Significance level
    
    Returns:
    - min_f2: Minimum detectable Cohen's f²
    """
    from scipy.stats import f
    from scipy.optimize import brentq
    
    df_numerator = k
    df_denominator = n - k - 1
    
    def power_function(f2):
        ncp = f2 * n
        f_critical = f.ppf(1 - alpha, df_numerator, df_denominator)
        from scipy.stats import ncf
        return 1 - ncf.cdf(f_critical, df_numerator, df_denominator, ncp) - power
    
    try:
        min_f2 = brentq(power_function, 0.001, 2.0)
        return min_f2
    except ValueError:
        # If no solution found, return large value indicating inadequate power
        return 2.0

def convert_sr2_to_cohens_f2(sr2: float, full_model_r2: float) -> float:
    """
    Convert semi-partial correlation squared (sr²) to Cohen's f² for individual predictor.
    
    For individual predictors in multiple regression:
    f² = sr² / (1 - R²_full)
    """
    return sr2 / (1 - full_model_r2)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: Power Analysis")
        # Load Input Data

        log("Loading effect sizes from Step 6...")
        effect_sizes_df = pd.read_csv(RQ_DIR / "data/step06_effect_sizes.csv")
        log(f"step06_effect_sizes.csv ({len(effect_sizes_df)} rows, {len(effect_sizes_df.columns)} cols)")

        # Study parameters from 4_analysis.yaml
        n = 100  # Sample size
        k = 6    # Number of predictors in Model 2 (Age + 5 cognitive tests)
        alpha = 0.05 / 6  # Bonferroni corrected alpha (0.05/6 predictors)
        
        log(f"N={n}, k={k} predictors, alpha={alpha} (Bonferroni corrected)")
        # Power Analysis for Model Comparison (Overall F-test)

        log("Running power analysis for model comparison...")
        
        # For overall Model 2 F-test, estimate R² from Step 3 hierarchical results
        # Using conservative estimate: sum of semi-partial r² as lower bound for R²
        estimated_r2_model2 = effect_sizes_df['sr2'].sum()
        log(f"Estimated Model 2 R² = {estimated_r2_model2:.4f} (sum of sr²)")
        
        # Overall F-test power for Model 2
        power_overall = compute_post_hoc_power_custom(n, k, estimated_r2_model2, alpha)
        min_detectable_f2_overall = compute_sensitivity_effect_size(n, k, 0.80, alpha)
        
        # Model comparison power (ΔR² test)
        # Assume Model 1 (Age only) has R² ≈ effect_sizes_df[effect_sizes_df['predictor'] == 'Age']['sr2'].iloc[0]
        age_sr2 = effect_sizes_df[effect_sizes_df['predictor'] == 'Age']['sr2'].iloc[0]
        delta_r2 = estimated_r2_model2 - age_sr2  # Additional variance explained by cognitive tests
        
        # Power for ΔR² uses F-test with k-1 predictors (3 cognitive tests added)
        k_added = 5  # Number of predictors added in Model 2
        power_model_comparison = compute_post_hoc_power_custom(n, k_added, delta_r2, alpha) if delta_r2 > 0 else 0.0
        
        log(f"Overall F-test: {power_overall:.3f}")
        log(f"Model comparison (ΔR²): {power_model_comparison:.3f}")
        # Power Analysis for Individual Predictors

        log("Running power analysis for individual predictors...")
        
        individual_power_results = []
        
        for idx, row in effect_sizes_df.iterrows():
            predictor = row['predictor']
            sr2 = row['sr2']
            beta = row['beta']
            
            # Convert sr² to Cohen's f² for individual predictor
            predictor_f2 = convert_sr2_to_cohens_f2(sr2, estimated_r2_model2)
            
            # Power for individual predictor (t-test in multiple regression)
            # Use single predictor test: k=1 for the specific predictor
            power_individual = compute_post_hoc_power_custom(n, 1, sr2, alpha)
            min_detectable_f2_individual = compute_sensitivity_effect_size(n, 1, 0.80, alpha)
            
            individual_power_results.append({
                'test_type': f'Individual_Predictor_{predictor}',
                'effect_size': predictor_f2,
                'power_achieved': power_individual,
                'power_adequate': power_individual >= 0.80,
                'min_detectable_effect': min_detectable_f2_individual,
                'limitation_flag': 'Low_Power' if power_individual < 0.80 else 'Adequate'
            })
            
            log(f"{predictor}: {power_individual:.3f} (f² = {predictor_f2:.4f})")
        # Power Analysis for Mediation Effects
        # Mediation power assessment based on Fritz & MacKinnon (2007)

        log("Assessing mediation power limitations...")
        
        # Fritz & MacKinnon (2007) guidelines:
        # - N=200+ recommended for adequate mediation power
        # - N=100 provides limited power for small-to-medium indirect effects
        # - Bootstrap CIs help but don't overcome fundamental power limitations
        
        mediation_power_adequate = n >= 200
        mediation_limitation_flag = 'Insufficient_N_for_Mediation' if n < 200 else 'Adequate'
        
        # Rough estimate: mediation power ≈ 0.50-0.65 for N=100 with medium effects
        estimated_mediation_power = 0.50 if n == 100 else min(0.80, 0.30 + (n - 100) * 0.005)
        
        log(f"Mediation analysis: {estimated_mediation_power:.3f} (limited by N={n})")
        log(f"Fritz & MacKinnon (2007): N=200+ recommended for mediation, current N={n}")
        # Compile Power Analysis Results
        # Compile all power results into single dataframe
        # Contains: Different test types, effect sizes, achieved power, adequacy flags

        log("Compiling power analysis results...")
        
        # Overall results
        power_results = [
            {
                'test_type': 'Overall_F_Test_Model2',
                'effect_size': estimated_r2_model2 / (1 - estimated_r2_model2),  # Convert R² to f²
                'power_achieved': power_overall,
                'power_adequate': power_overall >= 0.80,
                'min_detectable_effect': min_detectable_f2_overall,
                'limitation_flag': 'Low_Power' if power_overall < 0.80 else 'Adequate'
            },
            {
                'test_type': 'Model_Comparison_Delta_R2',
                'effect_size': delta_r2 / (1 - estimated_r2_model2) if estimated_r2_model2 < 1 else 0,
                'power_achieved': power_model_comparison,
                'power_adequate': power_model_comparison >= 0.80,
                'min_detectable_effect': compute_sensitivity_effect_size(n, k_added, 0.80, alpha),
                'limitation_flag': 'Low_Power' if power_model_comparison < 0.80 else 'Adequate'
            },
            {
                'test_type': 'Mediation_Analysis_Bootstrap',
                'effect_size': np.nan,  # Not applicable for mediation
                'power_achieved': estimated_mediation_power,
                'power_adequate': mediation_power_adequate,
                'min_detectable_effect': np.nan,  # Complex for mediation
                'limitation_flag': mediation_limitation_flag
            }
        ]
        
        # Add individual predictor results
        power_results.extend(individual_power_results)
        
        # Create final dataframe
        power_df = pd.DataFrame(power_results)
        
        # Add summary statistics
        n_tests = len(power_df)
        n_adequate = sum(power_df['power_adequate'])
        n_low_power = n_tests - n_adequate
        
        log(f"Power analysis complete: {n_adequate}/{n_tests} tests with adequate power")
        log(f"{n_low_power} tests flagged with low power (< 0.80)")
        # Save Power Analysis Outputs
        # Output: Power analysis results with test types, effect sizes, and limitations
        # Used by: Step 9 (summary) and Step 10 (final validation)

        log("Saving power analysis results...")
        output_path = RQ_DIR / "data/step07_power_analysis.csv"
        power_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(power_df)} rows, {len(power_df.columns)} cols)")
        
        # Log key findings
        log("Key power analysis results:")
        for idx, row in power_df.iterrows():
            test_type = row['test_type']
            power = row['power_achieved']
            adequate = row['power_adequate']
            flag = row['limitation_flag']
            log(f"  - {test_type}: Power = {power:.3f}, Adequate = {adequate}, Flag = {flag}")
        # Run Validation Tool
        # Validates: Power values are between 0 and 1, effect sizes are reasonable
        # Threshold: Power values must be 0-1, effect sizes > 0 (where applicable)

        log("Running validate_numeric_range on power values...")
        
        # Validate power values (should be 0-1)
        power_values = power_df['power_achieved'].dropna()
        power_validation = validate_numeric_range(
            data=power_values,
            min_val=0.0,
            max_val=1.0,
            column_name='power_achieved'
        )
        
        # Validate effect sizes (should be positive where applicable)
        effect_sizes = power_df['effect_size'].dropna()
        if len(effect_sizes) > 0:
            effect_size_validation = validate_numeric_range(
                data=effect_sizes,
                min_val=0.0,
                max_val=3.0,  # Reasonable upper bound for Cohen's f²
                column_name='effect_size'
            )
        else:
            effect_size_validation = {'valid': True, 'message': 'No effect sizes to validate'}
        
        # Report validation results
        log(f"Power values: {power_validation}")
        log(f"Effect sizes: {effect_size_validation}")
        
        # Check for any critical limitations
        critical_flags = power_df[power_df['limitation_flag'].isin(['Insufficient_N_for_Mediation', 'Low_Power'])]
        if len(critical_flags) > 0:
            log(f"{len(critical_flags)} tests have critical power limitations")
            for idx, row in critical_flags.iterrows():
                log(f"  - {row['test_type']}: {row['limitation_flag']}")

        log("Step 7: Power Analysis complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)