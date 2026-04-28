#!/usr/bin/env python3
"""power_analysis: Post-hoc power analysis and sensitivity testing for regression analysis"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_post_hoc_power

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: power_analysis")
        # Load Input Data

        log("Loading model results...")
        models_df = pd.read_csv(RQ_DIR / "data/step04_regression_models.csv")
        log(f"step04_regression_models.csv ({len(models_df)} rows, {len(models_df.columns)} cols)")
        
        log("Loading coefficient results...")
        coeffs_df = pd.read_csv(RQ_DIR / "data/step04_coefficients_ci.csv")
        log(f"step04_coefficients_ci.csv ({len(coeffs_df)} rows, {len(coeffs_df.columns)} cols)")

        # Extract full model parameters for power analysis
        full_model = models_df[models_df['model'] == 'Full'].iloc[0]
        n = int(full_model['N'])
        k_predictors = 4  # Age, Education, VR_Experience, Typical_Sleep
        r2_observed = full_model['R2']
        
        log(f"Full model parameters: N={n}, k={k_predictors}, R2={r2_observed:.4f}")
        # Run Analysis Tool - Overall Model Power

        log("Computing post-hoc power for overall model...")
        
        # Power for uncorrected alpha (standard significance threshold)
        power_uncorrected = compute_post_hoc_power(
            n=n,
            k_predictors=k_predictors,
            r2=r2_observed,
            alpha=0.05  # Standard uncorrected alpha level
        )
        log(f"Uncorrected power (alpha=0.05): {power_uncorrected:.4f}")
        
        # Power for corrected alpha (Bonferroni correction for 3 main predictors)
        power_corrected = compute_post_hoc_power(
            n=n,
            k_predictors=k_predictors,
            r2=r2_observed,
            alpha=0.0167  # 0.05/3 for Bonferroni correction
        )
        log(f"Corrected power (alpha=0.0167): {power_corrected:.4f}")
        # Individual Predictor Power Analysis
        # Approach: Approximate power for individual predictors using t-statistics
        # Rationale: Individual predictor significance tests use t-distribution

        log("Computing approximate power for individual predictors...")
        
        from scipy import stats
        
        power_results = []
        
        # Overall model power (primary result)
        power_results.append({
            'test': 'Overall_model',
            'observed_effect': r2_observed,
            'alpha': 0.05,
            'power': power_uncorrected,
            'interpretation': 'Adequate' if power_uncorrected >= 0.8 else ('Marginal' if power_uncorrected >= 0.6 else 'Inadequate')
        })
        
        # Individual predictor power approximations for main predictors
        main_predictors = ['Education_z', 'VR_Experience_z', 'Typical_Sleep_z']
        
        for predictor in main_predictors:
            coef_row = coeffs_df[coeffs_df['predictor'] == predictor].iloc[0]
            
            # Extract coefficient statistics (using direct indexing as per lesson #17)
            beta = coef_row['beta']
            se = coef_row['se']
            p_uncorrected = coef_row['p_uncorrected']
            
            # Compute t-statistic
            t_stat = abs(beta / se)
            
            # Approximate effect size for individual predictor
            # Using standardized beta as approximate effect size
            effect_size = abs(beta)
            
            # Approximate power using t-distribution
            # For two-tailed test with corrected alpha
            alpha_corrected = 0.0167
            df = n - k_predictors - 1
            t_critical = stats.t.ppf(1 - alpha_corrected/2, df)
            
            # Power approximation: probability that |t| > t_critical given observed effect
            # Using non-central t-distribution approximation
            ncp = t_stat  # Non-centrality parameter approximation
            power_approx = 1 - stats.t.cdf(t_critical, df, loc=ncp) + stats.t.cdf(-t_critical, df, loc=ncp)
            
            power_results.append({
                'test': f'{predictor}_individual',
                'observed_effect': effect_size,
                'alpha': alpha_corrected,
                'power': power_approx,
                'interpretation': 'Adequate' if power_approx >= 0.8 else ('Marginal' if power_approx >= 0.6 else 'Inadequate')
            })
            
            log(f"{predictor} power: {power_approx:.4f} (effect size: {effect_size:.4f})")
        # Sensitivity Analysis - Minimum Detectable Effects

        log("Computing sensitivity analysis - minimum detectable effects...")
        
        from statsmodels.stats.power import FTestAnovaPower
        
        power_test = FTestAnovaPower()
        
        # Minimum Cohen's f² for 80% power at different alpha levels
        min_f2_uncorrected = power_test.solve_power(
            effect_size=None,  # Solve for this
            nobs=n,
            alpha=0.05,
            power=0.8
        )
        
        min_f2_corrected = power_test.solve_power(
            effect_size=None,  # Solve for this
            nobs=n,
            alpha=0.0167,
            power=0.8
        )
        
        # Convert Cohen's f² to R² for interpretation
        # R² = f² / (1 + f²)
        min_r2_uncorrected = min_f2_uncorrected / (1 + min_f2_uncorrected)
        min_r2_corrected = min_f2_corrected / (1 + min_f2_corrected)
        
        # Effect size interpretation (Cohen's conventions)
        def interpret_f2(f2):
            if f2 >= 0.35:
                return "Large"
            elif f2 >= 0.15:
                return "Medium"
            elif f2 >= 0.02:
                return "Small"
            else:
                return "Negligible"
        
        sensitivity_results = [
            {
                'alpha_level': 0.05,
                'min_f2': min_f2_uncorrected,
                'min_r2': min_r2_uncorrected,
                'min_effect_label': interpret_f2(min_f2_uncorrected)
            },
            {
                'alpha_level': 0.0167,
                'min_f2': min_f2_corrected,
                'min_r2': min_r2_corrected,
                'min_effect_label': interpret_f2(min_f2_corrected)
            }
        ]
        
        log(f"Minimum detectable f² (alpha=0.05): {min_f2_uncorrected:.4f} ({interpret_f2(min_f2_uncorrected)} effect)")
        log(f"Minimum detectable f² (alpha=0.0167): {min_f2_corrected:.4f} ({interpret_f2(min_f2_corrected)} effect)")
        # Save Analysis Outputs
        # These outputs will be used by: Final results reporting and interpretation

        log("Saving power analysis results...")
        # Output: step08_power_analysis.csv
        # Contains: Post-hoc power for overall model and individual predictors
        # Columns: ['test', 'observed_effect', 'alpha', 'power', 'interpretation']
        power_df = pd.DataFrame(power_results)
        power_df.to_csv(RQ_DIR / "data/step08_power_analysis.csv", index=False, encoding='utf-8')
        log(f"step08_power_analysis.csv ({len(power_df)} rows, {len(power_df.columns)} cols)")

        log("Saving sensitivity analysis results...")
        # Output: step08_sensitivity_analysis.csv
        # Contains: Minimum detectable effect sizes at 80% power for both alpha levels
        # Columns: ['alpha_level', 'min_f2', 'min_r2', 'min_effect_label']
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df.to_csv(RQ_DIR / "data/step08_sensitivity_analysis.csv", index=False, encoding='utf-8')
        log(f"step08_sensitivity_analysis.csv ({len(sensitivity_df)} rows, {len(sensitivity_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Power values are between 0 and 1 (valid probabilities)
        # Note: Based on lesson #16, using manual validation since custom range parameter not supported

        log("Running power value validation...")
        
        # Manual validation of power values (0-1 range)
        power_values = power_df['power']
        within_bounds = power_values.between(0, 1)
        validation_result = {
            'valid': within_bounds.all(),
            'out_of_bounds_count': (~within_bounds).sum(),
            'min_value': power_values.min(),
            'max_value': power_values.max()
        }

        # Report validation results
        if validation_result['valid']:
            log(f"PASS: All power values within valid range [0,1]")
            log(f"Range: [{validation_result['min_value']:.4f}, {validation_result['max_value']:.4f}]")
        else:
            log(f"FAIL: {validation_result['out_of_bounds_count']} power values out of bounds")
            log(f"Range: [{validation_result['min_value']:.4f}, {validation_result['max_value']:.4f}]")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)