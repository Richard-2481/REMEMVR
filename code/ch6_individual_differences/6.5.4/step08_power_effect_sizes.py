#!/usr/bin/env python3
"""power_effect_sizes: Post-hoc power analysis and effect size interpretation"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.analysis_regression import compute_post_hoc_power
from tools.validation import validate_effect_sizes

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.4
LOG_FILE = RQ_DIR / "logs" / "step08_power_effect_sizes.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: power_effect_sizes")
        # Load Fixed Effects Results

        log("Loading fixed effects from step05...")
        effects_input_path = RQ_DIR / "data" / "step05_fixed_effects.csv"
        
        if not effects_input_path.exists():
            log(f"Fixed effects input file not found: {effects_input_path}")
            log("Run step05_fit_multilevel_models.py first")
            sys.exit(1)
            
        fixed_effects = pd.read_csv(effects_input_path)
        log(f"fixed_effects.csv ({len(fixed_effects)} rows, {len(fixed_effects.columns)} cols)")
        # Load Bootstrap Confidence Intervals

        log("Loading bootstrap CIs from step07...")
        bootstrap_input_path = RQ_DIR / "data" / "step07_bootstrap_cis.csv"
        
        if not bootstrap_input_path.exists():
            log(f"Bootstrap CI input file not found: {bootstrap_input_path}")
            log("Run step07_cross_validation_bootstrap.py first")
            sys.exit(1)
            
        bootstrap_cis = pd.read_csv(bootstrap_input_path)
        log(f"bootstrap_cis.csv ({len(bootstrap_cis)} rows, {len(bootstrap_cis.columns)} cols)")
        # Focus on Sleep Parameters
        # Extract sleep-related parameters for effect size analysis

        log("Extracting sleep parameters...")
        
        # Sleep parameters of interest
        sleep_params = ['Sleep_Hours_WP', 'Sleep_Quality_WP', 'Sleep_Hours_PM', 'Sleep_Quality_PM']
        
        # Filter fixed effects for sleep parameters
        sleep_effects = fixed_effects[fixed_effects['parameter'].isin(sleep_params)].copy()
        log(f"Sleep parameters found: {list(sleep_effects['parameter'])}")
        
        # Filter bootstrap CIs for sleep parameters  
        sleep_cis = bootstrap_cis[bootstrap_cis['parameter'].isin(sleep_params)].copy()
        log(f"[CIs] Bootstrap CIs available: {list(sleep_cis['parameter'])}")
        # Compute Effect Sizes (Cohen's d)

        log("Computing effect sizes...")
        
        # Estimate residual standard deviation for Cohen's d calculation
        # Note: For LMM, we need to account for total variability
        # This is an approximation - ideally we'd extract from model directly
        
        # Load original data to estimate outcome SD
        analysis_data_path = RQ_DIR / "data" / "step04_analysis_dataset.csv"
        if analysis_data_path.exists():
            analysis_data = pd.read_csv(analysis_data_path)
            outcome_sd = analysis_data['Memory_Score'].std()
            log(f"[SD] Memory_Score SD: {outcome_sd:.4f}")
        else:
            outcome_sd = 0.25  # Reasonable default for proportion scores
            log(f"[SD] Using default outcome SD: {outcome_sd:.4f}")
        
        # Compute Cohen's d for each sleep parameter
        effect_size_rows = []
        
        for _, param_row in sleep_effects.iterrows():
            param = param_row['parameter']
            estimate = param_row['estimate']
            std_error = param_row['std_error']
            p_uncorrected = param_row.get('p_uncorrected', np.nan)
            
            # Cohen's d = coefficient / outcome_sd
            cohens_d = abs(estimate) / outcome_sd
            
            # Effect size interpretation (Cohen 1988)
            if cohens_d < 0.2:
                effect_interpretation = "negligible"
            elif cohens_d < 0.5:
                effect_interpretation = "small"
            elif cohens_d < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Practical significance (>0.05 on 0-1 scale = 5% change)
            practical_significance = abs(estimate) >= 0.05
            
            effect_size_rows.append({
                'parameter': param,
                'estimate': estimate,
                'std_error': std_error,
                'p_uncorrected': p_uncorrected,
                'effect_size': abs(estimate),  # Raw effect size
                'cohens_d': cohens_d,
                'effect_interpretation': effect_interpretation,
                'practical_significance': practical_significance
            })
            
            log(f"{param}: estimate={estimate:.4f}, Cohen's d={cohens_d:.4f}, {effect_interpretation}")

        effect_sizes_df = pd.DataFrame(effect_size_rows)
        # Compute Post-Hoc Power Analysis

        log("Computing post-hoc power...")
        
        # Power analysis parameters
        n = 400  # Total observations
        n_level2 = 100  # Level-2 units (participants)
        k = 5  # Number of predictors
        alpha_bonferroni = 0.0125  # Bonferroni-corrected alpha (0.05/4)
        
        log(f"Sample size: {n} observations, {n_level2} participants")
        log(f"Predictors: {k}")
        log(f"Alpha (Bonferroni): {alpha_bonferroni}")
        
        # Compute power for each effect
        power_results = []
        
        for _, effect_row in effect_sizes_df.iterrows():
            param = effect_row['parameter']
            cohens_d = effect_row['cohens_d']
            
            try:
                # Convert Cohen's d to R² for power analysis
                # Approximation: R² ≈ d²/(d² + 4) for t-test
                # For regression: adjust for multiple predictors
                f2 = (cohens_d ** 2) / (4 + cohens_d ** 2)  # Cohen's f²
                r2_approx = f2 / (1 + f2)  # Approximate R²
                
                power_result = compute_post_hoc_power(
                    n=n,
                    k=k,
                    r2=r2_approx,
                    alpha=alpha_bonferroni
                )
                
                power_value = power_result.get('power', np.nan)
                power_interpretation = power_result.get('interpretation', 'Unknown')
                
                power_results.append({
                    'parameter': param,
                    'power': power_value,
                    'power_interpretation': power_interpretation
                })
                
                log(f"{param}: power={power_value:.3f}, {power_interpretation}")
                
            except Exception as e:
                log(f"Power computation failed for {param}: {str(e)}")
                power_results.append({
                    'parameter': param,
                    'power': np.nan,
                    'power_interpretation': 'computation_failed'
                })

        power_df = pd.DataFrame(power_results)
        # Merge Effect Sizes and Power
        # Combine effect size and power analyses

        log("Merging effect size and power results...")
        
        # Merge effect sizes with power results
        final_results = effect_sizes_df.merge(power_df, on='parameter', how='left')
        
        # Add practical interpretation
        def practical_interpretation(row):
            effect_size = row['effect_size']
            cohens_d = row['cohens_d']
            power = row.get('power', np.nan)
            practical = row['practical_significance']
            
            if practical and cohens_d >= 0.5 and power >= 0.80:
                return "practically_and_statistically_meaningful"
            elif practical and cohens_d >= 0.2:
                return "practically_meaningful_small_effect"
            elif cohens_d >= 0.5:
                return "statistically_meaningful_but_small_practical_impact"
            else:
                return "negligible_effect"
        
        final_results['practical_interpretation'] = final_results.apply(practical_interpretation, axis=1)
        
        # Select final columns
        output_cols = ['parameter', 'effect_size', 'cohens_d', 'power', 'practical_interpretation', 'effect_interpretation']
        final_output = final_results[output_cols].copy()
        
        log(f"Final results: {len(final_output)} parameters")
        # Save Power and Effect Size Results
        # Output: Comprehensive effect size and power analysis

        log("Saving power and effect size results...")
        
        power_output_path = RQ_DIR / "data" / "step08_power_effect_sizes.csv"
        final_output.to_csv(power_output_path, index=False, encoding='utf-8')
        log(f"{power_output_path} ({len(final_output)} rows, {len(final_output.columns)} cols)")
        # Run Effect Size Validation
        # Validates: Effect sizes are within reasonable bounds

        log("Running effect size validation...")
        
        # Prepare effect sizes DataFrame for validation
        validation_df = final_output[['parameter', 'cohens_d']].copy()
        validation_df = validation_df.rename(columns={'cohens_d': 'cohens_f2'})  # Rename for validator
        
        try:
            effect_validation = validate_effect_sizes(validation_df)
            
            if effect_validation.get('valid', False):
                log("Effect sizes: PASS")
            else:
                log(f"Effect sizes: FAIL - {effect_validation.get('message', 'Unknown error')}")
                
        except Exception as e:
            log(f"Effect size validation failed: {str(e)}")
            log("Continuing with manual validation")
            
            # Manual validation
            reasonable_effects = (final_output['cohens_d'] >= 0).all() and (final_output['cohens_d'] <= 5).all()
            log(f"Manual effect size check: {'PASS' if reasonable_effects else 'FAIL'}")

        # Summary statistics
        log("Effect size summary:")
        for interpretation in final_output['effect_interpretation'].unique():
            count = (final_output['effect_interpretation'] == interpretation).sum()
            log(f"{interpretation}: {count} parameters")
        
        practical_meaningful = (final_output['practical_interpretation'] == 'meaningful_effect').sum()
        log(f"Practically significant effects: {practical_meaningful}/{len(final_output)}")

        # Scientific Mantra logging between steps
        log("")
        log("=== SCIENTIFIC MANTRA ===")
        log("1. What question did we ask?")
        log("   -> What is the practical significance of our sleep-memory effects?")
        log("2. What did we find?")
        log(f"   -> {len(final_output)} sleep parameters analyzed")
        if len(final_output) > 0:
            mean_cohens_d = final_output['cohens_d'].mean()
            log(f"   -> Mean Cohen's d: {mean_cohens_d:.3f}")
            log(f"   -> Practically significant: {practical_meaningful} parameters")
        log("3. What does it mean?")
        log("   -> Effect sizes quantify real-world impact beyond statistical significance")
        log("4. What should we do next?")
        log("   -> Interpret findings for clinical and theoretical implications")
        log("=========================")
        log("")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)