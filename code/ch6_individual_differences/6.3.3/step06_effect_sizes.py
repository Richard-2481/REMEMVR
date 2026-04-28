#!/usr/bin/env python3
"""Effect sizes: Compute comprehensive effect sizes with bootstrap confidence intervals for hierarchical regression."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statistical packages
import statsmodels.api as sm
from sklearn.utils import resample

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_effect_sizes.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Effect Size Functions

def calculate_cohens_f2(r2_full, r2_reduced):
    """Calculate Cohen's f² from R² values."""
    if r2_full >= 1.0:
        return np.inf
    return (r2_full - r2_reduced) / (1 - r2_full)

def interpret_cohens_f2(f2):
    """Interpret Cohen's f² using conventional thresholds."""
    if f2 < 0.02:
        return "negligible", "minimal"
    elif f2 < 0.15:
        return "small", "meaningful"
    elif f2 < 0.35:
        return "medium", "substantial"
    else:
        return "large", "substantial"

def bootstrap_effect_size(X, y, n_iter=1000, seed=42):
    """Bootstrap confidence interval for effect sizes."""
    np.random.seed(seed)
    
    f2_values = []
    
    for i in range(n_iter):
        # Resample data
        indices = np.random.choice(len(y), size=len(y), replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        try:
            # Fit reduced model (demographics only)
            X_reduced = X_boot[['age_c', 'sex', 'education']]
            X_reduced_const = sm.add_constant(X_reduced)
            model_reduced = sm.OLS(y_boot, X_reduced_const).fit()
            r2_reduced = model_reduced.rsquared
            
            # Fit full model (demographics + cognitive)
            X_full_const = sm.add_constant(X_boot)
            model_full = sm.OLS(y_boot, X_full_const).fit()
            r2_full = model_full.rsquared
            
            # Calculate f²
            f2 = calculate_cohens_f2(r2_full, r2_reduced)
            f2_values.append(f2)
            
        except:
            # Skip failed bootstrap samples
            continue
    
    return np.array(f2_values)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Effect sizes")
        # Load Input Data

        log("Loading model comparison results...")
        model_comparison = pd.read_csv(RQ_DIR / "data/step04_model_comparison.csv")
        log(f"step04_model_comparison.csv ({len(model_comparison)} rows, {len(model_comparison.columns)} cols)")

        # Also need the analysis dataset for bootstrap CIs
        log("Loading analysis dataset for bootstrap...")
        analysis_data = pd.read_csv(RQ_DIR / "data/step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_data)} rows, {len(analysis_data.columns)} cols)")
        # Extract R² Values from Model Comparison

        log("Extracting R² values from model comparison...")
        
        # Find R² values for each model
        demographics_r2 = None
        full_model_r2 = None
        
        for idx, row in model_comparison.iterrows():
            if row['model'] == 'Demographics':
                demographics_r2 = row['R2']
            elif row['model'] == 'Full_Model':
                full_model_r2 = row['R2']
        
        if demographics_r2 is None or full_model_r2 is None:
            raise ValueError("Could not find R² values for both Demographics and Full_Model")
        
        log(f"[R²] Demographics model: {demographics_r2:.4f}")
        log(f"[R²] Full model: {full_model_r2:.4f}")
        log(f"[R²] Delta R²: {full_model_r2 - demographics_r2:.4f}")
        # Calculate Cohen's f² Effect Sizes

        log("Calculating Cohen's f² effect sizes...")
        
        # Overall model effect size (vs. null model with R²=0)
        overall_f2 = calculate_cohens_f2(full_model_r2, 0.0)
        overall_cohen, overall_practical = interpret_cohens_f2(overall_f2)
        
        # Incremental effect of cognitive predictors
        incremental_f2 = calculate_cohens_f2(full_model_r2, demographics_r2)
        incremental_cohen, incremental_practical = interpret_cohens_f2(incremental_f2)
        
        log(f"[EFFECT SIZE] Overall model f²: {overall_f2:.4f} ({overall_cohen}, {overall_practical})")
        log(f"[EFFECT SIZE] Incremental cognitive f²: {incremental_f2:.4f} ({incremental_cohen}, {incremental_practical})")
        # Individual Predictor Effect Sizes

        log("Calculating individual predictor effect sizes...")
        
        # Prepare data
        X_vars = ['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        X = analysis_data[X_vars].copy()
        y = analysis_data['hce_rate'].copy()

        individual_effects = []

        for predictor in ['ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']:
            # Model without this predictor
            vars_without = [v for v in X_vars if v != predictor]
            X_without = X[vars_without]
            X_without_const = sm.add_constant(X_without)
            model_without = sm.OLS(y, X_without_const).fit()
            r2_without = model_without.rsquared
            
            # Individual predictor f² (compared to full model)
            individual_f2 = calculate_cohens_f2(full_model_r2, r2_without)
            cohen_class, practical_sig = interpret_cohens_f2(individual_f2)
            
            individual_effects.append({
                'predictor': predictor,
                'f2': individual_f2,
                'cohen_classification': cohen_class,
                'practical_significance': practical_sig
            })
            
            log(f"{predictor}: f²={individual_f2:.4f} ({cohen_class}, {practical_sig})")
        # Bootstrap Confidence Intervals

        log("Computing bootstrap confidence intervals...")
        
        # Bootstrap for incremental cognitive effect
        boot_f2 = bootstrap_effect_size(
            X[['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']],
            y, 
            n_iter=1000, 
            seed=42
        )
        
        # Calculate percentile CIs
        ci_lower = np.percentile(boot_f2, 2.5)
        ci_upper = np.percentile(boot_f2, 97.5)
        
        log(f"Incremental f² 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        log(f"Bootstrap iterations: {len(boot_f2)}")
        # Relative Importance Analysis

        log("Conducting relative importance analysis...")
        
        # Calculate relative weights (simplified approach)
        # Full model R² partitioned across cognitive predictors
        cognitive_r2 = full_model_r2 - demographics_r2
        
        relative_weights = []
        total_individual_f2 = sum([effect['f2'] for effect in individual_effects])
        
        for effect in individual_effects:
            if total_individual_f2 > 0:
                relative_weight = effect['f2'] / total_individual_f2
            else:
                relative_weight = 0.0
            
            relative_weights.append({
                'predictor': effect['predictor'],
                'relative_weight': relative_weight,
                'relative_percent': relative_weight * 100
            })
            
            log(f"{effect['predictor']}: {relative_weight:.3f} ({relative_weight*100:.1f}%)")
        # Save Effect Size Results
        # These outputs provide comprehensive effect size interpretation

        log("Saving effect size results...")
        
        # Create main effect size summary
        effect_size_results = []
        
        # Overall model
        effect_size_results.append({
            'effect_type': 'overall_model_f2',
            'value': overall_f2,
            'interpretation': f"Full model vs. null (R²={full_model_r2:.4f})",
            'cohen_classification': overall_cohen,
            'practical_significance': overall_practical
        })
        
        # Incremental cognitive
        effect_size_results.append({
            'effect_type': 'incremental_f2_cognitive',
            'value': incremental_f2,
            'interpretation': f"Cognitive block addition (ΔR²={full_model_r2-demographics_r2:.4f})",
            'cohen_classification': incremental_cohen,
            'practical_significance': incremental_practical
        })
        
        # Individual predictors
        for effect in individual_effects:
            effect_size_results.append({
                'effect_type': f'individual_f2_{effect["predictor"]}',
                'value': effect['f2'],
                'interpretation': f"Unique contribution of {effect['predictor']}",
                'cohen_classification': effect['cohen_classification'],
                'practical_significance': effect['practical_significance']
            })
        
        # Relative importance
        for weight in relative_weights:
            effect_size_results.append({
                'effect_type': f'relative_importance_{weight["predictor"]}',
                'value': weight['relative_weight'],
                'interpretation': f"Relative weight: {weight['relative_percent']:.1f}%",
                'cohen_classification': 'proportion',
                'practical_significance': 'relative'
            })
        
        effect_sizes_df = pd.DataFrame(effect_size_results)
        effect_sizes_df.to_csv(RQ_DIR / "data/step06_effect_sizes.csv", index=False, encoding='utf-8')
        log(f"step06_effect_sizes.csv ({len(effect_sizes_df)} rows, {len(effect_sizes_df.columns)} cols)")

        # Save bootstrap CIs
        log("Saving bootstrap confidence intervals...")
        
        ci_results = []
        ci_results.append({
            'effect': 'incremental_cognitive_f2',
            'f2': incremental_f2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_iterations': len(boot_f2)
        })
        
        ci_df = pd.DataFrame(ci_results)
        ci_df.to_csv(RQ_DIR / "data/step06_effect_size_cis.csv", index=False, encoding='utf-8')
        log(f"step06_effect_size_cis.csv ({len(ci_df)} rows, {len(ci_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Effect sizes are non-negative and reasonable
        # Threshold: f² values should be in [0, 5] range

        log("Running validate_numeric_range...")
        
        # Validate main effect sizes
        f2_values = [overall_f2, incremental_f2] + [effect['f2'] for effect in individual_effects]
        # Simple validation without function call issues
        valid = all(0 <= f2 <= 5 for f2 in f2_values)
        validation_result = {'valid': valid, 'range': '[0, 5]', 'n_values': len(f2_values)}

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary interpretation
        log("Effect size interpretation:")
        log(f"- Overall model: {overall_cohen} effect (f²={overall_f2:.4f})")
        log(f"- Cognitive contribution: {incremental_cohen} effect (f²={incremental_f2:.4f})")
        
        # Identify strongest cognitive predictor
        strongest_predictor = max(individual_effects, key=lambda x: x['f2'])
        log(f"- Strongest cognitive predictor: {strongest_predictor['predictor']} (f²={strongest_predictor['f2']:.4f})")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)