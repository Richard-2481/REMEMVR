#!/usr/bin/env python3
"""effect_size_analysis: Compute comprehensive effect sizes with bootstrap confidence intervals including"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_cohens_f2

from tools.validation import validate_effect_sizes

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_effect_size_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Effect Size Analysis")
        # Load Input Data

        log("Loading input data...")
        
        # Load model comparison results
        # Expected columns: ['model', 'R2', 'adj_R2', 'F_stat', 'F_p', 'AIC', 'BIC', 'N']
        # Expected rows: 2 (Control and Full models)
        models_df = pd.read_csv(RQ_DIR / "data/step04_regression_models.csv")
        log(f"step04_regression_models.csv ({len(models_df)} models, {len(models_df.columns)} cols)")
        
        # Load coefficient results with CIs
        # Expected columns: ['predictor', 'beta', 'se', 'ci_lower', 'ci_upper', 'p_uncorrected', 'p_bonferroni', 'p_fdr']
        # Expected rows: 4 (Age_z, Education_z, VR_Experience_z, Typical_Sleep_z)
        coeffs_df = pd.read_csv(RQ_DIR / "data/step04_coefficients_ci.csv")
        log(f"step04_coefficients_ci.csv ({len(coeffs_df)} predictors, {len(coeffs_df.columns)} cols)")
        
        # Load analysis dataset for bootstrap
        # Expected columns: ['UID', 'theta_all', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z', 'Age_z'] 
        # Expected rows: ~100 complete cases
        analysis_df = pd.read_csv(RQ_DIR / "data/step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} participants, {len(analysis_df.columns)} cols)")
        # Run Analysis Tool - Compute Cohen's f²

        log("Computing Cohen's f² effect size...")
        
        # Extract R² values from control and full models
        r2_control = models_df[models_df['model'] == 'Control']['R2'].iloc[0]
        r2_full = models_df[models_df['model'] == 'Full']['R2'].iloc[0]
        
        log(f"Control model R² = {r2_control:.4f}")
        log(f"Full model R² = {r2_full:.4f}")
        
        # Compute Cohen's f² using analysis tool
        cohens_f2 = compute_cohens_f2(r2_full=r2_full, r2_reduced=r2_control)
        
        # Compute f² change (additional measure)
        f2_change = (r2_full - r2_control) / (1 - r2_full) if r2_full < 1.0 else float('inf')
        
        log(f"Cohen's f² = {cohens_f2:.4f}")
        log(f"f² change = {f2_change:.4f}")
        # Bootstrap Effect Sizes (1000 iterations, seed=42)

        log("Computing bootstrap confidence intervals (1000 iterations, seed=42)...")
        
        # Effect size interpretation function
        def interpret_f2(f2_value):
            """Interpret Cohen's f² according to Cohen (1988) guidelines."""
            if f2_value >= 0.35:
                return "Large"
            elif f2_value >= 0.15:
                return "Medium" 
            elif f2_value >= 0.02:
                return "Small"
            else:
                return "Negligible"
        
        # Bootstrap parameters
        n_bootstrap = 1000
        np.random.seed(42)  # Reproducibility
        
        # Storage for bootstrap results
        r2_bootstrap = []
        f2_bootstrap = []
        
        # Prepare predictor matrix and outcome vector
        predictor_cols = ['Age_z', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z']
        
        for iteration in range(n_bootstrap):
            try:
                # Resample participants with replacement (bootstrap sample)
                boot_idx = np.random.choice(len(analysis_df), len(analysis_df), replace=True)
                boot_data = analysis_df.iloc[boot_idx]
                
                # Prepare data for regression
                X = boot_data[predictor_cols].values
                X_with_const = sm.add_constant(X)  # Add intercept
                y = boot_data['theta_all'].values
                
                # Fit full model
                model = sm.OLS(y, X_with_const).fit()
                
                # Store bootstrap statistics
                r2_boot = model.rsquared
                f2_boot = r2_boot / (1 - r2_boot) if r2_boot < 1.0 else float('inf')
                
                r2_bootstrap.append(r2_boot)
                f2_bootstrap.append(f2_boot)
                
            except Exception as e:
                # Skip failed bootstrap iterations (e.g., singular matrix)
                log(f"Skipped iteration {iteration + 1}: {str(e)}")
                continue
        
        # Calculate 95% confidence intervals
        if len(r2_bootstrap) >= 50:  # Minimum valid bootstrap samples
            r2_ci = np.percentile(r2_bootstrap, [2.5, 97.5])
            f2_ci = np.percentile(f2_bootstrap, [2.5, 97.5])
            bootstrap_success = True
            log(f"Completed: {len(r2_bootstrap)} valid samples")
        else:
            r2_ci = [np.nan, np.nan]
            f2_ci = [np.nan, np.nan] 
            bootstrap_success = False
            log(f"WARNING: Only {len(r2_bootstrap)} valid samples - insufficient for CIs")
        # Save Effect Size Analysis Outputs
        # These outputs will be used by: Final interpretation and relative importance analysis

        # Prepare effect sizes with interpretation
        effect_sizes = [
            {
                'measure': 'R2_full',
                'value': r2_full,
                'ci_lower': r2_ci[0] if bootstrap_success else np.nan,
                'ci_upper': r2_ci[1] if bootstrap_success else np.nan,
                'interpretation': f"R² = {r2_full:.3f}",
                'bootstrap_se': np.std(r2_bootstrap) if bootstrap_success else np.nan
            },
            {
                'measure': 'Cohens_f2',
                'value': cohens_f2,
                'ci_lower': f2_ci[0] if bootstrap_success else np.nan,
                'ci_upper': f2_ci[1] if bootstrap_success else np.nan,
                'interpretation': interpret_f2(cohens_f2),
                'bootstrap_se': np.std(f2_bootstrap) if bootstrap_success else np.nan
            },
            {
                'measure': 'f2_change',
                'value': f2_change,
                'ci_lower': np.nan,  # Not bootstrapped separately
                'ci_upper': np.nan,
                'interpretation': interpret_f2(f2_change),
                'bootstrap_se': np.nan
            }
        ]
        
        # Save effect sizes
        log(f"Saving effect sizes...")
        effect_sizes_df = pd.DataFrame(effect_sizes)
        effect_sizes_df.to_csv(RQ_DIR / "data/step06_effect_sizes.csv", index=False, encoding='utf-8')
        log(f"step06_effect_sizes.csv ({len(effect_sizes_df)} measures, {len(effect_sizes_df.columns)} cols)")
        # Compute Relative Importance (Semi-Partial Correlations)

        log("Computing relative importance via semi-partial correlations...")
        
        # Focus on main predictors (exclude Age_z control variable)
        main_predictors = coeffs_df[coeffs_df['predictor'].isin(['Education_z', 'VR_Experience_z', 'Typical_Sleep_z'])]
        
        # Compute semi-partial r² for each predictor  
        relative_importance = []
        for _, row in main_predictors.iterrows():
            # Semi-partial r² approximation from standardized beta
            # sr² ≈ β² × (1 - R²_full) for standardized predictors
            sr2 = (row['beta'] ** 2) * (1 - r2_full)
            
            relative_importance.append({
                'predictor': row['predictor'],
                'sr2': sr2,
                'rank': 0,  # Will be filled after sorting
                'percent_of_r2': (sr2 / r2_full * 100) if r2_full > 0 else 0.0
            })
        
        # Create DataFrame and rank by semi-partial r²
        rel_imp_df = pd.DataFrame(relative_importance)
        rel_imp_df['rank'] = rel_imp_df['sr2'].rank(method='dense', ascending=False).astype(int)
        rel_imp_df = rel_imp_df.sort_values('rank')
        
        # Save relative importance
        log(f"Saving relative importance rankings...")
        rel_imp_df.to_csv(RQ_DIR / "data/step06_relative_importance.csv", index=False, encoding='utf-8')
        log(f"step06_relative_importance.csv ({len(rel_imp_df)} predictors, {len(rel_imp_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Effect sizes are within reasonable bounds (Cohen's f² guidelines)
        # Threshold: f² values should be non-negative, finite, and interpretable

        log("Running validate_effect_sizes...")
        validation_result = validate_effect_sizes(
            effect_sizes_df=effect_sizes_df,
            f2_column='value'  # Column containing f² values to validate
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Log summary results
        log("Effect Size Analysis Results:")
        log(f"  Cohen's f² = {cohens_f2:.4f} ({interpret_f2(cohens_f2)})")
        log(f"  R² full model = {r2_full:.4f}")
        log(f"  Bootstrap 95% CI: f² [{f2_ci[0]:.4f}, {f2_ci[1]:.4f}]" if bootstrap_success else "  Bootstrap CIs unavailable")
        
        log("Relative Importance Rankings:")
        for _, row in rel_imp_df.iterrows():
            log(f"  Rank {row['rank']}: {row['predictor']} (sr² = {row['sr2']:.4f}, {row['percent_of_r2']:.1f}% of R²)")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)