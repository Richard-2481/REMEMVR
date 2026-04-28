#!/usr/bin/env python3
"""correlations_bootstrap: Correlate continuous calibration quality (residuals) with cognitive reserve indicators"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.bootstrap import bootstrap_correlation_ci

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_correlations_bootstrap.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom Bootstrap Correlation Function (handles signature mismatch)

def compute_correlation_with_bootstrap(x, y, variable_pair, n_bootstrap=1000, seed=42):
    """
    Compute correlation with bootstrap CI using actual function signature.
    Based on actual signature: x, y, n_bootstrap, confidence, method, seed
    """
    # Use actual function signature from validation check
    result = bootstrap_correlation_ci(
        x=x,
        y=y,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        method='pearson',
        seed=seed
    )
    
    # Compute p-value using scipy
    # Handle different key names from bootstrap_correlation_ci
    corr_coeff = result.get('r', result.get('correlation', np.nan))
    ci_lower = result.get('ci_lower', result.get('ci_low', np.nan))
    ci_upper = result.get('ci_upper', result.get('ci_high', np.nan))
    
    n = len(x)
    # t-statistic for correlation
    if not np.isnan(corr_coeff) and abs(corr_coeff) < 1:
        t_stat = corr_coeff * np.sqrt((n - 2) / (1 - corr_coeff**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))  # Two-tailed
    else:
        p_value = np.nan
    
    return {
        'variable_pair': variable_pair,
        'r': corr_coeff,
        'p_uncorrected': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_boot': n_bootstrap,
        'bootstrap_distribution': result.get('bootstrap_distribution', [])
    }

def classify_effect_size(r):
    """Classify correlation effect size (Cohen's conventions)."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return 'negligible'
    elif abs_r < 0.3:
        return 'small'
    elif abs_r < 0.5:
        return 'medium'
    else:
        return 'large'

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Correlations with Bootstrap")
        # Load Input Data

        log("Loading calibration groups data...")
        df = pd.read_csv(RQ_DIR / "data" / "step02_calibration_groups.csv")
        log(f"step02_calibration_groups.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['residual', 'education', 'rpm', 'age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove any rows with missing data for correlations
        df_clean = df[required_cols].dropna()
        log(f"Clean data for correlations: {len(df_clean)} participants (removed {len(df) - len(df_clean)} with missing data)")
        # Run Bootstrap Correlations

        correlations = [
            ('residual', 'education'),
            ('residual', 'rpm'),
            ('residual', 'age')
        ]
        
        correlation_results = []
        bootstrap_distributions = []
        
        n_comparisons = 6  # 3 correlations + 3 ANOVAs (from step03) = 6 total tests
        alpha_bonferroni = 0.05 / n_comparisons

        for var1, var2 in correlations:
            log(f"Computing correlation: {var1} vs {var2}...")
            
            # Get clean data for this pair
            pair_data = df_clean[[var1, var2]].dropna()
            if len(pair_data) < 10:
                log(f"Insufficient data for {var1} vs {var2}: only {len(pair_data)} complete cases")
                continue
            
            x = pair_data[var1].values
            y = pair_data[var2].values
            variable_pair = f"{var1}_vs_{var2}"
            
            # Compute correlation with bootstrap
            result = compute_correlation_with_bootstrap(x, y, variable_pair, n_bootstrap=1000, seed=42)
            
            # Add Bonferroni-corrected p-value
            result['p_bonferroni'] = min(result['p_uncorrected'] * n_comparisons, 1.0)
            result['effect_size'] = classify_effect_size(result['r'])
            
            correlation_results.append(result)
            
            # Store bootstrap distribution
            if 'bootstrap_distribution' in result:
                for i, r_boot in enumerate(result['bootstrap_distribution']):
                    bootstrap_distributions.append({
                        'variable_pair': variable_pair,
                        'iteration': i,
                        'r_bootstrap': r_boot
                    })
            
            log(f"{variable_pair}: r = {result['r']:.3f}, p = {result['p_uncorrected']:.3f}, CI = [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        # Save Analysis Outputs
        # These outputs will be used by: Step 05 (effect sizes) and final reporting

        log("Saving correlation results...")
        
        # Save main correlation results
        if correlation_results:
            correlation_df = pd.DataFrame(correlation_results)
            # Remove bootstrap_distribution column for CSV (too large)
            correlation_df_save = correlation_df.drop('bootstrap_distribution', axis=1, errors='ignore')
        else:
            # Create empty DataFrame with expected columns if no results
            correlation_df_save = pd.DataFrame(columns=['variable_pair', 'r', 'p_uncorrected', 'p_bonferroni', 'ci_lower', 'ci_upper', 'n_boot', 'effect_size'])
        
        correlation_df_save.to_csv(RQ_DIR / "data" / "step04_correlations.csv", index=False, encoding='utf-8')
        log(f"step04_correlations.csv ({len(correlation_df_save)} rows, {len(correlation_df_save.columns)} cols)")
        
        # Save bootstrap distributions
        if bootstrap_distributions:
            bootstrap_df = pd.DataFrame(bootstrap_distributions)
        else:
            # Create empty DataFrame with expected columns if no results
            bootstrap_df = pd.DataFrame(columns=['variable_pair', 'iteration', 'r_bootstrap'])
        
        bootstrap_df.to_csv(RQ_DIR / "data" / "step04_bootstrap_distributions.csv", index=False, encoding='utf-8')
        log(f"step04_bootstrap_distributions.csv ({len(bootstrap_df)} rows, {len(bootstrap_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Dual p-value reporting compliance and required columns
        # Threshold: Checks for proper correlation test structure

        log("Running validate_correlation_test_d068...")
        
        validation_result = validate_correlation_test_d068(
            correlation_df=correlation_df_save,
            required_cols=['variable_pair', 'r', 'p_uncorrected', 'p_bonferroni']
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary statistics
        log("Correlation Results Summary:")
        if not correlation_df_save.empty:
            for _, row in correlation_df_save.iterrows():
                significance = ""
                if row['p_bonferroni'] < 0.05:
                    significance = " (significant after Bonferroni)"
                elif row['p_uncorrected'] < 0.05:
                    significance = " (significant uncorrected)"
                
                log(f"  {row['variable_pair']}: r = {row['r']:.3f}, p_uncorr = {row['p_uncorrected']:.3f}, p_bonf = {row['p_bonferroni']:.3f}, effect = {row['effect_size']}{significance}")
        else:
            log("  No correlation results computed")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)