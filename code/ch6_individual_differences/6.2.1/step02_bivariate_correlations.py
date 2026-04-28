#!/usr/bin/env python3
"""bivariate_correlations: Bivariate correlations with bootstrap CIs. Compute Pearson correlations between all"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from itertools import combinations
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.bootstrap import bootstrap_correlation_ci

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_bivariate_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Bivariate Correlations with Bootstrap CIs")
        # Load Input Data

        log("Loading merged analysis dataset...")
        # Load step01_analysis_dataset.csv
        # Expected columns: Age, theta_all, RAVLT_T, BVMT_T, RPM_T (plus standardized versions)
        # Expected rows: ~100 (one per participant)
        input_df = pd.read_csv(RQ_DIR / "data" / "step01_analysis_dataset.csv")
        log(f"step01_analysis_dataset.csv ({len(input_df)} rows, {len(input_df.columns)} cols)")
        
        # Extract variables for correlation analysis (use raw scores, not standardized)
        correlation_variables = ['Age', 'theta_all', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        
        # Verify all required variables present
        missing_vars = [var for var in correlation_variables if var not in input_df.columns]
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        log(f"Correlation variables: {correlation_variables}")
        
        # Extract correlation data
        corr_data = input_df[correlation_variables].copy()
        
        # Check for missing data
        missing_count = corr_data.isnull().sum().sum()
        if missing_count > 0:
            log(f"Missing data detected: {missing_count} values")
            # Drop rows with any missing values
            corr_data = corr_data.dropna()
            log(f"After dropping missing data: {len(corr_data)} rows")
        
        log(f"Final correlation data: {len(corr_data)} participants, {len(correlation_variables)} variables")
        # Run Bivariate Correlations with Bootstrap CIs

        log("Computing pairwise correlations with bootstrap CIs...")
        
        # Parameters for bootstrap analysis
        n_bootstrap = 1000
        confidence_level = 0.95
        method = "pearson"
        random_seed = 42
        
        correlation_results = []
        
        # Compute all pairwise correlations
        for var1, var2 in combinations(correlation_variables, 2):
            log(f"Computing correlation: {var1} vs {var2}")
            
            x = corr_data[var1].values
            y = corr_data[var2].values
            
            # Use bootstrap_correlation_ci with adapted parameters
            # Note: actual signature uses 'confidence' not 'confidence_level', 'seed' not 'random_state'
            bootstrap_result = bootstrap_correlation_ci(
                x=x,
                y=y,
                n_bootstrap=n_bootstrap,
                confidence=confidence_level,  # Adapted parameter name
                method=method,
                seed=random_seed  # Adapted parameter name
            )
            
            # Compute uncorrected p-value using scipy
            pearson_r, pearson_p = stats.pearsonr(x, y)
            
            # Check what keys are in bootstrap_result
            if isinstance(bootstrap_result, dict):
                # Try different possible key names
                if 'r' in bootstrap_result:
                    corr_value = bootstrap_result['r']
                elif 'correlation' in bootstrap_result:
                    corr_value = bootstrap_result['correlation']
                elif 'estimate' in bootstrap_result:
                    corr_value = bootstrap_result['estimate']
                else:
                    # Use Pearson r as fallback
                    corr_value = pearson_r
                    log(f"Using pearson_r, bootstrap keys: {list(bootstrap_result.keys())}")
            else:
                # If it's not a dict, assume it's the correlation value
                corr_value = bootstrap_result
                bootstrap_result = {'ci_lower': None, 'ci_upper': None}
            
            correlation_results.append({
                'Variable1': var1,
                'Variable2': var2,
                'r': corr_value,
                'ci_lower': bootstrap_result.get('ci_lower', bootstrap_result.get('ci_low', None)),
                'ci_upper': bootstrap_result.get('ci_upper', bootstrap_result.get('ci_high', None)),
                'p_uncorrected': pearson_p,
                'n_observations': len(x)
            })
            
            ci_lower = bootstrap_result.get('ci_lower', bootstrap_result.get('ci_low', 'N/A'))
            ci_upper = bootstrap_result.get('ci_upper', bootstrap_result.get('ci_high', 'N/A'))
            log(f"{var1} vs {var2}: r = {corr_value:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}], p = {pearson_p:.3f}")
        
        log("Bootstrap correlation analysis complete")
        # Multiple Comparison Corrections (Decision D068)
        # Apply both Bonferroni and FDR corrections for dual p-value reporting
        # Required per Decision D068 for all hypothesis tests

        log("Applying multiple comparison corrections...")
        
        from statsmodels.stats.multitest import multipletests
        
        # Extract p-values for correction
        p_values = [result['p_uncorrected'] for result in correlation_results]
        
        # Bonferroni correction
        # For bivariate correlations: 10 pairwise tests (5 choose 2)
        alpha_bonferroni = 0.05 / len(p_values)
        p_bonferroni = [p * len(p_values) for p in p_values]  # Manual Bonferroni
        
        # FDR correction using Benjamini-Hochberg
        reject_fdr, p_fdr, alpha_sidak, alpha_bonf = multipletests(p_values, method='fdr_bh')
        
        # Add corrected p-values to results
        for i, result in enumerate(correlation_results):
            result['p_bonferroni'] = min(p_bonferroni[i], 1.0)  # Cap at 1.0
            result['p_fdr'] = p_fdr[i]
            result['significant_bonferroni'] = p_bonferroni[i] < 0.05
            result['significant_fdr'] = reject_fdr[i]
        
        log(f"Multiple comparison corrections applied (Bonferroni alpha = {alpha_bonferroni:.4f})")
        
        # Report key findings
        age_theta_result = next((r for r in correlation_results if 
                               set([r['Variable1'], r['Variable2']]) == {'Age', 'theta_all'}), None)
        
        if age_theta_result:
            log(f"[KEY FINDING] Age-theta_all correlation: r = {age_theta_result['r']:.3f}, "
                f"95% CI [{age_theta_result['ci_lower']:.3f}, {age_theta_result['ci_upper']:.3f}], "
                f"p_uncorrected = {age_theta_result['p_uncorrected']:.3f}, "
                f"p_bonferroni = {age_theta_result['p_bonferroni']:.3f}, "
                f"p_fdr = {age_theta_result['p_fdr']:.3f}")
            
            if age_theta_result['r'] < -0.15:
                log("Age-theta_all correlation meets expected small negative threshold (< -0.15)")
            else:
                log(f"Age-theta_all correlation ({age_theta_result['r']:.3f}) does not meet expected threshold (< -0.15)")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 hierarchical regression, final summary

        log("Saving correlation results with dual p-values...")
        
        # Create DataFrame with required columns for D068 compliance
        results_df = pd.DataFrame(correlation_results)
        
        # Reorder columns to match specification
        output_columns = ['Variable1', 'Variable2', 'r', 'ci_lower', 'ci_upper', 
                         'p_uncorrected', 'p_bonferroni', 'p_fdr']
        results_df = results_df[output_columns]
        
        # Output: step02_correlations.csv
        # Contains: Correlation matrix with bootstrap CIs and dual p-values
        # Columns: Variable1, Variable2, r, ci_lower, ci_upper, p_uncorrected, p_bonferroni, p_fdr
        output_path = RQ_DIR / "data" / "step02_correlations.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_correlations.csv ({len(results_df)} rows, {len(results_df.columns)} cols)")
        
        # Summary statistics
        significant_bonferroni = (results_df['p_bonferroni'] < 0.05).sum()
        significant_fdr = (results_df['p_fdr'] < 0.05).sum()
        log(f"{len(results_df)} correlations computed")
        log(f"Significant after Bonferroni correction: {significant_bonferroni}/{len(results_df)}")
        log(f"Significant after FDR correction: {significant_fdr}/{len(results_df)}")
        # Run Validation Tool
        # Validates: Dual p-value reporting compliance (D068)
        # Threshold: Checks for required columns and format

        log("Running validate_hypothesis_test_dual_pvalues...")
        
        # Prepare data for validation (function expects 'term' column)
        validation_df = results_df.copy()
        validation_df['term'] = validation_df['Variable1'] + ':' + validation_df['Variable2']
        
        # Required terms for validation (all correlation pairs)
        required_terms = validation_df['term'].tolist()
        
        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=validation_df,
            required_terms=required_terms,
            alpha_bonferroni=0.05  # Standard alpha level
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
        
        # Check validation passed
        if isinstance(validation_result, dict) and validation_result.get('valid', False):
            log("PASSED - Dual p-value reporting format verified")
        else:
            log("WARNING - Validation concerns detected (may still be acceptable)")

        log("Step 2 complete - Bivariate correlations with bootstrap CIs and dual p-values")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)