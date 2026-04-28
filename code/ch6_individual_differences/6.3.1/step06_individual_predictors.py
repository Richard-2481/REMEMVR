#!/usr/bin/env python3
"""individual_predictors: Examine individual cognitive test predictors with corrected significance tests"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_multiple_regression

from tools.validation import validate_model_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.1
LOG_FILE = RQ_DIR / "logs" / "step06_individual_predictors.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Individual Predictors Analysis")
        # Load Analysis Dataset

        log("Loading analysis dataset...")
        df_analysis = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")
        log(f"Analysis dataset ({len(df_analysis)} rows, {len(df_analysis.columns)} cols)")
        
        # Verify required columns
        required_cols = ['confidence_theta', 'age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        missing_cols = [col for col in required_cols if col not in df_analysis.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"Sample size: {len(df_analysis)} participants")
        log(f"Predictors: {required_cols[1:]}")  # Exclude outcome variable
        # Prepare Data for Regression
        # Extract outcome and predictors
        y = df_analysis['confidence_theta']
        X = df_analysis[['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']]
        
        log(f"Outcome variable: confidence_theta (mean={y.mean():.3f}, sd={y.std():.3f})")
        log(f"Predictor matrix: {X.shape}")
        
        # Check for missing values
        missing_outcome = y.isna().sum()
        missing_predictors = X.isna().sum().sum()
        if missing_outcome > 0 or missing_predictors > 0:
            log(f"Missing values detected: outcome={missing_outcome}, predictors={missing_predictors}")
            # Drop missing cases
            complete_cases = df_analysis[required_cols].dropna()
            y = complete_cases['confidence_theta']
            X = complete_cases[['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T']]
            log(f"Complete cases: {len(complete_cases)} participants")
        # Run Multiple Regression Analysis
        # Use statsmodels directly due to function signature issues

        log("Running multiple regression analysis...")
        
        import statsmodels.api as sm
        
        # Add constant and fit model
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        log("Multiple regression complete")

        # Extract results
        r2 = results.rsquared
        adj_r2 = results.rsquared_adj
        f_stat = results.fvalue
        f_pval = results.f_pvalue
        
        log(f"Model R² = {r2:.4f} (adjusted = {adj_r2:.4f})")
        log(f"F({X.shape[1]}, {len(X)-X.shape[1]-1}) = {f_stat:.3f}, p = {f_pval:.6f}")
        # Compute VIF and Semi-partial Correlations
        # Calculate VIF for multicollinearity assessment
        log("Computing VIF values...")
        
        # Add constant for VIF calculation
        import statsmodels.api as sm
        X_with_const = sm.add_constant(X)
        
        vif_values = []
        for i in range(1, X_with_const.shape[1]):  # Skip constant
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_values.append(vif)
            log(f"{X.columns[i-1]}: {vif:.3f}")
        
        # Semi-partial correlations (sr²)
        log("Computing semi-partial correlations...")
        sr2_values = []
        
        # For each predictor, compute sr² by comparing full vs reduced model
        full_r2 = r2
        
        for i, predictor in enumerate(X.columns):
            # Reduced model without this predictor
            X_reduced = X.drop(columns=[predictor])
            reduced_result = fit_multiple_regression(
                X=X_reduced,
                y=y,
                add_constant=True,
                return_diagnostics=False
            )
            reduced_r2 = reduced_result['r2']
            sr2 = full_r2 - reduced_r2
            sr2_values.append(sr2)
            log(f"[SR2] {predictor}: {sr2:.6f}")
        # Apply Multiple Comparisons Corrections (Decision D068)
        # Dual p-values: uncorrected + Bonferroni
        # Bonferroni alpha = 0.000597 for cognitive tests

        log("Applying multiple comparisons corrections...")
        
        # Extract p-values (excluding intercept)
        p_uncorrected = []
        for i, predictor in enumerate(X.columns):
            # Get p-value for this predictor from results
            p_val = results.pvalues[i+1]  # +1 to skip intercept
            p_uncorrected.append(p_val)
        
        # Apply Bonferroni correction
        n_tests = 5  # Five cognitive tests: RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T
        alpha_bonferroni = 0.00179 / n_tests  # = 0.000358
        
        # For demographics (age, sex, education), use standard alpha = 0.05
        # For cognitive tests (RAVLT_T, BVMT_T, RPM_T), use Bonferroni corrected alpha
        p_bonferroni = []
        for i, predictor in enumerate(X.columns):
            p_val = p_uncorrected[i]
            if predictor in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
                # Cognitive test - use Bonferroni correction
                p_bonf = min(p_val * n_tests, 1.0)
            else:
                # Demographics - no correction needed
                p_bonf = p_val
            p_bonferroni.append(p_bonf)
        
        # Also compute FDR correction
        _, p_fdr, _, _ = multipletests(p_uncorrected, method='fdr_bh')
        
        log(f"Bonferroni alpha for cognitive tests: {alpha_bonferroni:.6f}")
        log(f"FDR correction applied to all {len(p_uncorrected)} predictors")
        # Create Individual Predictors Results DataFrame
        
        log("Creating individual predictors results...")
        
        # Extract coefficients and confidence intervals
        individual_results = []
        conf_int = results.conf_int()
        
        for i, predictor in enumerate(X.columns):
            result = {
                'predictor': predictor,
                'beta': results.params[i+1],  # +1 to skip intercept
                'se': results.bse[i+1],
                'ci_lower': conf_int.iloc[i+1, 0],
                'ci_upper': conf_int.iloc[i+1, 1],
                'sr2': sr2_values[i],
                'p_uncorrected': p_uncorrected[i],
                'p_bonferroni': p_bonferroni[i],
                'p_fdr': p_fdr[i],
                'vif': vif_values[i]
            }
            individual_results.append(result)
            
            # Log significance results
            if predictor in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
                alpha_used = alpha_bonferroni
                correction = "Bonferroni"
            else:
                alpha_used = 0.05
                correction = "None"
                
            significant = result['p_bonferroni'] < alpha_used
            log(f"{predictor}: β={result['beta']:.4f}, "
                f"p_uncorr={result['p_uncorrected']:.6f}, "
                f"p_bonf={result['p_bonferroni']:.6f} "
                f"({correction}), significant={significant}")
        
        individual_df = pd.DataFrame(individual_results)
        # Save Individual Predictors Results
        
        log("Saving individual predictors results...")
        output_path = RQ_DIR / "data" / "step06_individual_predictors.csv"
        individual_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(individual_df)} rows, {len(individual_df.columns)} cols)")
        # Save Assumption Diagnostics
        
        log("Saving assumption diagnostics...")
        
        diagnostics_text = f"""REGRESSION ASSUMPTION DIAGNOSTICS - RQ 7.3.1 STEP 06
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SUMMARY:
- Sample size: {len(X)} participants
- Predictors: {list(X.columns)}
- R² = {r2:.4f} (adjusted = {adj_r2:.4f})
- F({X.shape[1]}, {len(X)-X.shape[1]-1}) = {f_stat:.3f}, p = {f_pval:.6f}

MULTICOLLINEARITY (VIF):
"""
        
        for i, (predictor, vif) in enumerate(zip(X.columns, vif_values)):
            vif_concern = "HIGH" if vif > 5 else "OK"
            diagnostics_text += f"- {predictor}: VIF = {vif:.3f} ({vif_concern})\n"
        
        diagnostics_text += f"""
MULTIPLE COMPARISONS CORRECTION:
- Cognitive tests (RAVLT_T, BVMT_T, RPM_T): Bonferroni corrected
- Bonferroni alpha: {alpha_bonferroni:.6f}
- Demographics (age, sex, education): Uncorrected alpha = 0.05
- FDR correction also applied to all predictors

SEMI-PARTIAL CORRELATIONS (sr²):
"""
        
        for predictor, sr2 in zip(X.columns, sr2_values):
            diagnostics_text += f"- {predictor}: sr² = {sr2:.6f}\n"
        
        diagnostics_text += f"""
SIGNIFICANCE SUMMARY:
"""
        
        for _, row in individual_df.iterrows():
            predictor = row['predictor']
            if predictor in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
                alpha_used = alpha_bonferroni
                correction = "Bonferroni"
            else:
                alpha_used = 0.05
                correction = "None"
            
            significant = row['p_bonferroni'] < alpha_used
            sig_text = "SIGNIFICANT" if significant else "ns"
            
            diagnostics_text += (f"- {predictor}: β = {row['beta']:.4f} ± {row['se']:.4f}, "
                               f"p = {row['p_bonferroni']:.6f} ({correction}), {sig_text}\n")
        
        diag_path = RQ_DIR / "data" / "step06_assumption_diagnostics.txt"
        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write(diagnostics_text)
        log(f"{diag_path} ({len(diagnostics_text)} bytes)")
        # Run Validation Tool
        # Validates: Model convergence and numerical stability
        # Threshold: Default convergence criteria

        log("Running validate_model_convergence...")
        validation_result = validate_model_convergence(
            model_result=regression_result,
            convergence_threshold=1e-8
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary validation
        n_significant_bonferroni = sum(individual_df['p_bonferroni'] < 0.05)
        n_significant_cognitive = sum(individual_df[individual_df['predictor'].isin(['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T'])]['p_bonferroni'] < alpha_bonferroni)
        max_vif = max(vif_values)
        
        log(f"Predictors with p_bonferroni < 0.05: {n_significant_bonferroni}/6")
        log(f"Cognitive tests with p < {alpha_bonferroni:.6f}: {n_significant_cognitive}/5")
        log(f"Maximum VIF: {max_vif:.3f}")
        log(f"Individual predictors analyzed: {len(individual_df)} coefficients")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)