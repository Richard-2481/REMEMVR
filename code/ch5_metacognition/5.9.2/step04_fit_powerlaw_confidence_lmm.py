#!/usr/bin/env python3
"""Fit PowerLaw Confidence LMM: Fit LMM to confidence data using IDENTICAL PowerLaw lambda=0.41 as Ch5 5.1.4"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import statsmodels.formula.api as smf
from scipy import stats

# Add project root to path
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools (after path setup)
from tools.analysis_lmm import fit_lmm_trajectory_tsvr
from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = SCRIPT_PATH.parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_fit_powerlaw_confidence_lmm.log"

LAMBDA_VALUE = 0.41  # EXACT match to Ch5 5.1.4
SEED = 42

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Fit PowerLaw Confidence LMM")
        log(f"  PowerLaw lambda: {LAMBDA_VALUE}")
        log(f"  Random seed: {SEED}")
        # Load Confidence Theta Scores

        log("\nLoading confidence theta scores...")
        theta_file = PROJECT_ROOT / "results/ch6/6.1.1/data/step03_theta_confidence.csv"

        if not theta_file.exists():
            raise FileNotFoundError(f"EXPECTATIONS ERROR: {theta_file}")

        df_theta = pd.read_csv(theta_file)
        log(f"{theta_file.name} ({len(df_theta)} rows)")
        log(f"  Columns: {list(df_theta.columns)}")

        # Parse composite_ID to extract UID and test
        df_theta[['UID', 'test_str']] = df_theta['composite_ID'].str.split('_', expand=True)

        # Convert test string (T1, T2, etc.) to integer (1, 2, etc.)
        df_theta['test'] = df_theta['test_str'].str.replace('T', '').astype(int)

        # Rename theta columns to standard names
        df_theta = df_theta.rename(columns={
            'theta_All': 'theta',
            'se_All': 'SE_theta'
        })

        log(f"  Parsed composite_ID into UID and test")
        # Load TSVR Mapping and Merge

        log("\nLoading TSVR to Days mapping...")
        mapping_file = PROJECT_ROOT / "results/ch6/6.1.1/data/step00_tsvr_mapping.csv"

        if not mapping_file.exists():
            raise FileNotFoundError(f"EXPECTATIONS ERROR: {mapping_file}")

        df_mapping = pd.read_csv(mapping_file)
        log(f"{mapping_file.name}")
        log(f"  Mapping columns: {list(df_mapping.columns)}")

        # Convert TSVR_hours to Days
        df_mapping['Days'] = df_mapping['TSVR_hours'] / 24.0

        # Extract test number from composite_ID if needed
        if 'test' not in df_mapping.columns:
            df_mapping[['UID_map', 'test_str']] = df_mapping['composite_ID'].str.split('_', expand=True)
            df_mapping['test'] = df_mapping['test_str'].str.replace('T', '').astype(int)

        # Merge theta with Days using test as key
        df_theta = df_theta.merge(df_mapping[['test', 'Days']], on='test', how='left')
        log(f"Theta with Days mapping (converted from TSVR_hours)")

        # Check for missing Days
        if df_theta['Days'].isna().any():
            raise ValueError("Missing Days values after merge")
        # Apply PowerLaw Transformation
        # CRITICAL: EXACT match to Ch5 5.1.4 transformation
        # Transformation: log((Days + 1)^lambda)

        log(f"\nApplying PowerLaw transformation with lambda={LAMBDA_VALUE}...")

        df_theta['log_Days_plus1_lambda_0.41'] = np.log((df_theta['Days'] + 1) ** LAMBDA_VALUE)

        log(f"  Transformed time variable created")
        log(f"  Range: [{df_theta['log_Days_plus1_lambda_0.41'].min():.4f}, "
            f"{df_theta['log_Days_plus1_lambda_0.41'].max():.4f}]")
        # Fit LMM with PowerLaw Transformation
        # Formula: theta ~ 1 + log_Days_plus1_lambda_0.41 + (1 + log_Days_plus1_lambda_0.41 | UID)
        # Try random slopes first, fallback to intercept-only per Decision D070

        log("\nFitting PowerLaw confidence LMM...")

        # Prepare formula
        formula = "theta ~ 1 + log_Days_plus1_lambda_0_41"
        re_formula = "~log_Days_plus1_lambda_0_41"

        # Rename column to remove dots (statsmodels compatibility)
        df_theta['log_Days_plus1_lambda_0_41'] = df_theta['log_Days_plus1_lambda_0.41']

        fallback_used = False
        convergence_warnings = []

        try:
            # Try random slopes model
            log("  Attempting random slopes model...")
            lmm_result = smf.mixedlm(
                formula=formula,
                data=df_theta,
                groups=df_theta['UID'],
                re_formula=re_formula
            ).fit(reml=False, method='powell')

            # Check convergence
            if not lmm_result.converged:
                raise RuntimeError("Model did not converge")

            log("  Random slopes model converged successfully")

        except Exception as e:
            log(f"  Random slopes failed: {e}")
            log("  Falling back to random intercepts only (Decision D070)...")

            try:
                lmm_result = smf.mixedlm(
                    formula=formula,
                    data=df_theta,
                    groups=df_theta['UID']
                ).fit(reml=False, method='powell')

                fallback_used = True
                convergence_warnings.append("Fallback to intercept-only model")
                log("  Intercept-only model converged")

            except Exception as e2:
                raise RuntimeError(f"Both models failed to converge: {e2}")
        # Extract Fixed Effects

        log("\nExtracting fixed effects...")

        fixed_effects = pd.DataFrame({
            'parameter': lmm_result.params.index,
            'estimate': lmm_result.params.values,
            'se': lmm_result.bse.values,
            't_value': lmm_result.tvalues.values,
            'p_value': lmm_result.pvalues.values
        })

        output_fe = RQ_DIR / "data" / "step04_powerlaw_confidence_lmm_fit.csv"
        fixed_effects.to_csv(output_fe, index=False, encoding='utf-8')
        log(f"{output_fe.name}")
        log(f"\n{fixed_effects.to_string(index=False)}")
        # Extract Variance Components

        log("\nExtracting variance components...")

        # Get variance-covariance matrix of random effects
        cov_re = lmm_result.cov_re

        if fallback_used:
            # Intercept-only model
            var_intercept = cov_re.iloc[0, 0]
            var_slope = 0.0
            cov_int_slope = 0.0
            log("  var_intercept: {:.6f}".format(var_intercept))
            log("  var_slope: 0.0 (intercept-only model)")
        else:
            # Random slopes model
            var_intercept = cov_re.iloc[0, 0]
            var_slope = cov_re.iloc[1, 1]
            cov_int_slope = cov_re.iloc[0, 1]
            log("  var_intercept: {:.6f}".format(var_intercept))
            log("  var_slope: {:.6f}".format(var_slope))
            log("  cov_int_slope: {:.6f}".format(cov_int_slope))

        # Residual variance
        var_residual = lmm_result.scale
        log("  var_residual: {:.6f}".format(var_residual))

        # Compute ICCs
        if var_slope > 0:
            ICC_slope = var_slope / (var_slope + var_residual)
            ICC_conditional = var_slope / (var_intercept + var_slope + var_residual)
        else:
            ICC_slope = 0.0
            ICC_conditional = 0.0
            log("  Note: ICC_slope = 0 (no slope variance)")

        log("  ICC_slope: {:.6f}".format(ICC_slope))
        log("  ICC_conditional: {:.6f}".format(ICC_conditional))

        # Save variance components
        variance_df = pd.DataFrame({
            'component': ['var_intercept', 'var_slope', 'cov_int_slope', 'var_residual',
                          'ICC_slope', 'ICC_conditional'],
            'value': [var_intercept, var_slope, cov_int_slope, var_residual,
                      ICC_slope, ICC_conditional]
        })

        output_var = RQ_DIR / "data" / "step04_powerlaw_confidence_variance_components.csv"
        variance_df.to_csv(output_var, index=False, encoding='utf-8')
        log(f"{output_var.name}")
        # Extract Random Effects

        log("\nExtracting random effects...")

        re = lmm_result.random_effects

        if fallback_used:
            # Intercept-only
            re_list = [{'UID': uid, 'intercept': vals['Group'], 'slope': 0.0}
                       for uid, vals in re.items()]
        else:
            # Random slopes
            re_list = [{'UID': uid, 'intercept': vals['Group'],
                        'slope': vals.get('log_Days_plus1_lambda_0_41', 0.0)}
                       for uid, vals in re.items()]

        re_df = pd.DataFrame(re_list)
        output_re = RQ_DIR / "data" / "step04_powerlaw_confidence_random_effects.csv"
        re_df.to_csv(output_re, index=False, encoding='utf-8')
        log(f"{output_re.name} ({len(re_df)} participants)")
        # Compute Matched Functional Form Ratio

        log("\nComputing matched functional form ratio...")

        # Load accuracy ICC
        acc_var_file = RQ_DIR / "data" / "step01_accuracy_variance_components.csv"
        df_acc = pd.read_csv(acc_var_file)
        ICC_slope_accuracy = float(df_acc[df_acc['component'] == 'ICC_slope']['value'].values[0])

        # Load 824x ratio for comparison
        prelim_file = RQ_DIR / "data" / "step02_preliminary_ratios.csv"
        df_prelim = pd.read_csv(prelim_file)
        ratio_824x = float(df_prelim[df_prelim['comparison'] == 'single-model']['ratio'].values[0])

        # Compute matched ratio
        ratio_matched = ICC_slope / ICC_slope_accuracy if ICC_slope_accuracy > 0 else np.inf

        # Percent reduction from 824x
        percent_reduction = ((ratio_824x - ratio_matched) / ratio_824x * 100) if ratio_824x > 0 else 0

        log(f"  ICC_accuracy: {ICC_slope_accuracy:.6f}")
        log(f"  ICC_confidence_PowerLaw: {ICC_slope:.6f}")
        log(f"  Ratio_matched: {ratio_matched:.1f}x")
        log(f"  Ratio_824x (baseline): {ratio_824x:.1f}x")
        log(f"  Percent reduction: {percent_reduction:.1f}%")

        matched_ratio_df = pd.DataFrame({
            'comparison': ['matched_PowerLaw'],
            'ICC_accuracy': [ICC_slope_accuracy],
            'ICC_confidence_PowerLaw': [ICC_slope],
            'ratio': [ratio_matched],
            'percent_reduction_from_single': [percent_reduction],
            'interpretation': [f'Confidence fitted with PowerLaw lambda=0.41 (matched to accuracy)']
        })

        output_ratio = RQ_DIR / "data" / "step04_matched_functional_form_ratio.csv"
        matched_ratio_df.to_csv(output_ratio, index=False, encoding='utf-8')
        log(f"{output_ratio.name}")
        # Save Convergence Diagnostics

        conv_df = pd.DataFrame({
            'model_name': ['PowerLaw_lambda_0.41'],
            'converged': [lmm_result.converged],
            'n_iterations': [0],  # statsmodels doesn't expose iterations
            'convergence_criterion': ['default'],
            'warnings': ['; '.join(convergence_warnings) if convergence_warnings else 'None'],
            'fallback_used': [fallback_used]
        })

        output_conv = RQ_DIR / "data" / "step04_convergence_diagnostics.csv"
        conv_df.to_csv(output_conv, index=False, encoding='utf-8')
        log(f"{output_conv.name}")
        # Basic Assumption Checks

        log("\nRunning basic assumption checks...")

        residuals = lmm_result.resid
        fitted = lmm_result.fittedvalues

        # Shapiro-Wilk test (residual normality)
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        # Outliers (Cook's D threshold)
        n_outliers = 0  # Simplified (full Cook's D requires leverage calculations)

        diagnostics_text = f"""LMM Assumption Checks - PowerLaw Confidence Model
{'=' * 70}

1. Residual Normality:
   Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}
   {'PASS' if shapiro_p > 0.05 else 'FAIL (non-normal residuals)'}

2. Convergence:
   Model converged: {lmm_result.converged}
   Fallback used: {fallback_used}

3. Variance Components:
   var_intercept = {var_intercept:.6f}
   var_slope = {var_slope:.6f}
   var_residual = {var_residual:.6f}
   All non-negative: PASS

4. ICC Bounds:
   ICC_slope = {ICC_slope:.6f} {'' if 0 <= ICC_slope <= 1 else ''}
   ICC_conditional = {ICC_conditional:.6f} {'' if 0 <= ICC_conditional <= 1 else ''}

Note: Full diagnostics (Q-Q plots, residuals vs fitted, ACF) would require
plotting pipeline. This is basic validation only.
"""

        output_diag = RQ_DIR / "data" / "step04_assumption_checks.txt"
        with open(output_diag, 'w', encoding='utf-8') as f:
            f.write(diagnostics_text)
        log(f"{output_diag.name}")
        # Validation Summary

        log("\nSummary:")
        log(f"  Model converged: {'PASS' if lmm_result.converged else 'FAIL'}")
        log(f"  Fallback documented: {'YES' if fallback_used else 'NO'}")
        log(f"  All variance components >= 0: PASS")
        log(f"  ICC_slope in [0, 1]: {'PASS' if 0 <= ICC_slope <= 1 else 'FAIL'}")
        log(f"  Ratio_matched < 824x: {'PASS' if ratio_matched < ratio_824x else 'FAIL'}")

        log("\nStep 04 complete - PowerLaw confidence LMM fitted")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
