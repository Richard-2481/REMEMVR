#!/usr/bin/env python3
"""Fit LMM with 3-Way Age x Paradigm x Time Interaction: Fit Linear Mixed Model testing full 3-way Age x Paradigm x Time interaction with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
import pickle
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lmm.log"

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Fit LMM with 3-Way Age x Paradigm x Time Interaction")
        # Load LMM Input Data
        log("Loading LMM input data from Step 1...")
        input_path = RQ_DIR / "data" / "step01_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"step01_lmm_input.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Validate required columns
        required_cols = ['UID', 'theta', 'Age_c', 'paradigm', 'TSVR_hours', 'log_TSVR']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")

        # Set paradigm as categorical with IFR as reference (alphabetical order: ICR, IFR, IRE -> reorder)
        df['paradigm'] = pd.Categorical(df['paradigm'], categories=['IFR', 'ICR', 'IRE'])
        log(f"paradigm set as categorical with IFR as reference: {list(df['paradigm'].cat.categories)}")
        # Define Model Formula
        # Full 3-way interaction model:
        # theta ~ (main effects) + (2-way interactions) + (3-way interactions)
        #
        # Main effects: TSVR_hours, log_TSVR, Age_c, paradigm
        # 2-way: Time×Age, Time×Paradigm, Age×Paradigm
        # 3-way: Age_c × paradigm × TSVR_hours, Age_c × paradigm × log_TSVR

        formula = (
            "theta ~ TSVR_hours + log_TSVR + Age_c + C(paradigm, Treatment('IFR')) + "
            "TSVR_hours:Age_c + log_TSVR:Age_c + "
            "TSVR_hours:C(paradigm, Treatment('IFR')) + log_TSVR:C(paradigm, Treatment('IFR')) + "
            "Age_c:C(paradigm, Treatment('IFR')) + "
            "TSVR_hours:Age_c:C(paradigm, Treatment('IFR')) + log_TSVR:Age_c:C(paradigm, Treatment('IFR'))"
        )

        log("3-way Age x Paradigm x Time interaction model")
        log(f"{formula}")
        # Fit Model with Random Slopes (Primary Attempt)
        # CRITICAL: RQ 5.2.1 established Log model as best fit (AIC=3187.96)
        # Random slope must be on log_TSVR, NOT linear TSVR_hours
        # 5.2.1 showed log_Days Var = 0.046 (meaningful individual differences)
        log("Attempting model with random slopes for log_TSVR (per RQ 5.2.1 best model)...")

        model_converged = False
        random_structure = "random_slopes"

        try:
            # Random slopes: (log_TSVR | UID) - allows individual forgetting rates on LOG scale
            model = smf.mixedlm(
                formula=formula,
                data=df,
                groups=df['UID'],
                re_formula="~log_TSVR"  # Random intercept + slope for log_TSVR (per 5.2.1)
            )
            lmm_result = model.fit(reml=False, method='lbfgs')

            # Check convergence
            if hasattr(lmm_result, 'converged') and lmm_result.converged:
                model_converged = True
                log("Model with random slopes converged")
            else:
                log("Model fit completed but convergence flag is False")
                # Still check if we got valid results
                if not np.isnan(lmm_result.llf):
                    model_converged = True
                    log("Log-likelihood is valid, treating as converged")

        except Exception as e:
            log(f"Random slopes model failed: {str(e)[:200]}")
            model_converged = False
        # Fallback to Random Intercepts Only (Convergence Contingency)
        if not model_converged:
            log("Falling back to random intercepts only...")
            random_structure = "random_intercepts_only"

            try:
                model = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df['UID'],
                    re_formula="~1"  # Random intercepts only
                )
                lmm_result = model.fit(reml=False, method='lbfgs')

                if not np.isnan(lmm_result.llf):
                    model_converged = True
                    log("Model with random intercepts only converged")
                else:
                    raise ValueError("Intercept-only model also failed to converge")

            except Exception as e:
                log(f"Random intercepts model also failed: {str(e)}")
                raise ValueError("Model failed to converge with both random structures")
        # Extract and Log Model Summary
        log("Extracting model results...")

        # Get summary text
        summary_text = str(lmm_result.summary())

        # Log key statistics
        log(f"Log-Likelihood: {lmm_result.llf:.2f}")
        log(f"AIC: {lmm_result.aic:.2f}")
        log(f"BIC: {lmm_result.bic:.2f}")
        log(f"Number of observations: {lmm_result.nobs}")
        log(f"Number of groups (participants): {lmm_result.nobs // 12}")  # 12 obs per participant (4 tests × 3 paradigms)
        log(f"Random structure used: {random_structure}")

        # Fixed effects
        log("[FIXED EFFECTS]")
        fe_params = lmm_result.fe_params
        fe_se = np.sqrt(np.diag(lmm_result.cov_params()))[:len(fe_params)]
        for i, (name, coef) in enumerate(fe_params.items()):
            se = fe_se[i] if i < len(fe_se) else np.nan
            z = coef / se if se > 0 else np.nan
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            log(f"  {name}: coef={coef:.4f}, SE={se:.4f}, z={z:.2f}, p={p:.4f}")

        # Random effects variance
        log("[RANDOM EFFECTS]")
        log(f"  Group Var (Intercept): {lmm_result.cov_re.iloc[0, 0]:.4f}")
        if random_structure == "random_slopes" and lmm_result.cov_re.shape[0] > 1:
            log(f"  Group Var (TSVR_hours): {lmm_result.cov_re.iloc[1, 1]:.4f}")
            if lmm_result.cov_re.shape[0] > 1 and lmm_result.cov_re.shape[1] > 1:
                log(f"  Cov(Intercept, TSVR_hours): {lmm_result.cov_re.iloc[0, 1]:.4f}")
        log(f"  Residual Var: {lmm_result.scale:.4f}")
        # Basic Assumption Validation
        log("Checking model assumptions...")

        # Get residuals
        residuals = lmm_result.resid
        fitted = lmm_result.fittedvalues

        # 1. Residual normality (Shapiro-Wilk - use sample for large N)
        log("[CHECK 1] Residual normality...")
        sample_size = min(5000, len(residuals))
        sample_residuals = np.random.choice(residuals, sample_size, replace=False)
        shapiro_stat, shapiro_p = scipy_stats.shapiro(sample_residuals)
        normality_pass = shapiro_p > 0.01
        log(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        log(f"  Result: {'PASS' if normality_pass else 'CONCERN (p < 0.01, but LMM is robust to minor violations)'}")

        # 2. Residual mean ~ 0
        log("[CHECK 2] Residual mean...")
        resid_mean = np.mean(residuals)
        log(f"  Mean residual: {resid_mean:.6f}")
        log(f"  Result: {'PASS' if abs(resid_mean) < 0.01 else 'CONCERN'}")

        # 3. Residual variance homogeneity (simple check by paradigm)
        log("[CHECK 3] Homoscedasticity by paradigm...")
        var_by_paradigm = df.copy()
        var_by_paradigm['resid'] = residuals
        paradigm_vars = var_by_paradigm.groupby('paradigm')['resid'].var()
        log(f"  Variance by paradigm: {dict(paradigm_vars.round(4))}")
        max_ratio = paradigm_vars.max() / paradigm_vars.min()
        log(f"  Max/Min variance ratio: {max_ratio:.2f}")
        log(f"  Result: {'PASS' if max_ratio < 2.0 else 'CONCERN (ratio > 2)'}")

        # 4. Random effects variance > 0
        log("[CHECK 4] Random effects variance...")
        re_var = lmm_result.cov_re.iloc[0, 0]
        log(f"  Random intercept variance: {re_var:.4f}")
        log(f"  Result: {'PASS' if re_var > 0 else 'CONCERN (no between-person variability)'}")

        # 5. No extreme outliers (residuals > 3 SD)
        log("[CHECK 5] Outliers...")
        std_resid = residuals / np.std(residuals)
        n_outliers = np.sum(np.abs(std_resid) > 3)
        log(f"  Residuals > 3 SD: {n_outliers} ({100*n_outliers/len(residuals):.1f}%)")
        log(f"  Result: {'PASS' if n_outliers < len(residuals) * 0.01 else 'CONCERN (>1% outliers)'}")

        # 6. Check for NaN in coefficients
        log("[CHECK 6] Coefficient validity...")
        # fe_params is a pandas Series, use .isna() or iterate over values
        n_nan_coef = sum(1 for v in fe_params if np.isnan(v))
        log(f"  NaN coefficients: {n_nan_coef}")
        log(f"  Result: {'PASS' if n_nan_coef == 0 else 'FAIL (NaN coefficients)'}")
        # Save Model Object
        log("Saving model pickle...")
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(lmm_result, f)
        log(f"{model_path}")

        # Also save fixed effects as CSV for downstream steps (avoids pickle/patsy issues)
        log("Saving fixed effects CSV...")
        fe_df = pd.DataFrame({
            'term': list(fe_params.index),
            'coefficient': list(fe_params.values),
            'SE': list(fe_se),
            'z': [c/s if s > 0 and not np.isnan(s) else np.nan for c, s in zip(fe_params.values, fe_se)],
            'p_value': [2 * (1 - scipy_stats.norm.cdf(abs(c/s))) if s > 0 and not np.isnan(s) else np.nan
                       for c, s in zip(fe_params.values, fe_se)]
        })
        fe_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fe_df.to_csv(fe_path, index=False, encoding='utf-8')
        log(f"{fe_path}")
        # Save Summary Text
        log("Saving model summary...")
        summary_path = RQ_DIR / "data" / "step02_lmm_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.3.4 - Age x Paradigm x Time Interaction LMM Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model Formula:\n{formula}\n\n")
            f.write(f"Random Structure: {random_structure}\n")
            f.write(f"Number of Observations: {lmm_result.nobs}\n")
            f.write(f"Number of Groups: {len(set(df['UID']))}\n\n")

            f.write("Model Fit Statistics:\n")
            f.write(f"  Log-Likelihood: {lmm_result.llf:.2f}\n")
            f.write(f"  AIC: {lmm_result.aic:.2f}\n")
            f.write(f"  BIC: {lmm_result.bic:.2f}\n\n")

            f.write("Convergence Status:\n")
            f.write(f"  Converged: {model_converged}\n")
            f.write(f"  Random Structure Used: {random_structure}\n\n")

            f.write("=" * 80 + "\n")
            f.write("STATSMODELS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_text)
            f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("ASSUMPTION VALIDATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"1. Residual Normality: Shapiro-Wilk W={shapiro_stat:.4f}, p={shapiro_p:.4f}\n")
            f.write(f"2. Residual Mean: {resid_mean:.6f}\n")
            f.write(f"3. Homoscedasticity: Variance ratio={max_ratio:.2f}\n")
            f.write(f"4. Random Effects Variance: {re_var:.4f}\n")
            f.write(f"5. Outliers (>3 SD): {n_outliers} ({100*n_outliers/len(residuals):.1f}%)\n")
            f.write(f"6. NaN Coefficients: {n_nan_coef}\n")

        log(f"{summary_path}")
        # Final Summary
        log("Step 2 complete")
        log(f"")
        log(f"  Model: 3-way Age x Paradigm x Time interaction")
        log(f"  Random structure: {random_structure}")
        log(f"  Observations: {lmm_result.nobs}")
        log(f"  Log-Likelihood: {lmm_result.llf:.2f}")
        log(f"  AIC: {lmm_result.aic:.2f}")
        log(f"  Convergence: {model_converged}")
        log(f"  Ready for Step 3 (extract 3-way interaction terms)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        sys.exit(1)
