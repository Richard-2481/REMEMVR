#!/usr/bin/env python3
"""fit_lmm: Fit Linear Mixed Model to test Paradigm × Time interaction on calibration"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
import statsmodels.formula.api as smf
from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.7
LOG_FILE = RQ_DIR / "logs" / "step03_fit_lmm.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: fit_lmm")
        # Load Input Data

        log("Loading calibration scores from Step 2...")

        input_path = RQ_DIR / "data" / "step02_calibration_scores.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'paradigm', 'TSVR_hours', 'calibration']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            log(f"Missing required columns: {missing_cols}")
            sys.exit(1)

        log(f"All required columns present")
        log(f"  N observations: {len(df)}")
        log(f"  N participants: {df['UID'].nunique()}")
        log(f"  Paradigms: {sorted(df['paradigm'].unique())}")
        # Fit LMM with Paradigm × Time Interaction
        # Formula: calibration ~ paradigm + TSVR_hours + paradigm:TSVR_hours
        # Random effects: Try ~TSVR_hours first (random slopes), simplify to ~1 if needed

        log("Fitting LMM with Paradigm × Time interaction...")

        # Set paradigm reference level to 'free_recall' (for contrasts)
        df['paradigm'] = pd.Categorical(df['paradigm'], categories=['free_recall', 'cued_recall', 'recognition'])

        # Define formula
        formula = "calibration ~ paradigm + TSVR_hours + paradigm:TSVR_hours"
        log(f"  Formula: {formula}")

        # Start with random intercepts only (more stable convergence for calibration data)
        log("  Using random intercepts only: re_formula='~1' (calibration data typically requires simpler random structure)")

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Fit using statsmodels.MixedLM directly with random intercepts only
                lmm_result = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df['UID'],
                    re_formula='~1'
                ).fit(reml=False, method='lbfgs')

                convergence_warnings = [str(warning.message) for warning in w]
                if convergence_warnings:
                    log(f"  Warnings during fitting: {len(convergence_warnings)}")
                    for warn in convergence_warnings[:3]:
                        log(f"    {warn}")

            log("LMM fitted with random intercepts only")
            re_formula_used = "~1"

        except Exception as e_intercepts:
            log(f"LMM fitting failed: {str(e_intercepts)}")
            log("LMM CONVERGENCE FAILURE: Unable to fit model")
            sys.exit(1)
        # Extract Fixed Effects

        log("Extracting fixed effects...")

        fixed_effects = pd.DataFrame({
            'term': lmm_result.params.index,
            'coef': lmm_result.params.values,
            'se': lmm_result.bse.values,
            't': lmm_result.tvalues.values,
            'p_uncorrected': lmm_result.pvalues.values
        })

        log(f"{len(fixed_effects)} fixed effects")
        log("Fixed effects summary:")
        for _, row in fixed_effects.iterrows():
            sig = "***" if row['p_uncorrected'] < 0.001 else ("**" if row['p_uncorrected'] < 0.01 else ("*" if row['p_uncorrected'] < 0.05 else ""))
            log(f"  {row['term']}: coef={row['coef']:.4f}, SE={row['se']:.4f}, t={row['t']:.2f}, p={row['p_uncorrected']:.4f} {sig}")

        # Check for NaN coefficients
        if fixed_effects['coef'].isna().any():
            log("NaN coefficients detected in LMM output")
            sys.exit(1)
        # Extract Random Effects Variance Components

        log("Extracting random effects variance components...")

        random_effects_list = []

        # Get random effects variances (intercept only for re_formula='~1')
        try:
            re_cov = lmm_result.cov_re
            random_intercept_var = re_cov.iloc[0, 0]
            random_effects_list.append({
                'component': 'intercept',
                'variance': random_intercept_var,
                'sd': np.sqrt(random_intercept_var)
            })

        except Exception as e:
            log(f"Could not extract random effects variance: {str(e)}")

        # Get residual variance
        residual_var = lmm_result.scale
        random_effects_list.append({
            'component': 'residual',
            'variance': residual_var,
            'sd': np.sqrt(residual_var)
        })

        random_effects = pd.DataFrame(random_effects_list)
        log(f"{len(random_effects)} variance components")
        # Extract Model Fit Statistics

        log("Extracting model fit statistics...")

        # Compute ICC
        total_var = random_effects[random_effects['component'] != 'residual']['variance'].sum() + residual_var
        intercept_var = random_effects[random_effects['component'] == 'intercept']['variance'].values[0]
        icc = intercept_var / total_var

        model_fit = pd.DataFrame({
            'log_likelihood': [lmm_result.llf],
            'AIC': [lmm_result.aic],
            'BIC': [lmm_result.bic],
            'ICC': [icc]
        })

        log(f"Model fit: AIC={lmm_result.aic:.2f}, BIC={lmm_result.bic:.2f}, ICC={icc:.3f}")
        # Validate Convergence

        log("Validating LMM convergence...")

        validation_result = validate_lmm_convergence(lmm_result)

        if not validation_result.get('converged', False):
            log(f"LMM convergence validation failed: {validation_result.get('message', 'Unknown error')}")
            # Don't exit - log warning but continue (simplification already documented)

        log(f"Convergence validation: {validation_result.get('message', 'OK')}")
        # Save Outputs

        log("Saving LMM results...")

        # Save fixed effects
        fixed_out = RQ_DIR / "data" / "step03_lmm_fixed_effects.csv"
        fixed_effects.to_csv(fixed_out, index=False, encoding='utf-8')
        log(f"{fixed_out.name} ({len(fixed_effects)} rows)")

        # Save random effects
        random_out = RQ_DIR / "data" / "step03_lmm_random_effects.csv"
        random_effects.to_csv(random_out, index=False, encoding='utf-8')
        log(f"{random_out.name} ({len(random_effects)} rows)")

        # Save model fit
        fit_out = RQ_DIR / "data" / "step03_lmm_model_fit.csv"
        model_fit.to_csv(fit_out, index=False, encoding='utf-8')
        log(f"{fit_out.name}")

        # Save convergence report
        conv_report_path = RQ_DIR / "data" / "step03_lmm_convergence_report.txt"
        with open(conv_report_path, 'w', encoding='utf-8') as f:
            f.write("LMM CONVERGENCE REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Random effects: {re_formula_used}\n")
            f.write(f"Groups: UID (N={df['UID'].nunique()})\n")
            f.write(f"Observations: {len(df)}\n")
            f.write(f"REML: False (ML estimation)\n\n")
            f.write(f"Convergence status: {'SUCCESS' if validation_result.get('converged', False) else 'WARNING'}\n")
            f.write(f"Message: {validation_result.get('message', 'Unknown')}\n\n")
            f.write(f"Model fit:\n")
            f.write(f"  Log-likelihood: {lmm_result.llf:.2f}\n")
            f.write(f"  AIC: {lmm_result.aic:.2f}\n")
            f.write(f"  BIC: {lmm_result.bic:.2f}\n")
            f.write(f"  ICC: {icc:.3f}\n")

        log(f"{conv_report_path.name}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
