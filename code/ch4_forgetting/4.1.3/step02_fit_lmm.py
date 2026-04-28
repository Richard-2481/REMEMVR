#!/usr/bin/env python3
"""step02_fit_lmm: Fit Linear Mixed Model (LMM) testing age effects on baseline memory and"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM fitting
from statsmodels.formula.api import mixedlm
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Import validation tools
from tools.validation import validate_model_convergence, validate_lmm_assumptions_comprehensive

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit LMM (Age × Time Interaction with Lin+Log)")
        # Load Input Data

        log("Loading prepared LMM input data...")
        input_path = RQ_DIR / "data" / "step01_lmm_input_prepared.csv"
        df_lmm = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")

        # Validate expected columns present
        required_cols = ['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'theta',
                        'se_all', 'age', 'Age_c', 'Time', 'Time_log']
        missing_cols = [col for col in required_cols if col not in df_lmm.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")

        # Report data summary
        log(f"Data summary:")
        log(f"  - N participants: {df_lmm['UID'].nunique()}")
        log(f"  - N observations: {len(df_lmm)}")
        log(f"  - Age_c range: [{df_lmm['Age_c'].min():.2f}, {df_lmm['Age_c'].max():.2f}]")
        log(f"  - Age_c mean: {df_lmm['Age_c'].mean():.4f} (should be ~0)")
        log(f"  - Time range: [{df_lmm['Time'].min():.2f}, {df_lmm['Time'].max():.2f}] hours")
        log(f"  - Time_log range: [{df_lmm['Time_log'].min():.2f}, {df_lmm['Time_log'].max():.2f}]")
        # Fit LMM with Age × Time Interaction (Lin+Log)
        # Formula: theta ~ (Time + Time_log) * Age_c
        # Expands to: theta ~ Time + Time_log + Age_c + Time:Age_c + Time_log:Age_c
        # Random effects: (Time | UID) - random intercepts + random slopes for Time
        # REML: False - use ML for model comparison compatibility

        log("Fitting LMM with formula: theta ~ (Time + Time_log) * Age_c")
        log("Random effects: (Time | UID)")
        log("REML: False (using Maximum Likelihood)")

        # Define formula components
        formula = "theta ~ (Time + Time_log) * Age_c"
        re_formula = "Time"  # Random intercepts + random slopes for Time
        groups = df_lmm['UID']

        # Fit LMM
        log("Model fitting in progress (may take 30-90 seconds)...")
        lmm_model = mixedlm(formula=formula,
                           data=df_lmm,
                           groups=groups,
                           re_formula=re_formula,
                           missing='drop')
        lmm_result = lmm_model.fit(reml=False)
        log("LMM fitting complete")

        # Check convergence
        if hasattr(lmm_result, 'converged') and lmm_result.converged:
            log("Model converged successfully")
        elif hasattr(lmm_result, 'converged'):
            log("Model did NOT converge (converged=False)")
        else:
            log("Convergence status unknown (no converged attribute)")
        # Save LMM Model and Summary
        # Outputs:
        # 1. data/step02_lmm_model.pkl - Fitted model object for downstream use
        # 2. results/step02_lmm_summary.txt - Human-readable summary
        # 3. data/step02_fixed_effects.csv - Fixed effects table

        # Save fitted model as pickle
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"
        log(f"Saving fitted model to {model_path.name}...")
        lmm_result.save(str(model_path))
        log(f"{model_path.name}")

        # Save model summary as text
        summary_path = RQ_DIR / "results" / "step02_lmm_summary.txt"
        log(f"Saving model summary to {summary_path.name}...")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(str(lmm_result.summary()))
        log(f"{summary_path.name}")

        # Extract and save fixed effects table
        log("Extracting fixed effects table...")
        fixed_effects = lmm_result.fe_params
        fixed_effects_se = lmm_result.bse_fe
        fixed_effects_z = lmm_result.tvalues
        fixed_effects_p = lmm_result.pvalues

        # Create fixed effects DataFrame using index alignment
        df_fixed = pd.DataFrame({
            'term': fixed_effects.index.tolist(),
            'coef': fixed_effects.tolist(),
            'se': [fixed_effects_se[idx] for idx in fixed_effects.index],
            'z': [fixed_effects_z[idx] for idx in fixed_effects.index],
            'p': [fixed_effects_p[idx] for idx in fixed_effects.index]
        })

        # Save fixed effects table
        fixed_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        log(f"Saving fixed effects table to {fixed_path.name}...")
        df_fixed.to_csv(fixed_path, index=False, encoding='utf-8')
        log(f"{fixed_path.name} ({len(df_fixed)} rows)")

        # Report fixed effects summary
        log("Fixed effects summary:")
        for _, row in df_fixed.iterrows():
            sig_flag = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
            log(f"  - {row['term']}: coef={row['coef']:.4f}, SE={row['se']:.4f}, z={row['z']:.2f}, p={row['p']:.4f} {sig_flag}")
        # Run Validation - Model Convergence
        # Validates: lmm_result.converged == True

        log("Running validate_model_convergence...")
        convergence_result = validate_model_convergence(lmm_result)

        if convergence_result['valid']:
            log(f"PASS - {convergence_result['message']}")
        else:
            log(f"FAIL - {convergence_result['message']}")
            raise ValueError(f"Model convergence validation failed: {convergence_result['message']}")
        # Run Validation - LMM Assumptions (Comprehensive)
        # Validates:
        #   - Residual normality (Shapiro-Wilk)
        #   - Homoscedasticity (Breusch-Pagan)
        #   - Random effects normality
        #   - No strong autocorrelation (ACF lag-1 < 0.1)
        #   - No influential outliers (Cook's distance < 1.0)
        # Generates diagnostic plots in logs/ folder

        log("Running validate_lmm_assumptions_comprehensive...")
        log("This will generate 6 diagnostic plots in logs/ folder")

        assumptions_result = validate_lmm_assumptions_comprehensive(
            lmm_result=lmm_result,
            data=df_lmm,
            output_dir=RQ_DIR / "logs",
            acf_lag1_threshold=0.1,
            alpha=0.05
        )

        # Report validation results
        if assumptions_result['valid']:
            log(f"PASS - All LMM assumptions validated")
            log(f"Diagnostic plots saved:")
            for plot_path in assumptions_result.get('plot_paths', []):
                log(f"  - {Path(plot_path).name}")
        else:
            log(f"WARNING - {assumptions_result['message']}")
            log(f"Diagnostic details:")
            for key, value in assumptions_result.get('diagnostics', {}).items():
                log(f"  - {key}: {value}")
            log("Assumption violations noted but not fatal - proceeding with analysis")
            log("Violations will be reported in results summary")

        log("Step 02 complete - LMM fitted and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
