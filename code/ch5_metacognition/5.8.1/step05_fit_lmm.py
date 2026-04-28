#!/usr/bin/env python3
"""Fit LMM with Domain x Time Interaction: Fit Linear Mixed Model to test Domain × Time interaction on confidence theta scores."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory_tsvr

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.8.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Ensure immediate write
    print(msg, flush=True)  # Flush console output

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Fit LMM with Domain x Time Interaction")
        # Load Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Verify expected columns
        required_cols = ['composite_ID', 'UID', 'location', 'theta', 'log_TSVR']
        missing_cols = [col for col in required_cols if col not in lmm_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Domains: {sorted(lmm_input['location'].unique())}")
        log(f"Participants: {lmm_input['UID'].nunique()}")
        log(f"Observations: {len(lmm_input)}")
        # Run Analysis Tool

        log("Fitting LMM with formula: theta ~ C(location) * log_TSVR")
        log("Random effects: (1 | UID) - random intercept only")

        # Use statsmodels directly since data is already prepared
        import statsmodels.formula.api as smf

        # Model specification (use C() for categorical domain)
        formula = "theta ~ C(location) * log_TSVR"
        groups = "UID"
        reml = False  # ML estimation for model comparison compatibility

        # Fit LMM using statsmodels directly
        model = smf.mixedlm(
            formula=formula,
            data=lmm_input,
            groups=lmm_input[groups],
            re_formula="~1"  # Random intercept only (simpler, more stable)
        )
        lmm_result = model.fit(reml=reml)

        log("LMM fitting complete")
        log(f"Converged: {lmm_result.converged}")
        log(f"AIC: {lmm_result.aic:.2f}")
        log(f"BIC: {lmm_result.bic:.2f}")
        # Extract Fixed Effects (CORRECT PATTERN from execute.md)
        # CRITICAL: Extract fixed effects ONLY using len(model.exog_names)
        # WRONG approach: model.params includes random effects at the end
        # CORRECT approach: slice params/pvalues using n_fe from exog_names

        log("Extracting fixed effects coefficients...")

        # CORRECT: Get number of fixed effects from model design matrix
        n_fe = len(lmm_result.model.exog_names)
        log(f"Number of fixed effects: {n_fe}")
        log(f"Fixed effect terms: {lmm_result.model.exog_names}")

        # CORRECT: Slice params/pvalues/bse to get ONLY fixed effects
        fixed_params = lmm_result.params[:n_fe]
        fixed_se = lmm_result.bse[:n_fe]
        fixed_z = lmm_result.tvalues[:n_fe]  # t-values in statsmodels (asymptotically z)
        fixed_pvalues = lmm_result.pvalues[:n_fe]

        # Create fixed effects table
        fixed_effects_df = pd.DataFrame({
            'term': lmm_result.model.exog_names,
            'coef': fixed_params.values,
            'se': fixed_se.values,
            'z': fixed_z.values,
            'p_value': fixed_pvalues.values
        })

        log(f"Fixed effects table created ({len(fixed_effects_df)} rows)")

        # Log key results
        log("Fixed Effects:")
        for idx, row in fixed_effects_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            log(f"  {row['term']}: coef={row['coef']:.4f}, p={row['p_value']:.4f} {sig}")

        # Check for Domain × Time interaction
        interaction_terms = [term for term in fixed_effects_df['term'] if ':' in term and 'log_TSVR' in term]
        if interaction_terms:
            log(f"Domain x Time interaction terms found: {interaction_terms}")
            for term in interaction_terms:
                p_val = fixed_effects_df[fixed_effects_df['term'] == term]['p_value'].values[0]
                result = "NULL (p >= 0.05)" if p_val >= 0.05 else "SIGNIFICANT (p < 0.05)"
                log(f"{term}: p={p_val:.4f} -> {result}")
        else:
            log("No interaction terms found (unexpected for domain * log_TSVR model)")
        # Save Analysis Outputs
        # These outputs will be used by: Step 6 (post-hoc contrasts), plotting pipeline, results analysis

        log("Saving fixed effects table...")
        output_coef_path = RQ_DIR / "data" / "step05_lmm_coefficients.csv"
        fixed_effects_df.to_csv(output_coef_path, index=False, encoding='utf-8')
        log(f"{output_coef_path.name} ({len(fixed_effects_df)} rows, {len(fixed_effects_df.columns)} cols)")

        log("Saving full LMM summary...")
        output_summary_path = RQ_DIR / "data" / "step05_lmm_summary.txt"
        with open(output_summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Linear Mixed Model Summary\n")
            f.write("RQ: ch6/6.3.1 - Confidence Domain x Time Interaction\n")
            f.write("=" * 80 + "\n\n")
            f.write(str(lmm_result.summary()))
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Model Specification\n")
            f.write("=" * 80 + "\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Random effects: ~1 (random intercept only)\n")
            f.write(f"Groups: {groups}\n")
            f.write(f"REML: {reml}\n")
            f.write(f"Converged: {lmm_result.converged}\n")
            f.write(f"AIC: {lmm_result.aic:.4f}\n")
            f.write(f"BIC: {lmm_result.bic:.4f}\n")
            f.write(f"Log-Likelihood: {lmm_result.llf:.4f}\n")
            f.write(f"N observations: {lmm_result.nobs}\n")
            f.write(f"N groups: {len(lmm_result.model.group_labels)}\n")

        log(f"{output_summary_path.name}")
        # Run Validation Tool
        # Validates: Model convergence status, no optimization warnings
        # Threshold: converged=True required

        log("Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(lmm_result)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

            # Check convergence
            if not validation_result.get('converged', False):
                log("LMM did not converge - see summary for warnings")
                sys.exit(1)
        else:
            log(f"{validation_result}")

        log("Step 05 complete - Domain x Time interaction tested")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
