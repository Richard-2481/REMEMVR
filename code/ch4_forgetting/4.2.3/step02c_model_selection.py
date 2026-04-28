#!/usr/bin/env python3
"""Model Selection for Random Effects: Select optimal random effects structure via likelihood ratio test. Compares 3"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import select_lmm_random_structure_via_lrt

from tools.validation import validate_model_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02c_model_selection.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2c: Model Selection for Random Effects")
        # Load Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step01_lmm_input.csv"
        lmm_input = pd.read_csv(input_path)
        log(f"step01_lmm_input.csv ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Verify expected structure
        expected_rows = 1200  # 100 participants x 4 tests x 3 domains
        if len(lmm_input) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(lmm_input)}")

        required_cols = ['UID', 'theta', 'TSVR_hours', 'log_TSVR', 'Age_c', 'domain']
        missing_cols = [col for col in required_cols if col not in lmm_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Data shape: {lmm_input.shape}")
        log(f"Domains: {sorted(lmm_input['domain'].unique())}")
        log(f"N participants: {lmm_input['UID'].nunique()}")
        # Run Model Selection via Likelihood Ratio Test
        #   compares via LRT, selects simplest adequate model (parsimonious selection)

        log("Running random effects model selection...")

        # Fixed effects formula (3-way Age x Domain x Time interaction)
        # Note: This is the FULL fixed effects formula from Step 2 fit_lmm
        formula = (
            "theta ~ TSVR_hours + log_TSVR + Age_c + domain + "
            "TSVR_hours:Age_c + log_TSVR:Age_c + "
            "TSVR_hours:domain + log_TSVR:domain + Age_c:domain + "
            "TSVR_hours:Age_c:domain + log_TSVR:Age_c:domain"
        )

        log(f"Fixed effects formula: {formula}")
        log("Comparing 3 random structures:")
        log("  1. Full: random intercepts + slopes with correlation")
        log("  2. Uncorrelated: random intercepts + slopes without correlation")
        log("  3. Intercept-only: random intercepts only")
        log("Using ML estimation (REML=False) for valid LRT comparison")
        log("Parsimonious selection: prefer simpler if LRT p >= 0.05")

        selection_result = select_lmm_random_structure_via_lrt(
            data=lmm_input,
            formula=formula,
            groups='UID',
            reml=False  # ML estimation required for valid LRT
        )

        log("Model selection complete")
        log(f"Selected model: {selection_result['selected_model']}")
        # Save Model Selection Report
        # These outputs document the model selection process and selected model

        log("Saving model selection report...")

        # Save text report
        report_path = RQ_DIR / "results" / "step02c_model_selection.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 2c: RANDOM EFFECTS MODEL SELECTION\n")
            f.write("=" * 80 + "\n\n")

            f.write("OBJECTIVE:\n")
            f.write("Select optimal random effects structure via likelihood ratio test.\n")
            f.write("Compares nested models: Full vs Intercept-only.\n\n")

            f.write("METHOD:\n")
            f.write("- Likelihood Ratio Test (LRT) with chi-square distribution\n")
            f.write("- ML estimation (REML=False) for valid LRT comparison\n")
            f.write("- Parsimonious selection: prefer simpler if p >= 0.05\n")
            f.write("- Reference: Pinheiro & Bates (2000), Verbeke & Molenberghs (2000)\n\n")

            f.write("CANDIDATE MODELS FITTED:\n")
            for model_name, model_obj in selection_result['fitted_models'].items():
                if model_obj is not None:
                    f.write(f"  - {model_name}: Converged = {model_obj.converged}\n")
                    f.write(f"    AIC = {model_obj.aic:.2f}, BIC = {model_obj.bic:.2f}\n")
                    f.write(f"    Log-likelihood = {model_obj.llf:.2f}\n")
                else:
                    f.write(f"  - {model_name}: FAILED TO CONVERGE\n")
            f.write("\n")

            f.write("LIKELIHOOD RATIO TEST RESULTS:\n")
            lrt_df = selection_result['lrt_results']
            f.write(lrt_df.to_string(index=False))
            f.write("\n\n")

            f.write("SELECTED MODEL:\n")
            f.write(f"  {selection_result['selected_model']}\n\n")

            f.write("RATIONALE:\n")
            if selection_result['selected_model'] == 'Intercept-only':
                f.write("  Full model did not significantly improve fit over Intercept-only\n")
                f.write("  (LRT p >= 0.05). Selected simpler Intercept-only model per\n")
                f.write("  parsimonious selection principle.\n")
            elif selection_result['selected_model'] == 'Full':
                f.write("  Full model significantly improved fit over Intercept-only\n")
                f.write("  (LRT p < 0.05). Random slopes are necessary.\n")
            else:
                f.write("  Fallback model selected due to convergence issues.\n")
            f.write("\n")

            f.write("IMPLICATIONS:\n")
            if selection_result['selected_model'] == 'Intercept-only':
                f.write("  - Forgetting rates do not vary significantly across individuals\n")
                f.write("  - Individual differences primarily in baseline memory (intercepts)\n")
                f.write("  - Fixed effects reflect population-average trajectories\n")
            elif selection_result['selected_model'] == 'Full':
                f.write("  - Forgetting rates vary significantly across individuals\n")
                f.write("  - Both baseline memory and rates of forgetting show individual\n")
                f.write("    differences (random intercepts and slopes)\n")
                f.write("  - Fixed effects reflect conditional trajectories (controlling for\n")
                f.write("    random effects)\n")

            f.write("\n")
            f.write("=" * 80 + "\n")

        log(f"step02c_model_selection.txt")
        # Update Step 2 Outputs with Selected Model
        # Replace step02_lmm_model.pkl and step02_fixed_effects.csv with selected model

        log("Updating Step 2 outputs with selected model...")

        # Get selected model object
        selected_model_name = selection_result['selected_model']
        selected_model = selection_result['fitted_models'][selected_model_name]

        if selected_model is None:
            raise ValueError(f"Selected model '{selected_model_name}' failed to converge")

        # Save selected model as step02_lmm_model.pkl
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(selected_model, f)
        log(f"step02_lmm_model.pkl (selected model: {selected_model_name})")

        # Extract and save fixed effects table
        fixed_effects_df = pd.DataFrame({
            'term': selected_model.params.index,
            'estimate': selected_model.params.values,
            'se': selected_model.bse.values,
            'z': selected_model.tvalues.values,
            'p': selected_model.pvalues.values,
            'CI_lower': selected_model.conf_int()[0].values,
            'CI_upper': selected_model.conf_int()[1].values
        })

        fixed_effects_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fixed_effects_df.to_csv(fixed_effects_path, index=False, encoding='utf-8')
        log(f"step02_fixed_effects.csv ({len(fixed_effects_df)} terms)")
        # Run Validation Tool
        # Validates: Selected model converged successfully

        log("Running validate_model_convergence...")
        validation_result = validate_model_convergence(
            lmm_result=selected_model
        )

        # Report validation results
        if isinstance(validation_result, dict):
            log(f"Valid: {validation_result.get('valid', False)}")
            log(f"Converged: {validation_result.get('converged', False)}")
            log(f"Message: {validation_result.get('message', 'N/A')}")

            if not validation_result.get('valid', False):
                raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")
        else:
            log(f"{validation_result}")

        # Additional validation checks
        log("Checking LRT comparisons...")
        n_comparisons = len(selection_result['lrt_results'])
        log(f"Number of LRT comparisons: {n_comparisons}")

        if n_comparisons == 0:
            raise ValueError("No LRT comparisons performed")

        log("Checking selected model is refit with REML=False...")
        if selected_model.reml:
            raise ValueError("Selected model was fit with REML=True (should be REML=False for inference)")
        log("Selected model correctly fit with REML=False")

        log("Step 2c complete")
        log(f"Selected random effects structure: {selected_model_name}")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
