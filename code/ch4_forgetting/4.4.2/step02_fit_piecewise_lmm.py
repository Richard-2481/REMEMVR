#!/usr/bin/env python3
"""Fit Piecewise LMM with 3-Way Interaction: Fit piecewise linear mixed model testing the 3-way interaction:"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_lmm import fit_lmm_trajectory
from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq6
LOG_FILE = RQ_DIR / "logs" / "step02_fit_piecewise_lmm.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Fit Piecewise LMM with 3-Way Interaction")
        # Load Piecewise Input Data

        log("Loading piecewise LMM input from Step 1...")

        input_path = RQ_DIR / "data" / "step01_lmm_input_piecewise.csv"
        df_lmm = pd.read_csv(input_path, encoding='utf-8')

        log(f"{input_path.name} ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")

        # Restore categorical coding (lost in CSV save/load)
        df_lmm['Congruence'] = pd.Categorical(
            df_lmm['Congruence'],
            categories=['Common', 'Congruent', 'Incongruent'],
            ordered=False
        )
        df_lmm['Segment'] = pd.Categorical(
            df_lmm['Segment'],
            categories=['Early', 'Late'],
            ordered=False
        )

        log("Categorical coding restored (Congruence ref='Common', Segment ref='Early')")
        # Fit Piecewise LMM
        # Formula: theta ~ Days_within * Segment * Congruence (3-way interaction)
        # Random effects: ~Days_within (random intercepts + slopes)

        log("Fitting piecewise LMM with 3-way interaction...")
        log("theta ~ Days_within * C(Segment, Treatment('Early')) * C(Congruence, Treatment('Common'))")
        log("~Days_within (random intercepts + slopes by UID)")

        # Define formula (3-way interaction)
        formula = (
            "theta ~ Days_within * C(Segment, Treatment('Early')) * "
            "C(Congruence, Treatment('Common'))"
        )

        # Fit model using fit_lmm_trajectory (data already in long format with Days_within)
        lmm_model = fit_lmm_trajectory(
            data=df_lmm,
            formula=formula,
            groups='UID',
            re_formula='~Days_within',
            reml=False  # Use ML for model comparison
        )

        log("Model fitting complete")
        # Save Model Object
        # Output: data/step02_piecewise_lmm_model.pkl
        # Format: Pickle (MixedLMResults object)

        log("Saving fitted model object...")

        model_path = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
        lmm_model.save(str(model_path))

        log(f"{model_path.name}")
        # Save Model Summary
        # Output: results/step02_lmm_model_summary.txt
        # Contains: Fixed effects, random effects, fit statistics

        log("Saving model summary...")

        summary_path = RQ_DIR / "results" / "step02_lmm_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PIECEWISE LMM MODEL SUMMARY - RQ 5.6\n")
            f.write("=" * 80 + "\n\n")
            f.write("Formula:\n")
            f.write(f"  {formula}\n\n")
            f.write("Random Effects:\n")
            f.write("  ~Days_within | UID (random intercepts + slopes)\n\n")
            f.write("Model Summary:\n")
            f.write(str(lmm_model.summary()) + "\n")

        log(f"{summary_path.name}")
        # Validate LMM Convergence
        # Checks: Convergence, singular fit, gradient norm

        log("Validating model convergence...")

        validation_result = validate_lmm_convergence(lmm_model)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

            # Check for validation failures
            if 'passed' in validation_result and not validation_result['passed']:
                raise ValueError(
                    f"Validation failed: {validation_result.get('message', 'Unknown error')}"
                )
        else:
            log(f"{validation_result}")

        # Additional checks specific to 3-way interaction model
        n_fixed_effects = len(lmm_model.fe_params)
        if n_fixed_effects != 12:
            raise ValueError(
                f"Fixed effects count incorrect: expected 12 terms "
                f"(1 intercept + 4 main + 5 two-way + 2 three-way), found {n_fixed_effects}"
            )

        log(f"All 12 fixed effect terms present")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
