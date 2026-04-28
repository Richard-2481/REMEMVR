#!/usr/bin/env python3
"""Fit 5 Candidate LMM Models: Fit 5 candidate LMM models with different functional forms for forgetting curves:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# parents[4] = REMEMVR/ (code -> rq7 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import compare_lmm_models_by_aic

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
LOG_FILE = RQ_DIR / "logs" / "step05_lmm_fitting.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Fit 5 Candidate LMM Models")
        # Load LMM Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"LMM input data missing: {input_path}\n"
                                     "Run step04_prepare_lmm_input.py first")

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")
        log(f"  Columns: {lmm_input.columns.tolist()}")
        log(f"  Theta range: [{lmm_input['Theta'].min():.3f}, {lmm_input['Theta'].max():.3f}]")
        log(f"  Days range: [{lmm_input['Days'].min():.2f}, {lmm_input['Days'].max():.2f}]")
        # STEP 1.5: Transform Column Names for Tool Compatibility
        # The compare_lmm_models_by_aic tool expects specific column names:
        # - 'Ability' as outcome variable (instead of 'Theta')
        # - 'Days_sq' for quadratic term (instead of 'Days_squared')
        # - 'log_Days' for logarithmic term (instead of 'log_Days_plus1')

        log("Renaming columns for tool compatibility...")
        lmm_input = lmm_input.rename(columns={
            'Theta': 'Ability',
            'Days_squared': 'Days_sq',
            'log_Days_plus1': 'log_Days'
        })
        log("  Renamed: Theta -> Ability, Days_squared -> Days_sq, log_Days_plus1 -> log_Days")
        # Fit 5 Candidate Models
        #               fitted models + AIC comparison table + best model selection

        log("Fitting 5 candidate LMM models...")
        log("  Models:")
        log("    (1) Linear: Theta ~ Days")
        log("    (2) Quadratic: Theta ~ Days + Days_squared")
        log("    (3) Logarithmic: Theta ~ log_Days_plus1")
        log("    (4) LinLog: Theta ~ Days + log_Days_plus1")
        log("    (5) QuadLog: Theta ~ Days + Days_squared + log_Days_plus1")
        log("  Random effects: (1 | UID)")
        log("  REML: False (for AIC comparison)")

        # Create save directory for pickle output
        save_dir = RQ_DIR / "data"
        save_dir.mkdir(parents=True, exist_ok=True)

        comparison_results = compare_lmm_models_by_aic(
            data=lmm_input,
            n_factors=1,              # Single outcome variable (Theta)
            reference_group=None,     # No reference group (continuous outcome)
            groups='UID',             # Random intercepts by participant UID
            save_dir=save_dir         # Directory for pickle output
        )

        log("All 5 models fitted")
        log(f"  Best model: {comparison_results['best_model']}")
        # Save Model Comparison Results
        # Output will be used by: Step 6 (AIC model selection)

        # Save AIC comparison table
        comparison_output_path = RQ_DIR / "results" / "step05_model_comparison.csv"
        log(f"Saving model comparison table to {comparison_output_path.name}...")
        comparison_results['aic_comparison'].to_csv(
            comparison_output_path,
            index=False,
            encoding='utf-8'
        )
        log(f"{comparison_output_path.name}")
        log("")
        log("AIC Comparison:")
        log(comparison_results['aic_comparison'].to_string(index=False))
        log("")

        # Pickle file saved by compare_lmm_models_by_aic
        pickle_path = RQ_DIR / "data" / "step05_model_fits.pkl"
        if pickle_path.exists():
            log(f"Model fits pickle: {pickle_path.name}")
        else:
            log(f"Expected pickle file not found: {pickle_path.name}")
        # Validate Model Convergence
        # Validates: All models converged, AIC/BIC finite, no singular covariance

        log("Validating LMM convergence...")

        # Check convergence for each model
        all_converged = True
        for model_name, model_result in comparison_results['models'].items():
            try:
                validation_result = validate_lmm_convergence(lmm_result=model_result)

                if isinstance(validation_result, dict):
                    converged = validation_result.get('converged', False)
                    if converged:
                        log(f"{model_name}: Converged")
                    else:
                        log(f"{model_name}: Did not converge")
                        all_converged = False
                else:
                    log(f"{model_name}: {validation_result}")

            except Exception as e:
                log(f"{model_name}: Validation failed - {str(e)}")
                all_converged = False

        # Check AIC/BIC finite
        aic_comparison = comparison_results['aic_comparison']
        if aic_comparison[['AIC', 'BIC', 'log_likelihood']].isna().any().any():
            log("Some models have NaN AIC/BIC/log_likelihood")
            all_converged = False
        else:
            log("All AIC/BIC/log_likelihood values finite")

        if np.isinf(aic_comparison[['AIC', 'BIC', 'log_likelihood']]).any().any():
            log("Some models have Inf AIC/BIC/log_likelihood")
            all_converged = False
        else:
            log("All AIC/BIC/log_likelihood values finite (not Inf)")

        # Overall validation summary
        if all_converged:
            log("All 5 models converged successfully")
        else:
            log("Some models did not converge - check fitting diagnostics")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
