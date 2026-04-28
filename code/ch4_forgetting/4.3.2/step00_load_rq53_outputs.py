#!/usr/bin/env python3
"""Load RQ 5.3 Outputs: Load fitted LMM model and theta data from RQ 5.3 for contrast analysis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_convergence

# Statsmodels for loading model
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.2
RQ3_DIR = RQ_DIR.parent / "5.3.1"  # results/ch5/5.3.1 (dependency)
LOG_FILE = RQ_DIR / "logs" / "step00_load_rq53_outputs.log"


# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load RQ 5.3 Outputs")
        # Verify Dependency Files Exist

        log("Verifying RQ 5.3 dependency files...")

        model_path = RQ3_DIR / "data" / "step05_lmm_fitted_model.pkl"
        input_path = RQ3_DIR / "data" / "step04_lmm_input.csv"
        comparison_path = RQ3_DIR / "data" / "step05_model_comparison.csv"

        for path, name in [(model_path, "LMM model"),
                           (input_path, "LMM input data"),
                           (comparison_path, "Model comparison")]:
            if not path.exists():
                raise FileNotFoundError(f"RQ 5.3 dependency missing: {name} at {path}")
            log(f"Found {name}: {path}")
        # Load Model Comparison and Verify Best Model

        log("Loading model comparison results...")
        model_comparison = pd.read_csv(comparison_path)
        log(f"model_comparison: {len(model_comparison)} models compared")

        # Find best model (lowest AIC = first row after sorting by delta_AIC)
        best_model_name = model_comparison.loc[model_comparison['delta_AIC'] == 0, 'model_name'].iloc[0]
        best_aic = model_comparison.loc[model_comparison['delta_AIC'] == 0, 'AIC'].iloc[0]

        log(f"Best model: {best_model_name} (AIC = {best_aic:.2f})")

        if best_model_name != "Log":
            log(f"Expected Log model to be best, but found {best_model_name}")
            log("Proceeding anyway - contrast analysis will use whatever model was fit")
        # Load Fitted LMM Model

        log("Loading fitted LMM model...")
        lmm_model = MixedLMResults.load(str(model_path))
        log("LMM model object successfully loaded")

        # Verify model attributes exist
        if not hasattr(lmm_model, 'params'):
            raise ValueError("Model missing 'params' attribute")
        if not hasattr(lmm_model, 'cov_params'):
            raise ValueError("Model missing 'cov_params' method")
        if not hasattr(lmm_model, 'nobs'):
            raise ValueError("Model missing 'nobs' attribute")

        log(f"Model observations: {lmm_model.nobs}")
        log(f"Number of fixed effects: {len(lmm_model.params)}")
        # Load Theta Data

        log("Loading theta data...")
        theta_data = pd.read_csv(input_path)
        log(f"theta_data: {len(theta_data)} rows, {len(theta_data.columns)} cols")

        # Verify required columns
        required_cols = ["composite_ID", "UID", "test", "TSVR_hours", "TSVR_hours_log", "paradigm", "theta"]
        missing_cols = [c for c in required_cols if c not in theta_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check paradigm levels
        paradigms = theta_data['paradigm'].unique()
        log(f"Paradigm levels: {list(paradigms)}")

        if len(paradigms) != 3:
            raise ValueError(f"Expected 3 paradigm levels, found {len(paradigms)}: {paradigms}")
        # Run Validation Tool
        # Validates: Model convergence status

        log("Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(lmm_model)

        for key, value in validation_result.items():
            log(f"{key}: {value}")

        if not validation_result.get('converged', False):
            raise ValueError(f"Model validation failed: {validation_result.get('message', 'Unknown error')}")
        # Save Confirmation File
        # Output: data/step00_model_loaded.txt
        # Contains: Model details confirming successful load

        log("Writing confirmation file...")
        confirmation_path = RQ_DIR / "data" / "step00_model_loaded.txt"
        confirmation_path.parent.mkdir(parents=True, exist_ok=True)

        confirmation_text = f"""RQ 5.3.2 - Step 00: Model Load Confirmation
==============================================

Source Files:
- Model: {model_path}
- Input Data: {input_path}
- Model Comparison: {comparison_path}

Best Model: {best_model_name}
Best Model AIC: {best_aic:.4f}

Model Statistics:
- Number of observations: {lmm_model.nobs}
- Number of fixed effects: {len(lmm_model.params)}
- Converged: {validation_result.get('converged', False)}

Fixed Effect Names:
{chr(10).join(['- ' + str(p) for p in lmm_model.params.index])}

Paradigm Levels in Data:
{chr(10).join(['- ' + str(p) for p in sorted(paradigms)])}

Theta Data Summary:
- Total rows: {len(theta_data)}
- Unique participants: {theta_data['UID'].nunique()}
- Tests per participant: ~{len(theta_data) // theta_data['UID'].nunique() // len(paradigms)}

Validation Status: PASSED
"""

        with open(confirmation_path, 'w', encoding='utf-8') as f:
            f.write(confirmation_text)

        log(f"{confirmation_path}")

        log("Step 00 complete - RQ 5.3 outputs loaded and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
