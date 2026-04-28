#!/usr/bin/env python3
"""Fit Piecewise LMM with 3-Way Interaction: Fit piecewise Linear Mixed Model testing the 3-way interaction:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
import pickle

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_fit_piecewise_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Ensure log directory exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        log("Step 01: Fit Piecewise LMM with 3-Way Interaction")
        log("="*70)
        # Load Input Data

        log("Loading input data from step00...")

        input_path = RQ_DIR / "data" / "step00_piecewise_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_lmm = pd.read_csv(input_path)
        log(f"step00_piecewise_lmm_input.csv ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'domain', 'theta', 'Segment', 'Days_within']
        missing_cols = set(required_cols) - set(df_lmm.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")

        # Log data summary
        log(f"Unique UIDs: {df_lmm['UID'].nunique()}")
        log(f"Segments: {df_lmm['Segment'].unique().tolist()}")
        log(f"Domains: {df_lmm['domain'].unique().tolist()}")
        log(f"Days_within range: [{df_lmm['Days_within'].min():.2f}, {df_lmm['Days_within'].max():.2f}]")
        # Run Analysis Tool

        log("")
        log("Running fit_lmm_trajectory...")
        log("-"*70)

        # Define model parameters
        # Formula: 3-way interaction testing slope differences by Segment and Domain
        # Reference levels: Early segment, What domain (via Treatment coding)
        # This means (with When excluded, only What/Where remain):
        #   - Days_within main effect = slope in Early-What
        #   - Segment[T.Late] interaction = slope change from Early to Late in What
        #   - domain[T.where] interaction = slope difference from What in Early
        #   - 3-way = whether slope differences vary by segment

        formula = "theta ~ Days_within * C(Segment, Treatment('Early')) * C(domain, Treatment('what'))"
        groups = "UID"
        re_formula = "~Days_within"  # Random intercepts + slopes per participant
        reml = False  # Use ML for model comparison capability

        log(f"Formula: {formula}")
        log(f"Groups: {groups}")
        log(f"RE Formula: {re_formula}")
        log(f"REML: {reml}")

        # Fit the model
        lmm_model = fit_lmm_trajectory(
            data=df_lmm,
            formula=formula,
            groups=groups,
            re_formula=re_formula,
            reml=reml
        )

        log("Model fitting complete")
        # Save Analysis Outputs
        # These outputs will be used by:
        #   - Step 02: Extract slopes from model
        #   - Step 03: Compute contrasts using model
        #   - Step 05: Generate predictions for plots

        log("")
        log("Saving model outputs...")
        log("-"*70)

        # Save pickle model
        model_path = RQ_DIR / "data" / "step01_piecewise_lmm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(lmm_model, f)
        log(f"data/step01_piecewise_lmm_model.pkl")

        # Save model summary
        summary_path = RQ_DIR / "results" / "step01_piecewise_lmm_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_text = lmm_model.summary().as_text()
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PIECEWISE LMM MODEL SUMMARY\n")
            f.write("RQ 5.2: Consolidation Effects by Memory Domain\n")
            f.write("="*70 + "\n\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Groups: {groups}\n")
            f.write(f"Random Effects: {re_formula}\n")
            f.write(f"REML: {reml}\n\n")
            f.write("-"*70 + "\n")
            f.write("MODEL OUTPUT\n")
            f.write("-"*70 + "\n\n")
            f.write(summary_text)
        log(f"results/step01_piecewise_lmm_summary.txt")

        # Log key model statistics
        log("")
        log("Model Statistics:")
        log(f"  - Number of observations: {lmm_model.nobs}")
        log(f"  - Number of groups (UIDs): {len(lmm_model.model.group_labels)}")
        log(f"  - Log-likelihood: {lmm_model.llf:.4f}")
        log(f"  - AIC: {lmm_model.aic:.4f}")
        log(f"  - BIC: {lmm_model.bic:.4f}")
        # Run Validation Tool
        # Validates: Model convergence and structure
        # Criteria: Converged, no singular fit, all terms present

        log("")
        log("Running validate_lmm_convergence...")
        log("-"*70)

        validation_result = validate_lmm_convergence(lmm_model)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation: Check expected fixed effects
        log("")
        log("Verifying fixed effect structure...")

        fe_names = lmm_model.fe_params.index.tolist()
        log(f"Fixed effects ({len(fe_names)} terms):")
        for i, name in enumerate(fe_names):
            coef = lmm_model.fe_params[name]
            pval = lmm_model.pvalues[name]
            log(f"  {i+1}. {name}: {coef:.4f} (p={pval:.4f})")

        # Check we have the expected structure (intercept + 4 interaction terms for 2 domains)
        # With 2 domains (What/Where) and 2 segments (Early/Late):
        # Intercept, Days_within, Segment[T.Late], domain[T.where],
        # Days_within:Segment[T.Late], Days_within:domain[T.where],
        # Segment[T.Late]:domain[T.where], Days_within:Segment[T.Late]:domain[T.where]
        expected_min_terms = 5  # Intercept + at least 4 interaction terms
        if len(fe_names) >= expected_min_terms:
            log(f"Model has {len(fe_names)} fixed effects (expected >= {expected_min_terms})")
        else:
            log(f"Model has only {len(fe_names)} fixed effects (expected >= {expected_min_terms})")

        # Check random effects variance
        log("")
        log("Random effects structure...")
        re_cov = lmm_model.cov_re
        log(f"Random effects covariance matrix shape: {re_cov.shape}")

        # Check for singular fit (variance ~0)
        diag_var = np.diag(re_cov)
        log(f"Random effect variances: {diag_var}")

        if np.all(diag_var > 1e-6):
            log("No singular fit detected (all RE variances > 0)")
        else:
            log("Potential singular fit - some RE variances near zero")

        log("")
        log("="*70)
        log("Step 01 complete: Piecewise LMM fitted successfully")
        log("="*70)
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
