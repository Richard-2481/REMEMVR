#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA (Updated 2025-12-03 for When domain exclusion)
# =============================================================================
"""
Step ID: step03
Step Name: Fit Parallel LMMs (IRT Model + CTT Model)
RQ: results/ch5/5.2.4
Generated: 2025-11-29
Updated: 2025-12-03 (When domain exclusion applied)

PURPOSE:
Fit identical LMM structures for IRT and CTT scores to compare trajectory
patterns and statistical significance. This enables direct comparison of how
IRT theta scores vs CTT mean scores model memory forgetting over time.

**CRITICAL: When domain is EXCLUDED due to floor effects discovered in RQ 5.2.1**

EXPECTED INPUTS:
  - data/step00_irt_theta_loaded.csv
    Columns: ['composite_ID', 'theta_what', 'theta_where'] (NO theta_when)
    Format: Wide-format IRT theta scores from RQ 5.2.1
    Expected rows: 400 (100 participants × 4 tests)

  - data/step01_ctt_scores.csv
    Columns: ['composite_ID', 'UID', 'test', 'domain', 'CTT_score', 'n_items']
    Format: Long-format CTT mean scores per domain (When excluded)
    Expected rows: 800 (400 participant-tests × 2 domains)

  - data/step00_tsvr_loaded.csv
    Columns: ['composite_ID', 'UID', 'test', 'TSVR_hours']
    Format: Time variable (hours since encoding)
    Expected rows: 400 (100 participants × 4 tests)

EXPECTED OUTPUTS:
  - data/step03_irt_lmm_input.csv
    Columns: ['composite_ID', 'UID', 'test', 'domain', 'TSVR_hours', 'IRT_score']
    Format: Long-format IRT LMM input
    Expected rows: 800 (400 participant-tests × 2 domains - When excluded)

  - data/step03_ctt_lmm_input.csv
    Columns: ['composite_ID', 'UID', 'test', 'domain', 'TSVR_hours', 'CTT_score']
    Format: Long-format CTT LMM input
    Expected rows: 800 (400 participant-tests × 2 domains - When excluded)

  - results/step03_irt_lmm_summary.txt
    Format: IRT model summary (fixed effects, random effects, AIC, BIC)

  - results/step03_ctt_lmm_summary.txt
    Format: CTT model summary

  - results/step03_irt_lmm_fixed_effects.csv
    Columns: ['term', 'estimate', 'SE', 'z', 'p_uncorrected']
    Format: IRT model fixed effects table
    Expected rows: 8-12 (depends on random structure)

  - results/step03_ctt_lmm_fixed_effects.csv
    Columns: ['term', 'estimate', 'SE', 'z', 'p_uncorrected']
    Format: CTT model fixed effects table
    Expected rows: 8-12 (depends on random structure)

  - logs/step03_convergence_report.txt
    Format: Convergence decisions and random structure simplifications

VALIDATION CRITERIA:
  - Model converged (no convergence warnings)
  - No singular fit (random effects variance > 0)
  - Minimum 100 observations used
  - All fixed effects have finite estimates (no NaN/Inf)
  - BOTH models converged OR BOTH simplified to same random structure (parallelism requirement)

g_code REASONING:
- Approach: Fit identical LMM formula to IRT and CTT scores separately,
  allowing direct comparison of trajectory patterns and statistical significance.
  Formula: Score ~ log_TSVR * domain (Log model, per RQ 5.2.1 best fit)
  Random: log_TSVR | UID (random slopes on log-time, per RQ 5.2.1)

- Why this approach: RQ 5.2.1 identified Log model as best fit (AIC=3187.96, weight=0.619).
  Random slope on log_TSVR captures individual differences in forgetting rate.
  5.2.1 showed log_Days Var = 0.046 (meaningful individual variation).
  Domain interactions test if What/Where domains show different trajectories (When excluded).
  Parallel structure ensures fair IRT vs CTT comparison.

- When exclusion: Floor effects (6-9% probability at encoding) make forgetting analysis invalid.

- Data flow:
  IRT wide (theta_what, theta_where) -> long (domain, IRT_score) - NO theta_when
  CTT already long (domain, CTT_score) - When already excluded
  Merge both with TSVR on UID + test
  Fit identical models to IRT_score and CTT_score

- Expected performance: ~30 seconds per model (random slopes), ~10 seconds (intercepts-only)

IMPLEMENTATION NOTES:
- Analysis tool: statsmodels.regression.mixed_linear_model.MixedLM
- Validation tool: tools.validation.validate_lmm_convergence
- Parameters:
  - Formula: Score ~ (TSVR_hours + log(TSVR_hours + 1)) * domain
  - Random: TSVR_hours | UID (preferred), 1 | UID (fallback)
  - Method: REML
  - Convergence strategy: Attempt random slopes, simplify to intercepts-only if EITHER fails
  - Identical structure: CRITICAL - both models must use same random structure for fair comparison
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import warnings

# Add project root to path for imports
# CRITICAL: RQ scripts are in results/chX/rqY/code/ (4 levels deep from project root)
# Path hierarchy from script location:
#   parents[0] = code/ (immediate parent)
#   parents[1] = rqY/
#   parents[2] = chX/
#   parents[3] = results/
#   parents[4] = REMEMVR/ (project root - THIS is what we need for imports)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults

# Import validation tool
from tools.validation import validate_lmm_convergence

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_lmm.log"

# =============================================================================
# FOLDER CONVENTIONS (MANDATORY - NO EXCEPTIONS)
# =============================================================================
#
# code/   : ONLY .py scripts (generated by g_code)
# data/   : ALL data outputs (.csv, .pkl, .txt) - ANY file produced by code
# logs/   : ONLY .log files - execution logs
# plots/  : ONLY image files (.png, .pdf, .svg) - actual plot images
# results/: ONLY final summary reports (.md, .html)
# docs/   : RQ documentation (concept, plan, analysis specs)
#
# NAMING CONVENTION (MANDATORY):
# ALL files in data/ and logs/ MUST be prefixed with step number:
#   - stepXX_descriptive_name.csv
#   - stepXX_descriptive_name.pkl
#   - stepXX_descriptive_name.log
#
# Examples:
#   CORRECT: data/step03_irt_lmm_input.csv
#   CORRECT: results/step03_irt_lmm_summary.txt
#   WRONG:   results/irt_lmm_input.csv  (CSV should be in data/)
#   WRONG:   data/lmm_input.csv         (missing step prefix)

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 03: Fit Parallel LMMs (IRT + CTT)")

        # =========================================================================
        # STEP 1: Load Input Data
        # =========================================================================
        # Expected: IRT theta scores (wide format), CTT scores (long format), TSVR
        # Purpose: Prepare data for LMM fitting with identical structures

        log("[LOAD] Loading input data...")

        # Load IRT theta scores (wide format)
        irt_theta = pd.read_csv(RQ_DIR / "data/step00_irt_theta_loaded.csv")
        log(f"[LOADED] IRT theta ({len(irt_theta)} rows, {len(irt_theta.columns)} cols)")

        # Load CTT scores (already long format)
        ctt_scores = pd.read_csv(RQ_DIR / "data/step01_ctt_scores.csv")
        log(f"[LOADED] CTT scores ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")

        # Load TSVR time variable
        tsvr_data = pd.read_csv(RQ_DIR / "data/step00_tsvr_loaded.csv")
        log(f"[LOADED] TSVR data ({len(tsvr_data)} rows, {len(tsvr_data.columns)} cols)")

        # =========================================================================
        # STEP 2: Prepare LMM Input Datasets
        # =========================================================================
        # Tool: Manual pandas operations (reshape IRT to long, merge with TSVR)
        # What it does: Creates parallel long-format datasets for IRT and CTT
        # Expected output: Two identical-structure datasets with different score columns

        log("[PREPARE] Reshaping IRT theta to long format (When excluded)...")

        # Reshape IRT theta from wide to long (NO theta_when)
        # Wide: composite_ID, theta_what, theta_where
        # Long: composite_ID, domain, IRT_score
        irt_long = pd.melt(
            irt_theta,
            id_vars=['composite_ID'],
            value_vars=['theta_what', 'theta_where'],  # NO theta_when
            var_name='theta_col',
            value_name='IRT_score'
        )

        # Map theta column names to domain names (When excluded)
        domain_mapping = {
            'theta_what': 'What',
            'theta_where': 'Where'
            # 'theta_when': 'When' - EXCLUDED
        }
        irt_long['domain'] = irt_long['theta_col'].map(domain_mapping)
        irt_long = irt_long.drop(columns=['theta_col'])

        log(f"[RESHAPED] IRT long format ({len(irt_long)} rows - When excluded)")

        # Extract UID and test from composite_ID for both datasets
        log("[PREPARE] Extracting UID and test from composite_ID...")

        # IRT long: add UID and test columns
        irt_long[['UID', 'test']] = irt_long['composite_ID'].str.split('_', expand=True)
        irt_long['test'] = irt_long['test'].astype(int)  # Convert to int to match TSVR

        # CTT already has UID and test columns from step01
        # Just need to ensure composite_ID exists
        if 'composite_ID' not in ctt_scores.columns:
            ctt_scores['composite_ID'] = ctt_scores['UID'] + '_' + ctt_scores['test'].astype(str)

        # Capitalize domain names to match IRT (what -> What, where -> Where)
        ctt_scores['domain'] = ctt_scores['domain'].str.capitalize()

        log("[PREPARE] Merging with TSVR time variable...")

        # Merge IRT with TSVR on UID + test
        irt_lmm_input = pd.merge(
            irt_long,
            tsvr_data[['UID', 'test', 'TSVR_hours']],
            on=['UID', 'test'],
            how='left'
        )

        # Merge CTT with TSVR on UID + test
        ctt_lmm_input = pd.merge(
            ctt_scores,
            tsvr_data[['UID', 'test', 'TSVR_hours']],
            on=['UID', 'test'],
            how='left'
        )

        # Reorder columns to match specification
        irt_lmm_input = irt_lmm_input[['composite_ID', 'UID', 'test', 'domain', 'TSVR_hours', 'IRT_score']]
        ctt_lmm_input = ctt_lmm_input[['composite_ID', 'UID', 'test', 'domain', 'TSVR_hours', 'CTT_score']]

        log(f"[PREPARED] IRT LMM input ({len(irt_lmm_input)} rows)")
        log(f"[PREPARED] CTT LMM input ({len(ctt_lmm_input)} rows)")

        # Check for missing TSVR values
        irt_missing_tsvr = irt_lmm_input['TSVR_hours'].isna().sum()
        ctt_missing_tsvr = ctt_lmm_input['TSVR_hours'].isna().sum()
        if irt_missing_tsvr > 0 or ctt_missing_tsvr > 0:
            log(f"[WARNING] Missing TSVR values: IRT={irt_missing_tsvr}, CTT={ctt_missing_tsvr}")

        # =========================================================================
        # STEP 3: Save LMM Input Datasets
        # =========================================================================
        # These inputs will be used by: Step 4 (assumption validation), downstream analyses

        log("[SAVE] Saving LMM input datasets...")

        irt_lmm_input.to_csv(RQ_DIR / "data/step03_irt_lmm_input.csv", index=False, encoding='utf-8')
        log(f"[SAVED] data/step03_irt_lmm_input.csv ({len(irt_lmm_input)} rows, {len(irt_lmm_input.columns)} cols)")

        ctt_lmm_input.to_csv(RQ_DIR / "data/step03_ctt_lmm_input.csv", index=False, encoding='utf-8')
        log(f"[SAVED] data/step03_ctt_lmm_input.csv ({len(ctt_lmm_input)} rows, {len(ctt_lmm_input.columns)} cols)")

        # =========================================================================
        # STEP 4: Create Log-Transformed Time Variable
        # =========================================================================
        # Purpose: Dual time predictors (linear + log) capture both immediate drop and gradual decay

        log("[PREPARE] Creating log-transformed time variable...")

        irt_lmm_input['log_TSVR'] = np.log(irt_lmm_input['TSVR_hours'] + 1)
        ctt_lmm_input['log_TSVR'] = np.log(ctt_lmm_input['TSVR_hours'] + 1)

        log("[PREPARED] Log-transformed TSVR created")

        # =========================================================================
        # STEP 5: Fit LMM Models (with convergence strategy)
        # =========================================================================
        # Tool: statsmodels MixedLM
        # What it does: Fits identical LMM formula to IRT and CTT scores
        # Expected output: Two fitted models with same random structure
        # Convergence strategy: Attempt random slopes, fallback to intercepts-only if either fails

        log("[ANALYSIS] Fitting LMM models...")
        log("[STRATEGY] Convergence: Attempt random slopes, simplify if either model fails")

        # Define formulas
        # CRITICAL: RQ 5.2.1 identified Log model as best fit (AIC=3187.96)
        # Random slope must be on log_TSVR, NOT linear TSVR_hours
        # 5.2.1 showed log_Days Var = 0.046 (meaningful individual differences in log-decay)
        fixed_formula = "score ~ log_TSVR * C(domain)"
        re_formula_slopes = "log_TSVR"  # Random slopes on LOG time (per 5.2.1)
        re_formula_intercepts = "1"  # Intercepts-only (fallback)

        convergence_report = []

        # PHASE 1: Attempt random slopes for BOTH models
        log("[PHASE 1] Attempting random slopes: TSVR_hours | UID")

        try:
            # Fit IRT model with random slopes
            log("[FIT] IRT model with random slopes...")
            irt_lmm_input_copy = irt_lmm_input.rename(columns={'IRT_score': 'score'})

            irt_model_slopes = MixedLM.from_formula(
                fixed_formula,
                data=irt_lmm_input_copy,
                groups=irt_lmm_input_copy['UID'],
                re_formula=re_formula_slopes
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                irt_result_slopes = irt_model_slopes.fit(reml=True)
                irt_warnings = [str(warning.message) for warning in w]

            irt_slopes_converged = irt_result_slopes.converged
            log(f"[IRT SLOPES] Converged: {irt_slopes_converged}")
            if irt_warnings:
                log(f"[IRT SLOPES] Warnings: {irt_warnings}")

            convergence_report.append(f"IRT model with random slopes: converged={irt_slopes_converged}")
            if irt_warnings:
                convergence_report.append(f"  Warnings: {irt_warnings}")

        except Exception as e:
            log(f"[IRT SLOPES] Failed: {str(e)}")
            irt_slopes_converged = False
            irt_result_slopes = None
            convergence_report.append(f"IRT model with random slopes: FAILED ({str(e)})")

        try:
            # Fit CTT model with random slopes
            log("[FIT] CTT model with random slopes...")
            ctt_lmm_input_copy = ctt_lmm_input.rename(columns={'CTT_score': 'score'})

            ctt_model_slopes = MixedLM.from_formula(
                fixed_formula,
                data=ctt_lmm_input_copy,
                groups=ctt_lmm_input_copy['UID'],
                re_formula=re_formula_slopes
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ctt_result_slopes = ctt_model_slopes.fit(reml=True)
                ctt_warnings = [str(warning.message) for warning in w]

            ctt_slopes_converged = ctt_result_slopes.converged
            log(f"[CTT SLOPES] Converged: {ctt_slopes_converged}")
            if ctt_warnings:
                log(f"[CTT SLOPES] Warnings: {ctt_warnings}")

            convergence_report.append(f"CTT model with random slopes: converged={ctt_slopes_converged}")
            if ctt_warnings:
                convergence_report.append(f"  Warnings: {ctt_warnings}")

        except Exception as e:
            log(f"[CTT SLOPES] Failed: {str(e)}")
            ctt_slopes_converged = False
            ctt_result_slopes = None
            convergence_report.append(f"CTT model with random slopes: FAILED ({str(e)})")

        # DECISION: Use random slopes if BOTH converged, else use intercepts-only for BOTH
        if irt_slopes_converged and ctt_slopes_converged:
            log("[DECISION] BOTH models converged with random slopes -> Use random slopes")
            irt_result_final = irt_result_slopes
            ctt_result_final = ctt_result_slopes
            final_random_structure = "random slopes (TSVR_hours | UID)"
            convergence_report.append("DECISION: BOTH converged with slopes -> Use random slopes for both")
        else:
            log("[DECISION] At least one model failed with random slopes -> Simplify BOTH to intercepts-only")
            convergence_report.append("DECISION: At least one model failed with slopes -> Simplify BOTH to intercepts-only")

            # PHASE 2: Fit intercepts-only for BOTH models
            log("[PHASE 2] Fitting intercepts-only: 1 | UID")

            try:
                log("[FIT] IRT model with intercepts-only...")
                irt_model_intercepts = MixedLM.from_formula(
                    fixed_formula,
                    data=irt_lmm_input_copy,
                    groups=irt_lmm_input_copy['UID'],
                    re_formula=re_formula_intercepts
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    irt_result_intercepts = irt_model_intercepts.fit(reml=True)
                    irt_warnings_int = [str(warning.message) for warning in w]

                log(f"[IRT INTERCEPTS] Converged: {irt_result_intercepts.converged}")
                if irt_warnings_int:
                    log(f"[IRT INTERCEPTS] Warnings: {irt_warnings_int}")

                convergence_report.append(f"IRT model with intercepts-only: converged={irt_result_intercepts.converged}")
                if irt_warnings_int:
                    convergence_report.append(f"  Warnings: {irt_warnings_int}")

                irt_result_final = irt_result_intercepts

            except Exception as e:
                log(f"[ERROR] IRT intercepts-only failed: {str(e)}")
                convergence_report.append(f"IRT model with intercepts-only: FAILED ({str(e)})")
                raise ValueError(f"IRT model failed to converge even with intercepts-only: {str(e)}")

            try:
                log("[FIT] CTT model with intercepts-only...")
                ctt_model_intercepts = MixedLM.from_formula(
                    fixed_formula,
                    data=ctt_lmm_input_copy,
                    groups=ctt_lmm_input_copy['UID'],
                    re_formula=re_formula_intercepts
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ctt_result_intercepts = ctt_model_intercepts.fit(reml=True)
                    ctt_warnings_int = [str(warning.message) for warning in w]

                log(f"[CTT INTERCEPTS] Converged: {ctt_result_intercepts.converged}")
                if ctt_warnings_int:
                    log(f"[CTT INTERCEPTS] Warnings: {ctt_warnings_int}")

                convergence_report.append(f"CTT model with intercepts-only: converged={ctt_result_intercepts.converged}")
                if ctt_warnings_int:
                    convergence_report.append(f"  Warnings: {ctt_warnings_int}")

                ctt_result_final = ctt_result_intercepts

            except Exception as e:
                log(f"[ERROR] CTT intercepts-only failed: {str(e)}")
                convergence_report.append(f"CTT model with intercepts-only: FAILED ({str(e)})")
                raise ValueError(f"CTT model failed to converge even with intercepts-only: {str(e)}")

            final_random_structure = "intercepts-only (1 | UID)"

        log(f"[DONE] Analysis complete with {final_random_structure}")

        # =========================================================================
        # STEP 6: Save Model Summaries and Fixed Effects
        # =========================================================================
        # These outputs will be used by: Step 5 (coefficient comparison), Step 6 (AIC/BIC)

        log("[SAVE] Saving model summaries...")

        # Save IRT model summary
        with open(RQ_DIR / "results/step03_irt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(irt_result_final.summary()))
        log("[SAVED] results/step03_irt_lmm_summary.txt")

        # Save CTT model summary
        with open(RQ_DIR / "results/step03_ctt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(ctt_result_final.summary()))
        log("[SAVED] results/step03_ctt_lmm_summary.txt")

        # Extract and save fixed effects tables
        log("[EXTRACT] Extracting fixed effects...")

        # IRT fixed effects (use params method for robust extraction)
        irt_fe = pd.DataFrame({
            'term': irt_result_final.fe_params.index.tolist(),
            'estimate': irt_result_final.fe_params.values.tolist(),
            'SE': irt_result_final.bse_fe.values.tolist(),
            'z': [irt_result_final.fe_params[i] / irt_result_final.bse_fe[i] for i in range(len(irt_result_final.fe_params))],
            'p_uncorrected': irt_result_final.pvalues[:len(irt_result_final.fe_params)].tolist()
        })
        irt_fe.to_csv(RQ_DIR / "results/step03_irt_lmm_fixed_effects.csv", index=False, encoding='utf-8')
        log(f"[SAVED] results/step03_irt_lmm_fixed_effects.csv ({len(irt_fe)} coefficients)")

        # CTT fixed effects (use params method for robust extraction)
        ctt_fe = pd.DataFrame({
            'term': ctt_result_final.fe_params.index.tolist(),
            'estimate': ctt_result_final.fe_params.values.tolist(),
            'SE': ctt_result_final.bse_fe.values.tolist(),
            'z': [ctt_result_final.fe_params[i] / ctt_result_final.bse_fe[i] for i in range(len(ctt_result_final.fe_params))],
            'p_uncorrected': ctt_result_final.pvalues[:len(ctt_result_final.fe_params)].tolist()
        })
        ctt_fe.to_csv(RQ_DIR / "results/step03_ctt_lmm_fixed_effects.csv", index=False, encoding='utf-8')
        log(f"[SAVED] results/step03_ctt_lmm_fixed_effects.csv ({len(ctt_fe)} coefficients)")

        # Save convergence report
        log("[SAVE] Saving convergence report...")
        with open(RQ_DIR / "logs/step03_convergence_report.txt", 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONVERGENCE REPORT - Step 03 LMM Fitting\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Final random structure: {final_random_structure}\n\n")
            f.write("Convergence Details:\n")
            f.write("-" * 80 + "\n")
            for line in convergence_report:
                f.write(f"{line}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"IRT model AIC: {irt_result_final.aic:.2f}\n")
            f.write(f"IRT model BIC: {irt_result_final.bic:.2f}\n")
            f.write(f"CTT model AIC: {ctt_result_final.aic:.2f}\n")
            f.write(f"CTT model BIC: {ctt_result_final.bic:.2f}\n")
            f.write("=" * 80 + "\n")
        log("[SAVED] logs/step03_convergence_report.txt")

        # Save model objects (for step 4 assumption validation)
        log("[SAVE] Saving model objects for assumption validation...")
        irt_result_final.save(str(RQ_DIR / "data/step03_irt_lmm_model.pkl"))
        log("[SAVED] data/step03_irt_lmm_model.pkl")

        ctt_result_final.save(str(RQ_DIR / "data/step03_ctt_lmm_model.pkl"))
        log("[SAVED] data/step03_ctt_lmm_model.pkl")

        # =========================================================================
        # STEP 7: Run Validation Tool
        # =========================================================================
        # Tool: tools.validation.validate_lmm_convergence
        # Validates: Model convergence, no singular fit, finite estimates
        # Threshold: Both models must pass validation

        log("[VALIDATION] Running validate_lmm_convergence...")

        # Validate IRT model
        irt_validation = validate_lmm_convergence(irt_result_final)
        log(f"[VALIDATION IRT] Converged: {irt_validation.get('converged', False)}")
        log(f"[VALIDATION IRT] Message: {irt_validation.get('message', 'No message')}")

        # Validate CTT model
        ctt_validation = validate_lmm_convergence(ctt_result_final)
        log(f"[VALIDATION CTT] Converged: {ctt_validation.get('converged', False)}")
        log(f"[VALIDATION CTT] Message: {ctt_validation.get('message', 'No message')}")

        # Check parallelism requirement (both models use same random structure)
        log(f"[VALIDATION] Parallel structure: {final_random_structure}")
        log("[VALIDATION] CRITICAL: Both models use identical random structure (parallelism requirement satisfied)")

        # Overall validation
        if not irt_validation.get('converged', False):
            raise ValueError(f"IRT model validation failed: {irt_validation.get('message', 'Unknown error')}")
        if not ctt_validation.get('converged', False):
            raise ValueError(f"CTT model validation failed: {ctt_validation.get('message', 'Unknown error')}")

        log("[SUCCESS] Step 03 complete")
        log(f"[SUMMARY] Final models use: {final_random_structure}")
        log(f"[SUMMARY] IRT: {len(irt_fe)} fixed effects, AIC={irt_result_final.aic:.2f}")
        log(f"[SUMMARY] CTT: {len(ctt_fe)} fixed effects, AIC={ctt_result_final.aic:.2f}")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
