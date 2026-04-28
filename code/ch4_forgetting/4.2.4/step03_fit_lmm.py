#!/usr/bin/env python3
"""Fit Parallel LMMs (IRT Model + CTT Model): Fit identical LMM structures for IRT and CTT scores to compare trajectory"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Fit Parallel LMMs (IRT + CTT)")
        # Load Input Data

        log("Loading input data...")

        # Load IRT theta scores (wide format)
        irt_theta = pd.read_csv(RQ_DIR / "data/step00_irt_theta_loaded.csv")
        log(f"IRT theta ({len(irt_theta)} rows, {len(irt_theta.columns)} cols)")

        # Load CTT scores (already long format)
        ctt_scores = pd.read_csv(RQ_DIR / "data/step01_ctt_scores.csv")
        log(f"CTT scores ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")

        # Load TSVR time variable
        tsvr_data = pd.read_csv(RQ_DIR / "data/step00_tsvr_loaded.csv")
        log(f"TSVR data ({len(tsvr_data)} rows, {len(tsvr_data.columns)} cols)")
        # Prepare LMM Input Datasets

        log("Reshaping IRT theta to long format (When excluded)...")

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

        log(f"IRT long format ({len(irt_long)} rows - When excluded)")

        # Extract UID and test from composite_ID for both datasets
        log("Extracting UID and test from composite_ID...")

        # IRT long: add UID and test columns
        irt_long[['UID', 'test']] = irt_long['composite_ID'].str.split('_', expand=True)
        irt_long['test'] = irt_long['test'].astype(int)  # Convert to int to match TSVR

        # CTT already has UID and test columns from step01
        # Just need to ensure composite_ID exists
        if 'composite_ID' not in ctt_scores.columns:
            ctt_scores['composite_ID'] = ctt_scores['UID'] + '_' + ctt_scores['test'].astype(str)

        # Capitalize domain names to match IRT (what -> What, where -> Where)
        ctt_scores['domain'] = ctt_scores['domain'].str.capitalize()

        log("Merging with TSVR time variable...")

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

        log(f"IRT LMM input ({len(irt_lmm_input)} rows)")
        log(f"CTT LMM input ({len(ctt_lmm_input)} rows)")

        # Check for missing TSVR values
        irt_missing_tsvr = irt_lmm_input['TSVR_hours'].isna().sum()
        ctt_missing_tsvr = ctt_lmm_input['TSVR_hours'].isna().sum()
        if irt_missing_tsvr > 0 or ctt_missing_tsvr > 0:
            log(f"Missing TSVR values: IRT={irt_missing_tsvr}, CTT={ctt_missing_tsvr}")
        # Save LMM Input Datasets
        # These inputs will be used by: Step 4 (assumption validation), downstream analyses

        log("Saving LMM input datasets...")

        irt_lmm_input.to_csv(RQ_DIR / "data/step03_irt_lmm_input.csv", index=False, encoding='utf-8')
        log(f"data/step03_irt_lmm_input.csv ({len(irt_lmm_input)} rows, {len(irt_lmm_input.columns)} cols)")

        ctt_lmm_input.to_csv(RQ_DIR / "data/step03_ctt_lmm_input.csv", index=False, encoding='utf-8')
        log(f"data/step03_ctt_lmm_input.csv ({len(ctt_lmm_input)} rows, {len(ctt_lmm_input.columns)} cols)")
        # Create Log-Transformed Time Variable

        log("Creating log-transformed time variable...")

        irt_lmm_input['log_TSVR'] = np.log(irt_lmm_input['TSVR_hours'] + 1)
        ctt_lmm_input['log_TSVR'] = np.log(ctt_lmm_input['TSVR_hours'] + 1)

        log("Log-transformed TSVR created")
        # Fit LMM Models (with convergence strategy)
        # Convergence strategy: Attempt random slopes, fallback to intercepts-only if either fails

        log("Fitting LMM models...")
        log("Convergence: Attempt random slopes, simplify if either model fails")

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
            log("IRT model with random slopes...")
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
            log("CTT model with random slopes...")
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
            log("BOTH models converged with random slopes -> Use random slopes")
            irt_result_final = irt_result_slopes
            ctt_result_final = ctt_result_slopes
            final_random_structure = "random slopes (TSVR_hours | UID)"
            convergence_report.append("DECISION: BOTH converged with slopes -> Use random slopes for both")
        else:
            log("At least one model failed with random slopes -> Simplify BOTH to intercepts-only")
            convergence_report.append("DECISION: At least one model failed with slopes -> Simplify BOTH to intercepts-only")

            # PHASE 2: Fit intercepts-only for BOTH models
            log("[PHASE 2] Fitting intercepts-only: 1 | UID")

            try:
                log("IRT model with intercepts-only...")
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
                log(f"IRT intercepts-only failed: {str(e)}")
                convergence_report.append(f"IRT model with intercepts-only: FAILED ({str(e)})")
                raise ValueError(f"IRT model failed to converge even with intercepts-only: {str(e)}")

            try:
                log("CTT model with intercepts-only...")
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
                log(f"CTT intercepts-only failed: {str(e)}")
                convergence_report.append(f"CTT model with intercepts-only: FAILED ({str(e)})")
                raise ValueError(f"CTT model failed to converge even with intercepts-only: {str(e)}")

            final_random_structure = "intercepts-only (1 | UID)"

        log(f"Analysis complete with {final_random_structure}")
        # Save Model Summaries and Fixed Effects
        # These outputs will be used by: Step 5 (coefficient comparison), Step 6 (AIC/BIC)

        log("Saving model summaries...")

        # Save IRT model summary
        with open(RQ_DIR / "results/step03_irt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(irt_result_final.summary()))
        log("results/step03_irt_lmm_summary.txt")

        # Save CTT model summary
        with open(RQ_DIR / "results/step03_ctt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(ctt_result_final.summary()))
        log("results/step03_ctt_lmm_summary.txt")

        # Extract and save fixed effects tables
        log("Extracting fixed effects...")

        # IRT fixed effects (use params method for robust extraction)
        irt_fe = pd.DataFrame({
            'term': irt_result_final.fe_params.index.tolist(),
            'estimate': irt_result_final.fe_params.values.tolist(),
            'SE': irt_result_final.bse_fe.values.tolist(),
            'z': [irt_result_final.fe_params[i] / irt_result_final.bse_fe[i] for i in range(len(irt_result_final.fe_params))],
            'p_uncorrected': irt_result_final.pvalues[:len(irt_result_final.fe_params)].tolist()
        })
        irt_fe.to_csv(RQ_DIR / "results/step03_irt_lmm_fixed_effects.csv", index=False, encoding='utf-8')
        log(f"results/step03_irt_lmm_fixed_effects.csv ({len(irt_fe)} coefficients)")

        # CTT fixed effects (use params method for robust extraction)
        ctt_fe = pd.DataFrame({
            'term': ctt_result_final.fe_params.index.tolist(),
            'estimate': ctt_result_final.fe_params.values.tolist(),
            'SE': ctt_result_final.bse_fe.values.tolist(),
            'z': [ctt_result_final.fe_params[i] / ctt_result_final.bse_fe[i] for i in range(len(ctt_result_final.fe_params))],
            'p_uncorrected': ctt_result_final.pvalues[:len(ctt_result_final.fe_params)].tolist()
        })
        ctt_fe.to_csv(RQ_DIR / "results/step03_ctt_lmm_fixed_effects.csv", index=False, encoding='utf-8')
        log(f"results/step03_ctt_lmm_fixed_effects.csv ({len(ctt_fe)} coefficients)")

        # Save convergence report
        log("Saving convergence report...")
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
        log("logs/step03_convergence_report.txt")

        # Save model objects (for step 4 assumption validation)
        log("Saving model objects for assumption validation...")
        irt_result_final.save(str(RQ_DIR / "data/step03_irt_lmm_model.pkl"))
        log("data/step03_irt_lmm_model.pkl")

        ctt_result_final.save(str(RQ_DIR / "data/step03_ctt_lmm_model.pkl"))
        log("data/step03_ctt_lmm_model.pkl")
        # Run Validation Tool
        # Validates: Model convergence, no singular fit, finite estimates
        # Threshold: Both models must pass validation

        log("Running validate_lmm_convergence...")

        # Validate IRT model
        irt_validation = validate_lmm_convergence(irt_result_final)
        log(f"[VALIDATION IRT] Converged: {irt_validation.get('converged', False)}")
        log(f"[VALIDATION IRT] Message: {irt_validation.get('message', 'No message')}")

        # Validate CTT model
        ctt_validation = validate_lmm_convergence(ctt_result_final)
        log(f"[VALIDATION CTT] Converged: {ctt_validation.get('converged', False)}")
        log(f"[VALIDATION CTT] Message: {ctt_validation.get('message', 'No message')}")

        # Check parallelism requirement (both models use same random structure)
        log(f"Parallel structure: {final_random_structure}")
        log("CRITICAL: Both models use identical random structure (parallelism requirement satisfied)")

        # Overall validation
        if not irt_validation.get('converged', False):
            raise ValueError(f"IRT model validation failed: {irt_validation.get('message', 'Unknown error')}")
        if not ctt_validation.get('converged', False):
            raise ValueError(f"CTT model validation failed: {ctt_validation.get('message', 'Unknown error')}")

        log("Step 03 complete")
        log(f"Final models use: {final_random_structure}")
        log(f"IRT: {len(irt_fe)} fixed effects, AIC={irt_result_final.aic:.2f}")
        log(f"CTT: {len(ctt_fe)} fixed effects, AIC={ctt_result_final.aic:.2f}")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
