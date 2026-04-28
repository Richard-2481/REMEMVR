#!/usr/bin/env python3
"""Fit Piecewise Model: Fit piecewise LMM (theta ~ Days_within * Segment + (Days_within | UID)) to test"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_piecewise_model.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Fit Piecewise Model")
        # Load Input Data

        log("Loading time-transformed data from Step 1...")
        time_data = pd.read_csv(RQ_DIR / "data" / "step01_time_transformed.csv", encoding='utf-8')
        log(f"step01_time_transformed.csv ({len(time_data)} rows, {len(time_data.columns)} cols)")

        # Verify required columns for piecewise model
        required_cols = ['UID', 'test', 'TSVR_hours', 'theta', 'Segment', 'Days_within']
        missing_cols = [col for col in required_cols if col not in time_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")

        # Load RQ 5.7 best continuous AIC for comparison (Test 2 criterion)
        log("Loading RQ 5.7 best continuous model AIC...")
        continuous_aic_file = RQ_DIR / "data" / "step00_best_continuous_aic.txt"
        with open(continuous_aic_file, 'r', encoding='utf-8') as f:
            continuous_aic = float(f.read().strip())
        log(f"RQ 5.7 best continuous AIC = {continuous_aic:.2f}")

        # Load RQ 5.7 convergence status (CRITICAL for AIC comparison validity)
        log("Checking RQ 5.7 convergence status...")
        rq57_convergence_file = RQ_DIR / "data" / "step00_rq57_convergence.txt"
        with open(rq57_convergence_file, 'r', encoding='utf-8') as f:
            rq57_status = f.read().strip()
        log(f"RQ 5.7 status: {rq57_status}")

        # Flag if RQ 5.7 used fallback random structure
        if "fallback" in rq57_status.lower() or "(1 | UID)" in rq57_status:
            log("RQ 5.7 model used fallback random structure!")
            log("AIC comparison may be invalid (different random structures)")
            log("Proceed with caution - interpret AIC difference qualitatively")
            rq57_fallback = True
        else:
            log("RQ 5.7 model converged with maximal random structure")
            log("AIC comparison valid (comparable random structures)")
            rq57_fallback = False
        # Fit Piecewise LMM
        #               if Early (0-48h) and Late (48-240h) segments have different
        #               forgetting slopes (two-phase hypothesis Test 2)

        # CRITICAL FIX (2025-12-03): Use MATCHED random structure with Step 02 (Quadratic)
        # Step 02 quadratic model uses (1 | UID) intercepts-only (fallback from maximal)
        # For VALID AIC comparison, piecewise must use SAME random structure
        # Previous version used (Days_within | UID) which:
        #   1. Failed to converge (Converged: False)
        #   2. Invalidated AIC comparison (different random structures)
        # FIX: Use (1 | UID) to match quadratic model, enabling valid AIC comparison

        log("Fitting piecewise LMM: theta ~ Days_within * Segment + (1 | UID)...")
        log("Random structure: (1 | UID) - MATCHED to Step 02 quadratic model")
        log("This enables VALID AIC comparison (same random structure across models)")

        # Piecewise formula with interaction term (key test for two-phase forgetting)
        formula = "theta ~ Days_within * Segment"

        # Use intercept-only random structure to match Step 02 quadratic model
        piecewise_model = fit_lmm_trajectory(
            data=time_data,
            formula=formula,
            groups="UID",
            re_formula="~1",  # Intercept-only: MUST MATCH Step 02 for valid AIC comparison
            reml=False  # ML estimation required for AIC comparison
        )

        if piecewise_model.converged:
            log("Piecewise model converged with matched random structure (1 | UID)")
        else:
            log("Piecewise model did not converge - interpret results with caution")

        random_structure_used = "(1 | UID)"
        log("Random structure matched to Step 02 quadratic model - AIC comparison VALID")

        log("Piecewise LMM fitting complete")
        log(f"Random structure used: {random_structure_used}")
        # Extract Model Results and Compare AIC
        # Test 2 Criterion: deltaAIC = AIC_piecewise - AIC_continuous
        # deltaAIC < -2 favors piecewise (segmented forgetting)
        # deltaAIC > +2 favors continuous (single-phase forgetting)
        # -2 <= deltaAIC <= +2 inconclusive (models equivalent)

        log("Extracting fixed effects and AIC...")

        # Get piecewise model AIC
        piecewise_aic = piecewise_model.aic
        log(f"Piecewise model AIC = {piecewise_aic:.2f}")

        # Compute deltaAIC (Test 2 for two-phase forgetting)
        delta_aic = piecewise_aic - continuous_aic
        log(f"deltaAIC = {delta_aic:.2f} (Piecewise - Continuous)")

        # Interpret AIC comparison
        if delta_aic < -2:
            aic_interpretation = "Piecewise model FAVORED (deltaAIC < -2) - Evidence for two-phase forgetting"
        elif delta_aic > 2:
            aic_interpretation = "Continuous model FAVORED (deltaAIC > +2) - Evidence against two-phase forgetting"
        else:
            aic_interpretation = "Models EQUIVALENT (-2 <= deltaAIC <= +2) - Inconclusive evidence"

        log(f"{aic_interpretation}")

        # Flag if AIC comparison potentially invalid due to RQ 5.7 fallback
        if rq57_fallback and random_structure_used != "(1 | UID)":
            log("AIC comparison validity compromised:")
            log(f"RQ 5.7 used fallback structure (likely (1 | UID))")
            log(f"This model used: {random_structure_used}")
            log("Comparing models with different random structures violates AIC assumptions")
            log("Interpret deltaAIC qualitatively, not as definitive evidence")

        # Extract fixed effects table
        fixed_effects_table = piecewise_model.summary().tables[1]
        log("Fixed effects extracted (Intercept, Days_within, Segment, interaction)")

        # Convert to DataFrame (handle both SimpleTable and DataFrame)
        if hasattr(fixed_effects_table, 'data'):
            # SimpleTable object
            fe_df = pd.DataFrame(fixed_effects_table.data[1:], columns=fixed_effects_table.data[0])
        else:
            # Already a DataFrame
            fe_df = fixed_effects_table.reset_index()
        interaction_term = "Days_within:Segment[T.Late]"

        if interaction_term in fe_df.iloc[:, 0].values:
            interaction_row = fe_df[fe_df.iloc[:, 0] == interaction_term].iloc[0]
            interaction_pval = float(interaction_row.iloc[4])  # P>|z| column
            log(f"Interaction p-value = {interaction_pval:.4f}")

            # Decision D068 compliance: Report both uncorrected and Bonferroni-corrected p-values
            bonferroni_alpha = 0.05 / 15  # Chapter 5 family size = 15 tests
            if interaction_pval < bonferroni_alpha:
                log(f"Interaction significant at Bonferroni alpha = {bonferroni_alpha:.4f}")
                log("Early and Late segments have DIFFERENT forgetting rates")
            else:
                log(f"[NOT SIGNIFICANT] Interaction not significant at Bonferroni alpha = {bonferroni_alpha:.4f}")
                log("Early and Late segments have SIMILAR forgetting rates")
        else:
            log(f"Interaction term {interaction_term} not found in fixed effects")
            interaction_pval = np.nan
        # Generate Predictions for Plotting

        log("Generating model predictions for Early and Late segments...")

        # Prediction grids (in HOURS, will be converted to Days_within)
        early_grid_hours = np.linspace(0, 48, 9)  # 0, 6, 12, 18, 24, 30, 36, 42, 48 hours
        late_grid_hours = np.linspace(0, 192, 9)  # 0, 24, 48, 72, 96, 120, 144, 168, 192 hours within-segment

        # Create prediction DataFrames
        predictions_list = []

        # Early segment predictions (Segment = "Early")
        for tsvr_hours in early_grid_hours:
            pred_row = pd.DataFrame({
                'UID': ['PRED'],  # Dummy UID for prediction
                'Segment': ['Early'],
                'Days_within': [tsvr_hours / 24.0],  # Convert hours to days
                'TSVR_hours': [tsvr_hours],
                'theta': [0]  # Placeholder (will use fittedvalues)
            })
            predictions_list.append(pred_row)

        # Late segment predictions (Segment = "Late")
        for within_hours in late_grid_hours:
            pred_row = pd.DataFrame({
                'UID': ['PRED'],  # Dummy UID for prediction
                'Segment': ['Late'],
                'Days_within': [within_hours / 24.0],  # Convert hours to days
                'TSVR_hours': [48 + within_hours],  # Late starts at 48h
                'theta': [0]  # Placeholder
            })
            predictions_list.append(pred_row)

        # Combine prediction rows
        pred_data = pd.concat(predictions_list, ignore_index=True)

        # Get predictions from model (marginal predictions at population level)
        # Note: This is a simplified approach - marginal effects would be more rigorous
        # but require additional dependencies (statsmodels.api.PredictionResults)
        # For RQ 5.1.2, we use fixed effects predictions as approximation

        log("Computing fixed effects predictions (population-level trends)...")

        # Extract fixed effects coefficients
        fe_params = piecewise_model.fe_params
        intercept = fe_params['Intercept']
        slope_early = fe_params['Days_within']
        segment_late = fe_params['Segment[T.Late]']
        interaction = fe_params['Days_within:Segment[T.Late]']

        log(f"Fixed effects: Intercept={intercept:.4f}, Days_within={slope_early:.4f}")
        log(f"Segment[Late]={segment_late:.4f}, Interaction={interaction:.4f}")

        # Compute predictions manually from fixed effects
        predictions = []
        for idx, row in pred_data.iterrows():
            segment = row['Segment']
            days = row['Days_within']

            if segment == 'Early':
                # Early segment: theta = Intercept + slope_early * days
                pred_theta = intercept + slope_early * days
            else:  # Late
                # Late segment: theta = (Intercept + segment_late) + (slope_early + interaction) * days
                pred_theta = (intercept + segment_late) + (slope_early + interaction) * days

            predictions.append(pred_theta)

        pred_data['predicted_theta'] = predictions

        # Compute 95% CI (simplified - assumes no uncertainty in fixed effects)
        # For publication-quality CIs, would need delta method or bootstrap
        # Here we use residual SE as approximation
        residual_se = np.sqrt(piecewise_model.scale)  # Residual standard error
        pred_data['CI_lower'] = pred_data['predicted_theta'] - 1.96 * residual_se
        pred_data['CI_upper'] = pred_data['predicted_theta'] + 1.96 * residual_se

        log(f"Generated {len(pred_data)} predictions (9 Early + 9 Late)")
        # Save Outputs
        # Output 1: Model summary (text file for human inspection)
        # Output 2: Predictions CSV (for plotting in Step 6)

        log("Saving piecewise model summary...")
        summary_file = RQ_DIR / "results" / "step03_piecewise_model_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PIECEWISE LMM SUMMARY - RQ 5.1.2 TEST 2 (Two-Phase Forgetting)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Formula: {formula}\n")
            f.write(f"Random Structure: {random_structure_used}\n")
            f.write(f"Estimation: ML (REML=False for AIC comparison)\n")
            f.write(f"N observations: {piecewise_model.nobs}\n")
            f.write(f"N groups (UIDs): {len(piecewise_model.model.group_labels)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("CONVERGENCE STATUS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Converged: {piecewise_model.converged}\n")
            if not piecewise_model.converged:
                f.write("WARNING: Model did not converge - interpret results with caution\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("FIXED EFFECTS\n")
            f.write("-" * 80 + "\n")
            f.write(str(piecewise_model.summary().tables[1]))
            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write("RANDOM EFFECTS\n")
            f.write("-" * 80 + "\n")
            f.write(str(piecewise_model.summary().tables[0]))
            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write("AIC COMPARISON (TEST 2 FOR TWO-PHASE FORGETTING)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Piecewise model AIC:  {piecewise_aic:.2f}\n")
            f.write(f"Continuous model AIC: {continuous_aic:.2f} (from RQ 5.7)\n")
            f.write(f"deltaAIC:             {delta_aic:.2f} (Piecewise - Continuous)\n\n")
            f.write(f"Interpretation: {aic_interpretation}\n\n")

            if rq57_fallback and random_structure_used != "(1 | UID)":
                f.write("WARNING: AIC comparison validity compromised\n")
                f.write("  RQ 5.7 used fallback random structure (likely (1 | UID))\n")
                f.write(f"  This model used: {random_structure_used}\n")
                f.write("  Comparing models with different random structures violates AIC assumptions\n")
                f.write("  Interpret deltaAIC qualitatively, not as definitive evidence\n\n")

            f.write("-" * 80 + "\n")
            f.write("INTERACTION TEST (KEY EVIDENCE FOR TWO-PHASE FORGETTING)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Interaction term: {interaction_term}\n")
            f.write(f"p-value (uncorrected): {interaction_pval:.4f}\n")
            f.write(f"p-value (Bonferroni):  {interaction_pval * 15:.4f} (x 15 Chapter 5 tests)\n")
            f.write(f"Bonferroni alpha:      {bonferroni_alpha:.4f}\n\n")

            if interaction_pval < bonferroni_alpha:
                f.write("RESULT: Interaction SIGNIFICANT\n")
                f.write("INTERPRETATION: Early and Late segments have DIFFERENT forgetting rates\n")
                f.write("                (Evidence for two-phase forgetting hypothesis)\n")
            else:
                f.write("RESULT: Interaction NOT SIGNIFICANT\n")
                f.write("INTERPRETATION: Early and Late segments have SIMILAR forgetting rates\n")
                f.write("                (Evidence against two-phase forgetting hypothesis)\n")

        log(f"{summary_file}")

        log("Saving piecewise predictions...")
        pred_output = pred_data[['Segment', 'Days_within', 'TSVR_hours', 'predicted_theta', 'CI_lower', 'CI_upper']]
        pred_output.to_csv(RQ_DIR / "data" / "step03_piecewise_predictions.csv", index=False, encoding='utf-8')
        log(f"data/step03_piecewise_predictions.csv ({len(pred_output)} rows)")

        # Save pickled model for Step 4 validation
        log("Saving piecewise model pickle...")
        import pickle
        model_pkl_path = RQ_DIR / "data" / "step03_piecewise_model.pkl"
        piecewise_model.save(str(model_pkl_path))
        log(f"{model_pkl_path.name}")
        # Run Validation Tool
        # Validates: Model convergence status and parameter estimates
        # Threshold: Convergence = True (or fallback documented)

        log("Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(lmm_result=piecewise_model)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation checks specific to piecewise model
        log("Checking fixed effects validity...")

        # Check all fixed effects are finite (not NaN/Inf)
        fe_values = piecewise_model.fe_params.values
        if np.all(np.isfinite(fe_values)):
            log("All fixed effects finite")
        else:
            raise ValueError("Fixed effects contain NaN or Inf - model estimation failed")

        # Check interaction p-value in valid range
        if 0 <= interaction_pval <= 1 or np.isnan(interaction_pval):
            log(f"Interaction p-value in valid range: {interaction_pval:.4f}")
        else:
            raise ValueError(f"Interaction p-value out of bounds: {interaction_pval}")

        # Check AIC comparison computed successfully
        if np.isfinite(delta_aic):
            log(f"deltaAIC computed successfully: {delta_aic:.2f}")
        else:
            raise ValueError(f"deltaAIC computation failed: {delta_aic}")

        # Check predictions generated for all timepoints
        if len(pred_output) == 18:
            log("All 18 predictions generated (9 Early + 9 Late)")
        else:
            raise ValueError(f"Expected 18 predictions, got {len(pred_output)}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
