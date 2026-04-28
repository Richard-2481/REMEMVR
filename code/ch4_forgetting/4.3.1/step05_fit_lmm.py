#!/usr/bin/env python3
"""Fit LMM and Select Best Model: Fit 5 candidate LMM trajectory models with Paradigm x Time interaction, select best by AIC."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
import pickle

# parents[4] = REMEMVR/ (code -> rq3 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import compare_lmm_models_by_aic, extract_fixed_effects_from_lmm

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1
LOG_FILE = RQ_DIR / "logs" / "step05_fit_lmm.log"


# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Fit LMM and Select Best Model")
        log("=" * 60)
        # Load Input Data

        log("\nLoading LMM input data from step04...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_lmm = pd.read_csv(input_path, encoding='utf-8')
        log(f"step04_lmm_input.csv ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")
        log(f"  Columns: {df_lmm.columns.tolist()}")
        log(f"  Unique UIDs: {df_lmm['UID'].nunique()}")
        log(f"  Paradigms: {sorted(df_lmm['paradigm'].unique().tolist())}")
        log(f"  TSVR_hours range: {df_lmm['TSVR_hours'].min():.1f} - {df_lmm['TSVR_hours'].max():.1f}")
        # Transform Data for Tool Compatibility
        # The compare_lmm_models_by_aic tool expects specific column names:
        # - 'Ability' as outcome variable
        # - 'Factor' as paradigm variable
        # - 'Days', 'Days_sq', 'log_Days' as time variables
        # We need to rename/create these columns from our input data

        log("\nPreparing data for LMM tool...")

        # Rename columns to match tool expectations
        df_lmm_transformed = df_lmm.copy()

        # Convert TSVR_hours to Days (Decision D070: use actual hours / 24)
        df_lmm_transformed['Days'] = df_lmm_transformed['TSVR_hours'] / 24.0
        df_lmm_transformed['Days_sq'] = df_lmm_transformed['Days'] ** 2
        df_lmm_transformed['log_Days'] = np.log(df_lmm_transformed['Days'] + 1)

        # Rename paradigm -> Factor (capitalize first letter of each word for treatment coding)
        # free_recall -> Free_Recall, cued_recall -> Cued_Recall, recognition -> Recognition
        # Using title case on underscored string produces Free_Recall format
        df_lmm_transformed['Factor'] = df_lmm_transformed['paradigm'].str.replace('_', ' ').str.title().str.replace(' ', '_')

        # Rename theta -> Ability
        df_lmm_transformed['Ability'] = df_lmm_transformed['theta']

        log(f"  Created time variables: Days, Days_sq, log_Days")
        log(f"  Days range: {df_lmm_transformed['Days'].min():.2f} - {df_lmm_transformed['Days'].max():.2f}")
        log(f"  Factor levels: {sorted(df_lmm_transformed['Factor'].unique().tolist())}")

        # Determine reference group from transformed Factor levels
        factor_levels = sorted(df_lmm_transformed['Factor'].unique().tolist())
        reference_group = factor_levels[1]  # Free_Recall (alphabetically middle)
        log(f"  Using reference_group: '{reference_group}'")
        # Run Analysis Tool (compare_lmm_models_by_aic)

        log("\nRunning compare_lmm_models_by_aic...")
        log(f"  n_factors: 3 (free_recall, cued_recall, recognition)")
        log(f"  reference_group: '{reference_group}' (treatment coding)")
        log(f"  groups: 'UID' (random effects grouping)")

        # Create save directory for model files
        save_dir = RQ_DIR / "data"
        save_dir.mkdir(parents=True, exist_ok=True)

        comparison_results = compare_lmm_models_by_aic(
            data=df_lmm_transformed,       # Long-format LMM data
            n_factors=3,                   # Three paradigms (free_recall, cued_recall, recognition)
            reference_group=reference_group,  # Treatment coding: Free_Recall as reference
            groups='UID',                  # Grouping variable for random effects
            save_dir=save_dir              # Directory to save fitted models
        )

        log("Model comparison complete")

        # Extract results
        fitted_models = comparison_results['models']
        aic_df = comparison_results['aic_comparison']
        best_model_name = comparison_results['best_model_name']
        best_result = comparison_results['best_model']

        # Check if best model is None (all models failed)
        if best_result is None:
            raise ValueError(f"All 5 candidate models failed to fit. Cannot proceed.")

        log(f"\nBest model: {best_model_name}")
        log(f"  AIC: {best_result.aic:.2f}")
        log(f"  BIC: {best_result.bic:.2f}")
        log(f"  Log-Likelihood: {best_result.llf:.2f}")
        # Save Analysis Outputs
        # These outputs will be used by: step06 (post-hoc contrasts) and step07 (plots)

        # Save AIC comparison table
        aic_output_path = RQ_DIR / "data" / "step05_model_comparison.csv"

        # Add convergence column to AIC comparison
        aic_df['converged'] = aic_df['model_name'].apply(
            lambda x: fitted_models.get(x) is not None and
                      getattr(fitted_models.get(x), 'converged', False) if fitted_models.get(x) is not None else False
        )

        aic_df.to_csv(aic_output_path, index=False, encoding='utf-8')
        log(f"\nSaved AIC comparison: {aic_output_path}")
        log(f"  {len(aic_df)} models compared")
        log(f"  AIC Comparison Table:")
        for _, row in aic_df.iterrows():
            log(f"    {row['model_name']}: AIC={row['AIC']:.2f}, delta={row['delta_AIC']:.2f}, weight={row['AIC_weight']:.3f}")

        # Save model summary text file
        summary_output_path = RQ_DIR / "results" / "step05_lmm_model_summary.txt"
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"BEST MODEL: {best_model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(best_result.summary().as_text())
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("AIC COMPARISON\n")
            f.write("=" * 60 + "\n\n")
            f.write(aic_df.to_string(index=False))
        log(f"Saved model summary: {summary_output_path}")

        # Save fixed effects to CSV
        log("\nExtracting fixed effects from best model...")
        fe_df = extract_fixed_effects_from_lmm(best_result)
        # Rename columns to match 4_analysis.yaml specification
        fe_df.columns = ['effect', 'coefficient', 'std_error', 'z_value', 'p_value', 'CI_lower', 'CI_upper']
        fe_output_path = RQ_DIR / "data" / "step05_fixed_effects.csv"
        fe_df.to_csv(fe_output_path, index=False, encoding='utf-8')
        log(f"Saved fixed effects: {fe_output_path}")
        log(f"  {len(fe_df)} fixed effect terms")

        # Save fitted model using statsmodels save method for step06 post-hoc contrasts
        model_output_path = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"
        best_result.save(str(model_output_path))
        log(f"Saved fitted model: {model_output_path}")
        # Run Validation Tool
        # Validates: Model convergence and fit quality
        # Criteria: converged=True, no singular fit, finite estimates

        log("\nRunning validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(lmm_result=best_result)

        # Report validation results
        for key, value in validation_result.items():
            log(f"  {key}: {value}")

        # Additional validation checks
        log("\nAdditional checks:")

        # Check number of converged models
        n_converged = sum(1 for m in fitted_models.values() if m is not None and getattr(m, 'converged', False))
        log(f"  Converged models: {n_converged}/5")
        if n_converged < 3:
            raise ValueError(f"Only {n_converged}/5 models converged. Need at least 3.")

        # Check best model convergence
        if not validation_result.get('converged', False):
            raise ValueError(f"Best model ({best_model_name}) did not converge")

        # Check for singular fit (random effects variance > 0)
        re_cov = best_result.cov_re
        if isinstance(re_cov, pd.DataFrame):
            re_var = re_cov.values[0, 0] if re_cov.shape[0] > 0 else 0
        else:
            re_var = re_cov[0, 0] if re_cov.shape[0] > 0 else 0
        log(f"  Random effects variance: {re_var:.4f}")
        if re_var <= 0:
            log("Singular fit detected (random effects variance = 0)")

        # Check fixed effects are finite
        fe_params = best_result.params
        if fe_params.isna().any():
            raise ValueError("Some fixed effects have NaN estimates")
        if np.isinf(fe_params).any():
            raise ValueError("Some fixed effects have infinite estimates")
        log(f"  Fixed effects: All {len(fe_params)} estimates are finite")

        # Verify all 5 models have valid AIC
        invalid_aic = aic_df[np.isinf(aic_df['AIC'])]
        if len(invalid_aic) > 0:
            log(f"{len(invalid_aic)} models have invalid AIC (inf)")
        else:
            log(f"  All 5 models have valid AIC values: ")

        log("\n" + "=" * 60)
        log("Step 05 complete")
        log("=" * 60)
        log(f"\nOutputs:")
        log(f"  - {aic_output_path}")
        log(f"  - {summary_output_path}")
        log(f"  - {fe_output_path}")
        log(f"  - {model_output_path}")
        log(f"\nBest Model: {best_model_name}")
        log(f"  Formula: Ability ~ Days * C(Factor, Treatment('{reference_group}'))")
        log(f"  AIC: {best_result.aic:.2f}")

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
