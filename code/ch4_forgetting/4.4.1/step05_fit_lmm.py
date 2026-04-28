#!/usr/bin/env python3
"""LMM Model Fitting and Selection: Fit 5 candidate LMMs with Congruence x Time interactions and select best by AIC."""

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
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_fit_lmm.log"

# Input files
INPUT_LMM = RQ_DIR / "data" / "step04_lmm_input.csv"

# Output files
OUTPUT_COMPARISON = RQ_DIR / "results" / "step05_model_comparison.csv"
OUTPUT_SUMMARY = RQ_DIR / "results" / "step05_lmm_model_summary.txt"
OUTPUT_MODEL = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"

# Model save directory (for all candidate models)
MODEL_SAVE_DIR = RQ_DIR / "data"

# Candidate models - Congruence x Time interactions
# Reference = common (schema-neutral baseline)
CANDIDATE_MODELS = {
    "Linear": {
        "formula": "theta ~ TSVR_hours * C(congruence, Treatment('common'))",
        "re_formula": "~TSVR_hours"
    },
    "Quadratic": {
        "formula": "theta ~ (TSVR_hours + TSVR_sq) * C(congruence, Treatment('common'))",
        "re_formula": "~TSVR_hours"
    },
    "Log": {
        "formula": "theta ~ TSVR_log * C(congruence, Treatment('common'))",
        "re_formula": "~TSVR_log"
    },
    "Lin+Log": {
        "formula": "theta ~ (TSVR_hours + TSVR_log) * C(congruence, Treatment('common'))",
        "re_formula": "~TSVR_hours"
    },
    "Quad+Log": {
        "formula": "theta ~ (TSVR_hours + TSVR_sq + TSVR_log) * C(congruence, Treatment('common'))",
        "re_formula": "~TSVR_hours"
    }
}

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# LMM Fitting Function

def fit_single_lmm(data, formula, groups, re_formula, reml=False):
    """Fit a single LMM and return the result."""
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data[groups],
        re_formula=re_formula
    )

    try:
        result = model.fit(method=['lbfgs'], reml=reml)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {str(e)}")

    if not result.converged:
        warnings.warn(f"Model did not converge. Formula: {formula}")

    return result

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: LMM Model Fitting and Selection")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Input Data
        log("\nLoading LMM input data...")

        df_lmm = pd.read_csv(INPUT_LMM)
        log(f"{INPUT_LMM.name} ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")

        # Ensure congruence is categorical with correct order
        df_lmm["congruence"] = pd.Categorical(
            df_lmm["congruence"],
            categories=["common", "congruent", "incongruent"],
            ordered=True
        )

        log(f"  Unique UIDs: {df_lmm['UID'].nunique()}")
        log(f"  Congruence levels: {df_lmm['congruence'].cat.categories.tolist()}")
        # Fit Candidate Models
        log("\nFitting candidate models...")

        fitted_models = {}
        aics = {}

        for model_name, config in CANDIDATE_MODELS.items():
            log(f"\n  Fitting {model_name} model...")

            # Check if saved model exists
            model_path = MODEL_SAVE_DIR / f"step05_lmm_{model_name}.pkl"

            if model_path.exists():
                log(f"    Loading existing model from {model_path.name}")
                try:
                    result = MixedLMResults.load(str(model_path))
                    fitted_models[model_name] = result
                    aics[model_name] = result.aic
                    log(f"    Loaded: AIC = {result.aic:.2f}")
                    continue
                except Exception as e:
                    log(f"    Failed to load saved model: {e}")
                    log(f"    Re-fitting model...")

            # Fit model
            try:
                result = fit_single_lmm(
                    data=df_lmm,
                    formula=config["formula"],
                    groups="UID",
                    re_formula=config["re_formula"],
                    reml=False
                )

                fitted_models[model_name] = result
                aics[model_name] = result.aic

                # Save model
                MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
                result.save(str(model_path))
                log(f"    Fitted: AIC = {result.aic:.2f}")
                log(f"    Saved: {model_path.name}")

            except Exception as e:
                log(f"    {model_name} model failed: {str(e)}")
                fitted_models[model_name] = None
                aics[model_name] = np.inf
        # Create AIC Comparison Table
        log("\nCreating AIC comparison table...")

        aic_df = pd.DataFrame({
            "model_name": list(aics.keys()),
            "AIC": list(aics.values())
        })

        # Sort by AIC
        aic_df = aic_df.sort_values("AIC").reset_index(drop=True)

        # Calculate delta AIC and weights
        aic_df["delta_AIC"] = aic_df["AIC"] - aic_df["AIC"].min()
        aic_df["AIC_weight"] = np.exp(-aic_df["delta_AIC"] / 2)
        aic_df["AIC_weight"] = aic_df["AIC_weight"] / aic_df["AIC_weight"].sum()

        # Add BIC
        bics = {}
        for model_name, result in fitted_models.items():
            if result is not None:
                bics[model_name] = result.bic
            else:
                bics[model_name] = np.inf

        aic_df["BIC"] = aic_df["model_name"].map(bics)

        # Reorder columns
        aic_df = aic_df[["model_name", "AIC", "BIC", "delta_AIC", "AIC_weight"]]

        log("\n  AIC Comparison:")
        for _, row in aic_df.iterrows():
            log(f"    {row['model_name']}: AIC={row['AIC']:.2f}, delta={row['delta_AIC']:.2f}, weight={row['AIC_weight']:.3f}")
        # Identify Best Model
        best_model_name = aic_df.iloc[0]["model_name"]
        best_model = fitted_models[best_model_name]

        log(f"\nBest model: {best_model_name}")
        log(f"  AIC: {best_model.aic:.2f}")
        log(f"  BIC: {best_model.bic:.2f}")
        # Save Outputs
        log("\nSaving output files...")

        # Ensure directories exist
        (RQ_DIR / "results").mkdir(parents=True, exist_ok=True)
        (RQ_DIR / "data").mkdir(parents=True, exist_ok=True)

        # Save comparison table
        aic_df.to_csv(OUTPUT_COMPARISON, index=False, encoding='utf-8')
        log(f"{OUTPUT_COMPARISON.name}")

        # Save model summary
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"BEST MODEL: {best_model_name}\n")
            f.write("=" * 70 + "\n\n")
            f.write(best_model.summary().as_text())
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("AIC COMPARISON\n")
            f.write("=" * 70 + "\n\n")
            f.write(aic_df.to_string(index=False))
        log(f"{OUTPUT_SUMMARY.name}")

        # Save best model (copy to standard location)
        best_model.save(str(OUTPUT_MODEL))
        log(f"{OUTPUT_MODEL.name}")
        # Validation
        log("\nValidating results...")

        # Check model convergence
        if best_model.converged:
            log("Best model converged")
        else:
            log("Best model did NOT converge")

        # Check AIC is finite
        if np.isfinite(best_model.aic):
            log(f"AIC is finite: {best_model.aic:.2f}")
        else:
            raise ValueError("Best model has non-finite AIC")

        # Check fixed effects are finite
        fe_nan = best_model.params.isna().sum()
        fe_inf = np.isinf(best_model.params).sum()
        if fe_nan > 0 or fe_inf > 0:
            log(f"Fixed effects have NaN ({fe_nan}) or Inf ({fe_inf})")
        else:
            log("All fixed effects are finite")

        # Check random effect variance
        re_var = best_model.cov_re.iloc[0, 0] if hasattr(best_model.cov_re, 'iloc') else best_model.cov_re[0, 0]
        if re_var > 0:
            log(f"Random effect variance: {re_var:.4f}")
        else:
            log(f"Random effect variance <= 0: {re_var:.4f}")

        # Show key fixed effects
        log("\n  Key Fixed Effects:")
        for param in best_model.params.index:
            coef = best_model.params[param]
            pval = best_model.pvalues[param]
            sig = "*" if pval < 0.05 else ""
            log(f"    {param}: {coef:.4f} (p={pval:.4f}){sig}")

        log("\nStep 05 complete (LMM Model Fitting)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
