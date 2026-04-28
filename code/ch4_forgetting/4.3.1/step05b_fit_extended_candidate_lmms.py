#!/usr/bin/env python3
"""
Extended Model Comparison for RQ 5.3.1

PURPOSE:
Test power law and other functional forms for Paradigm x Time trajectory analysis.
Original step05 tested only 5 models and selected Log with 99.99% weight.
This extends to 17+ models to verify if power law models challenge Log dominance.

CONTEXT:
- RQ 5.3.1: Does forgetting trajectory differ by paradigm (free_recall, cued_recall, recognition)?
- Original best model: Log (AIC=2346.60, weight=99.99%)
- RQ 5.1.1 finding: Power law models beat Log by ΔAIC=2.97 when testing single trajectory

QUESTION: Do power law models also outperform Log in multi-group (Paradigm x Time) analysis?

INPUTS:
  - data/step04_lmm_input.csv (same as step05)

OUTPUTS:
  - data/step05b_extended_model_fits.pkl
  - results/step05b_extended_model_comparison.csv
  - logs/step05b_extended_lmm_fitting.log
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05b_extended_lmm_fitting.log"

# Logging

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5b: Extended Model Comparison (Paradigm x Time)")
        # Load LMM Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"LMM input data missing: {input_path}")

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"{len(lmm_input)} rows, {len(lmm_input.columns)} cols")
        log(f"  Paradigms: {lmm_input['paradigm'].unique()}")
        log(f"  N subjects: {lmm_input['UID'].nunique()}")

        # Rename for compatibility (convert TSVR hours to Days)
        lmm_input['Days'] = lmm_input['TSVR_hours'] / 24.0
        lmm_input['Days_sq'] = lmm_input['Days'] ** 2
        lmm_input['log_Days'] = np.log(lmm_input['Days'] + 1)

        # Rename columns
        lmm_input = lmm_input.rename(columns={
            'theta': 'Ability',
            'paradigm': 'Factor'
        })
        # Add New Time Transformations

        log("Adding new time transformations...")

        # Power law variants
        lmm_input['log_log_Days'] = np.log(lmm_input['log_Days'] + 1)
        lmm_input['sqrt_Days'] = np.sqrt(lmm_input['Days'])
        lmm_input['cbrt_Days'] = np.cbrt(lmm_input['Days'])
        lmm_input['recip_Days'] = 1.0 / (lmm_input['Days'] + 1)
        lmm_input['Days_pow_neg05'] = (lmm_input['Days'] + 1) ** (-0.5)
        lmm_input['Days_pow_neg03'] = (lmm_input['Days'] + 1) ** (-0.3)
        lmm_input['Days_pow_neg07'] = (lmm_input['Days'] + 1) ** (-0.7)
        lmm_input['neg_Days'] = -lmm_input['Days']

        log("  Added transformations (same as RQ 5.1.1)")
        # Define Extended Model Set (WITH Paradigm Interaction)

        log("Defining extended model set...")
        log("  NOTE: All models include Paradigm x Time interaction")

        # Reference group for Factor (paradigm)
        factor_term = "C(Factor, Treatment('free_recall'))"

        # Build models with interactions
        models = {
            # ---- ORIGINAL 5 MODELS ----
            'Linear': f'Ability ~ Days * {factor_term}',
            'Quadratic': f'Ability ~ (Days + Days_sq) * {factor_term}',
            'Log': f'Ability ~ log_Days * {factor_term}',
            'Lin+Log': f'Ability ~ (Days + log_Days) * {factor_term}',
            'Quad+Log': f'Ability ~ (Days + Days_sq + log_Days) * {factor_term}',

            # ---- POWER LAW VARIANTS ----
            'PowerLaw_LogLog': f'Ability ~ log_log_Days * {factor_term}',
            'PowerLaw_Alpha05': f'Ability ~ Days_pow_neg05 * {factor_term}',
            'PowerLaw_Alpha03': f'Ability ~ Days_pow_neg03 * {factor_term}',
            'PowerLaw_Alpha07': f'Ability ~ Days_pow_neg07 * {factor_term}',
            'PowerLaw_Combined': f'Ability ~ (log_Days + log_log_Days) * {factor_term}',

            # ---- ROOT MODELS ----
            'SquareRoot': f'Ability ~ sqrt_Days * {factor_term}',
            'CubeRoot': f'Ability ~ cbrt_Days * {factor_term}',
            'SquareRoot+Log': f'Ability ~ (sqrt_Days + log_Days) * {factor_term}',

            # ---- RECIPROCAL MODELS ----
            'Reciprocal': f'Ability ~ recip_Days * {factor_term}',
            'Recip+Log': f'Ability ~ (recip_Days + log_Days) * {factor_term}',

            # ---- EXPONENTIAL PROXY ----
            'Exponential': f'Ability ~ neg_Days * {factor_term}',
            'Exp+Log': f'Ability ~ (neg_Days + log_Days) * {factor_term}',
        }

        log(f"  Total models to fit: {len(models)}")
        # Fit All Models

        log("Fitting all models...")

        fitted_models = {}
        model_stats = []

        save_dir = RQ_DIR / "data"
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, formula in models.items():
            log(f"  Fitting {model_name}...")

            try:
                # Fit model with random intercepts only (for simplicity)
                result = fit_lmm_trajectory(
                    data=lmm_input,
                    formula=formula,
                    groups='UID',
                    re_formula='~1',  # Random intercept only
                    reml=False
                )

                fitted_models[model_name] = result

                model_stats.append({
                    'model_name': model_name,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'log_likelihood': result.llf,
                    'n_params': result.params.shape[0],
                    'converged': result.converged
                })

                log(f"    ✓ AIC={result.aic:.2f}")

            except Exception as e:
                log(f"    ✗ FAILED: {str(e)}")
                fitted_models[model_name] = None
                model_stats.append({
                    'model_name': model_name,
                    'AIC': np.inf,
                    'BIC': np.inf,
                    'log_likelihood': -np.inf,
                    'n_params': np.nan,
                    'converged': False
                })
        # Compute AIC Comparison Metrics

        log("Computing AIC comparison metrics...")

        comparison_df = pd.DataFrame(model_stats)

        # Remove failed models
        comparison_df = comparison_df[comparison_df['AIC'] != np.inf].copy()

        # Sort by AIC
        comparison_df = comparison_df.sort_values('AIC').reset_index(drop=True)

        # Compute delta AIC
        aic_min = comparison_df['AIC'].min()
        comparison_df['delta_AIC'] = comparison_df['AIC'] - aic_min

        # Compute Akaike weights
        comparison_df['akaike_weight'] = np.exp(-0.5 * comparison_df['delta_AIC'])
        weight_sum = comparison_df['akaike_weight'].sum()
        comparison_df['akaike_weight'] = comparison_df['akaike_weight'] / weight_sum

        # Compute cumulative weights
        comparison_df['cumulative_weight'] = comparison_df['akaike_weight'].cumsum()

        log("Extended Model Comparison:")
        log("")
        log(comparison_df[['model_name', 'AIC', 'delta_AIC', 'akaike_weight']].to_string(index=False))
        log("")
        # Identify Best Model

        best_model_name = comparison_df.iloc[0]['model_name']
        best_model_aic = comparison_df.iloc[0]['AIC']
        best_model_weight = comparison_df.iloc[0]['akaike_weight']

        log(f"[BEST MODEL] {best_model_name}")
        log(f"  AIC: {best_model_aic:.2f}")
        log(f"  Weight: {best_model_weight:.4f} ({best_model_weight*100:.1f}%)")

        # Find original Log model for comparison
        log_model_row = comparison_df[comparison_df['model_name'] == 'Log']
        if len(log_model_row) > 0:
            log_rank = log_model_row.index[0] + 1
            log_aic = log_model_row.iloc[0]['AIC']
            log_weight = log_model_row.iloc[0]['akaike_weight']
            log_delta = log_model_row.iloc[0]['delta_AIC']

            log("")
            log(f"Original 'Log' model:")
            log(f"  Rank: #{log_rank} of {len(comparison_df)}")
            log(f"  AIC: {log_aic:.2f} (Δ = {log_delta:.2f})")
            log(f"  Weight: {log_weight:.4f} ({log_weight*100:.1f}%)")
            log(f"  Original weight (5-model comparison): 99.99%")

        # Check power law models
        power_law_models = comparison_df[comparison_df['model_name'].str.contains('PowerLaw')]
        if len(power_law_models) > 0:
            log("")
            log("[POWER LAW MODELS]")
            for idx, row in power_law_models.iterrows():
                rank = idx + 1
                log(f"  #{rank}: {row['model_name']}")
                log(f"       AIC={row['AIC']:.2f}, Δ={row['delta_AIC']:.2f}, weight={row['akaike_weight']:.4f}")
        # Save Results

        log("Saving results...")

        # Save comparison table
        comparison_output = RQ_DIR / "results" / "step05b_extended_model_comparison.csv"
        comparison_df.to_csv(comparison_output, index=False, encoding='utf-8')
        log(f"  ✓ {comparison_output.name}")

        # Save fitted models
        pickle_path = RQ_DIR / "data" / "step05b_extended_model_fits.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(fitted_models, f)
        log(f"  ✓ {pickle_path.name}")
        # Summary Statistics

        log("")
        log("")
        log(f"  Total models tested: {len(models)}")
        log(f"  Successful fits: {len(comparison_df)}")
        log(f"  Failed fits: {len(models) - len(comparison_df)}")
        log(f"  Best model: {best_model_name}")
        log(f"  Best AIC: {best_model_aic:.2f}")
        log(f"  Original Log model rank: #{log_rank if len(log_model_row) > 0 else 'N/A'}")

        # Compare to RQ 5.1.1 findings
        log("")
        log("RQ 5.1.1 vs 5.3.1 Comparison:")
        log("  RQ 5.1.1 (single trajectory): Power law wins, Log ranked #10")
        log("  RQ 5.3.1 (Paradigm x Time): [results above]")

        # Check if any power law model beat Log
        if len(power_law_models) > 0:
            best_power_law_rank = power_law_models.index[0] + 1
            if best_power_law_rank < log_rank:
                log(f"  ✓ POWER LAW WINS: Best power law (#{best_power_law_rank}) beats Log (#{log_rank})")
            else:
                log(f"  ✗ LOG WINS: Log (#{log_rank}) beats best power law (#{best_power_law_rank})")

        log("Step 5b complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
