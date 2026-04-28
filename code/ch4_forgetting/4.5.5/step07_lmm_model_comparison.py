#!/usr/bin/env python3
"""LMM Model Comparison: Fit parallel Linear Mixed Models on z-standardized measurements (IRT, Full CTT,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory_tsvr

from tools.validation import validate_model_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5
LOG_FILE = RQ_DIR / "logs" / "step07_lmm_model_comparison.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: LMM Model Comparison")
        # Load Input Data

        log("Loading input data...")

        # Load standardized scores (test format: 'T1', 'T2', 'T3', 'T4')
        standardized_scores_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        df_scores = pd.read_csv(standardized_scores_path, encoding='utf-8')
        log(f"standardized_scores: {len(df_scores)} rows, {len(df_scores.columns)} cols")
        log(f"Columns: {df_scores.columns.tolist()}")
        log(f"Test format in scores: {df_scores['test'].unique()}")

        # Load TSVR mapping (test format: '1', '2', '3', '4')
        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"tsvr_mapping: {len(df_tsvr)} rows, {len(df_tsvr.columns)} cols")
        log(f"Test format in TSVR: {df_tsvr['test'].unique()}")
        # Merge Data and Create Time Variable
        # Critical: Convert TSVR test format '1','2','3','4' to 'T1','T2','T3','T4'
        # to match standardized_scores format for merge
        # Time = TSVR_hours / 24 (convert to days for interpretability)

        log("Converting TSVR test format and merging datasets...")

        # Convert TSVR test format: '1' -> 'T1', '2' -> 'T2', etc.
        df_tsvr['test'] = 'T' + df_tsvr['test'].astype(str)
        log(f"TSVR test format after conversion: {df_tsvr['test'].unique()}")

        # Merge on (UID, test)
        df_merged = df_scores.merge(
            df_tsvr[['UID', 'test', 'TSVR_hours']],
            on=['UID', 'test'],
            how='left'
        )
        log(f"Combined dataset: {len(df_merged)} rows")

        # Check for missing TSVR values after merge
        missing_tsvr = df_merged['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            log(f"{missing_tsvr} rows missing TSVR_hours after merge")
            sys.exit(1)

        # Create Time variable (days)
        df_merged['Time'] = df_merged['TSVR_hours'] / 24.0
        log(f"Time variable created: min={df_merged['Time'].min():.2f}, max={df_merged['Time'].max():.2f} days")
        # Fit 6 Parallel LMMs
        # Models:
        #   - Source_IRT:           outcome=irt_z,           filter=location_type=='source'
        #   - Source_Full_CTT:      outcome=ctt_full_z,      filter=location_type=='source'
        #   - Source_Purified_CTT:  outcome=ctt_purified_z,  filter=location_type=='source'
        #   - Destination_IRT:      outcome=irt_z,           filter=location_type=='destination'
        #   - Destination_Full_CTT: outcome=ctt_full_z,      filter=location_type=='destination'
        #   - Destination_Purified_CTT: outcome=ctt_purified_z, filter=location_type=='destination'
        # Formula: score ~ Time (random intercepts and slopes per UID)
        # REML=False required for valid AIC comparison

        log("Fitting 6 parallel LMMs...")

        # Define models to fit
        models = [
            {'name': 'Source_IRT',              'location': 'source',      'outcome': 'irt_z'},
            {'name': 'Source_Full_CTT',         'location': 'source',      'outcome': 'ctt_full_z'},
            {'name': 'Source_Purified_CTT',     'location': 'source',      'outcome': 'ctt_purified_z'},
            {'name': 'Destination_IRT',         'location': 'destination', 'outcome': 'irt_z'},
            {'name': 'Destination_Full_CTT',    'location': 'destination', 'outcome': 'ctt_full_z'},
            {'name': 'Destination_Purified_CTT','location': 'destination', 'outcome': 'ctt_purified_z'},
        ]

        # Fit all models
        fitted_models = {}
        aic_values = {}
        convergence_status = {}
        n_obs = {}

        for model_spec in models:
            model_name = model_spec['name']
            location = model_spec['location']
            outcome = model_spec['outcome']

            log(f"Fitting {model_name}...")

            # Filter data to location type
            df_filtered = df_merged[df_merged['location_type'] == location].copy()
            n_obs[model_name] = len(df_filtered)
            log(f"{model_name}: {n_obs[model_name]} observations")

            # Rename outcome column to 'score' for consistent formula
            df_filtered['score'] = df_filtered[outcome]

            # Fit LMM using fit_lmm_trajectory_tsvr
            # Note: fit_lmm_trajectory_tsvr expects theta_scores with specific columns
            # but we can use it by preparing data in expected format
            # Actually, for this step we'll use statsmodels directly since we have
            # standardized data and just need simple LMM fitting

            # Fit using statsmodels MixedLM directly
            formula = "score ~ Time"
            try:
                md = MixedLM.from_formula(
                    formula=formula,
                    data=df_filtered,
                    groups=df_filtered['UID'],
                    re_formula='~Time'
                )
                result = md.fit(reml=False)  # ML estimation for AIC comparison

                fitted_models[model_name] = result
                aic_values[model_name] = result.aic
                convergence_status[model_name] = result.converged

                log(f"{model_name}: AIC={result.aic:.2f}, converged={result.converged}")

            except Exception as e:
                log(f"{model_name} failed to fit: {str(e)}")
                fitted_models[model_name] = None
                aic_values[model_name] = np.nan
                convergence_status[model_name] = False
        # Compute AIC Comparisons Per Location Type
        # delta_aic_full_purified = AIC(Purified) - AIC(Full)
        # Positive delta = Purified has WORSE fit (paradox confirmed)
        # Interpretation per Burnham & Anderson (2002):
        #   |delta| <= 2: "No difference"
        #   delta > 2: "Full CTT favored" (Purified worse)
        #   delta < -2: "Purified CTT favored" (Purified better)

        log("Computing AIC comparisons...")

        results = []
        for location in ['source', 'destination']:
            # Extract AICs for this location
            aic_irt = aic_values[f'{location.capitalize()}_IRT']
            aic_full = aic_values[f'{location.capitalize()}_Full_CTT']
            aic_purified = aic_values[f'{location.capitalize()}_Purified_CTT']

            # Compute delta
            delta_aic = aic_purified - aic_full

            # Interpret per Burnham & Anderson (2002)
            if np.isnan(delta_aic):
                interpretation = "ERROR - AIC computation failed"
            elif abs(delta_aic) <= 2:
                interpretation = "No difference"
            elif delta_aic > 2:
                interpretation = "Full CTT favored"
            else:  # delta_aic < -2
                interpretation = "Purified CTT favored"

            # Check if all models converged
            converged_irt = convergence_status[f'{location.capitalize()}_IRT']
            converged_full = convergence_status[f'{location.capitalize()}_Full_CTT']
            converged_purified = convergence_status[f'{location.capitalize()}_Purified_CTT']
            converged_all = converged_irt and converged_full and converged_purified

            # Get n_observations (should be same for all models in location)
            n_observations = n_obs[f'{location.capitalize()}_IRT']

            results.append({
                'location_type': location,
                'aic_irt': aic_irt,
                'aic_ctt_full': aic_full,
                'aic_ctt_purified': aic_purified,
                'delta_aic_full_purified': delta_aic,
                'interpretation': interpretation,
                'n_observations': n_observations,
                'converged_all': converged_all
            })

            log(f"{location}: delta_aic={delta_aic:.2f}, interpretation='{interpretation}'")
        # Save Results
        # Output: data/step07_lmm_model_comparison.csv
        # Contains: AIC comparison results for 2 location types

        log("Saving AIC comparison results...")
        df_results = pd.DataFrame(results)
        output_path = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_results)} rows, {len(df_results.columns)} cols)")
        # Run Validation Tool
        # Validates: All 6 models converged, all AIC values finite
        # Threshold: converged_all = True, no NaN/Inf AIC values

        log("Running validate_model_convergence...")

        # Validate each fitted model
        validation_results = []
        for model_name, result in fitted_models.items():
            if result is not None:
                val_result = validate_model_convergence(result)
                validation_results.append({
                    'model': model_name,
                    'valid': val_result['valid'],
                    'converged': val_result['converged'],
                    'message': val_result['message']
                })
                log(f"{model_name}: {val_result['message']}")
            else:
                validation_results.append({
                    'model': model_name,
                    'valid': False,
                    'converged': False,
                    'message': 'Model fitting failed'
                })
                log(f"{model_name}: Model fitting failed")

        # Check overall validation status
        all_valid = all(v['valid'] for v in validation_results)
        all_aic_finite = all(np.isfinite(aic) for aic in aic_values.values())

        if not all_valid:
            log("Some models did not converge - check results carefully")
        if not all_aic_finite:
            log("Some AIC values are NaN or infinite - cannot compare models")
            sys.exit(1)

        # Final validation: check converged_all in results
        for _, row in df_results.iterrows():
            if row['converged_all']:
                log(f"{row['location_type']}: All 3 models converged")
            else:
                log(f"{row['location_type']}: Not all models converged")

        log("Step 7 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
