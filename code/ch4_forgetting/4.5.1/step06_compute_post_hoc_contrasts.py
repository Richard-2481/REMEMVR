#!/usr/bin/env python3
"""Post-Hoc Contrasts with Dual P-Values (Decision D068): Extract LocationType main effect and LocationType × Time interaction from the"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy import stats

from tools.validation import validate_contrasts_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1
LOG_FILE = RQ_DIR / "logs" / "step06_compute_post_hoc_contrasts.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Post-Hoc Contrasts with Dual P-Values")
        # Load Best LMM Model

        log("Checking for Step 5 outputs...")

        # NOTE: We don't load the pickled model directly due to patsy environment issues
        # Instead, we use the pre-extracted coefficients from step05_lmm_coefficients.csv
        model_path = RQ_DIR / "data/step05_lmm_fitted_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Run Step 5 first.")
        log(f"Model file exists: {model_path.name}")

        # Load model comparison to confirm best model
        comparison_path = RQ_DIR / "data/step05_model_comparison.csv"
        model_comparison = pd.read_csv(comparison_path)
        best_model_name = model_comparison.iloc[0]['model_name']
        log(f"Best model: {best_model_name} (AIC: {model_comparison.iloc[0]['AIC']:.2f})")

        # Load original data for effect size calculations
        lmm_data_path = RQ_DIR / "data/step04_lmm_input.csv"
        lmm_data = pd.read_csv(lmm_data_path)
        log(f"{lmm_data_path.name} ({len(lmm_data)} rows)")
        # Extract Fixed Effects Coefficients

        log("Extracting fixed effects coefficients...")

        # Load pre-extracted coefficients (avoids patsy environment issues with pickled model)
        coef_path = RQ_DIR / "data" / "step05_lmm_coefficients.csv"
        if not coef_path.exists():
            raise FileNotFoundError(f"Coefficients file not found: {coef_path}. Run Step 5 first.")

        fe_df = pd.read_csv(coef_path)
        log(f"step05_lmm_coefficients.csv ({len(fe_df)} parameters)")

        # Identify coefficients of interest
        # Main effect: LocationType[T.source] (source vs destination baseline)
        # Interaction: log_Days_plus1:LocationType[T.source] (slope difference)
        # Note: Destination is the reference category in this model

        main_effect_row = fe_df[fe_df['parameter'] == 'LocationType[T.source]']
        interaction_row = fe_df[fe_df['parameter'] == 'log_Days_plus1:LocationType[T.source]']

        if main_effect_row.empty:
            raise ValueError("LocationType[T.source] coefficient not found in model")
        if interaction_row.empty:
            raise ValueError("log_Days_plus1:LocationType[T.source] coefficient not found in model")

        # Extract coefficient values from pre-saved CSV
        contrasts_data = []

        # Main effect (source vs destination)
        main_coef = float(main_effect_row.iloc[0]['coefficient'])
        main_se = float(main_effect_row.iloc[0]['std_error'])
        main_z = float(main_effect_row.iloc[0]['z_value'])
        main_p = float(main_effect_row.iloc[0]['p_value'])
        main_ci_lower = float(main_effect_row.iloc[0]['ci_lower'])
        main_ci_upper = float(main_effect_row.iloc[0]['ci_upper'])

        contrasts_data.append({
            'test_name': 'LocationType_main_effect',
            'coefficient': main_coef,
            'SE': main_se,
            'z': main_z,
            'p_uncorrected': main_p,
            'CI_lower': main_ci_lower,
            'CI_upper': main_ci_upper
        })

        log(f"Main effect: coef={main_coef:.4f}, z={main_z:.3f}, p={main_p:.4f}")

        # Interaction
        int_coef = float(interaction_row.iloc[0]['coefficient'])
        int_se = float(interaction_row.iloc[0]['std_error'])
        int_z = float(interaction_row.iloc[0]['z_value'])
        int_p = float(interaction_row.iloc[0]['p_value'])
        int_ci_lower = float(interaction_row.iloc[0]['ci_lower'])
        int_ci_upper = float(interaction_row.iloc[0]['ci_upper'])

        contrasts_data.append({
            'test_name': 'LocationType_x_Time_interaction',
            'coefficient': int_coef,
            'SE': int_se,
            'z': int_z,
            'p_uncorrected': int_p,
            'CI_lower': int_ci_lower,
            'CI_upper': int_ci_upper
        })

        log(f"Interaction: coef={int_coef:.4f}, z={int_z:.3f}, p={int_p:.4f}")

        # Create contrasts DataFrame
        contrasts = pd.DataFrame(contrasts_data)
        # Apply Bonferroni Correction (Decision D068)
        # Bonferroni correction: p_bonferroni = min(p_uncorrected × n_tests, 1.0)
        # n_tests = 2 (main effect + interaction)

        log("Applying Bonferroni correction (n_tests=2)...")

        n_tests = 2
        contrasts['p_bonferroni'] = contrasts['p_uncorrected'].apply(
            lambda p: min(p * n_tests, 1.0)
        )

        log(f"Bonferroni correction applied")
        log(f"  Main effect: p_uncorr={contrasts.loc[0, 'p_uncorrected']:.4f}, "
            f"p_bonf={contrasts.loc[0, 'p_bonferroni']:.4f}")
        log(f"  Interaction: p_uncorr={contrasts.loc[1, 'p_uncorrected']:.4f}, "
            f"p_bonf={contrasts.loc[1, 'p_bonferroni']:.4f}")
        # Compute Effect Sizes at Key Timepoints
        # Compute Cohen's d for source vs. destination at Days 0, 1, 3, 6
        # Using marginal means from model predictions

        log("Computing effect sizes at Days 0, 1, 3, 6...")

        timepoints = [0, 1, 3, 6]
        effect_sizes_data = []

        for day in timepoints:
            # Subset data to current timepoint (approximate match due to TSVR variation)
            # Use Days column (TSVR_hours / 24) for matching
            day_data = lmm_data[np.abs(lmm_data['Days'] - day) < 0.5].copy()

            if len(day_data) == 0:
                log(f"No data found for Day {day}, skipping...")
                continue

            # Split by LocationType
            source_data = day_data[day_data['LocationType'] == 'source']['theta']
            destination_data = day_data[day_data['LocationType'] == 'destination']['theta']

            # Compute means
            source_mean = source_data.mean()
            destination_mean = destination_data.mean()
            mean_diff = source_mean - destination_mean

            # Compute pooled SD for Cohen's d
            # Formula: SD_pooled = sqrt(((n1-1)*SD1^2 + (n2-1)*SD2^2) / (n1 + n2 - 2))
            n1 = len(source_data)
            n2 = len(destination_data)
            sd1 = source_data.std()
            sd2 = destination_data.std()

            if n1 > 1 and n2 > 1:
                pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
                cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
            else:
                cohens_d = np.nan
                pooled_sd = np.nan

            # Compute 95% CI for Cohen's d
            # Using delta method approximation
            if not np.isnan(cohens_d) and pooled_sd > 0:
                se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
                ci_lower = cohens_d - 1.96 * se_d
                ci_upper = cohens_d + 1.96 * se_d
            else:
                ci_lower = np.nan
                ci_upper = np.nan

            effect_sizes_data.append({
                'timepoint': f'Day{day}',
                'source_mean': source_mean,
                'destination_mean': destination_mean,
                'mean_difference': mean_diff,
                'cohens_d': cohens_d,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper
            })

            log(f"Day {day}: source_mean={source_mean:.3f}, "
                f"dest_mean={destination_mean:.3f}, d={cohens_d:.3f}")

        effect_sizes = pd.DataFrame(effect_sizes_data)
        log(f"Effect sizes computed for {len(effect_sizes)} timepoints")
        # Save Analysis Outputs
        # These outputs will be used by: rq_inspect for validation, results analysis for reporting

        log("Saving outputs...")

        # Save contrasts
        contrasts_path = RQ_DIR / "data/step06_post_hoc_contrasts.csv"
        contrasts.to_csv(contrasts_path, index=False, encoding='utf-8')
        log(f"{contrasts_path.name} ({len(contrasts)} rows, {len(contrasts.columns)} cols)")

        # Save effect sizes
        effect_sizes_path = RQ_DIR / "data/step06_effect_sizes.csv"
        effect_sizes.to_csv(effect_sizes_path, index=False, encoding='utf-8')
        log(f"{effect_sizes_path.name} ({len(effect_sizes)} rows, {len(effect_sizes.columns)} cols)")
        # Run Validation Tool
        # Validates: Decision D068 compliance (dual p-value reporting)
        # Threshold: Both p_uncorrected and p_bonferroni must be present

        log("Running validate_contrasts_d068...")
        validation_result = validate_contrasts_d068(contrasts_df=contrasts)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check validation passed
        if not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")

        if not validation_result.get('d068_compliant', False):
            raise ValueError(f"Decision D068 compliance failed: {validation_result.get('message', 'Missing dual p-values')}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
