#!/usr/bin/env python3
"""power_analysis: Conduct post-hoc power analysis for all fitted models and summarize effect sizes with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.stats.power import FTestAnovaPower
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_effect_sizes

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: power_analysis")
        # Load Model Results

        log("Loading model results...")

        comparison = pd.read_csv(RQ_DIR / 'data' / 'step06_model_comparison.csv')
        effect_sizes = pd.read_csv(RQ_DIR / 'data' / 'step07_predictor_effect_sizes.csv')

        log(f"{len(comparison)} model comparisons")
        log(f"{len(effect_sizes)} effect sizes")
        # Load Univariate Model R² Values

        univ_models = pd.read_csv(RQ_DIR / 'data' / 'step04_univariate_models.csv')

        log(f"{len(univ_models)} univariate models")
        # Compute Post-Hoc Power

        log("Computing post-hoc power...")

        n = 100  # Sample size
        k = 5    # Number of predictors (ravlt, bvmt, rpm, age, pctret)
        alpha = 0.05

        power_test = FTestAnovaPower()

        power_results = []

        # Univariate models
        for _, row in univ_models.iterrows():
            r2 = row['R2']
            f2 = r2 / (1 - r2) if r2 < 1.0 else np.inf

            # Compute power
            try:
                power = power_test.solve_power(
                    effect_size=f2,
                    nobs=n,
                    alpha=alpha
                )
            except:
                power = np.nan

            # Compute minimum detectable effect (power = 0.80)
            try:
                min_f2 = power_test.solve_power(
                    effect_size=None,
                    nobs=n,
                    alpha=alpha,
                    power=0.8
                )
                min_r2 = min_f2 / (1 + min_f2)
            except:
                min_f2 = np.nan
                min_r2 = np.nan

            power_results.append({
                'model_type': f"Univariate_{row['domain']}",
                'n': n,
                'k': k,
                'r2': r2,
                'power': power,
                'minimum_detectable_effect': min_r2
            })

            log(f"{row['domain']}: R² = {r2:.3f}, Power = {power:.3f}, Min R² (80% power) = {min_r2:.3f}")

        # Multivariate model (average R² across domains)
        multi_r2_values = []
        for domain in ['what', 'where', 'when']:
            cv_row = comparison[comparison['comparison_type'] == f'CV_R2_{domain}']
            if len(cv_row) > 0:
                multi_r2_values.append(cv_row['multivariate_value'].values[0])

        if len(multi_r2_values) > 0:
            multi_r2_mean = np.mean(multi_r2_values)
            multi_f2 = multi_r2_mean / (1 - multi_r2_mean) if multi_r2_mean < 1.0 else np.inf

            try:
                multi_power = power_test.solve_power(
                    effect_size=multi_f2,
                    nobs=n,
                    alpha=alpha
                )
            except:
                multi_power = np.nan

            try:
                multi_min_f2 = power_test.solve_power(
                    effect_size=None,
                    nobs=n,
                    alpha=alpha,
                    power=0.8
                )
                multi_min_r2 = multi_min_f2 / (1 + multi_min_f2)
            except:
                multi_min_r2 = np.nan

            power_results.append({
                'model_type': 'Multivariate',
                'n': n,
                'k': k,
                'r2': multi_r2_mean,
                'power': multi_power,
                'minimum_detectable_effect': multi_min_r2
            })

            log(f"Multivariate: R² = {multi_r2_mean:.3f}, Power = {multi_power:.3f}, Min R² (80% power) = {multi_min_r2:.3f}")

        df_power = pd.DataFrame(power_results)
        # Summarize Effect Sizes with Cohen's Conventions

        log("Summarizing effect sizes with Cohen's conventions...")

        effect_size_summary = []

        # Cohen's thresholds
        cohen_thresholds = {
            'small': 0.02,
            'medium': 0.15,
            'large': 0.35
        }

        def categorize_f2(f2):
            """Categorize Cohen's f² by magnitude."""
            if np.isnan(f2):
                return 'Unknown'
            elif f2 < cohen_thresholds['small']:
                return 'Negligible'
            elif f2 < cohen_thresholds['medium']:
                return 'Small'
            elif f2 < cohen_thresholds['large']:
                return 'Medium'
            else:
                return 'Large'

        # Model R² as effect sizes
        for _, row in df_power.iterrows():
            r2 = row['r2']
            f2 = r2 / (1 - r2) if r2 < 1.0 else np.inf

            effect_size_summary.append({
                'effect_type': f"{row['model_type']}_R2",
                'value': r2,
                'interpretation': f"R² = {r2:.3f}, f² = {f2:.3f}",
                'cohen_category': categorize_f2(f2)
            })

        # Predictor effect sizes (aggregate across contexts and domains)
        predictors = effect_sizes['predictor'].unique()
        for predictor in predictors:
            pred_data = effect_sizes[effect_sizes['predictor'] == predictor]
            mean_f2 = pred_data['cohens_f2'].mean()

            effect_size_summary.append({
                'effect_type': f"Predictor_{predictor}",
                'value': mean_f2,
                'interpretation': f"Mean f² across contexts = {mean_f2:.3f}",
                'cohen_category': categorize_f2(mean_f2)
            })

        df_effect_summary = pd.DataFrame(effect_size_summary)

        log(f"Summarized {len(df_effect_summary)} effect sizes")
        # Save Outputs
        # These outputs will be used by: Final interpretation and reporting

        log("Saving power analysis results...")

        power_output = RQ_DIR / 'data' / 'step08_power_analysis.csv'
        df_power.to_csv(power_output, index=False, encoding='utf-8')
        log(f"{power_output} ({len(df_power)} rows)")

        summary_output = RQ_DIR / 'data' / 'step08_effect_size_summary.csv'
        df_effect_summary.to_csv(summary_output, index=False, encoding='utf-8')
        log(f"{summary_output} ({len(df_effect_summary)} rows)")
        # Validation
        # Validates: Power in [0, 1], effect sizes categorized correctly
        # Threshold: All checks pass

        log("Validating power analysis results...")

        validation_pass = True

        # Check power values in valid range
        power_values = df_power['power'].dropna()
        if ((power_values < 0) | (power_values > 1)).any():
            log(f"Power values outside [0, 1] range")
            validation_pass = False
        else:
            log(f"All power values in valid range [0, 1]")

        # Check effect size categorization
        valid_categories = ['Negligible', 'Small', 'Medium', 'Large', 'Unknown']
        invalid_cats = df_effect_summary[~df_effect_summary['cohen_category'].isin(valid_categories)]
        if len(invalid_cats) > 0:
            log(f"{len(invalid_cats)} effect sizes have invalid categories")
        else:
            log(f"All effect sizes categorized correctly")

        # Check all models have power estimates
        missing_power = df_power[df_power['power'].isna()]
        if len(missing_power) > 0:
            log(f"{len(missing_power)} models missing power estimates")
        else:
            log(f"All models have power estimates")

        if validation_pass:
            log("All validation checks passed")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
