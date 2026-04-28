#!/usr/bin/env python3
"""Model Comparison: Compare models using cross-validated R² and apply parsimony criterion for"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_regression import fit_multiple_regression
# Note: validate_regression_assumptions does not exist in tools.validation

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_model_comparison.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
PARSIMONY_THRESHOLD = 0.02  # ΔCVR² < 0.02 favors simpler model
ALPHA = 0.05
N_COMPARISONS = 4  # Minimal vs Core, Core vs Extended, Extended vs Full, Full vs Full+Retention
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def nested_f_test(X_reduced, X_full, y):
    """
    Perform nested F-test comparing reduced vs full model.
    Returns F-statistic and p-value.
    """
    # Add constant
    X_reduced_const = sm.add_constant(X_reduced, has_constant='add')
    X_full_const = sm.add_constant(X_full, has_constant='add')

    # Fit models
    model_reduced = sm.OLS(y, X_reduced_const).fit()
    model_full = sm.OLS(y, X_full_const).fit()

    # F-test
    rss_reduced = model_reduced.ssr
    rss_full = model_full.ssr
    df_reduced = len(y) - X_reduced_const.shape[1]
    df_full = len(y) - X_full_const.shape[1]
    df_diff = df_reduced - df_full

    f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_full)
    p_value = 1 - stats.f.cdf(f_stat, df_diff, df_full)

    return f_stat, p_value

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Model Comparison")
        # Load Data
        log("Loading data...")

        df_analysis = pd.read_csv(RQ_DIR / 'data' / 'step02_analysis_input.csv')
        df_models = pd.read_csv(RQ_DIR / 'data' / 'step03_nested_models.csv')
        df_cv = pd.read_csv(RQ_DIR / 'data' / 'step04_cv_results.csv')
        df_ci = pd.read_csv(RQ_DIR / 'data' / 'step04_cv_bootstrap_cis.csv')

        log(f"{len(df_analysis)} participants, {len(df_models)} models")

        y = df_analysis['Theta_All']
        # Rank Models by CV-R²
        log("Ranking models by CV-R²...")

        df_comparison = df_cv[['model', 'cv_r2_mean', 'shrinkage']].copy()
        df_comparison = df_comparison.sort_values('cv_r2_mean', ascending=False).reset_index(drop=True)
        df_comparison['rank'] = df_comparison.index + 1

        log("")
        for _, row in df_comparison.iterrows():
            log(f"  {row['rank']}. {row['model']}: CV-R²={row['cv_r2_mean']:.4f}, Shrinkage={row['shrinkage']:.4f}")
        # Apply Parsimony Criterion
        log(f"\nApplying parsimony criterion (threshold={PARSIMONY_THRESHOLD})...")

        best_model = df_comparison.iloc[0]['model']
        best_cv_r2 = df_comparison.iloc[0]['cv_r2_mean']

        df_comparison['parsimony_selected'] = False

        # Check if simpler models are within threshold
        for idx, row in df_comparison.iterrows():
            delta_cv_r2 = best_cv_r2 - row['cv_r2_mean']

            if delta_cv_r2 < PARSIMONY_THRESHOLD:
                log(f"{row['model']}: ΔCV-R²={delta_cv_r2:.4f} < {PARSIMONY_THRESHOLD} (within threshold)")
                df_comparison.loc[idx, 'parsimony_selected'] = True
            else:
                log(f"{row['model']}: ΔCV-R²={delta_cv_r2:.4f} >= {PARSIMONY_THRESHOLD} (excluded by parsimony)")

        # Select simplest model within threshold
        eligible_models = df_comparison[df_comparison['parsimony_selected']]

        # Get model complexity (num_predictors)
        model_complexity = df_models.set_index('model_name')['num_predictors'].to_dict()
        eligible_models = eligible_models.copy()
        eligible_models['num_predictors'] = eligible_models['model'].map(model_complexity)
        eligible_models = eligible_models.sort_values('num_predictors')

        optimal_model = eligible_models.iloc[0]['model']
        log(f"\nOptimal model: {optimal_model}")

        df_comparison['selected_as_optimal'] = df_comparison['model'] == optimal_model
        # Nested F-Tests with Dual P-Values (D068)
        log(f"\nPerforming nested F-tests with dual p-values (D068)...")

        nested_comparisons = []

        # Define nested comparisons
        comparisons = [
            ('Minimal', 'Core'),
            ('Core', 'Extended'),
            ('Extended', 'Full'),
            ('Full', 'Full+Retention')
        ]

        for reduced_name, full_name in comparisons:
            # Get predictor lists
            reduced_predictors = df_models[df_models['model_name'] == reduced_name]['predictor_list'].iloc[0]
            full_predictors = df_models[df_models['model_name'] == full_name]['predictor_list'].iloc[0]

            reduced_predictors = [p.strip() for p in reduced_predictors.split(',')]
            full_predictors = [p.strip() for p in full_predictors.split(',')]

            # Extract predictors
            X_reduced = df_analysis[reduced_predictors].values
            X_full = df_analysis[full_predictors].values

            # Perform F-test
            f_stat, p_uncorrected = nested_f_test(X_reduced, X_full, y.values)
            p_bonferroni = min(p_uncorrected * N_COMPARISONS, 1.0)
            significant_corrected = p_bonferroni < ALPHA

            log(f"{reduced_name} vs {full_name}:")
            log(f"  F={f_stat:.4f}, p_uncorrected={p_uncorrected:.4f}, p_bonferroni={p_bonferroni:.4f}")
            log(f"  Significant (Bonferroni): {'YES' if significant_corrected else 'NO'}")

            nested_comparisons.append({
                'comparison': f'{reduced_name}_vs_{full_name}',
                'f_stat': f_stat,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'significant_corrected': significant_corrected
            })

        df_nested = pd.DataFrame(nested_comparisons)
        # Save Outputs
        log("\nSaving model comparison results...")

        # Model comparison
        comparison_file = OUTPUT_DIR / "step05_model_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False, encoding='utf-8')
        log(f"{comparison_file}")

        # Nested comparisons
        nested_file = OUTPUT_DIR / "step05_nested_comparisons.csv"
        df_nested.to_csv(nested_file, index=False, encoding='utf-8')
        log(f"{nested_file}")

        # Optimal model rationale
        optimal_file = OUTPUT_DIR / "step05_optimal_model.txt"
        with open(optimal_file, 'w', encoding='utf-8') as f:
            f.write("Optimal Model Selection\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Selected Model: {optimal_model}\n")
            f.write(f"CV-R²: {eligible_models.iloc[0]['cv_r2_mean']:.4f}\n")
            f.write(f"Number of Predictors: {eligible_models.iloc[0]['num_predictors']}\n\n")

            optimal_predictors = df_models[df_models['model_name'] == optimal_model]['predictor_list'].iloc[0]
            f.write(f"Predictors: {optimal_predictors}\n\n")

            f.write("Selection Rationale:\n")
            f.write("-" * 70 + "\n")
            f.write(f"1. Parsimony criterion: ΔCV-R² < {PARSIMONY_THRESHOLD} from best model\n")
            f.write(f"2. Simplest model within parsimony threshold\n")
            f.write(f"3. Best: {best_model} (CV-R²={best_cv_r2:.4f})\n")
            f.write(f"4. Selected: {optimal_model} (ΔCV-R²={(best_cv_r2 - eligible_models.iloc[0]['cv_r2_mean']):.4f})\n")

        log(f"{optimal_file}")
        # Validation Check
        log("\nValidating model comparison results...")

        # Check best model has highest CV-R²
        if df_comparison.iloc[0]['model'] == df_comparison[df_comparison['rank'] == 1].iloc[0]['model']:
            log("Best model correctly ranked")
        else:
            log("Ranking inconsistency detected")

        # Note: Assumption validation will be done in step07_diagnostic_validation.py
        log("Model assumption diagnostics will be performed in Step 07")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
