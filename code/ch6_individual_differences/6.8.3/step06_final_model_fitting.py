#!/usr/bin/env python3
"""Final Model Fitting: Fit optimal model on full dataset and extract detailed coefficient information"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_regression import bootstrap_regression_ci

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_final_model_fitting.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def compute_semipartial_correlations(model, X):
    """
    Compute semi-partial correlations (unique variance contributions).
    sr² = (R²_full - R²_without_predictor)
    """
    X_const = sm.add_constant(X, has_constant='add')
    y = model.model.endog

    r2_full = model.rsquared
    semipartial_r2s = []

    for i, predictor in enumerate(X.columns):
        # Fit model without this predictor
        X_reduced = X.drop(columns=[predictor])
        X_reduced_const = sm.add_constant(X_reduced, has_constant='add')

        model_reduced = sm.OLS(y, X_reduced_const).fit()
        r2_reduced = model_reduced.rsquared

        sr2 = r2_full - r2_reduced
        semipartial_r2s.append({
            'predictor': predictor,
            'sr2': sr2,
            'percent_unique_variance': sr2 * 100
        })

    return pd.DataFrame(semipartial_r2s)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Final Model Fitting")
        # Load Data and Optimal Model Specification
        log("Loading data and optimal model specification...")

        df_analysis = pd.read_csv(RQ_DIR / 'data' / 'step02_analysis_input.csv')
        df_models = pd.read_csv(RQ_DIR / 'data' / 'step03_nested_models.csv')
        df_comparison = pd.read_csv(RQ_DIR / 'data' / 'step05_model_comparison.csv')

        # Get optimal model
        optimal_model_name = df_comparison[df_comparison['selected_as_optimal']].iloc[0]['model']
        log(f"Optimal model: {optimal_model_name}")

        # Get predictors
        optimal_predictors_str = df_models[df_models['model_name'] == optimal_model_name]['predictor_list'].iloc[0]
        optimal_predictors = [p.strip() for p in optimal_predictors_str.split(',')]
        log(f"Predictors ({len(optimal_predictors)}): {optimal_predictors}")

        # Extract data
        X = df_analysis[optimal_predictors]
        y = df_analysis['Theta_All']

        log(f"N={len(df_analysis)}, Predictors={len(optimal_predictors)}, Outcome=Theta_All")
        # Fit Final Model
        log("Fitting final model on full dataset...")

        X_const = sm.add_constant(X, has_constant='add')
        model_final = sm.OLS(y, X_const).fit()

        r2 = model_final.rsquared
        adj_r2 = model_final.rsquared_adj
        f_stat = model_final.fvalue
        f_pvalue = model_final.f_pvalue

        log(f"R²={r2:.4f}, Adj R²={adj_r2:.4f}")
        log(f"F({model_final.df_model:.0f}, {model_final.df_resid:.0f})={f_stat:.4f}, p={f_pvalue:.4e}")
        # Bootstrap Confidence Intervals
        log(f"Computing {N_BOOTSTRAP} bootstrap CIs for coefficients...")

        bootstrap_result = bootstrap_regression_ci(
            X=X.values,
            y=y.values,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_STATE,
            alpha=1 - CONFIDENCE_LEVEL
        )

        log(f"Bootstrap complete")
        # Extract Coefficients with Dual P-Values (D068)
        log("Extracting coefficients with dual p-values...")

        n_predictors = len(optimal_predictors)
        coefficients = []

        # Bonferroni correction
        p_bonferroni_threshold = 0.05 / n_predictors

        # Extract coefficients (skip intercept for predictor-level reporting)
        for i, predictor in enumerate(['const'] + optimal_predictors):
            beta = model_final.params[i]
            se = model_final.bse[i]
            p_uncorrected = model_final.pvalues[i]
            p_bonferroni = min(p_uncorrected * n_predictors, 1.0)

            # Get bootstrap CI
            ci_lower = bootstrap_result['ci_lower'][i]
            ci_upper = bootstrap_result['ci_upper'][i]

            log(f"{predictor}: β={beta:.4f}, SE={se:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            log(f"       p_uncorr={p_uncorrected:.4f}, p_bonf={p_bonferroni:.4f}")

            coefficients.append({
                'predictor': predictor,
                'beta': beta,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni
            })

        df_coefficients = pd.DataFrame(coefficients)
        # Compute Semi-Partial Correlations
        log("Computing unique variance contributions...")

        df_semipartial = compute_semipartial_correlations(model_final, X)

        log("Unique variance:")
        for _, row in df_semipartial.iterrows():
            log(f"  {row['predictor']}: sr²={row['sr2']:.4f} ({row['percent_unique_variance']:.2f}%)")

        sum_sr2 = df_semipartial['sr2'].sum()
        log(f"Sum of sr²: {sum_sr2:.4f} (Model R²: {r2:.4f})")
        # Save Outputs
        log("\nSaving final model results...")

        # Model summary
        summary_data = [{
            'r2': r2,
            'adj_r2': adj_r2,
            'f_stat': f_stat,
            'model_p_value': f_pvalue
        }]
        df_summary = pd.DataFrame(summary_data)
        summary_file = OUTPUT_DIR / "step06_final_model_summary.csv"
        df_summary.to_csv(summary_file, index=False, encoding='utf-8')
        log(f"{summary_file}")

        # Coefficients
        coef_file = OUTPUT_DIR / "step06_coefficients.csv"
        df_coefficients.to_csv(coef_file, index=False, encoding='utf-8')
        log(f"{coef_file}")

        # Semi-partial correlations
        semipartial_file = OUTPUT_DIR / "step06_semipartial_correlations.csv"
        df_semipartial.to_csv(semipartial_file, index=False, encoding='utf-8')
        log(f"{semipartial_file}")
        # Validation Checks
        log("\nValidating final model results...")

        # Check R² reasonable
        cv_r2 = df_comparison[df_comparison['model'] == optimal_model_name]['cv_r2_mean'].iloc[0]
        r2_diff = abs(r2 - cv_r2)

        if r2_diff < 0.10:
            log(f"R²={r2:.4f} close to CV-R²={cv_r2:.4f} (diff={r2_diff:.4f})")
        else:
            log(f"R²={r2:.4f} differs from CV-R²={cv_r2:.4f} by {r2_diff:.4f}")

        # Check bootstrap CI widths
        df_coefficients['ci_width'] = df_coefficients['ci_upper'] - df_coefficients['ci_lower']
        max_ci_width = df_coefficients['ci_width'].max()

        if max_ci_width < 2.0:
            log(f"Bootstrap CIs stable (max width={max_ci_width:.4f})")
        else:
            log(f"Wide bootstrap CIs detected (max width={max_ci_width:.4f})")

        # Check semi-partial R² sum
        if abs(sum_sr2 - r2) < 0.05:
            log(f"Sum of sr²={sum_sr2:.4f} approximately equals R²={r2:.4f}")
        else:
            log(f"Sum of sr²={sum_sr2:.4f} vs R²={r2:.4f} (difference due to shared variance)")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
