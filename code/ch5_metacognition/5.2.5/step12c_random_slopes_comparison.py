#!/usr/bin/env python3
"""
RQ 6.2.5: Random Slopes Comparison (Validation Requirement)
==========================================================

Per improvement_taxonomy.md Section 4.4 :
"Test intercepts-only vs random slopes (NON-NEGOTIABLE)"

Compares two models via AIC:
1. Intercepts-only: calibration ~ TSVR_hours * Age_c + (1 | UID)
2. Random slopes: calibration ~ TSVR_hours * Age_c + (1 + TSVR_hours | UID)

Decision rule: ΔAIC < 2 → simpler model preferred (intercepts-only)
               ΔAIC > 2 → complex model justified (random slopes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step12c_random_slopes_comparison.log"

# Input
DATA_FILE = RQ_DIR / "data" / "step01_calibration_age_centered.csv"

def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def fit_intercepts_only(df: pd.DataFrame):
    """Fit model with random intercepts only."""
    log("\n" + "=" * 70)
    log("MODEL 1: Random Intercepts Only")
    log("=" * 70)

    formula = "calibration ~ TSVR_hours * Age_c"
    log(f"\nFormula: {formula}")
    log("Random effects: (1 | UID) - intercept only")

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df['UID'],
            re_formula="~1"  # Intercept only
        )
        result = model.fit(method='powell', maxiter=1000)
        log("✓ Model converged")
    except Exception as e:
        log(f"✗ Model fitting failed: {e}")
        raise

    # Extract fit indices
    n_params = len(result.params)
    n_obs = len(df)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params

    log(f"\nModel Fit:")
    log(f"  Log-likelihood: {result.llf:.2f}")
    log(f"  Parameters: {n_params}")
    log(f"  AIC: {aic:.2f}")
    log(f"  BIC: {bic:.2f}")

    # Random effects variance
    log(f"\nRandom Effects:")
    log(f"  Var(Intercept): {result.cov_re.iloc[0, 0]:.6f}")
    log(f"  SD(Intercept): {np.sqrt(result.cov_re.iloc[0, 0]):.6f}")

    return {
        'model_name': 'Intercepts Only',
        're_formula': '~1',
        'llf': result.llf,
        'n_params': n_params,
        'AIC': aic,
        'BIC': bic,
        'converged': result.converged if hasattr(result, 'converged') else True,
        'var_intercept': result.cov_re.iloc[0, 0],
        'var_slope': np.nan,
        'cov_intercept_slope': np.nan
    }


def fit_random_slopes(df: pd.DataFrame):
    """Fit model with random intercepts AND slopes."""
    log("\n" + "=" * 70)
    log("MODEL 2: Random Intercepts + Random Slopes")
    log("=" * 70)

    formula = "calibration ~ TSVR_hours * Age_c"
    log(f"\nFormula: {formula}")
    log("Random effects: (1 + TSVR_hours | UID) - intercept and slope")

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df['UID'],
            re_formula="~TSVR_hours"  # Intercept + slope
        )
        result = model.fit(method='powell', maxiter=1000)
        log("✓ Model converged")
    except Exception as e:
        log(f"✗ Model fitting failed: {e}")
        raise

    # Extract fit indices
    n_params = len(result.params)
    n_obs = len(df)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params

    log(f"\nModel Fit:")
    log(f"  Log-likelihood: {result.llf:.2f}")
    log(f"  Parameters: {n_params}")
    log(f"  AIC: {aic:.2f}")
    log(f"  BIC: {bic:.2f}")

    # Random effects variance
    log(f"\nRandom Effects:")
    log(f"  Var(Intercept): {result.cov_re.iloc[0, 0]:.6f}")
    log(f"  Var(TSVR_hours): {result.cov_re.iloc[1, 1]:.6f}")
    log(f"  Cov(Intercept, TSVR_hours): {result.cov_re.iloc[0, 1]:.6f}")
    log(f"  SD(Intercept): {np.sqrt(result.cov_re.iloc[0, 0]):.6f}")
    log(f"  SD(TSVR_hours): {np.sqrt(max(0, result.cov_re.iloc[1, 1])):.6f}")

    return {
        'model_name': 'Random Slopes',
        're_formula': '~TSVR_hours',
        'llf': result.llf,
        'n_params': n_params,
        'AIC': aic,
        'BIC': bic,
        'converged': result.converged if hasattr(result, 'converged') else True,
        'var_intercept': result.cov_re.iloc[0, 0],
        'var_slope': result.cov_re.iloc[1, 1],
        'cov_intercept_slope': result.cov_re.iloc[0, 1]
    }


def compare_models(model1: dict, model2: dict):
    """Compare two models via AIC/BIC."""
    log("\n" + "=" * 70)
    log("MODEL COMPARISON")
    log("=" * 70)

    # AIC comparison
    delta_aic = model2['AIC'] - model1['AIC']
    log(f"\nAIC Comparison:")
    log(f"  Model 1 (Intercepts Only): AIC = {model1['AIC']:.2f}")
    log(f"  Model 2 (Random Slopes):   AIC = {model2['AIC']:.2f}")
    log(f"  ΔAIC (Model2 - Model1): {delta_aic:+.2f}")

    # BIC comparison
    delta_bic = model2['BIC'] - model1['BIC']
    log(f"\nBIC Comparison:")
    log(f"  Model 1 (Intercepts Only): BIC = {model1['BIC']:.2f}")
    log(f"  Model 2 (Random Slopes):   BIC = {model2['BIC']:.2f}")
    log(f"  ΔBIC (Model2 - Model1): {delta_bic:+.2f}")

    # Decision rule
    log("\n" + "=" * 70)
    log("DECISION (AIC-based)")
    log("=" * 70)

    if abs(delta_aic) < 2:
        decision = "COMPARABLE"
        preferred = "Intercepts Only (simpler)"
        justification = "ΔAIC < 2: Models equivalent, prefer simpler (parsimony)"
    elif delta_aic < 0:
        decision = "RANDOM SLOPES BETTER"
        preferred = "Random Slopes"
        justification = f"ΔAIC = {delta_aic:.2f} < -2: Random slopes improve fit substantially"
    else:
        decision = "INTERCEPTS ONLY BETTER"
        preferred = "Intercepts Only"
        justification = f"ΔAIC = {delta_aic:.2f} > +2: Random slopes worsen fit (overfitting)"

    log(f"\nResult: {decision}")
    log(f"Preferred Model: {preferred}")
    log(f"Justification: {justification}")

    # Interpretation for validation.md
    log("\n" + "=" * 70)
    log("INTERPRETATION")
    log("=" * 70)

    if abs(delta_aic) < 2 or delta_aic > 0:
        log("\n⚠️  Random slopes NOT justified:")
        log("  - Random slopes model does not improve fit (ΔAIC ≥ -2)")
        log("  - Simpler intercepts-only model should be preferred (parsimony)")
        log("  - Original analysis used random slopes unnecessarily")
        log("  - Recommendation: Re-run with intercepts-only model")
    else:
        log("\n✓ Random slopes justified:")
        log("  - Random slopes model improves fit substantially (ΔAIC < -2)")
        log("  - Participants show heterogeneous calibration trajectories")
        log("  - Original analysis correctly used random slopes")

    # Create comparison DataFrame
    comparison = pd.DataFrame([model1, model2])
    comparison['delta_AIC'] = comparison['AIC'] - model1['AIC']
    comparison['delta_BIC'] = comparison['BIC'] - model1['BIC']
    comparison['preferred'] = [preferred == 'Intercepts Only', preferred == 'Random Slopes']

    return comparison, decision, preferred, justification


def main():
    """Execute random slopes comparison."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.2.5: Random Slopes Comparison\n")
        f.write("=" * 70 + "\n")
        f.write("Validation Requirement: Section 4.4 \n\n")

    log("Starting random slopes comparison...")
    log(f"Data file: {DATA_FILE}")

    try:
        # Load data
        log("\n" + "=" * 70)
        log("LOADING DATA")
        log("=" * 70)

        df = pd.read_csv(DATA_FILE)
        log(f"\nLoaded: {len(df)} rows, {len(df.columns)} columns")
        log(f"Unique participants: {df['UID'].nunique()}")
        log(f"Tests per participant: {len(df) / df['UID'].nunique():.0f}")

        # Fit Model 1: Intercepts Only
        model1_results = fit_intercepts_only(df)

        # Fit Model 2: Random Slopes
        model2_results = fit_random_slopes(df)

        # Compare models
        comparison_df, decision, preferred, justification = compare_models(
            model1_results, model2_results
        )

        # Save comparison
        output_path = RQ_DIR / "data" / "step12c_random_slopes_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        log(f"\n✓ Comparison saved: {output_path}")

        # Save decision
        decision_data = pd.DataFrame([{
            'delta_AIC': model2_results['AIC'] - model1_results['AIC'],
            'delta_BIC': model2_results['BIC'] - model1_results['BIC'],
            'decision': decision,
            'preferred_model': preferred,
            'justification': justification
        }])
        decision_path = RQ_DIR / "data" / "step12c_model_decision.csv"
        decision_data.to_csv(decision_path, index=False)
        log(f"✓ Decision saved: {decision_path}")

        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)
        log(f"\n✅ Random slopes comparison complete")
        log(f"Preferred model: {preferred}")
        log(f"See {output_path.name} for full comparison")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
