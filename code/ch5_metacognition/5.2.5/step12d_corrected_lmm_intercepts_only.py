#!/usr/bin/env python3
"""
RQ 6.2.5: Corrected LMM with Intercepts Only (PLATINUM Fix)
===========================================================

Per step12c random slopes comparison: ΔAIC = +0.47 (slopes NOT justified)

Corrected model: calibration ~ TSVR_hours * Age_c + (1 | UID)

This replaces the original step02_fit_lmm analysis which incorrectly used
random slopes. Results should be qualitatively similar (both NULL) but
this is the statistically justified model per parsimony principle.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

# ============================================================================
# CONFIGURATION
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step12d_corrected_lmm.log"
BONFERRONI_ALPHA = 0.05 / 3  # 0.0167 for 3 comparisons per Decision D068

# Input
DATA_FILE = RQ_DIR / "data" / "step01_calibration_age_centered.csv"


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def fit_corrected_lmm(df: pd.DataFrame):
    """
    Fit corrected LMM with intercepts only.

    Model: calibration ~ TSVR_hours * Age_c + (1 | UID)
    """
    log("\n" + "=" * 70)
    log("CORRECTED LMM: Random Intercepts Only")
    log("=" * 70)

    formula = "calibration ~ TSVR_hours * Age_c"
    log(f"\nFormula: {formula}")
    log("Random effects: (1 | UID) - intercept only (per step12c comparison)")

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df['UID'],
            re_formula="~1"  # Intercept only (CORRECTED)
        )
        result = model.fit(method='powell', maxiter=1000)
        log("\n✓ Model converged successfully")
    except Exception as e:
        log(f"✗ ERROR: Model fitting failed: {e}")
        raise

    # Save full summary
    summary_path = RQ_DIR / "data" / "step12d_corrected_lmm_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"✓ Full summary saved: {summary_path}")

    # Extract fixed effects
    log("\n--- Fixed Effects ---")
    n_fe = len(result.model.exog_names)
    fe_names = result.model.exog_names
    fe_params = result.params[:n_fe]
    fe_bse = result.bse[:n_fe]
    fe_tvalues = result.tvalues[:n_fe]
    fe_pvalues = result.pvalues[:n_fe]

    fixed_effects = pd.DataFrame({
        'term': fe_names,
        'estimate': fe_params.values,
        'se': fe_bse.values,
        'z_value': fe_tvalues.values,
        'p_value': fe_pvalues.values
    })

    for _, row in fixed_effects.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        log(f"  {row['term']:20s}: β = {row['estimate']:8.5f}, SE = {row['se']:.5f}, z = {row['z_value']:6.2f}, p = {row['p_value']:.4f} {sig}")

    # Random effects variance
    log("\n--- Random Effects ---")
    re_var = result.cov_re
    var_intercept = re_var.iloc[0, 0]
    sd_intercept = np.sqrt(var_intercept)
    log(f"Var(Intercept): {var_intercept:.6f}")
    log(f"SD(Intercept): {sd_intercept:.6f}")

    # Model fit indices
    log("\n--- Model Fit ---")
    n_params = len(result.params)
    n_obs = len(df)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params
    log(f"Log-likelihood: {result.llf:.2f}")
    log(f"AIC: {aic:.2f}")
    log(f"BIC: {bic:.2f}")

    # Add dual p-values per Decision D068
    log("\n--- Age Effects with Dual P-Values (Decision D068) ---")
    log(f"Bonferroni alpha = {BONFERRONI_ALPHA:.4f} (0.05 / 3 comparisons)")

    age_terms = ['Age_c', 'TSVR_hours:Age_c', 'Age_c:TSVR_hours']
    age_effects = fixed_effects[fixed_effects['term'].isin(age_terms)].copy()

    age_effects['p_uncorrected'] = age_effects['p_value']
    age_effects['p_bonferroni'] = np.minimum(age_effects['p_value'] * 3, 1.0)
    age_effects['sig_uncorrected'] = age_effects['p_value'] < 0.05
    age_effects['sig_bonferroni'] = age_effects['p_bonferroni'] < 0.05

    for _, row in age_effects.iterrows():
        sig_unc = "YES" if row['sig_uncorrected'] else "NO"
        sig_bon = "YES" if row['sig_bonferroni'] else "NO"
        log(f"\n{row['term']}:")
        log(f"  Estimate: {row['estimate']:.6f}")
        log(f"  SE: {row['se']:.6f}")
        log(f"  z: {row['z_value']:.2f}")
        log(f"  p_uncorrected: {row['p_uncorrected']:.4f} (sig @ 0.05: {sig_unc})")
        log(f"  p_bonferroni: {row['p_bonferroni']:.4f} (sig @ 0.05: {sig_bon})")

    # Key hypothesis test
    interaction_mask = age_effects['term'].str.contains('TSVR_hours') & age_effects['term'].str.contains('Age_c')
    interaction_row = age_effects[interaction_mask].iloc[0]

    log("\n" + "=" * 70)
    log("PRIMARY HYPOTHESIS TEST: Age × Time Interaction")
    log("=" * 70)

    log(f"\nEstimate: {interaction_row['estimate']:.6f}")
    log(f"p_uncorrected: {interaction_row['p_uncorrected']:.4f}")
    log(f"p_bonferroni: {interaction_row['p_bonferroni']:.4f}")

    if interaction_row['sig_bonferroni']:
        conclusion = "SIGNIFICANT - Age moderates calibration trajectory"
    else:
        conclusion = "NULL - Age-invariant calibration trajectory (parallels Ch5 accuracy findings)"

    log(f"\nCONCLUSION: {conclusion}")

    # Save outputs
    fe_path = RQ_DIR / "data" / "step12d_corrected_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False)
    log(f"\n✓ Fixed effects saved: {fe_path}")

    age_path = RQ_DIR / "data" / "step12d_corrected_age_effects.csv"
    age_effects.to_csv(age_path, index=False)
    log(f"✓ Age effects saved: {age_path}")

    return result, fixed_effects, age_effects, interaction_row


def main():
    """Execute corrected LMM analysis."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.2.5: Corrected LMM (Intercepts Only)\n")
        f.write("=" * 70 + "\n")
        f.write("PLATINUM Fix: Random slopes NOT justified (ΔAIC = +0.47)\n\n")

    log("Starting corrected LMM analysis...")
    log(f"Data file: {DATA_FILE}")

    try:
        # Load data
        log("\n" + "=" * 70)
        log("LOADING DATA")
        log("=" * 70)

        df = pd.read_csv(DATA_FILE)
        log(f"\nLoaded: {len(df)} rows, {len(df.columns)} columns")
        log(f"Unique participants: {df['UID'].nunique()}")

        # Fit corrected model
        result, fixed_effects, age_effects, interaction = fit_corrected_lmm(df)

        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        log(f"\n✅ Corrected LMM complete")
        log(f"Primary finding: Age × Time p_bonferroni = {interaction['p_bonferroni']:.4f}")
        log(f"Conclusion: {'NULL' if not interaction['sig_bonferroni'] else 'SIGNIFICANT'}")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
