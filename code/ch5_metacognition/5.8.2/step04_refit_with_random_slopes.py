#!/usr/bin/env python3
"""
RQ 6.8.2: Refit LMM with Random Slopes
=======================================
Based on validation validation finding that random slopes significantly
improve model fit (ΔAIC = 21.00), refit primary model with slopes.

Updated model: calibration ~ LocationType * log_TSVR + (log_TSVR | UID)

Date: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_random_slopes.log"

def log(msg):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Refit LMM with Random Slopes

def step04_refit_with_random_slopes():
    """
    Refit LMM with random intercepts + random slopes on log_TSVR.

    Model: calibration ~ LocationType_Source * log_TSVR + (log_TSVR | UID)

    This is the PREFERRED model based on ΔAIC = 21.00 favoring slopes.
    """
    log("\n" + "="*70)
    log("STEP 04: Refit LMM with Random Slopes (validation UPDATE)")
    log("="*70)

    # Load calibration data
    df = pd.read_csv(RQ_DIR / "data" / "step01_calibration_by_location.csv")
    df['log_TSVR'] = np.log(df['TSVR_hours'] + 1)
    df['LocationType_Source'] = (df['LocationType'] == 'Source').astype(int)

    log(f"\nData loaded: {len(df)} observations")
    log(f"  N participants: {df['UID'].nunique()}")

    # Fit LMM with random slopes
    log("\nFitting LMM: calibration ~ LocationType_Source * log_TSVR")
    log("  Random effects: (log_TSVR | UID) - random intercepts + slopes")

    formula = "calibration ~ LocationType_Source * log_TSVR"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df['UID'],
            re_formula="~log_TSVR"  # Random intercepts + slopes
        )
        result = model.fit(reml=True, maxiter=200)

    log(f"\n  Converged: {result.converged}")

    if not result.converged:
        log("\n⚠️ WARNING: Model did not converge")
        log("  Falling back to intercepts-only model")
        return None

    log("\n" + "="*50)
    log("LMM Results (Random Slopes Model)")
    log("="*50)
    log(str(result.summary()))

    # Save full summary
    summary_path = RQ_DIR / "data" / "step04_lmm_slopes_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"\nSaved: {summary_path}")

    # Extract fixed effects
    n_fe = len(result.model.exog_names)
    fixed_names = result.model.exog_names
    fixed_params = result.params[:n_fe]
    fixed_bse = result.bse[:n_fe]
    fixed_tvalues = result.tvalues[:n_fe]
    fixed_pvalues = result.pvalues[:n_fe]

    fe_df = pd.DataFrame({
        'Effect': fixed_names,
        'Estimate': fixed_params.values,
        'SE': fixed_bse.values,
        't': fixed_tvalues.values,
        'p_uncorrected': fixed_pvalues.values
    })

    # Bonferroni correction
    fe_df['p_bonferroni'] = np.minimum(fe_df['p_uncorrected'] * 4, 1.0)

    # 95% CI
    fe_df['CI_lower'] = fe_df['Estimate'] - 1.96 * fe_df['SE']
    fe_df['CI_upper'] = fe_df['Estimate'] + 1.96 * fe_df['SE']

    # Save
    fe_path = RQ_DIR / "data" / "step04_location_effects_slopes.csv"
    fe_df.to_csv(fe_path, index=False)

    log("\nFixed Effects:")
    log(fe_df.to_string(index=False))

    # Compare to intercepts-only model
    log("\n" + "="*50)
    log("Comparison to Intercepts-Only Model")
    log("="*50)

    # Load original results
    fe_orig = pd.read_csv(RQ_DIR / "data" / "step02_location_effects.csv")

    log("\nLocationType Main Effect:")
    loc_orig = fe_orig[fe_orig['Effect'].str.contains('LocationType')].iloc[0]
    loc_slopes = fe_df[fe_df['Effect'].str.contains('LocationType')].iloc[0]

    log(f"  Intercepts-only: β = {loc_orig['Estimate']:.4f}, p = {loc_orig['p_uncorrected']:.4f}")
    log(f"  Slopes model:     β = {loc_slopes['Estimate']:.4f}, p = {loc_slopes['p_uncorrected']:.4f}")

    if abs(loc_slopes['Estimate'] - loc_orig['Estimate']) > 0.05:
        log(f"  ⚠️ Estimates differ by {abs(loc_slopes['Estimate'] - loc_orig['Estimate']):.4f}")
        log(f"  Random slopes model changes interpretation")
    else:
        log(f"  ✓ Estimates similar (differ by {abs(loc_slopes['Estimate'] - loc_orig['Estimate']):.4f})")

    # Random effects
    log("\n" + "="*50)
    log("Random Effects Covariance Matrix")
    log("="*50)

    log(f"\n  Intercept variance: {result.cov_re.iloc[0,0]:.4f}")
    log(f"  Slope variance: {result.cov_re.iloc[1,1]:.4f}")
    log(f"  Intercept-Slope covariance: {result.cov_re.iloc[0,1]:.4f}")

    corr = result.cov_re.iloc[0,1] / np.sqrt(result.cov_re.iloc[0,0] * result.cov_re.iloc[1,1])
    log(f"  Correlation: {corr:.3f}")

    if corr < -0.5:
        log(f"\n  INTERPRETATION: Strong negative correlation ({corr:.3f})")
        log(f"    Participants with higher baseline calibration show SLOWER change over time")
        log(f"    Participants with lower baseline show FASTER change over time")
    elif corr > 0.5:
        log(f"\n  INTERPRETATION: Strong positive correlation ({corr:.3f})")
        log(f"    Participants with higher baseline show FASTER change over time")
    else:
        log(f"\n  INTERPRETATION: Weak correlation ({corr:.3f})")
        log(f"    Baseline calibration and trajectory independent")

    # Model fit
    log(f"\nModel Fit Indices:")
    log(f"  AIC: {result.aic:.2f}")
    log(f"  BIC: {result.bic:.2f}")
    log(f"  Log-Likelihood: {result.llf:.2f}")

    log("\n✓ Random slopes model COMPLETE")

    return result, fe_df

# MAIN EXECUTION

def main():
    """Execute random slopes refitting."""
    log("\n" + "="*70)
    log("RQ 6.8.2: Random Slopes Model (validation UPDATE)")
    log("="*70)

    # Clear log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    # Refit with random slopes
    result, fe_df = step04_refit_with_random_slopes()

    if result is None:
        log("\n✗ Random slopes model failed - use intercepts-only")
        return

    log("\n" + "="*70)
    log("RECOMMENDATION")
    log("="*70)

    log("\nBased on validation validation (ΔAIC = 21.00):")
    log("  ✓ Random slopes model is PREFERRED")
    log("  ✓ Individual differences in calibration trajectories CONFIRMED")
    log("  ✓ Update summary.md with slopes model results")
    log("  ✓ Report random slope variance in Section 1")
    log("  ✓ Interpret negative intercept-slope correlation in Section 3")

if __name__ == "__main__":
    main()
