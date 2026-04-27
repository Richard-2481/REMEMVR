#!/usr/bin/env python3
"""
===============================================================================
RQ 5.4.3 - Step 02: Fit LMM with 3-Way Age x Congruence x Time Interaction
===============================================================================

PURPOSE:
    Fit Linear Mixed Model with 3-way Age x Congruence x Time interaction.
    Random intercepts and slopes for recip_TSVR by participant.
    UPDATED: RQ 5.4.1 ROOT established Recip+Log as two-process forgetting model
    (rapid 1/(t+1) + slow log(t+1)). Random slope on recip_TSVR per ROOT spec.

INPUTS:
    - data/step01_lmm_input.csv (1200 rows, long format)

OUTPUTS:
    - data/step02_lmm_model.pkl (pickled model object)
    - data/step02_lmm_model_summary.txt (model summary text)
    - data/step02_fixed_effects.csv (24 fixed effects terms, including recip_TSVR)

MODEL FORMULA:
    theta ~ recip_TSVR + log_TSVR + Age_c + C(congruence, Treatment('Common')) +
            Age_c:recip_TSVR + Age_c:log_TSVR +
            C(congruence):recip_TSVR + C(congruence):log_TSVR +
            Age_c:C(congruence) +
            Age_c:C(congruence):recip_TSVR + Age_c:C(congruence):log_TSVR

    Random: ~recip_TSVR | UID (random intercepts and slopes per RQ 5.4.1 ROOT)

===============================================================================
"""

import sys
import traceback
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.4.3"
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step02_fit_lmm.log"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

logger = Logger(LOG_FILE)
log = logger.log

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def main():
    log("[START] Step 02: Fit LMM with 3-Way Interaction")
    log("")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    log("[STEP 1] Load LMM Input Data")
    log("-" * 70)

    lmm_input = pd.read_csv(DATA_DIR / "step01_lmm_input.csv", encoding='utf-8')
    log(f"[LOADED] LMM input: {len(lmm_input)} rows, {len(lmm_input.columns)} columns")
    log(f"[INFO] Columns: {list(lmm_input.columns)}")
    log(f"[INFO] Unique UIDs: {lmm_input['UID'].nunique()}")
    log(f"[INFO] Congruence levels: {sorted(lmm_input['congruence'].unique())}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 2: Create Dummy Variables for Congruence
    # -------------------------------------------------------------------------
    log("[STEP 2] Create Dummy Variables for Congruence")
    log("-" * 70)

    # Reference category: Common (as specified in concept.md)
    # Create explicit dummy variables for cleaner formula
    lmm_input['Congruent'] = (lmm_input['congruence'] == 'Congruent').astype(int)
    lmm_input['Incongruent'] = (lmm_input['congruence'] == 'Incongruent').astype(int)

    log(f"[CREATED] Dummy variables: Congruent, Incongruent (reference: Common)")
    log(f"[INFO] Congruent sum: {lmm_input['Congruent'].sum()} (expected: 400)")
    log(f"[INFO] Incongruent sum: {lmm_input['Incongruent'].sum()} (expected: 400)")
    log("")

    # -------------------------------------------------------------------------
    # STEP 3: Fit LMM Model
    # -------------------------------------------------------------------------
    log("[STEP 3] Fit LMM Model")
    log("-" * 70)

    # Formula: Full 3-way interaction model
    # Using explicit dummy variables for clarity
    formula = """
    theta ~ 1 + recip_TSVR + log_TSVR + Age_c +
            Congruent + Incongruent +
            Age_c:recip_TSVR + Age_c:log_TSVR +
            Congruent:recip_TSVR + Congruent:log_TSVR +
            Incongruent:recip_TSVR + Incongruent:log_TSVR +
            Age_c:Congruent + Age_c:Incongruent +
            Age_c:Congruent:recip_TSVR + Age_c:Congruent:log_TSVR +
            Age_c:Incongruent:recip_TSVR + Age_c:Incongruent:log_TSVR
    """.strip().replace('\n', ' ')

    log(f"[INFO] Model formula (fixed effects - Recip+Log two-process):")
    log(f"  theta ~ 1 + recip_TSVR + log_TSVR + Age_c + Congruent + Incongruent")
    log(f"        + Age_c:recip_TSVR + Age_c:log_TSVR")
    log(f"        + Congruent:recip_TSVR + Congruent:log_TSVR")
    log(f"        + Incongruent:recip_TSVR + Incongruent:log_TSVR")
    log(f"        + Age_c:Congruent + Age_c:Incongruent")
    log(f"        + Age_c:Congruent:recip_TSVR + Age_c:Congruent:log_TSVR")
    log(f"        + Age_c:Incongruent:recip_TSVR + Age_c:Incongruent:log_TSVR")
    log("")
    # UPDATED: RQ 5.4.1 ROOT established Recip+Log as two-process forgetting model
    # Random slope on recip_TSVR (rapid component)
    log(f"[INFO] Random effects: ~recip_TSVR | UID (per RQ 5.4.1 ROOT Recip+Log model)")
    log(f"[INFO] Two-process forgetting: rapid 1/(t+1) + slow log(t+1)")
    log("")

    log("[INFO] Fitting model (this may take a moment)...")

    # Try to fit with random slopes on log_TSVR (per RQ 5.4.1 best model)
    try:
        model = smf.mixedlm(
            formula=formula,
            data=lmm_input,
            groups=lmm_input['UID'],
            re_formula='~recip_TSVR'  # UPDATED: recip_TSVR for two-process forgetting (per 5.4.1 ROOT)
        )
        result = model.fit(method='lbfgs', maxiter=500)
        random_structure = "random intercepts + slopes for recip_TSVR (per 5.4.1 ROOT)"
        log(f"[SUCCESS] Model fitted with random slopes on recip_TSVR (two-process forgetting)")

    except Exception as e:
        log(f"[WARNING] Random slopes failed: {e}")
        log("[INFO] Trying random intercepts only...")

        try:
            model = smf.mixedlm(
                formula=formula,
                data=lmm_input,
                groups=lmm_input['UID']
            )
            result = model.fit(method='lbfgs', maxiter=500)
            random_structure = "random intercepts only"
            log(f"[SUCCESS] Model fitted with random intercepts only")

        except Exception as e2:
            log(f"[FAIL] Model fitting failed: {e2}")
            return False

    # Check convergence
    converged = result.converged
    log(f"[INFO] Model converged: {converged}")

    if not converged:
        log("[WARNING] Model did not converge - results may be unreliable")
    log("")

    # -------------------------------------------------------------------------
    # STEP 4: Extract Fixed Effects
    # -------------------------------------------------------------------------
    log("[STEP 4] Extract Fixed Effects")
    log("-" * 70)

    # Get summary table
    summary = result.summary()
    log(f"[INFO] Model summary available")

    # Extract fixed effects - need to align indices properly
    fe_params = result.fe_params

    # Get summary table which has all aligned values
    summary_df = result.summary().tables[1]

    # Extract from summary table (more reliable alignment)
    # The summary table has: Coef., Std.Err., z, P>|z|, [0.025, 0.975]
    fixed_effects_list = []

    for term in fe_params.index:
        coef = fe_params[term]
        # Get SE, z, p from the result object using the aligned param indices
        try:
            idx = list(result.params.index).index(term)
            se = result.bse.iloc[idx]
            z = result.tvalues.iloc[idx]
            p = result.pvalues.iloc[idx]
        except (ValueError, IndexError):
            # Fallback - extract from fe-specific attributes
            se = result.bse_fe.get(term, np.nan) if hasattr(result.bse_fe, 'get') else result.bse_fe[list(fe_params.index).index(term)]
            z = coef / se if se > 0 else np.nan
            # Use scipy for p-value calculation
            from scipy import stats
            p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

        fixed_effects_list.append({
            'term': term,
            'coef': coef,
            'se': se,
            'z': z,
            'p': p
        })

    fixed_effects = pd.DataFrame(fixed_effects_list)

    log(f"[EXTRACTED] Fixed effects: {len(fixed_effects)} terms")
    log("")

    # Print key terms
    log("[INFO] Fixed Effects Table:")
    log("-" * 70)
    for _, row in fixed_effects.iterrows():
        sig = "*" if row['p'] < 0.05 else ""
        log(f"  {row['term']:40s} coef={row['coef']:8.4f} SE={row['se']:6.4f} p={row['p']:.4f}{sig}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 5: Model Diagnostics
    # -------------------------------------------------------------------------
    log("[STEP 5] Model Diagnostics")
    log("-" * 70)

    # Model fit statistics
    log(f"[INFO] Log-Likelihood: {result.llf:.2f}")
    log(f"[INFO] AIC: {result.aic:.2f}")
    log(f"[INFO] BIC: {result.bic:.2f}")
    log("")

    # Random effects variance
    log("[INFO] Random Effects:")
    log(f"  Group Var (UID): {result.cov_re.iloc[0, 0]:.4f}")
    if result.cov_re.shape[0] > 1:
        log(f"  TSVR_hours Var: {result.cov_re.iloc[1, 1]:.4f}")
        log(f"  Covariance: {result.cov_re.iloc[0, 1]:.4f}")
    log(f"  Residual Var: {result.scale:.4f}")
    log("")

    # Sample size info
    log(f"[INFO] Sample Size:")
    log(f"  Observations: {result.nobs}")
    n_groups = lmm_input['UID'].nunique()
    log(f"  Groups (UIDs): {n_groups}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 6: Validate LMM Assumptions (Basic Checks)
    # -------------------------------------------------------------------------
    log("[STEP 6] Validate LMM Assumptions (Basic Checks)")
    log("-" * 70)

    # Get residuals
    residuals = result.resid

    # Residual mean (should be ~0)
    resid_mean = residuals.mean()
    log(f"[CHECK] Residual mean: {resid_mean:.6f} (should be ~0)")
    if abs(resid_mean) < 0.01:
        log(f"[PASS] Residual mean approximately zero")
    else:
        log(f"[WARNING] Residual mean not zero - may indicate model misspecification")

    # Residual variance
    resid_std = residuals.std()
    log(f"[INFO] Residual SD: {resid_std:.4f}")

    # Check for extreme residuals (potential outliers)
    n_extreme = (np.abs(residuals) > 3 * resid_std).sum()
    log(f"[CHECK] Observations with |residual| > 3 SD: {n_extreme}")
    if n_extreme < 0.01 * len(residuals):
        log(f"[PASS] Few extreme residuals (<1% of observations)")
    else:
        log(f"[WARNING] {n_extreme} extreme residuals - consider outlier analysis")

    log("")

    # -------------------------------------------------------------------------
    # STEP 7: Save Outputs
    # -------------------------------------------------------------------------
    log("[STEP 7] Save Outputs")
    log("-" * 70)

    # Save fixed effects to CSV
    fe_path = DATA_DIR / "step02_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False, encoding='utf-8')
    log(f"[SAVED] Fixed effects: {fe_path}")
    log(f"  {len(fixed_effects)} rows, {len(fixed_effects.columns)} columns")

    # Save model summary to text file
    summary_path = DATA_DIR / "step02_lmm_model_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ 5.4.3 - LMM Model Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model Formula:\n{formula}\n\n")
        f.write(f"Random Structure: {random_structure}\n")
        f.write(f"Converged: {converged}\n\n")
        f.write(f"Model Fit Statistics:\n")
        f.write(f"  Log-Likelihood: {result.llf:.2f}\n")
        f.write(f"  AIC: {result.aic:.2f}\n")
        f.write(f"  BIC: {result.bic:.2f}\n\n")
        f.write(f"Sample Size:\n")
        f.write(f"  Observations: {result.nobs}\n")
        f.write(f"  Groups (UIDs): {n_groups}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Fixed Effects\n")
        f.write("=" * 70 + "\n")
        f.write(str(result.summary().tables[1]))
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("Random Effects\n")
        f.write("=" * 70 + "\n")
        f.write(str(result.cov_re))
        f.write(f"\n\nResidual Variance: {result.scale:.4f}\n")
    log(f"[SAVED] Model summary: {summary_path}")

    # Save model object (pickle) - with workaround for patsy issue
    model_path = DATA_DIR / "step02_lmm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result, f)
    log(f"[SAVED] Model object: {model_path}")
    log("")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    log("[SUMMARY]")
    log("-" * 70)
    log(f"Model: 3-way Age x Congruence x Time LMM")
    log(f"Random Structure: {random_structure}")
    log(f"Converged: {converged}")
    log("")
    log(f"Fixed Effects: {len(fixed_effects)} terms")
    log(f"  - Main effects: Intercept, TSVR_hours, log_TSVR, Age_c, Congruent, Incongruent")
    log(f"  - 2-way interactions: Age_c x Time, Congruence x Time, Age_c x Congruence")
    log(f"  - 3-way interactions: Age_c x Congruence x Time (4 terms)")
    log("")
    log(f"Model Fit:")
    log(f"  Log-Likelihood: {result.llf:.2f}")
    log(f"  AIC: {result.aic:.2f}")
    log(f"  BIC: {result.bic:.2f}")
    log("")

    # Key hypothesis tests (3-way interactions)
    log("[KEY FINDINGS] 3-Way Interaction Terms:")
    for _, row in fixed_effects.iterrows():
        if 'Age_c' in row['term'] and ('Congruent' in row['term'] or 'Incongruent' in row['term']) and ('TSVR' in row['term'] or 'log_TSVR' in row['term']):
            sig = "SIGNIFICANT" if row['p'] < 0.05 else "not significant"
            log(f"  {row['term']}: p={row['p']:.4f} ({sig})")
    log("")
    log("[SUCCESS] Step 02 complete - LMM model fitted")

    return True

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        success = main()
        logger.close()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"[ERROR] Unexpected error: {e}")
        log(traceback.format_exc())
        logger.close()
        sys.exit(1)
