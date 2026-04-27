#!/usr/bin/env python3
"""
RQ 5.3.7 Step 02: Fit Paradigm-Stratified LMMs with Random Slopes

PURPOSE:
Fit three separate LMM models (one per paradigm: free_recall, cued_recall, recognition)
with random slopes to extract paradigm-specific variance components for ICC computation.

INPUTS:
- data/step00_theta_scores_validated.csv (1200 rows x 8 columns)
  Columns: composite_ID, UID, test, TSVR_hours, TSVR_hours_sq, TSVR_hours_log, paradigm, theta
- data/step01_model_metadata.yaml (functional form = Log)

OUTPUTS:
- data/step02_variance_components.csv (15 rows: 5 components x 3 paradigms)
  Columns: paradigm, component, estimate
  Components: var_intercept, var_slope, cov_int_slope, corr_int_slope, var_residual
- data/step02_lmm_free_recall_model.pkl (statsmodels MixedLMResults object)
- data/step02_lmm_cued_recall_model.pkl
- data/step02_lmm_recognition_model.pkl
- data/step02_model_summaries.txt (concatenated summaries for all 3 models)

ANALYSIS:
For each paradigm (400 rows):
1. Filter data to paradigm subset
2. Create log_TSVR column: log(TSVR_hours + 1)
3. Fit LMM: theta ~ log_TSVR with random intercepts + slopes per UID
   Formula: theta ~ log_TSVR
   Groups: UID
   re_formula: ~log_TSVR (correlated random intercepts + slopes)
4. Extract variance components from fitted model
5. Compute correlation from covariance: corr = cov / sqrt(var_int * var_slope)
6. Save model object using .save() method (NOT pickle - prevents patsy errors)

CONVERGENCE CONTINGENCY:
If random slopes fail to converge:
1. Try alternative optimizers (default lbfgs -> bobyqa -> powell)
2. If all optimizers fail: Fall back to intercept-only model
3. Document convergence status in output
4. Set slope variance/covariance/correlation to NaN for failed models

VALIDATION:
- All variance components >= 0 (variances cannot be negative)
- Correlation in [-1, 1] range
- All 3 paradigms present in output
- Models saved successfully
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import traceback
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults

# Add project root to path (script is 4 levels deep: REMEMVR/results/ch5/5.3.7/code)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.7
LOG_FILE = RQ_DIR / "logs" / "step02_fit_paradigm_lmms.log"

# Paradigms to analyze (from step00 validation)
PARADIGMS = ['free_recall', 'cued_recall', 'recognition']

# =============================================================================
# Logging
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 02: Fit Paradigm-Stratified LMMs with Random Slopes")

        # =====================================================================
        # STEP 1: Load Data
        # =====================================================================
        log("[LOAD] Loading theta scores and model metadata...")

        # Load theta scores (1200 rows)
        theta_data = pd.read_csv(RQ_DIR / "data" / "step00_theta_scores_validated.csv")
        log(f"[LOADED] Theta scores: {len(theta_data)} rows, {len(theta_data.columns)} columns")
        log(f"[INFO] Paradigm counts: {theta_data['paradigm'].value_counts().to_dict()}")

        # Load model metadata to confirm functional form
        with open(RQ_DIR / "data" / "step01_model_metadata.yaml", 'r') as f:
            metadata = yaml.safe_load(f)
        log(f"[INFO] Best functional form from RQ 5.3.1: {metadata['functional_form']}")
        log(f"[INFO] Time transformation: {metadata['time_transformation']}")

        # Create log_TSVR column for all rows (NOT per paradigm - consistent across all)
        theta_data['log_TSVR'] = np.log(theta_data['TSVR_hours'] + 1)
        log(f"[CREATED] log_TSVR column: log(TSVR_hours + 1)")
        log(f"[INFO] log_TSVR range: [{theta_data['log_TSVR'].min():.3f}, {theta_data['log_TSVR'].max():.3f}]")

        # =====================================================================
        # STEP 2: Fit LMM for Each Paradigm
        # =====================================================================
        log("[ANALYSIS] Fitting LMM for each paradigm with random slopes...")

        variance_components = []
        model_summaries = []

        for paradigm in PARADIGMS:
            log(f"\n{'='*70}")
            log(f"[PARADIGM] {paradigm}")
            log(f"{'='*70}")

            # Filter to paradigm subset (400 rows)
            paradigm_data = theta_data[theta_data['paradigm'] == paradigm].copy()
            log(f"[FILTER] N = {len(paradigm_data)} observations for {paradigm}")
            log(f"[INFO] UIDs: {paradigm_data['UID'].nunique()} unique participants")

            # Fit LMM with random intercepts + slopes
            # Formula: theta ~ log_TSVR
            # Groups: UID
            # re_formula: ~log_TSVR (correlated random effects)
            log(f"[FIT] Attempting LMM with random slopes (correlated)...")

            convergence_status = "pending"
            model = None
            result = None

            # Try multiple optimizers
            optimizers = ['lbfgs', 'bfgs', 'powell']

            for optimizer in optimizers:
                try:
                    log(f"[TRY] Optimizer: {optimizer}")

                    # Fit model
                    model = MixedLM.from_formula(
                        formula='theta ~ log_TSVR',
                        data=paradigm_data,
                        groups=paradigm_data['UID'],
                        re_formula='~log_TSVR'  # Random intercepts + slopes (correlated)
                    )

                    result = model.fit(method=optimizer, reml=False)

                    # Check convergence
                    if result.converged:
                        log(f"[SUCCESS] Model converged with {optimizer}")
                        convergence_status = f"converged_{optimizer}"
                        break
                    else:
                        log(f"[WARNING] Model did not converge with {optimizer}")

                except Exception as e:
                    log(f"[ERROR] Optimizer {optimizer} failed: {str(e)}")
                    continue

            # If all optimizers failed, fall back to intercept-only model
            if result is None or not result.converged:
                log(f"[FALLBACK] All optimizers failed, trying intercept-only model...")

                try:
                    model = MixedLM.from_formula(
                        formula='theta ~ log_TSVR',
                        data=paradigm_data,
                        groups=paradigm_data['UID']
                        # No re_formula = random intercepts only
                    )

                    result = model.fit(reml=False)

                    if result.converged:
                        log(f"[FALLBACK SUCCESS] Intercept-only model converged")
                        convergence_status = "fallback_intercept_only"
                    else:
                        log(f"[FAILURE] Intercept-only model did not converge")
                        convergence_status = "failed"

                except Exception as e:
                    log(f"[FAILURE] Intercept-only model failed: {str(e)}")
                    convergence_status = "failed"
                    result = None

            # =====================================================================
            # STEP 3: Extract Variance Components
            # =====================================================================
            if result is not None and result.converged:
                log(f"[EXTRACT] Extracting variance components for {paradigm}...")

                # Get variance components from cov_re
                # cov_re is a DataFrame, convert to numpy array
                cov_re = result.cov_re.values  # Convert DataFrame to numpy array
                var_residual = result.scale  # Residual variance

                log(f"[DEBUG] cov_re shape: {cov_re.shape}")
                log(f"[DEBUG] cov_re:\n{cov_re}")

                if cov_re.shape == (2, 2):
                    # Random slopes model
                    var_intercept = cov_re[0, 0]
                    var_slope = cov_re[1, 1]
                    cov_int_slope = cov_re[0, 1]

                    # Compute correlation from covariance
                    if var_intercept > 0 and var_slope > 0:
                        corr_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope)
                    else:
                        corr_int_slope = np.nan

                    log(f"[VARIANCE] var_intercept = {var_intercept:.6f}")
                    log(f"[VARIANCE] var_slope = {var_slope:.6f}")
                    log(f"[COVARIANCE] cov_int_slope = {cov_int_slope:.6f}")
                    log(f"[CORRELATION] corr_int_slope = {corr_int_slope:.6f}")
                    log(f"[VARIANCE] var_residual = {var_residual:.6f}")

                elif cov_re.shape == (1, 1):
                    # Intercept-only model (fallback)
                    var_intercept = cov_re[0, 0]
                    var_slope = np.nan
                    cov_int_slope = np.nan
                    corr_int_slope = np.nan

                    log(f"[VARIANCE] var_intercept = {var_intercept:.6f}")
                    log(f"[VARIANCE] var_slope = NaN (intercept-only model)")
                    log(f"[COVARIANCE] cov_int_slope = NaN (intercept-only model)")
                    log(f"[CORRELATION] corr_int_slope = NaN (intercept-only model)")
                    log(f"[VARIANCE] var_residual = {var_residual:.6f}")

                else:
                    log(f"[ERROR] Unexpected cov_re shape: {cov_re.shape}")
                    var_intercept = np.nan
                    var_slope = np.nan
                    cov_int_slope = np.nan
                    corr_int_slope = np.nan

                # Append to variance components list
                variance_components.append({
                    'paradigm': paradigm,
                    'component': 'var_intercept',
                    'estimate': var_intercept
                })
                variance_components.append({
                    'paradigm': paradigm,
                    'component': 'var_slope',
                    'estimate': var_slope
                })
                variance_components.append({
                    'paradigm': paradigm,
                    'component': 'cov_int_slope',
                    'estimate': cov_int_slope
                })
                variance_components.append({
                    'paradigm': paradigm,
                    'component': 'corr_int_slope',
                    'estimate': corr_int_slope
                })
                variance_components.append({
                    'paradigm': paradigm,
                    'component': 'var_residual',
                    'estimate': var_residual
                })

                # =====================================================================
                # STEP 4: Save Model Object
                # =====================================================================
                log(f"[SAVE] Saving model for {paradigm}...")

                # Use .save() method (NOT pickle - prevents patsy errors per REMEMVR conventions)
                model_path = RQ_DIR / "data" / f"step02_lmm_{paradigm}_model.pkl"
                result.save(str(model_path))
                log(f"[SAVED] {model_path.name}")

                # =====================================================================
                # STEP 5: Collect Model Summary
                # =====================================================================
                summary_text = f"\n{'='*70}\n"
                summary_text += f"PARADIGM: {paradigm}\n"
                summary_text += f"{'='*70}\n"
                summary_text += f"Convergence Status: {convergence_status}\n"
                summary_text += f"AIC: {result.aic:.2f}\n"
                summary_text += f"BIC: {result.bic:.2f}\n"
                summary_text += f"Log-Likelihood: {result.llf:.2f}\n"
                summary_text += f"\nVariance Components:\n"
                summary_text += f"  var_intercept:  {var_intercept:.6f}\n"
                summary_text += f"  var_slope:      {var_slope:.6f}\n"
                summary_text += f"  cov_int_slope:  {cov_int_slope:.6f}\n"
                summary_text += f"  corr_int_slope: {corr_int_slope:.6f}\n"
                summary_text += f"  var_residual:   {var_residual:.6f}\n"
                summary_text += f"\nFixed Effects:\n"
                summary_text += str(result.summary().tables[1])
                summary_text += "\n"
                summary_text += "\n"

                model_summaries.append(summary_text)

            else:
                log(f"[ERROR] Model did not converge for {paradigm}")
                log(f"[SKIP] Cannot extract variance components for failed model")

                # Append NaN for all components
                for component in ['var_intercept', 'var_slope', 'cov_int_slope', 'corr_int_slope', 'var_residual']:
                    variance_components.append({
                        'paradigm': paradigm,
                        'component': component,
                        'estimate': np.nan
                    })

                # Add summary note
                summary_text = f"\n{'='*70}\n"
                summary_text += f"PARADIGM: {paradigm}\n"
                summary_text += f"{'='*70}\n"
                summary_text += f"Convergence Status: {convergence_status}\n"
                summary_text += f"ERROR: Model failed to converge\n"
                summary_text += "\n"

                model_summaries.append(summary_text)

        # =====================================================================
        # STEP 6: Save Variance Components
        # =====================================================================
        log(f"\n[SAVE] Saving variance components...")

        df_variance = pd.DataFrame(variance_components)
        output_path = RQ_DIR / "data" / "step02_variance_components.csv"
        df_variance.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {output_path.name} ({len(df_variance)} rows)")
        log(f"[INFO] Components per paradigm: {df_variance.groupby('paradigm').size().to_dict()}")

        # =====================================================================
        # STEP 7: Save Model Summaries
        # =====================================================================
        log(f"[SAVE] Saving model summaries...")

        summaries_path = RQ_DIR / "data" / "step02_model_summaries.txt"
        with open(summaries_path, 'w', encoding='utf-8') as f:
            f.write("RQ 5.3.7 Step 02: Paradigm-Stratified LMM Model Summaries\n")
            f.write("="*70 + "\n\n")
            for summary in model_summaries:
                f.write(summary)
        log(f"[SAVED] {summaries_path.name}")

        # =====================================================================
        # STEP 8: Final Validation
        # =====================================================================
        log(f"\n[VALIDATION] Checking variance components...")

        # Check all paradigms present
        paradigms_found = df_variance['paradigm'].unique()
        if len(paradigms_found) == 3:
            log(f"[PASS] All 3 paradigms present: {list(paradigms_found)}")
        else:
            log(f"[WARNING] Missing paradigms: expected 3, found {len(paradigms_found)}")

        # Check variance components (excluding NaN for failed models)
        for paradigm in PARADIGMS:
            paradigm_vars = df_variance[df_variance['paradigm'] == paradigm]

            var_int = paradigm_vars[paradigm_vars['component'] == 'var_intercept']['estimate'].values[0]
            var_slope = paradigm_vars[paradigm_vars['component'] == 'var_slope']['estimate'].values[0]
            var_resid = paradigm_vars[paradigm_vars['component'] == 'var_residual']['estimate'].values[0]
            corr = paradigm_vars[paradigm_vars['component'] == 'corr_int_slope']['estimate'].values[0]

            # Check if model converged (non-NaN values)
            if not np.isnan(var_int):
                # Variance components must be non-negative
                if var_int >= 0 and var_resid >= 0:
                    log(f"[PASS] {paradigm}: All variances non-negative")
                else:
                    log(f"[FAIL] {paradigm}: Negative variance detected")

                # Correlation must be in [-1, 1] (if not NaN)
                if not np.isnan(corr):
                    if -1 <= corr <= 1:
                        log(f"[PASS] {paradigm}: Correlation in valid range [{corr:.3f}]")
                    else:
                        log(f"[FAIL] {paradigm}: Correlation out of range [{corr:.3f}]")
                else:
                    log(f"[INFO] {paradigm}: Correlation is NaN (intercept-only model)")
            else:
                log(f"[WARNING] {paradigm}: Model did not converge (all components NaN)")

        log(f"\n[SUCCESS] Step 02 complete")
        log(f"[INFO] Variance components saved: {output_path}")
        log(f"[INFO] Model summaries saved: {summaries_path}")
        log(f"[INFO] Models saved: 3 .pkl files in data/")

        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
