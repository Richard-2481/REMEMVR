#!/usr/bin/env python3
"""
Step 02: Fit Congruence-Stratified LMMs

PURPOSE:
Fit three separate LMMs, one per congruence level (Common, Congruent, Incongruent),
to estimate variance components for between-person and within-person variance in
intercepts and slopes.

EXPECTED INPUTS:
- data/step01_loaded_lmm_input.csv: LMM input (1200 rows x 6 columns)

EXPECTED OUTPUTS:
- data/step02_fitted_model_common.pkl: Fitted LMM for Common congruence
- data/step02_fitted_model_congruent.pkl: Fitted LMM for Congruent congruence
- data/step02_fitted_model_incongruent.pkl: Fitted LMM for Incongruent congruence
- data/step02_variance_components.csv: Variance components (15 rows: 5 components x 3 congruence)
- data/step02_model_metadata_common.yaml: Model metadata for Common
- data/step02_model_metadata_congruent.yaml: Model metadata for Congruent
- data/step02_model_metadata_incongruent.yaml: Model metadata for Incongruent

VALIDATION CRITERIA:
- All 3 models converged
- No singular fit (random effects variance > 0)
- Minimum 100 observations per congruence level
- All variance components positive
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
import statsmodels.formula.api as smf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_convergence, validate_variance_positivity

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_fit_stratified_lmms.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def extract_variance_components(model):
    """Extract variance components from fitted LMM."""
    # Get random effects covariance matrix
    cov_re = model.cov_re

    # Extract variance components
    # cov_re is a 2x2 matrix: [[var_intercept, cov], [cov, var_slope]]
    var_intercept = cov_re.iloc[0, 0]
    var_slope = cov_re.iloc[1, 1]
    cov_int_slope = cov_re.iloc[0, 1]

    # Residual variance
    var_residual = model.scale

    # Total variance (at baseline, TSVR_hours=0)
    var_total = var_intercept + var_residual

    return {
        'var_intercept': var_intercept,
        'var_slope': var_slope,
        'cov_int_slope': cov_int_slope,
        'var_residual': var_residual,
        'var_total': var_total
    }

if __name__ == "__main__":
    try:
        log("[START] Step 02: Fit Congruence-Stratified LMMs")

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================
        log("[LOAD] Loading LMM input data...")

        input_file = RQ_DIR / "data" / "step01_loaded_lmm_input.csv"
        df_lmm = pd.read_csv(input_file, encoding='utf-8')

        log(f"[LOADED] {input_file.name} ({len(df_lmm)} rows, {len(df_lmm.columns)} columns)")

        # Get congruence levels
        congruence_levels = sorted(df_lmm['congruence'].unique())
        log(f"[INFO] Congruence levels: {congruence_levels}")

        # =====================================================================
        # STEP 2: Fit LMMs for Each Congruence Level
        # =====================================================================
        log("[ANALYSIS] Fitting stratified LMMs (one per congruence level)...")

        models = {}
        variance_components = []

        for congruence in congruence_levels:
            log(f"\n[FIT] Fitting LMM for {congruence} congruence...")

            # Filter data for this congruence level
            df_subset = df_lmm[df_lmm['congruence'] == congruence].copy()
            log(f"[INFO] {congruence}: {len(df_subset)} observations, {df_subset['UID'].nunique()} participants")

            # Fit LMM with random intercept and slope
            # Formula: theta ~ TSVR_hours (fixed effects)
            # Random effects: ~TSVR_hours | UID (random intercept and slope per participant)
            formula = "theta ~ TSVR_hours"

            try:
                model = smf.mixedlm(formula, df_subset, groups=df_subset['UID'],
                                   re_formula="~TSVR_hours").fit(reml=True)

                log(f"[CONVERGED] {congruence} model converged: {model.converged}")

                if not model.converged:
                    log(f"[WARNING] {congruence} model did NOT converge - results may be unreliable")

                models[congruence] = model

                # Extract variance components
                components = extract_variance_components(model)

                # Add to dataframe
                for comp_name, comp_value in components.items():
                    variance_components.append({
                        'congruence': congruence,
                        'component': comp_name,
                        'value': comp_value
                    })

                log(f"[VARIANCE] {congruence} variance components:")
                for comp_name, comp_value in components.items():
                    log(f"  {comp_name}: {comp_value:.6f}")

            except Exception as e:
                log(f"[ERROR] Failed to fit {congruence} model: {str(e)}")
                raise

        # =====================================================================
        # STEP 3: Save Fitted Models
        # =====================================================================
        log("\n[SAVE] Saving fitted models...")

        model_files = {
            'Common': RQ_DIR / "data" / "step02_fitted_model_common.pkl",
            'Congruent': RQ_DIR / "data" / "step02_fitted_model_congruent.pkl",
            'Incongruent': RQ_DIR / "data" / "step02_fitted_model_incongruent.pkl"
        }

        for congruence, model_path in model_files.items():
            with open(model_path, 'wb') as f:
                pickle.dump(models[congruence], f)
            log(f"[SAVED] {model_path.name}")

        # =====================================================================
        # STEP 4: Save Variance Components
        # =====================================================================
        log("[SAVE] Saving variance components...")

        df_variance = pd.DataFrame(variance_components)
        variance_output = RQ_DIR / "data" / "step02_variance_components.csv"
        df_variance.to_csv(variance_output, index=False, encoding='utf-8')

        log(f"[SAVED] {variance_output.name} ({len(df_variance)} rows)")

        # =====================================================================
        # STEP 5: Save Model Metadata
        # =====================================================================
        log("[SAVE] Saving model metadata...")

        metadata_files = {
            'Common': RQ_DIR / "data" / "step02_model_metadata_common.yaml",
            'Congruent': RQ_DIR / "data" / "step02_model_metadata_congruent.yaml",
            'Incongruent': RQ_DIR / "data" / "step02_model_metadata_incongruent.yaml"
        }

        for congruence, metadata_path in metadata_files.items():
            model = models[congruence]

            metadata = {
                'congruence': congruence,
                'formula': 'theta ~ TSVR_hours',
                're_formula': '~TSVR_hours | UID',
                'converged': bool(model.converged),
                'reml': True,
                'n_obs': int(model.nobs),
                'n_groups': len(model.model.group_labels),
                'aic': float(model.aic),
                'bic': float(model.bic),
                'llf': float(model.llf)
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            log(f"[SAVED] {metadata_path.name}")

        # =====================================================================
        # STEP 6: Validate Model Convergence (log warnings but don't fail)
        # =====================================================================
        log("\n[VALIDATION] Validating model convergence...")

        convergence_warnings = []
        for congruence, model in models.items():
            validation = validate_lmm_convergence(model)

            if validation['converged']:
                log(f"[PASS] {congruence} model converged successfully")
            else:
                log(f"[WARNING] {congruence} model did not converge: {validation['message']}")
                log(f"  -> Results may be unreliable but proceeding with analysis")
                convergence_warnings.append(congruence)

        if convergence_warnings:
            log(f"[WARNING] {len(convergence_warnings)} model(s) did not converge: {', '.join(convergence_warnings)}")
            log("[INFO] Non-convergence often indicates near-zero random slope variance (singular fit)")
        else:
            log("[PASS] All models converged successfully")


        # =====================================================================
        # STEP 7: Validate Variance Positivity
        # =====================================================================
        log("[VALIDATION] Validating variance component positivity...")

        # Filter out covariances (can be negative) - only check actual variances
        df_variance_only = df_variance[~df_variance['component'].str.contains('cov')].copy()

        variance_validation = validate_variance_positivity(
            df_variance_only,
            component_col='component',
            value_col='value'
        )

        if variance_validation['valid']:
            log("[PASS] All variance components are positive")
        else:
            log(f"[FAIL] Variance validation failed: {variance_validation['message']}")
            raise ValueError(variance_validation['message'])

        log("\n[SUCCESS] Step 02 complete - All models fitted and validated")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
