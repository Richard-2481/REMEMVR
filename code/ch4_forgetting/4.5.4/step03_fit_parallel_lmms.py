#!/usr/bin/env python3
"""Fit Parallel LMMs (IRT-based and CTT-based): Fit identical Linear Mixed Models to IRT theta scores and CTT mean scores for"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM fitting
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

from tools.validation import validate_model_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_parallel_lmms.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Fit Parallel LMMs (IRT-based and CTT-based)")
        # Load Input Data

        log("Loading IRT theta scores...")
        irt_data = pd.read_csv(RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv")
        log(f"step00_irt_theta_from_rq551.csv ({len(irt_data)} rows, {len(irt_data.columns)} cols)")
        log(f"IRT theta range: [{irt_data['irt_theta'].min():.2f}, {irt_data['irt_theta'].max():.2f}]")

        log("Loading CTT mean scores...")
        ctt_data = pd.read_csv(RQ_DIR / "data" / "step01_ctt_scores.csv")
        log(f"step01_ctt_scores.csv ({len(ctt_data)} rows, {len(ctt_data.columns)} cols)")
        log(f"CTT score range: [{ctt_data['ctt_mean_score'].min():.3f}, {ctt_data['ctt_mean_score'].max():.3f}]")
        # Prepare Data for LMM Fitting
        # Create log_TSVR transformation for nonlinear time effects
        # Note: TSVR already in hours (0-168), add 1 to avoid log(0)

        log("Creating log_TSVR transformation...")
        irt_data['log_TSVR'] = np.log(irt_data['TSVR_hours'] + 1)
        ctt_data['log_TSVR'] = np.log(ctt_data['TSVR_hours'] + 1)
        log(f"log_TSVR range (IRT): [{irt_data['log_TSVR'].min():.3f}, {irt_data['log_TSVR'].max():.3f}]")
        log(f"log_TSVR range (CTT): [{ctt_data['log_TSVR'].min():.3f}, {ctt_data['log_TSVR'].max():.3f}]")

        # Verify data integrity
        log("Checking data integrity...")
        assert len(irt_data) == 800, f"Expected 800 IRT rows, got {len(irt_data)}"
        assert len(ctt_data) == 800, f"Expected 800 CTT rows, got {len(ctt_data)}"
        assert irt_data['irt_theta'].notna().all(), "Missing values in irt_theta"
        assert ctt_data['ctt_mean_score'].notna().all(), "Missing values in ctt_mean_score"
        assert set(irt_data['location_type'].unique()) == {'source', 'destination'}, "Invalid location types (IRT)"
        assert set(ctt_data['location_type'].unique()) == {'source', 'destination'}, "Invalid location types (CTT)"
        log("Data integrity checks passed")
        # Fit IRT-Based LMM (Full Random Structure)
        # Model: irt_theta ~ LocationType * log_TSVR
        # Random: ~log_TSVR | UID (random intercepts + slopes)
        # REML: False (for AIC/BIC comparison in Step 6)

        log("Fitting IRT-based LMM with full random structure...")
        log("Formula: irt_theta ~ C(location_type, Treatment('source')) * log_TSVR")
        log("Random: ~log_TSVR (random intercepts + slopes for TSVR)")
        log("Groups: UID")
        log("REML: False (for AIC comparison)")

        try:
            irt_model_full = smf.mixedlm(
                formula="irt_theta ~ C(location_type, Treatment('source')) * log_TSVR",
                data=irt_data,
                groups=irt_data['UID'],
                re_formula="~log_TSVR"
            )
            irt_result_full = irt_model_full.fit(reml=False)
            irt_converged = irt_result_full.converged
            log(f"IRT model (full) convergence: {irt_converged}")
        except Exception as e:
            log(f"IRT model (full) fitting failed: {str(e)}")
            irt_converged = False
            irt_result_full = None
        # Fit CTT-Based LMM (Full Random Structure)
        # Model: ctt_mean_score ~ LocationType * log_TSVR
        # Random: ~log_TSVR | UID (random intercepts + slopes)
        # REML: False (for AIC/BIC comparison in Step 6)

        log("Fitting CTT-based LMM with full random structure...")
        log("Formula: ctt_mean_score ~ C(location_type, Treatment('source')) * log_TSVR")
        log("Random: ~log_TSVR (random intercepts + slopes for TSVR)")
        log("Groups: UID")
        log("REML: False (for AIC comparison)")

        try:
            ctt_model_full = smf.mixedlm(
                formula="ctt_mean_score ~ C(location_type, Treatment('source')) * log_TSVR",
                data=ctt_data,
                groups=ctt_data['UID'],
                re_formula="~log_TSVR"
            )
            ctt_result_full = ctt_model_full.fit(reml=False)
            ctt_converged = ctt_result_full.converged
            log(f"CTT model (full) convergence: {ctt_converged}")
        except Exception as e:
            log(f"CTT model (full) fitting failed: {str(e)}")
            ctt_converged = False
            ctt_result_full = None
        # Handle Convergence Failures (Symmetric Simplification)
        # CRITICAL: If EITHER model fails, simplify BOTH to ~1|UID (random intercepts only)

        simplified = False

        if not irt_converged or not ctt_converged:
            log("At least one model failed to converge with full random structure")
            log("Simplifying BOTH models to ~1|UID (random intercepts only)")
            simplified = True

            # Fit IRT model with simplified random structure
            log("Re-fitting IRT-based LMM with simplified random structure...")
            log("Random: ~1 (random intercepts only)")

            irt_model_simple = smf.mixedlm(
                formula="irt_theta ~ C(location_type, Treatment('source')) * log_TSVR",
                data=irt_data,
                groups=irt_data['UID']
                # re_formula defaults to "~1" (random intercepts only)
            )
            irt_result = irt_model_simple.fit(reml=False)
            log(f"IRT model (simplified) convergence: {irt_result.converged}")

            # Fit CTT model with simplified random structure
            log("Re-fitting CTT-based LMM with simplified random structure...")
            log("Random: ~1 (random intercepts only)")

            ctt_model_simple = smf.mixedlm(
                formula="ctt_mean_score ~ C(location_type, Treatment('source')) * log_TSVR",
                data=ctt_data,
                groups=ctt_data['UID']
                # re_formula defaults to "~1" (random intercepts only)
            )
            ctt_result = ctt_model_simple.fit(reml=False)
            log(f"CTT model (simplified) convergence: {ctt_result.converged}")

            if not irt_result.converged or not ctt_result.converged:
                raise ValueError("Models failed to converge even with simplified random structure")

        else:
            # Both models converged with full random structure
            log("Both models converged with full random structure")
            irt_result = irt_result_full
            ctt_result = ctt_result_full
        # Extract Fixed Effects Tables for Step 5 Comparison
        # Save fixed effects as CSV so Step 5 doesn't need to unpickle models

        log("Extracting fixed effects from IRT model...")
        irt_fe = irt_result.fe_params
        irt_se = irt_result.bse_fe
        # Note: tvalues and pvalues include random effects, need to slice to fixed effects only
        n_fe = len(irt_fe)
        irt_z = irt_result.tvalues.iloc[:n_fe]
        irt_p = irt_result.pvalues.iloc[:n_fe]
        irt_ci = irt_result.conf_int().iloc[:n_fe]

        irt_coef_df = pd.DataFrame({
            'term': irt_fe.index,
            'coef': irt_fe.values,
            'std_err': irt_se.values,
            'z': irt_z.values,
            'p_value': irt_p.values,
            'ci_lower': irt_ci.iloc[:, 0].values,
            'ci_upper': irt_ci.iloc[:, 1].values
        })
        log(f"IRT model: {len(irt_coef_df)} fixed effects")

        log("Extracting fixed effects from CTT model...")
        ctt_fe = ctt_result.fe_params
        ctt_se = ctt_result.bse_fe
        # Note: tvalues and pvalues include random effects, need to slice to fixed effects only
        n_fe_ctt = len(ctt_fe)
        ctt_z = ctt_result.tvalues.iloc[:n_fe_ctt]
        ctt_p = ctt_result.pvalues.iloc[:n_fe_ctt]
        ctt_ci = ctt_result.conf_int().iloc[:n_fe_ctt]

        ctt_coef_df = pd.DataFrame({
            'term': ctt_fe.index,
            'coef': ctt_fe.values,
            'std_err': ctt_se.values,
            'z': ctt_z.values,
            'p_value': ctt_p.values,
            'ci_lower': ctt_ci.iloc[:, 0].values,
            'ci_upper': ctt_ci.iloc[:, 1].values
        })
        log(f"CTT model: {len(ctt_coef_df)} fixed effects")
        # Save Model Outputs

        log("Saving IRT model...")
        with open(RQ_DIR / "data" / "step03_irt_lmm_model.pkl", 'wb') as f:
            pickle.dump(irt_result, f)
        log("data/step03_irt_lmm_model.pkl")

        log("Saving CTT model...")
        with open(RQ_DIR / "data" / "step03_ctt_lmm_model.pkl", 'wb') as f:
            pickle.dump(ctt_result, f)
        log("data/step03_ctt_lmm_model.pkl")

        log("Saving IRT model summary...")
        with open(RQ_DIR / "data" / "step03_irt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(irt_result.summary()))
        log("data/step03_irt_lmm_summary.txt")

        log("Saving CTT model summary...")
        with open(RQ_DIR / "data" / "step03_ctt_lmm_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(ctt_result.summary()))
        log("data/step03_ctt_lmm_summary.txt")

        log("Saving IRT fixed effects table...")
        irt_coef_df.to_csv(RQ_DIR / "data" / "step03_irt_coefficients.csv", index=False, encoding='utf-8')
        log(f"data/step03_irt_coefficients.csv ({len(irt_coef_df)} rows)")

        log("Saving CTT fixed effects table...")
        ctt_coef_df.to_csv(RQ_DIR / "data" / "step03_ctt_coefficients.csv", index=False, encoding='utf-8')
        log(f"data/step03_ctt_coefficients.csv ({len(ctt_coef_df)} rows)")

        # Create model metadata
        log("Creating model metadata...")
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'formula': "score ~ C(location_type, Treatment('source')) * log_TSVR",
            'random_structure': '~1 (intercepts only)' if simplified else '~log_TSVR (intercepts + slopes)',
            'simplified': simplified,
            'reml': False,
            'irt_model': {
                'converged': bool(irt_result.converged),
                'aic': float(irt_result.aic),
                'bic': float(irt_result.bic),
                'n_obs': int(irt_result.nobs),
                'n_groups': len(irt_result.model.group_labels)
            },
            'ctt_model': {
                'converged': bool(ctt_result.converged),
                'aic': float(ctt_result.aic),
                'bic': float(ctt_result.bic),
                'n_obs': int(ctt_result.nobs),
                'n_groups': len(ctt_result.model.group_labels)
            }
        }

        with open(RQ_DIR / "data" / "step03_model_metadata.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        log("data/step03_model_metadata.yaml")
        # Run Validation
        # Validate: Both models converged, AIC/BIC finite, 4 fixed effects

        log("Running validation checks...")

        # Validate IRT model convergence
        irt_validation = validate_model_convergence(irt_result)
        if not irt_validation['valid']:
            raise ValueError(f"IRT model validation failed: {irt_validation['message']}")
        log(f"IRT model convergence: {irt_validation['message']}")

        # Validate CTT model convergence
        ctt_validation = validate_model_convergence(ctt_result)
        if not ctt_validation['valid']:
            raise ValueError(f"CTT model validation failed: {ctt_validation['message']}")
        log(f"CTT model convergence: {ctt_validation['message']}")

        # Validate AIC/BIC are finite
        assert np.isfinite(irt_result.aic), f"IRT model AIC is not finite: {irt_result.aic}"
        assert np.isfinite(irt_result.bic), f"IRT model BIC is not finite: {irt_result.bic}"
        assert np.isfinite(ctt_result.aic), f"CTT model AIC is not finite: {ctt_result.aic}"
        assert np.isfinite(ctt_result.bic), f"CTT model BIC is not finite: {ctt_result.bic}"
        log(f"IRT model AIC={irt_result.aic:.2f}, BIC={irt_result.bic:.2f}")
        log(f"CTT model AIC={ctt_result.aic:.2f}, BIC={ctt_result.bic:.2f}")

        # Validate 4 fixed effects (Intercept, LocationType, log_TSVR, interaction)
        assert len(irt_coef_df) == 4, f"Expected 4 IRT fixed effects, got {len(irt_coef_df)}"
        assert len(ctt_coef_df) == 4, f"Expected 4 CTT fixed effects, got {len(ctt_coef_df)}"
        log("Both models have 4 fixed effects (Intercept, LocationType, log_TSVR, interaction)")

        # Report validation summary
        log("All validation checks passed:")
        log("  IRT model converged")
        log("  CTT model converged")
        log("  Both models have identical random structure (symmetric)")
        log("  AIC and BIC are finite for both models")
        log("  Both models have 4 fixed effects")

        log("Step 3 complete")
        log(f"Random structure: {'Simplified (~1)' if simplified else 'Full (~log_TSVR)'}")
        log("Next: Run step04_validate_lmm_assumptions.py")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
