#!/usr/bin/env python3
"""extract_variance_components: Extract variance-covariance matrix from random effects and residual variance"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import pickle
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_variance_positivity

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_variance_extraction.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract Variance Components from LMM")
        # Load Model Object from Step 1

        log("Loading model metadata from Step 1...")

        # Read metadata to get model source path
        metadata_path = RQ_DIR / "data" / "step01_model_metadata.yaml"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        log(f"Model metadata: {metadata_path}")
        log(f"  Model source: {metadata['model_source']}")
        log(f"  Model type: {metadata['model_type']}")
        log(f"  Converged: {metadata['converged']}")
        log(f"  Participants: {metadata['n_participants']}")
        log(f"  Observations: {metadata['n_observations']}")

        # Load model object from pickle
        model_path = RQ_DIR.parent.parent / metadata['model_source'].replace('results/', '')
        log(f"Loading model object from {model_path}...")

        # Statsmodels pickle workaround: manually bypass patsy formula re-evaluation
        # The model object is valid for variance extraction even if formula fails to re-evaluate
        log("Using statsmodels pickle workaround for patsy compatibility")

        lmm_model = None
        model_loaded = False

        # Monkey-patch statsmodels data class to skip formula re-evaluation
        try:
            from statsmodels.base import data
            original_setstate = data.ModelData.__setstate__

            def patched_setstate(self, d):
                """Skip formula re-evaluation that causes patsy errors"""
                try:
                    # Try normal __setstate__
                    original_setstate(self, d)
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'f_locals'" in str(e):
                        # Expected error - skip formula re-evaluation
                        # Manually set attributes we need (frame, orig_endog, orig_exog)
                        self.__dict__.update({k: v for k, v in d.items() if k != 'formula'})
                        log("Skipped patsy formula re-evaluation (not needed for variance extraction)")
                    else:
                        raise

            # Apply patch
            data.ModelData.__setstate__ = patched_setstate

            # Now load pickle
            with open(model_path, 'rb') as f:
                lmm_model = pickle.load(f)
                model_loaded = True
                log("Model loaded successfully with patsy workaround")

            # Restore original __setstate__
            data.ModelData.__setstate__ = original_setstate

        except Exception as e:
            log(f"Failed to load model even with patsy workaround: {str(e)}")
            log("Cannot proceed without model object")
            import traceback
            log(traceback.format_exc())
            sys.exit(1)

        log(f"Model object successfully loaded")
        log(f"  Type: {type(lmm_model).__name__}")
        # Extract Variance Components
        #               and residual variance from model fit

        log("Extracting variance components from LMM...")

        # Extract random effects covariance matrix
        # cov_re is a DataFrame with random effects covariance (intercept, slope)
        cov_re = lmm_model.cov_re
        log(f"Random effects covariance matrix shape: {cov_re.shape}")
        log(f"Covariance matrix:\n{cov_re}")

        # Extract variance components from diagonal and off-diagonal elements
        # Diagonal [0,0] = variance of random intercepts
        # Diagonal [1,1] = variance of random slopes
        # Off-diagonal [0,1] = covariance between intercepts and slopes
        var_intercept = cov_re.iloc[0, 0]
        var_slope = cov_re.iloc[1, 1]
        cov_int_slope = cov_re.iloc[0, 1]

        log(f"var_intercept (baseline variance): {var_intercept:.6f}")
        log(f"var_slope (forgetting rate variance): {var_slope:.6f}")
        log(f"cov_int_slope (intercept-slope covariance): {cov_int_slope:.6f}")

        # Extract residual variance
        # scale attribute contains residual variance (within-person error)
        var_residual = lmm_model.scale
        log(f"var_residual (within-person variance): {var_residual:.6f}")

        # Compute correlation from covariance
        # Formula: cor = cov / sqrt(var1 * var2)
        # This standardizes covariance to [-1, 1] range for interpretability
        cor_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope)
        log(f"cor_int_slope (standardized correlation): {cor_int_slope:.6f}")
        log(f"  Formula: cov_int_slope / sqrt(var_intercept * var_slope)")
        log(f"  = {cov_int_slope:.6f} / sqrt({var_intercept:.6f} * {var_slope:.6f})")
        log(f"  = {cov_int_slope:.6f} / {np.sqrt(var_intercept * var_slope):.6f}")

        log("Variance components extracted")
        # Save Variance Components to CSV
        # Output: data/step02_variance_components.csv
        # Contains: 5 rows (one per variance component)
        # Columns: component (string), estimate (float)

        log("Creating variance components DataFrame...")

        # Create DataFrame with component names and estimates
        variance_components = pd.DataFrame({
            'component': [
                'var_intercept',
                'var_slope',
                'cov_int_slope',
                'var_residual',
                'cor_int_slope'
            ],
            'estimate': [
                var_intercept,
                var_slope,
                cov_int_slope,
                var_residual,
                cor_int_slope
            ]
        })

        log(f"Variance components DataFrame ({len(variance_components)} rows)")
        log(f"\n{variance_components.to_string(index=False)}")

        # Save to CSV
        output_path = RQ_DIR / "data" / "step02_variance_components.csv"
        variance_components.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(variance_components)} rows, {len(variance_components.columns)} cols)")
        # Run Validation Tool
        # Validates: All variance components > 0, correlation in [-1, 1]
        # Threshold: Variance > 0 (strict positivity), |correlation| <= 1

        log("Running validate_variance_positivity...")

        # Filter to only variance components (exclude covariance and correlation)
        # Covariance and correlation can be negative, so we only validate variances
        variance_only = variance_components[
            variance_components['component'].str.contains('var_')
        ].copy()

        log(f"Checking {len(variance_only)} variance components (excluding cov/cor)")

        validation_result = validate_variance_positivity(
            variance_df=variance_only,
            component_col='component',
            value_col='estimate'
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - All variance components valid")
            log(f"  Message: {validation_result['message']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            if 'negative_components' in validation_result and validation_result['negative_components']:
                log(f"  Negative/zero components: {validation_result['negative_components']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation for correlation bounds (not in validate_variance_positivity)
        if abs(cor_int_slope) > 1.0:
            log(f"FAIL - Correlation out of bounds: {cor_int_slope}")
            raise ValueError(f"Correlation out of bounds: cor_int_slope = {cor_int_slope}, expected in [-1, 1]")
        else:
            log(f"PASS - Correlation in bounds: {cor_int_slope} in [-1, 1]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
