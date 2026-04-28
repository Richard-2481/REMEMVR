#!/usr/bin/env python3
"""Extract Individual Random Effects: Extract participant-specific random intercepts and slopes for descriptive"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import yaml
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_random_effects_extraction.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Extract Individual Random Effects")
        # Load Model Metadata from Step 1

        log("Loading model metadata from Step 1...")
        metadata_path = RQ_DIR / "data" / "step01_model_metadata.yaml"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}. Run step01 first.")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        log(f"Model metadata: {metadata_path.name}")
        log(f"  Model source: {metadata['model_source']}")
        log(f"  Model type: {metadata['model_type']}")
        log(f"  Participants: {metadata['n_participants']}")
        log(f"  Observations: {metadata['n_observations']}")
        log(f"  Converged: {metadata['converged']}")

        if not metadata['converged']:
            raise ValueError("RQ 5.1.1 model did not converge. Cannot extract random effects from failed model.")
        # Load LMM Model from RQ 5.1.1 Pickle

        log("Loading LMM model from RQ 5.1.1...")

        # Construct path to RQ 5.7 model pickle (now RQ 5.1.1 in new hierarchy)
        # CHANGED: Using Lin+Log model instead of Log (Log model has singular covariance, ΔAIC=0.8)
        rq7_model_path = RQ_DIR.parent / "5.1.1" / "data" / "lmm_Lin+Log.pkl"

        if not rq7_model_path.exists():
            raise FileNotFoundError(f"RQ 5.1.1 model not found: {rq7_model_path}. Run RQ 5.1.1 first.")

        # Statsmodels pickle workaround: manually bypass patsy formula re-evaluation
        log("Using statsmodels pickle workaround for patsy compatibility")

        lmm_model = None

        # Monkey-patch statsmodels data class to skip formula re-evaluation
        try:
            from statsmodels.base import data
            original_setstate = data.ModelData.__setstate__

            def patched_setstate(self, d):
                """Skip formula re-evaluation that causes patsy errors"""
                try:
                    original_setstate(self, d)
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'f_locals'" in str(e):
                        self.__dict__.update({k: v for k, v in d.items() if k != 'formula'})
                        log("Skipped patsy formula re-evaluation")
                    else:
                        raise

            # Apply patch
            data.ModelData.__setstate__ = patched_setstate

            # Load pickle
            with open(rq7_model_path, 'rb') as f:
                lmm_model = pickle.load(f)
                log("Model loaded successfully with patsy workaround")

            # Restore original
            data.ModelData.__setstate__ = original_setstate

        except Exception as e:
            log(f"Failed to load model: {str(e)}")
            import traceback
            log(traceback.format_exc())
            raise

        log(f"LMM model from {rq7_model_path}")
        log(f"  Model converged: {lmm_model.converged}")
        # Extract Random Effects for Each Participant

        log("Extracting random effects from model...")

        # Extract random effects dictionary from model
        # Format: {UID: DataFrame with columns [Group, Group x TSVR_hours]}
        random_effects_dict = lmm_model.random_effects

        log(f"  Found random effects for {len(random_effects_dict)} participants")

        # Extract fixed effects (population-average intercept and slope)
        fixed_intercept = lmm_model.fe_params['Intercept']

        # Get slope parameter name (could be 'log_TSVR' or similar)
        # Find the time variable parameter (not Intercept)
        slope_params = [p for p in lmm_model.fe_params.index if p != 'Intercept']
        if len(slope_params) == 0:
            raise ValueError("No slope parameter found in fixed effects (expected time variable)")

        # Use first non-intercept parameter as slope (should be only one for simple model)
        slope_param_name = slope_params[0]
        fixed_slope = lmm_model.fe_params[slope_param_name]

        log(f"  Fixed effects:")
        log(f"    Intercept: {fixed_intercept:.4f}")
        log(f"    Slope ({slope_param_name}): {fixed_slope:.4f}")

        # Initialize lists for DataFrame construction
        uids = []
        random_intercepts = []
        random_slopes = []
        total_intercepts = []
        total_slopes = []

        # Extract random effects for each participant
        for uid, re_df in random_effects_dict.items():
            # Random effects can be either Series or DataFrame
            # If Series: index = ['Group', 'Group x slope_var'], values = [intercept, slope]
            # If DataFrame: columns = same as Series index

            if isinstance(re_df, pd.Series):
                # Series format: index contains 'Group' and slope variable name
                random_intercept = re_df['Group']
                # Find slope - it's the other index value (not 'Group')
                slope_keys = [key for key in re_df.index if key != 'Group']
                if len(slope_keys) == 0:
                    log(f"UID {uid}: index = {re_df.index.tolist()}")
                    raise ValueError(f"No random slope found for {uid}")
                random_slope = re_df[slope_keys[0]]
            else:
                # DataFrame format (less common but possible)
                random_intercept = re_df['Group'].iloc[0]
                slope_cols = [col for col in re_df.columns if 'Group x' in col]
                if len(slope_cols) == 0:
                    raise ValueError(f"No random slope column found for {uid}")
                random_slope = re_df[slope_cols[0]].iloc[0]

            # Compute total effects (fixed + random)
            total_intercept = fixed_intercept + random_intercept
            total_slope = fixed_slope + random_slope

            # Append to lists
            uids.append(uid)
            random_intercepts.append(random_intercept)
            random_slopes.append(random_slope)
            total_intercepts.append(total_intercept)
            total_slopes.append(total_slope)

        # Create DataFrame
        df_random_effects = pd.DataFrame({
            'UID': uids,
            'random_intercept': random_intercepts,
            'random_slope': random_slopes,
            'total_intercept': total_intercepts,
            'total_slope': total_slopes
        })

        log(f"Random effects for {len(df_random_effects)} participants")
        log(f"  Columns: {list(df_random_effects.columns)}")
        # Compute Descriptive Statistics

        log("Computing descriptive statistics...")

        # Descriptive statistics for random slopes
        slope_mean = df_random_effects['random_slope'].mean()
        slope_std = df_random_effects['random_slope'].std()
        slope_min = df_random_effects['random_slope'].min()
        slope_max = df_random_effects['random_slope'].max()
        slope_q1 = df_random_effects['random_slope'].quantile(0.25)
        slope_median = df_random_effects['random_slope'].median()
        slope_q3 = df_random_effects['random_slope'].quantile(0.75)

        log(f"  Random slopes distribution:")
        log(f"    Mean: {slope_mean:.4f}")
        log(f"    SD: {slope_std:.4f}")
        log(f"    Min: {slope_min:.4f}")
        log(f"    Q1: {slope_q1:.4f}")
        log(f"    Median: {slope_median:.4f}")
        log(f"    Q3: {slope_q3:.4f}")
        log(f"    Max: {slope_max:.4f}")
        # Save Analysis Outputs
        # These outputs will be used by:
        #   - Step 5 (correlation test and visualization)
        #   - RQ 5.14 (K-means clustering analysis)

        log("Saving outputs...")

        # Output 1: Random effects CSV (CRITICAL for RQ 5.14)
        output_csv = RQ_DIR / "data" / "step04_random_effects.csv"
        df_random_effects.to_csv(output_csv, index=False, encoding='utf-8')
        log(f"{output_csv.name} ({len(df_random_effects)} rows, {len(df_random_effects.columns)} cols)")
        log(f"  CRITICAL: This file is REQUIRED INPUT for RQ 5.14 (K-means clustering)")

        # Output 2: Descriptive statistics TXT
        descriptives_txt = RQ_DIR / "results" / "step04_random_slopes_descriptives.txt"
        with open(descriptives_txt, 'w', encoding='utf-8') as f:
            f.write("Random Slopes Descriptive Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Sample size: {len(df_random_effects)} participants\n\n")
            f.write("Distribution:\n")
            f.write(f"  Mean:   {slope_mean:.4f}\n")
            f.write(f"  SD:     {slope_std:.4f}\n")
            f.write(f"  Min:    {slope_min:.4f}\n")
            f.write(f"  Q1:     {slope_q1:.4f}\n")
            f.write(f"  Median: {slope_median:.4f}\n")
            f.write(f"  Q3:     {slope_q3:.4f}\n")
            f.write(f"  Max:    {slope_max:.4f}\n\n")
            f.write("Interpretation:\n")
            f.write("  Negative slopes = faster forgetting than average\n")
            f.write("  Positive slopes = slower forgetting than average\n")
            f.write("  Zero slope = average forgetting rate\n")

        log(f"{descriptives_txt.name}")
        # Run Validation Tool
        # Validates: Required columns present, no missing data
        # Threshold: All 5 columns required, no NaN allowed

        log("Running validate_data_columns...")

        validation_result = validate_data_columns(
            df=df_random_effects,
            required_columns=['UID', 'random_intercept', 'random_slope', 'total_intercept', 'total_slope']
        )

        # Report validation results
        log(f"Validation result: {validation_result['valid']}")
        if validation_result['valid']:
            log(f"All required columns present ({validation_result['n_required']} columns)")
            log(f"No missing columns")
        else:
            log(f"FAILED: {validation_result['message']}")
            log(f"Missing columns: {validation_result['missing_columns']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation: row count
        if len(df_random_effects) != 100:
            log(f"WARNING: Expected 100 participants, found {len(df_random_effects)}")
            raise ValueError(f"Expected 100 participants, found {len(df_random_effects)}")
        else:
            log(f"Participant count correct: 100")

        # Additional validation: no NaN in critical columns
        nan_intercepts = df_random_effects['random_intercept'].isna().sum()
        nan_slopes = df_random_effects['random_slope'].isna().sum()

        if nan_intercepts > 0 or nan_slopes > 0:
            log(f"FAILED: NaN values detected")
            log(f"  NaN in random_intercept: {nan_intercepts}")
            log(f"  NaN in random_slope: {nan_slopes}")
            raise ValueError(f"NaN values in random effects: {nan_intercepts} intercepts, {nan_slopes} slopes")
        else:
            log(f"No NaN values in random effects")

        # Additional validation: no duplicate UIDs
        duplicates = df_random_effects['UID'].duplicated().sum()
        if duplicates > 0:
            log(f"FAILED: Duplicate UIDs detected ({duplicates} duplicates)")
            raise ValueError(f"Duplicate UIDs: {duplicates} participants appear multiple times")
        else:
            log(f"No duplicate UIDs")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
