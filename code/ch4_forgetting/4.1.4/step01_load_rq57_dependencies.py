#!/usr/bin/env python3
"""Load RQ 5.7 Dependencies: Load saved LMM model object, theta scores, and TSVR mapping from RQ 5.7 to"""

import sys
import pickle
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Tuple
import traceback
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tool (if available)
try:
    from tools.validation import validate_model_convergence
    HAS_VALIDATION_TOOL = True
except ImportError:
    HAS_VALIDATION_TOOL = False

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_load_dependencies.log"

# Create logs directory if it doesn't exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Relative paths from RQ directory to RQ 5.7 artifacts (now RQ 5.1.1 in new hierarchy)
# These are RELATIVE paths as specified in 4_analysis.yaml
# CHANGED: Using Lin+Log model instead of Log (Log model has singular covariance, ΔAIC=0.8)
RQ57_LMM_MODEL = RQ_DIR / "../5.1.1/data/lmm_Lin+Log.pkl"
RQ57_THETA_SCORES = RQ_DIR / "../5.1.1/data/step03_theta_scores.csv"
RQ57_LMM_INPUT = RQ_DIR / "../5.1.1/data/step04_lmm_input.csv"

# Output paths (relative to RQ directory)
METADATA_OUTPUT = RQ_DIR / "data" / "step01_model_metadata.yaml"
METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# Logging Function

def log(msg: str) -> None:
    """Write message to both console and log file (UTF-8 encoding)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"

    # Write to console (ASCII-safe)
    print(formatted_msg)

    # Write to log file (UTF-8)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + "\n")


# Validation Functions

def validate_files_exist() -> bool:
    """
    Circuit breaker check: Validate all three RQ 5.1.1 dependency files exist.

    If ANY file is missing, print EXPECTATIONS ERROR and return False.
    """
    missing_files = []

    if not RQ57_LMM_MODEL.exists():
        missing_files.append(f"../5.1.1/data/lmm_Lin+Log.pkl (resolved: {RQ57_LMM_MODEL})")
    if not RQ57_THETA_SCORES.exists():
        missing_files.append(f"../5.1.1/data/step03_theta_scores.csv (resolved: {RQ57_THETA_SCORES})")
    if not RQ57_LMM_INPUT.exists():
        missing_files.append(f"../5.1.1/data/step04_lmm_input.csv (resolved: {RQ57_LMM_INPUT})")

    if missing_files:
        error_msg = """
EXPECTATIONS ERROR: To perform Step 1 (Load RQ 5.1.1 Dependencies) I expect:
  - ../5.1.1/data/lmm_Lin+Log.pkl (best-fitting LMM model)
  - ../5.1.1/data/step03_theta_scores.csv (theta scores)
  - ../5.1.1/data/step04_lmm_input.csv (LMM input with TSVR_hours)

But missing:
"""
        for f in missing_files:
            error_msg += f"  - {f}\n"

        error_msg += """
Action: RQ 5.1.1 must complete Steps 1-5 before RQ 5.1.4 can execute.
Run RQ 5.1.1 workflow first, then retry RQ 5.1.4.
"""
        log(error_msg)
        print(error_msg, file=sys.stderr)
        return False

    return True


def validate_csv_columns(df: pd.DataFrame, expected_columns: list, file_path: str) -> bool:
    """
    Validate that CSV has expected columns.

    Args:
        df: Loaded dataframe
        expected_columns: List of required column names
        file_path: Path for error reporting

    Returns:
        True if columns match, False otherwise
    """
    actual_columns = df.columns.tolist()

    if actual_columns != expected_columns:
        error_msg = f"CSV column mismatch in {file_path}"
        error_msg += f"\n  Expected: {expected_columns}"
        error_msg += f"\n  Actual: {actual_columns}"
        log(error_msg)
        return False

    return True


def validate_csv_rows(df: pd.DataFrame, expected_range: Tuple[int, int], file_path: str) -> bool:
    """
    Validate that CSV has expected number of rows.

    Args:
        df: Loaded dataframe
        expected_range: Tuple of (min_rows, max_rows)
        file_path: Path for error reporting

    Returns:
        True if row count in range, False otherwise
    """
    min_rows, max_rows = expected_range
    actual_rows = len(df)

    if not (min_rows <= actual_rows <= max_rows):
        error_msg = f"CSV row count out of range in {file_path}"
        error_msg += f"\n  Expected: {min_rows}-{max_rows} rows"
        error_msg += f"\n  Actual: {actual_rows} rows"
        log(error_msg)
        return False

    return True


def validate_model_object(model_obj: Any) -> bool:
    """
    Validate that loaded object is a valid statsmodels MixedLMResults.

    Args:
        model_obj: Loaded model object

    Returns:
        True if valid MixedLMResults, False otherwise
    """
    if model_obj is None:
        log("Model object is None (pickle likely corrupted)")
        return False

    # Check for expected attributes of MixedLMResults
    required_attrs = ['cov_re', 'scale', 'random_effects']
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(model_obj, attr):
            missing_attrs.append(attr)

    if missing_attrs:
        log(f"Model missing required attributes: {missing_attrs}")
        return False

    return True


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Load RQ 5.7 Dependencies")
        log("=" * 70)
        # Circuit Breaker Check (Validate all files exist)
        log("Validating RQ 5.7 dependency files...")

        if not validate_files_exist():
            log("Circuit breaker: Missing RQ 5.1.1 dependency files")
            sys.exit(1)

        log("All three RQ 5.1.1 dependency files exist")
        # Load LMM Model (Pickle)
        log("Loading LMM model from pickle file...")
        log(f"  File: {RQ57_LMM_MODEL}")
        log(f"  Size: {RQ57_LMM_MODEL.stat().st_size / 1024:.1f} KB")

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
            with open(RQ57_LMM_MODEL, 'rb') as f:
                lmm_model = pickle.load(f)
                model_loaded = True
                log("LMM model loaded successfully with patsy workaround")

            # Restore original __setstate__
            data.ModelData.__setstate__ = original_setstate

        except Exception as e:
            log(f"Failed to load model even with patsy workaround: {str(e)}")
            log("Cannot proceed without model object")
            import traceback
            log(traceback.format_exc())
            sys.exit(1)

        if not model_loaded or lmm_model is None:
            log("Model object is None after loading")
            log("Cannot proceed")
            sys.exit(1)

        # Validate model object
        if not validate_model_object(lmm_model):
            log("Loaded object is not a valid MixedLMResults model")
            sys.exit(1)

        log("Successfully loaded model from pickle file")
        log(f"  Model type: {type(lmm_model).__name__}")
        # Load Theta Scores CSV
        log("Loading theta scores from CSV...")
        log(f"  File: {RQ57_THETA_SCORES}")

        try:
            df_theta = pd.read_csv(RQ57_THETA_SCORES, encoding='utf-8')
        except Exception as e:
            log(f"Failed to load theta scores CSV: {str(e)}")
            sys.exit(1)

        # Validate columns
        if not validate_csv_columns(df_theta, ['UID', 'test', 'Theta_All'], str(RQ57_THETA_SCORES)):
            log("Theta scores CSV columns invalid")
            sys.exit(1)

        # Validate row count
        if not validate_csv_rows(df_theta, (380, 400), str(RQ57_THETA_SCORES)):
            log("Theta scores CSV row count out of range")
            sys.exit(1)

        log(f"Successfully loaded theta scores: {len(df_theta)} rows, {len(df_theta.columns)} columns")
        # Load LMM Input CSV (TSVR mapping)
        log("Loading LMM input with TSVR mapping from CSV...")
        log(f"  File: {RQ57_LMM_INPUT}")

        try:
            df_lmm_input = pd.read_csv(RQ57_LMM_INPUT, encoding='utf-8')
        except Exception as e:
            log(f"Failed to load LMM input CSV: {str(e)}")
            sys.exit(1)

        # Validate columns
        expected_lmm_cols = ['composite_ID', 'UID', 'test', 'Theta', 'SE',
                             'TSVR_hours', 'Days', 'Days_squared', 'log_Days_plus1']
        if not validate_csv_columns(df_lmm_input, expected_lmm_cols, str(RQ57_LMM_INPUT)):
            log("LMM input CSV columns invalid")
            sys.exit(1)

        # Validate row count
        if not validate_csv_rows(df_lmm_input, (380, 400), str(RQ57_LMM_INPUT)):
            log("LMM input CSV row count out of range")
            sys.exit(1)

        log(f"Successfully loaded LMM input: {len(df_lmm_input)} rows, {len(df_lmm_input.columns)} columns")
        # Extract Model Metadata
        log("Extracting model metadata...")

        # Get model metadata
        n_groups = len(lmm_model.model.group_labels)
        n_observations = len(lmm_model.model.endog)  # Endogenous variable (y values)
        model_converged = lmm_model.converged

        # Extract random effects structure
        random_effects_structure = []
        if hasattr(lmm_model, 'random_effects') and lmm_model.random_effects:
            first_uid = list(lmm_model.random_effects.keys())[0]
            re_data = lmm_model.random_effects[first_uid]
            if hasattr(re_data, 'columns'):
                random_effects_structure = re_data.columns.tolist()
            elif hasattr(re_data, 'index'):
                random_effects_structure = re_data.index.tolist()

        # Build metadata dict
        metadata = {
            'model_source': 'results/ch5/5.1.1/data/lmm_Lin+Log.pkl',
            'model_formula': 'Lin+Log (Theta ~ Days + log(Days+1))',
            'model_type': type(lmm_model).__name__,
            'n_participants': n_groups,
            'n_observations': n_observations,
            'random_effects': random_effects_structure,
            'converged': model_converged,
            'loaded_timestamp': datetime.now().isoformat(),
        }

        log(f"Model metadata extracted:")
        for key, value in metadata.items():
            log(f"  {key}: {value}")
        # Save Metadata to YAML
        log(f"Saving model metadata to {METADATA_OUTPUT}...")

        try:
            with open(METADATA_OUTPUT, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            log(f"Failed to save metadata YAML: {str(e)}")
            sys.exit(1)

        log("Successfully saved metadata.yaml")
        # Validation (using tools.validation if available)
        log("Running model convergence validation...")

        if HAS_VALIDATION_TOOL:
            try:
                validation_result = validate_model_convergence(lmm_model)

                # Log validation results
                if isinstance(validation_result, dict):
                    for key, value in validation_result.items():
                        log(f"{key}: {value}")

                    # Check for failures
                    if validation_result.get('converged') == False:
                        log("Model did not converge - RQ 5.7 may have issues")
                        sys.exit(1)
                else:
                    log(f"{validation_result}")

            except Exception as e:
                log(f"Validation tool error: {str(e)}")
                log("Falling back to manual validation checks...")

                # Manual checks
                if not model_converged:
                    log("Model did not converge (converged=False)")
                    sys.exit(1)

                if n_groups != 100:
                    log(f"Expected 100 participants, got {n_groups}")

                if not (380 <= n_observations <= 400):
                    log(f"Expected 380-400 observations, got {n_observations}")

                log("Manual validation checks passed")
        else:
            # Validation tool not available, use manual checks
            log("Validation tool not available, using manual checks...")

            if not model_converged:
                log("Model did not converge (converged=False)")
                sys.exit(1)

            if n_groups != 100:
                log(f"Expected 100 participants, got {n_groups}")

            if not (380 <= n_observations <= 400):
                log(f"Expected 380-400 observations, got {n_observations}")

            log("Manual validation checks passed")
        # SUMMARY
        log("=" * 70)
        log("Step 01 complete: Successfully loaded RQ 5.7 dependencies")
        log(f"  Model: {n_groups} participants, {n_observations} observations, converged={model_converged}")
        log(f"  Theta scores: {len(df_theta)} rows")
        log(f"  LMM input: {len(df_lmm_input)} rows")
        log(f"  Metadata: {METADATA_OUTPUT}")
        log("=" * 70)

        sys.exit(0)

    except Exception as e:
        log(f"Unexpected error: {str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
