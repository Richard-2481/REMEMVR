#!/usr/bin/env python3
"""IRT Calibration Pass 2 (Purified Items): Re-calibrate single-factor IRT model using ONLY high-quality items identified"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# parents[4] = REMEMVR/ (code -> rq7 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import calibrate_irt

from tools.validation import validate_irt_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
LOG_FILE = RQ_DIR / "logs" / "step03_calibration.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: IRT Calibration Pass 2 (Purified Items)")
        # Load Input Data

        log("Loading raw VR item responses...")
        irt_input_path = PROJECT_ROOT / "results" / "ch5" / "rq1" / "data" / "step00_irt_input.csv"

        if not irt_input_path.exists():
            raise FileNotFoundError(f"IRT input data missing: {irt_input_path}\n"
                                     "Expected from RQ 5.1 step00")

        irt_data_raw = pd.read_csv(irt_input_path, encoding='utf-8')
        log(f"{irt_input_path.name} ({len(irt_data_raw)} rows, {len(irt_data_raw.columns)} cols)")

        log("Loading purified item list...")
        purified_path = RQ_DIR / "data" / "step02_purified_items.csv"

        if not purified_path.exists():
            raise FileNotFoundError(f"Purified items missing: {purified_path}\n"
                                     "Run step02_purify_items.py first")

        purified_items = pd.read_csv(purified_path, encoding='utf-8')
        log(f"{purified_path.name} ({len(purified_items)} items retained)")

        # Extract item names for calibration
        purified_item_names = purified_items['item_name'].tolist()
        log(f"  Purified items: {purified_item_names[:5]}... ({len(purified_item_names)} total)")
        # Transform to Long Format for IRT
        # IRT requires long format: composite_ID, UID, test, item_name, score

        log("Converting wide format to long format for IRT...")
        # Only keep purified items
        id_vars = ['composite_ID']
        df_long = irt_data_raw.melt(
            id_vars=id_vars,
            value_vars=purified_item_names,
            var_name='item_name',
            value_name='score'
        )

        # Drop missing responses
        n_before = len(df_long)
        df_long = df_long.dropna(subset=['score'])
        n_after = len(df_long)
        log(f"Long format: {n_after} observations ({n_before - n_after} missing values dropped)")

        # Ensure scores are integers
        df_long['score'] = df_long['score'].astype(int)

        # Split composite_ID into UID and test for calibrate_irt API
        df_long[['UID', 'test']] = df_long['composite_ID'].str.split('_', n=1, expand=True)
        log(f"Split composite_ID into UID and test columns")
        # Run Pass 2 IRT Calibration

        log("Running Pass 2 IRT calibration (purified items only)...")

        # Configure groups: Only purified items assigned to 'All' factor
        groups = {
            "All": purified_item_names
        }

        # Configure IRT model (same settings as Pass 1)
        config = {
            "factors": ["All"],
            "correlated_factors": False,  # Single factor (no correlations)
            "device": "cpu",
            "seed": 42,
            "model_fit": {
                "batch_size": 2048,
                "iw_samples": 10,
                "mc_samples": 1
            },
            "model_scores": {
                "scoring_batch_size": 2048,
                "mc_samples": 10,
                "iw_samples": 10
            }
        }

        theta_scores, item_params = calibrate_irt(
            df_long=df_long,
            groups=groups,
            config=config
        )

        log("Pass 2 calibration complete")
        log(f"  Theta scores: {len(theta_scores)} rows")
        log(f"  Item parameters: {len(item_params)} items")
        # Save Pass 2 Outputs
        # These outputs will be used by: Step 4 (LMM input preparation)

        # Save theta scores (FINAL estimates for LMM)
        theta_output_path = RQ_DIR / "data" / "step03_theta_scores.csv"
        log(f"Saving Pass 2 theta scores to {theta_output_path.name}...")
        theta_scores.to_csv(theta_output_path, index=False, encoding='utf-8')
        log(f"{theta_output_path.name} ({len(theta_scores)} rows, {len(theta_scores.columns)} cols)")
        log(f"  Theta_All range: [{theta_scores['Theta_All'].min():.3f}, {theta_scores['Theta_All'].max():.3f}]")
        if 'SE_All' in theta_scores.columns:
            log(f"  SE_All range: [{theta_scores['SE_All'].min():.3f}, {theta_scores['SE_All'].max():.3f}]")
        else:
            log(f"  SE_All: Not available with minimal settings")

        # Save item parameters (Pass 2 estimates)
        params_output_path = RQ_DIR / "logs" / "step03_item_parameters.csv"
        log(f"Saving Pass 2 item parameters to {params_output_path.name}...")
        item_params.to_csv(params_output_path, index=False, encoding='utf-8')
        log(f"{params_output_path.name} ({len(item_params)} items, {len(item_params.columns)} cols)")
        log(f"  Discrimination (a) range: [{item_params['a'].min():.3f}, {item_params['a'].max():.3f}]")
        log(f"  Difficulty (b) range: [{item_params['b'].min():.3f}, {item_params['b'].max():.3f}]")
        # Validate Pass 2 Results
        # Validates: Model convergence, theta/SE ranges, item count match, SE improvement
        # Threshold: Same as Pass 1, plus SE improvement check

        log("Validating Pass 2 calibration results...")

        # Load Pass 1 theta scores for SE comparison
        pass1_theta_path = RQ_DIR / "data" / "step01_theta_scores.csv"
        if pass1_theta_path.exists():
            pass1_theta = pd.read_csv(pass1_theta_path, encoding='utf-8')
            se_comparison = theta_scores.merge(
                pass1_theta[['composite_ID', 'SE_All']],
                on='composite_ID',
                suffixes=('_pass2', '_pass1')
            )
            se_improved = (se_comparison['SE_All_pass2'] <= se_comparison['SE_All_pass1']).mean()
            log(f"SE improvement: {se_improved*100:.1f}% of cases have SE_pass2 <= SE_pass1")
        else:
            log("Pass 1 theta scores not found - skipping SE comparison")

        validation_result = validate_irt_convergence(
            results={
                "item_params": item_params,
                "theta_scores": theta_scores,
                "log_file": str(LOG_FILE)
            }
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check item count matches purified list
        if len(item_params) == len(purified_items):
            log(f"Item count matches purified list ({len(item_params)} items)")
        else:
            log(f"Item count mismatch - expected {len(purified_items)}, got {len(item_params)}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
