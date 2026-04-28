#!/usr/bin/env python3
"""IRT Calibration Pass 1 (All Items): Calibrate 2-dimensional GRM on all 36 items (source and destination factors)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import calibrate_irt

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_irt_calibration_pass1.log"


# IRT TESTING WORKFLOW 
# Phase 1 (MINIMAL TEST - run this first):
#   Set: iw_samples=10, mc_samples=1 (in model_fit and model_scores)
#   Runtime: ~10 minutes (validates entire pipeline)
#   Expected: Convergence may not be perfect (acceptable for testing)
#
# Phase 2 (PRODUCTION - only after Phase 1 passes):
#   Set: iw_samples=100, mc_samples=100 (in model_fit and model_scores)
#   Runtime: ~60 minutes (production-quality theta scores)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: IRT Calibration Pass 1 (All Items)")
        log("=" * 70)
        # Load Input Data

        log("\nLoading input data...")

        # Load wide-format IRT input
        irt_input_wide = pd.read_csv(RQ_DIR / "data/step00_irt_input.csv")
        log(f"step00_irt_input.csv ({len(irt_input_wide)} rows, {len(irt_input_wide.columns)} cols)")

        # Load Q-matrix
        q_matrix = pd.read_csv(RQ_DIR / "data/step00_q_matrix.csv")
        log(f"step00_q_matrix.csv ({len(q_matrix)} items)")
        # Convert Wide to Long Format
        # calibrate_irt expects long format with columns: [UID, test, item_name, score]
        # We need to split composite_ID back to UID and test

        log("\nConverting wide format to long format...")

        # Extract UID and test from composite_ID (format: "A010_1" -> UID="A010", test="1")
        irt_input_wide['UID'] = irt_input_wide['composite_ID'].str.split('_').str[0]
        irt_input_wide['test'] = irt_input_wide['composite_ID'].str.split('_').str[1].astype(int)

        # Melt item columns to long format
        item_cols = [col for col in irt_input_wide.columns if col.startswith('TQ_')]
        df_long = irt_input_wide.melt(
            id_vars=['UID', 'test'],
            value_vars=item_cols,
            var_name='item_name',
            value_name='score'
        )

        # Remove NaN responses (missing data)
        n_before = len(df_long)
        df_long = df_long.dropna(subset=['score'])
        n_after = len(df_long)
        log(f"Wide -> Long: {n_before} -> {n_after} observations ({n_before - n_after} NaN removed)")
        # Define Groups for 2-Factor Model
        # Source factor: items with "-U-" in name
        # Destination factor: items with "-D-" in name

        log("\nDefining factor groups...")
        groups = {
            'source': ['-U-'],
            'destination': ['-D-']
        }
        log(f"Groups: {groups}")
        # Configure IRT Model Settings
        # CRITICAL: Using MINIMAL settings first to validate pipeline
        # Change to production settings after testing

        log("\nConfiguring IRT model settings...")
        config = {
            'factors': ['source', 'destination'],
            'correlated_factors': True,
            'device': 'cpu',
            'seed': 42,
            'model_fit': {
                'batch_size': 2048,
                'iw_samples': 10,    # MINIMAL: Change to 100 for production
                'mc_samples': 1      # MINIMAL: Change to 100 for production (scoring only)
            },
            'model_scores': {
                'scoring_batch_size': 2048,
                'mc_samples': 10,    # MINIMAL: Change to 100 for production
                'iw_samples': 10     # MINIMAL: Change to 100 for production
            }
        }
        log("Using MINIMAL settings for pipeline validation")
        log("model_fit: batch_size=2048, iw_samples=10, mc_samples=1")
        log("model_scores: scoring_batch_size=2048, mc_samples=10, iw_samples=10")
        log("Change to production settings after testing (iw_samples=100, mc_samples=100)")
        # Run IRT Calibration

        log("\nRunning IRT calibration (2-factor GRM)...")
        log("This may take ~10 minutes with minimal settings...")

        df_thetas, df_items = calibrate_irt(
            df_long=df_long,
            groups=groups,
            config=config
        )

        log(f"IRT calibration complete")
        log(f"Theta scores: {len(df_thetas)} rows, {len(df_thetas.columns)} cols")
        log(f"Item parameters: {len(df_items)} rows, {len(df_items.columns)} cols")
        # Reformat Outputs to Match Expected Schema
        # calibrate_irt returns:
        #   - df_thetas: [UID, test, Theta_source, Theta_destination, ...]
        #   - df_items: [item_name, Difficulty, Overall_Discrimination, ...]
        # We need:
        #   - theta: [composite_ID, theta_source, theta_destination, se_source, se_destination]
        #   - item_params: [item_tag, factor, a, b]

        log("\nReformatting outputs to match expected schema...")

        # --- Reformat Theta Scores ---
        # Recreate composite_ID from UID and test
        df_thetas['composite_ID'] = df_thetas['UID'].astype(str) + '_' + df_thetas['test'].astype(str)

        # Rename columns to lowercase format
        theta_output = pd.DataFrame({
            'composite_ID': df_thetas['composite_ID'],
            'theta_source': df_thetas['Theta_source'],
            'theta_destination': df_thetas['Theta_destination']
        })

        # Add standard errors (if available from calibrate_irt, otherwise compute placeholders)
        # Note: calibrate_irt may not return SE directly - checking columns
        if 'SE_source' in df_thetas.columns:
            theta_output['se_source'] = df_thetas['SE_source']
            theta_output['se_destination'] = df_thetas['SE_destination']
            log("Using SE values from calibrate_irt")
        else:
            # Placeholder SEs (will be replaced with actual SEs in future implementation)
            # Using conservative estimate: SE = 0.5 (typical for IRT theta estimates)
            theta_output['se_source'] = 0.5
            theta_output['se_destination'] = 0.5
            log("SE not available from calibrate_irt - using placeholder SE=0.5")

        # --- Reformat Item Parameters ---
        # calibrate_irt returns item parameters in wide format with columns per difficulty category
        # We need to reshape to long format with one row per item

        # Extract item tag (remove any prefixes added by calibrate_irt)
        df_items['item_tag'] = df_items['item_name']

        # Determine factor for each item from Q-matrix
        item_factor_map = {}
        for _, row in q_matrix.iterrows():
            if row['source'] == 1:
                item_factor_map[row['item_tag']] = 'source'
            elif row['destination'] == 1:
                item_factor_map[row['item_tag']] = 'destination'

        df_items['factor'] = df_items['item_tag'].map(item_factor_map)

        # Extract discrimination (a) and difficulty (b)
        # calibrate_irt returns Overall_Discrimination and Difficulty columns
        item_params = pd.DataFrame({
            'item_tag': df_items['item_tag'],
            'factor': df_items['factor'],
            'a': df_items['Overall_Discrimination'],
            'b': df_items['Difficulty']
        })

        log(f"Item parameters reformatted: {len(item_params)} items")
        log(f"Theta scores reformatted: {len(theta_output)} composite_IDs")
        # Save Outputs
        # These outputs will be used by: Step 2 (purify items) and Step 3 (Pass 2 calibration)

        log("\nSaving outputs...")

        # Save item parameters
        item_params_path = RQ_DIR / "data/step01_pass1_item_params.csv"
        item_params.to_csv(item_params_path, index=False, encoding='utf-8')
        log(f"{item_params_path.name} ({len(item_params)} rows, {len(item_params.columns)} cols)")

        # Save theta scores
        theta_path = RQ_DIR / "data/step01_pass1_theta.csv"
        theta_output.to_csv(theta_path, index=False, encoding='utf-8')
        log(f"{theta_path.name} ({len(theta_output)} rows, {len(theta_output.columns)} cols)")

        # Save diagnostics (text file with summary statistics)
        diagnostics_path = RQ_DIR / "data/step01_pass1_diagnostics.txt"
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            f.write("IRT Calibration Pass 1 Diagnostics\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Items calibrated: {len(item_params)}\n")
            f.write(f"Composite IDs: {len(theta_output)}\n")
            f.write(f"Factors: source, destination\n")
            f.write(f"Correlated factors: True\n\n")
            f.write("Item Parameter Summary:\n")
            f.write(f"  Discrimination (a): min={item_params['a'].min():.3f}, max={item_params['a'].max():.3f}, mean={item_params['a'].mean():.3f}\n")
            f.write(f"  Difficulty (b): min={item_params['b'].min():.3f}, max={item_params['b'].max():.3f}, mean={item_params['b'].mean():.3f}\n\n")
            f.write("Theta Score Summary:\n")
            f.write(f"  Source: min={theta_output['theta_source'].min():.3f}, max={theta_output['theta_source'].max():.3f}, mean={theta_output['theta_source'].mean():.3f}\n")
            f.write(f"  Destination: min={theta_output['theta_destination'].min():.3f}, max={theta_output['theta_destination'].max():.3f}, mean={theta_output['theta_destination'].mean():.3f}\n")
        log(f"{diagnostics_path.name}")
        # Validation
        # Validate: All items present, all composite_IDs present, parameter bounds

        log("\nValidating outputs...")

        validation_passed = True

        # Check item count
        if len(item_params) != 36:
            log(f"[VALIDATION ERROR] Expected 36 items, got {len(item_params)}")
            validation_passed = False
        else:
            log(f"All 36 items present in item_params")

        # Check composite_ID count
        if len(theta_output) != 400:
            log(f"[VALIDATION ERROR] Expected 400 composite_IDs, got {len(theta_output)}")
            validation_passed = False
        else:
            log(f"All 400 composite_IDs present in theta")

        # Check discrimination bounds
        a_min, a_max = item_params['a'].min(), item_params['a'].max()
        if a_min < 0.0 or a_max > 10.0:
            log(f"[VALIDATION WARNING] Discrimination (a) outside [0.0, 10.0]: min={a_min:.3f}, max={a_max:.3f}")
        else:
            log(f"Discrimination (a) in [0.0, 10.0]: min={a_min:.3f}, max={a_max:.3f}")

        # Check difficulty bounds
        b_min, b_max = item_params['b'].min(), item_params['b'].max()
        if b_min < -6.0 or b_max > 6.0:
            log(f"[VALIDATION WARNING] Difficulty (b) outside [-6.0, 6.0]: min={b_min:.3f}, max={b_max:.3f}")
        else:
            log(f"Difficulty (b) in [-6.0, 6.0]: min={b_min:.3f}, max={b_max:.3f}")

        # Check theta bounds
        theta_source_min, theta_source_max = theta_output['theta_source'].min(), theta_output['theta_source'].max()
        theta_dest_min, theta_dest_max = theta_output['theta_destination'].min(), theta_output['theta_destination'].max()
        if theta_source_min < -4 or theta_source_max > 4 or theta_dest_min < -4 or theta_dest_max > 4:
            log(f"[VALIDATION WARNING] Theta outside [-4, 4]:")
            log(f"  Source: min={theta_source_min:.3f}, max={theta_source_max:.3f}")
            log(f"  Destination: min={theta_dest_min:.3f}, max={theta_dest_max:.3f}")
        else:
            log(f"Theta in [-4, 4] for both factors")

        # Print summary statistics
        log("\nDescriptive Statistics:")
        log(f"  Items: {len(item_params)} (source: {sum(item_params['factor'] == 'source')}, destination: {sum(item_params['factor'] == 'destination')})")
        log(f"  Discrimination (a): mean={item_params['a'].mean():.3f}, SD={item_params['a'].std():.3f}")
        log(f"  Difficulty (b): mean={item_params['b'].mean():.3f}, SD={item_params['b'].std():.3f}")
        log(f"  Theta (source): mean={theta_output['theta_source'].mean():.3f}, SD={theta_output['theta_source'].std():.3f}")
        log(f"  Theta (destination): mean={theta_output['theta_destination'].mean():.3f}, SD={theta_output['theta_destination'].std():.3f}")

        if validation_passed:
            log("\nStep 01 complete - all validations passed")
        else:
            log("\nStep 01 complete - some validations failed (see above)")

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
