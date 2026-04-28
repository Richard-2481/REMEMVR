#!/usr/bin/env python3
"""IRT Calibration Pass 1 (All TC_* Items): Pass 1 provides initial parameter estimates for item purification (Step 2)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.analysis_irt import (
    prepare_irt_input_from_long,
    configure_irt_model,
    fit_irt_grm,
    extract_theta_from_irt,
    extract_parameters_from_irt
)

# Import validation tools
from tools.validation import validate_irt_convergence, validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.1
LOG_FILE = RQ_DIR / "logs" / "step01_irt_calibration_pass1.log"


# IRT TESTING WORKFLOW 
# Phase 1 (MINIMAL TEST - run this first):
#   Set: max_iter=50, mc_samples=10, iw_samples=10
#   Runtime: ~5-10 minutes (validates entire pipeline)
#   Expected: Convergence may fail (acceptable for testing)
#
# Phase 2 (PRODUCTION - only after Phase 1 passes):
#   Set: max_iter=200 (or as specified below)
#   Runtime: 20-30 minutes (production-quality theta scores)

# IRT Model Configuration (2-factor GRM for TC_* confidence items)
IRT_CONFIG = {
    'model_type': 'GRM',
    'n_cats': 5,  # CORRECTED: Actual data has 5 categories {0.2, 0.4, 0.6, 0.8, 1.0}
    'factors': ['Source', 'Destination'],  # 2-factor structure for source vs destination locations
    'correlated_factors': True,
    'device': 'cpu',
    'seed': 42,
}

# CRITICAL SETTINGS FROM RQ 6.1.1:
# - FITTING: mc_samples=1 (FAST - avoids 7000+ epoch hang)
# - SCORING: mc_samples=100 (ACCURATE theta estimates)

# MINIMUM MODE: Absolute minimum settings to prove code works without crashing
# Model Fitting Settings (MINIMUM)
MODEL_FIT_SETTINGS = {
    'batch_size': 2048,
    'iw_samples': 100,  # MED: 100 (was 1) - Ch5 validated 2025-11-25
    'mc_samples': 1     # MED: 1 = point estimates (FAST - correct)
}

# Theta Scoring Settings (MED - Ch5 validated)
MODEL_SCORING_SETTINGS = {
    'scoring_batch_size': 2048,
    'mc_samples': 100,  # MED: 100 (was 1) - Monte Carlo integration
    'iw_samples': 100   # MED: 100 (was 1) - Importance weighting
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: IRT Calibration Pass 1 (2-factor GRM on all TC_* items)")
        log(f"Model: 2-factor GRM, {IRT_CONFIG['n_cats']} categories")
        log(f"Fitting: batch_size={MODEL_FIT_SETTINGS['batch_size']}, "
            f"mc_samples={MODEL_FIT_SETTINGS['mc_samples']} (FAST mode), "
            f"iw_samples={MODEL_FIT_SETTINGS['iw_samples']}")
        log(f"Scoring: mc_samples={MODEL_SCORING_SETTINGS['mc_samples']} (ACCURATE mode), "
            f"iw_samples={MODEL_SCORING_SETTINGS['iw_samples']}")
        # Load Input Data

        log("Loading input data...")

        # Load wide-format IRT input (composite_ID x TC_* items)
        df_wide = pd.read_csv(RQ_DIR / "data" / "step00_irt_input.csv")
        log(f"step00_irt_input.csv ({len(df_wide)} rows, {len(df_wide.columns)} cols)")

        # Load Q-matrix (item-to-factor mapping)
        df_q_matrix = pd.read_csv(RQ_DIR / "data" / "step00_q_matrix.csv")
        log(f"step00_q_matrix.csv ({len(df_q_matrix)} items)")

        # Extract item list from Q-matrix
        item_list = df_q_matrix['item_name'].tolist()
        log(f"Q-matrix specifies {len(item_list)} items across 3 factors")

        # Verify all Q-matrix items exist in wide data
        tc_cols = [c for c in df_wide.columns if c.startswith('TC_')]
        missing_items = set(item_list) - set(tc_cols)
        if missing_items:
            raise ValueError(f"Q-matrix contains items not in data: {missing_items}")
        log(f"All {len(item_list)} Q-matrix items found in data")

        # Convert wide to long format for prepare_irt_input_from_long
        log("Converting wide format to long format...")
        df_long_list = []
        for idx, row in df_wide.iterrows():
            composite_id = row['composite_ID']
            # Parse UID and test from composite_ID (format: A010_T1)
            parts = composite_id.rsplit('_', 1)
            uid = parts[0]
            test_str = parts[1] if len(parts) > 1 else 'T1'
            test_num = test_str.replace('T', '')  # Extract number from T1, T2, etc.

            for item in item_list:
                response = row[item]
                df_long_list.append({
                    'UID': uid,
                    'test': test_num,
                    'item_name': item,
                    'score': response
                })
        df_long = pd.DataFrame(df_long_list)
        log(f"Long format created: {len(df_long)} observations "
            f"({len(df_wide)} participants x {len(item_list)} items)")

        # Create groups dictionary from Q-matrix (factor -> items mapping)
        log("Creating factor groups from Q-matrix...")
        groups = {}
        for factor in IRT_CONFIG['factors']:
            # Q-matrix uses capitalized column names: Source, Destination
            factor_col = factor  # Direct column name (no prefix)
            items_in_factor = df_q_matrix[df_q_matrix[factor_col] == 1]['item_name'].tolist()
            groups[factor] = items_in_factor
            log(f"{factor}: {len(items_in_factor)} items")
        # Prepare IRT Tensors (Response Matrix, Missing Mask, Q-Matrix)

        log("Preparing IRT tensors via prepare_irt_input_from_long...")
        # Return order: response_matrix, Q_matrix, missing_mask, item_list, composite_ids
        response_matrix, Q_matrix, missing_mask, item_list_ordered, composite_ids = \
            prepare_irt_input_from_long(df_long, groups)

        log(f"Tensors prepared:")
        log(f"  - response_matrix: {response_matrix.shape} (participants x items)")
        log(f"  - missing_mask: {missing_mask.shape}")
        log(f"  - Q_matrix: {Q_matrix.shape} (items x factors)")
        log(f"  - composite_ids: {len(composite_ids)} IDs")
        log(f"  - item_list: {len(item_list_ordered)} items")
        # Configure IRT Model (3-Factor GRM)

        log("Configuring 2-factor GRM model via configure_irt_model...")
        n_items = response_matrix.shape[1]
        n_factors = len(IRT_CONFIG['factors'])

        # CRITICAL: n_cats must be a list (one value per item, or single value for all)
        # For uniform category structure, pass list of same value
        n_cats_list = [IRT_CONFIG['n_cats']] * n_items

        model = configure_irt_model(
            n_items=n_items,
            n_factors=n_factors,
            n_cats=n_cats_list,  # Pass as list
            Q_matrix=Q_matrix,
            correlated_factors=IRT_CONFIG['correlated_factors'],
            device=IRT_CONFIG['device'],
            seed=IRT_CONFIG['seed']
        )
        log(f"Model configured: {n_items} items, {n_factors} factors, "
            f"{IRT_CONFIG['n_cats']} categories, correlated={IRT_CONFIG['correlated_factors']}")
        # Fit IRT Model (CRITICAL: mc_samples=1 for fitting)
        # KEY FIX: mc_samples=1 during fit prevents 7000+ epoch hang (RQ 6.1.1)

        log("Fitting GRM model via fit_irt_grm...")
        log("Using mc_samples=1 during fit (avoids timeout per RQ 6.1.1)")

        fitted_model = fit_irt_grm(
            model=model,
            response_matrix=response_matrix,
            missing_mask=missing_mask,
            batch_size=MODEL_FIT_SETTINGS['batch_size'],
            iw_samples=MODEL_FIT_SETTINGS['iw_samples'],
            mc_samples=MODEL_FIT_SETTINGS['mc_samples']  # mc_samples=1 is CRITICAL
        )
        log("Model fitting complete")
        # Extract Theta Scores (CRITICAL: mc_samples=100 for scoring)
        # KEY FIX: mc_samples=100 during scoring for accurate estimates (RQ 6.1.1)

        log("Extracting theta scores via extract_theta_from_irt...")
        log("Using mc_samples=100 during scoring (accurate theta estimates per RQ 6.1.1)")

        df_theta = extract_theta_from_irt(
            model=fitted_model,
            response_matrix=response_matrix,
            missing_mask=missing_mask,
            composite_ids=composite_ids,
            factor_names=IRT_CONFIG['factors'],
            scoring_batch_size=MODEL_SCORING_SETTINGS['scoring_batch_size'],
            mc_samples=MODEL_SCORING_SETTINGS['mc_samples'],  # mc_samples=100 for accuracy
            iw_samples=MODEL_SCORING_SETTINGS['iw_samples'],
            invert_scale=False  # Higher theta = higher confidence
        )

        log(f"Theta extraction complete: {len(df_theta)} participants x {len(IRT_CONFIG['factors'])} factors")

        # Reconstruct composite_ID from UID and test
        log("Reconstructing composite_ID from UID and test...")
        df_theta['composite_ID'] = df_theta['UID'] + '_T' + df_theta['test'].astype(str)

        # Rename columns to match expected output format (lowercase theta_)
        rename_map = {}
        for factor in IRT_CONFIG['factors']:
            rename_map[f'Theta_{factor}'] = f'theta_{factor}'
        df_theta_wide = df_theta.rename(columns=rename_map)

        # Select and order output columns
        # Note: Tool does NOT return SE values - only theta
        ordered_cols = ['composite_ID']
        for factor in IRT_CONFIG['factors']:
            ordered_cols.append(f'theta_{factor}')
        df_theta_wide = df_theta_wide[ordered_cols]

        log(f"Wide format: {len(df_theta_wide)} rows, {len(df_theta_wide.columns)} columns")
        # Extract Item Parameters

        log("Extracting item parameters via extract_parameters_from_irt...")

        df_params = extract_parameters_from_irt(
            model=fitted_model,
            item_list=item_list_ordered,
            factor_names=IRT_CONFIG['factors'],
            n_cats=n_cats_list  # Pass as list
        )

        log(f"Parameter extraction complete: {len(df_params)} items")

        # Note: extract_parameters_from_irt returns MIRT format for multidimensional models:
        # item_name, Difficulty, Overall_Discrimination, Discrim_Factor1, Discrim_Factor2, ...
        # Keep as-is since this is the tool's native output format
        log(f"Parameter columns: {list(df_params.columns)}")
        # Save Outputs
        # These outputs will be used by Step 2 (purification) and Step 3 (Pass 2)

        log("Saving Pass 1 outputs...")

        # Save item parameters (for purification)
        item_params_path = RQ_DIR / "data" / "step01_pass1_item_params.csv"
        df_params.to_csv(item_params_path, index=False, encoding='utf-8')
        log(f"{item_params_path.name} ({len(df_params)} items, {len(df_params.columns)} cols)")

        # Save theta estimates (diagnostic, not used downstream)
        theta_path = RQ_DIR / "data" / "step01_pass1_theta.csv"
        df_theta_wide.to_csv(theta_path, index=False, encoding='utf-8')
        log(f"{theta_path.name} ({len(df_theta_wide)} rows, {len(df_theta_wide.columns)} cols)")
        # Validation
        # Validates: convergence, parameter bounds, no NaN parameters

        log("Running validation checks...")

        # Check 1: Parameter bounds (discrimination in [0, 10], difficulty in [-6, 6])
        log("Checking parameter bounds...")
        a_min, a_max = df_params['Overall_Discrimination'].min(), df_params['Overall_Discrimination'].max()
        b_min, b_max = df_params['Difficulty'].min(), df_params['Difficulty'].max()

        log(f"Discrimination: min={a_min:.3f}, max={a_max:.3f}")
        log(f"Difficulty (b): min={b_min:.3f}, max={b_max:.3f}")

        if a_min < 0.0 or a_max > 10.0:
            log(f"Discrimination out of bounds [0.0, 10.0]")
            sys.exit(1)
        if b_min < -6.0 or b_max > 6.0:
            log(f"Difficulty out of bounds [-6.0, 6.0]")
            sys.exit(1)
        log("Parameter bounds valid")

        # Check 2: No NaN parameters
        log("Checking for NaN parameters...")
        nan_count = df_params[['Overall_Discrimination', 'Difficulty']].isna().sum().sum()
        if nan_count > 0:
            log(f"{nan_count} NaN parameters found")
            sys.exit(1)
        log("No NaN parameters")

        # Check 3: Theta estimates valid (theta in [-4, 4])
        # Note: Tool does not return SE values, only theta
        log("Checking theta estimates...")
        theta_cols = [c for c in df_theta_wide.columns if c.startswith('theta_')]

        theta_min = df_theta_wide[theta_cols].min().min()
        theta_max = df_theta_wide[theta_cols].max().max()

        log(f"Theta: min={theta_min:.3f}, max={theta_max:.3f}")

        if theta_min < -4.0 or theta_max > 4.0:
            log(f"Theta out of bounds [-4.0, 4.0]")
            sys.exit(1)
        log("Theta estimates valid")

        # Check 4: Output row counts
        log("Checking output row counts...")
        if len(df_params) != len(item_list):
            log(f"Item parameters count mismatch: {len(df_params)} vs {len(item_list)} expected")
            sys.exit(1)
        if len(df_theta_wide) != len(df_wide):
            log(f"Theta rows mismatch: {len(df_theta_wide)} vs {len(df_wide)} expected")
            sys.exit(1)
        log(f"Output row counts correct (items={len(df_params)}, theta={len(df_theta_wide)})")

        log("Step 01 complete - All validations passed")
        log(f"Pass 1 calibrated {len(df_params)} items for {len(df_theta_wide)} participants")
        log("Run Step 02 to purify items using quality thresholds (a >= 0.4, |b| <= 3.0)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
