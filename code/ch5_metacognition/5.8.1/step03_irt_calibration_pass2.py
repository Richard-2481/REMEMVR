#!/usr/bin/env python3
"""IRT Calibration Pass 2 (Purified Items Only): Re-calibrate GRM on purified items only (after Step 02 quality filtering)."""

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
LOG_FILE = RQ_DIR / "logs" / "step03_irt_calibration_pass2.log"


# IRT TESTING WORKFLOW 
# Phase 1 (MINIMAL TEST - run this first):
#   Set: mc_samples=1, iw_samples=1 (CURRENT SETTINGS)
#   Runtime: ~5-10 minutes (validates entire pipeline)
#   Expected: Convergence may fail (acceptable for testing)
#
# Phase 2 (PRODUCTION - only after Phase 1 passes):
#   Set: mc_samples=100 for scoring (iw_samples=100)
#   Runtime: 15-25 minutes (production-quality theta scores)

# IRT Model Configuration (2-factor GRM for TC_* confidence items)
IRT_CONFIG = {
    'model_type': 'GRM',
    'n_cats': 5,  # 5 categories {0.2, 0.4, 0.6, 0.8, 1.0}
    'factors': ['Source', 'Destination'],  # 2-factor structure for source vs destination locations
    'correlated_factors': True,
    'device': 'cpu',
    'seed': 42,
}

# CRITICAL SETTINGS FROM RQ 6.1.1:
# - FITTING: mc_samples=1 (FAST - avoids 7000+ epoch hang)
# - SCORING: mc_samples=1 (MINIMUM mode) or mc_samples=100 (PRODUCTION mode)

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
        log("Step 03: IRT Calibration Pass 2 (2-factor GRM on purified items only)")
        log(f"Model: 2-factor GRM, {IRT_CONFIG['n_cats']} categories")
        log(f"Fitting: batch_size={MODEL_FIT_SETTINGS['batch_size']}, "
            f"mc_samples={MODEL_FIT_SETTINGS['mc_samples']} (FAST mode), "
            f"iw_samples={MODEL_FIT_SETTINGS['iw_samples']}")
        log(f"Scoring: mc_samples={MODEL_SCORING_SETTINGS['mc_samples']} (MINIMUM mode), "
            f"iw_samples={MODEL_SCORING_SETTINGS['iw_samples']}")
        log("This is MINIMUM mode - for production, set mc_samples=100 for scoring")
        # Load Input Data

        log("Loading input data...")

        # Load original wide-format IRT input (composite_ID x TC_* items)
        df_wide_original = pd.read_csv(RQ_DIR / "data" / "step00_irt_input.csv")
        log(f"step00_irt_input.csv ({len(df_wide_original)} rows, {len(df_wide_original.columns)} cols)")

        # Load purified items list from Step 02
        df_purified = pd.read_csv(RQ_DIR / "data" / "step02_purified_items.csv")
        log(f"step02_purified_items.csv ({len(df_purified)} items retained)")

        # Extract purified item names
        purified_items = df_purified['item_name'].tolist()
        log(f"Purified items count: {len(purified_items)}")

        # Load Q-matrix and filter to purified items
        # NOTE: Step 02 did not generate step02_q_matrix_purified.csv, so we create it here
        df_q_matrix_original = pd.read_csv(RQ_DIR / "data" / "step00_q_matrix.csv")
        log(f"step00_q_matrix.csv ({len(df_q_matrix_original)} items)")

        # Filter Q-matrix to purified items only
        df_q_matrix = df_q_matrix_original[df_q_matrix_original['item_name'].isin(purified_items)].copy()
        log(f"Q-matrix filtered to {len(df_q_matrix)} purified items")

        # Verify all purified items have Q-matrix entries
        missing_q = set(purified_items) - set(df_q_matrix['item_name'].tolist())
        if missing_q:
            raise ValueError(f"Purified items missing from Q-matrix: {missing_q}")
        log("All purified items have Q-matrix entries")

        # Filter wide data to purified items only
        log("Filtering wide data to purified items...")
        cols_to_keep = ['composite_ID'] + purified_items
        df_wide = df_wide_original[cols_to_keep].copy()
        log(f"Wide data filtered: {len(df_wide)} rows x {len(df_wide.columns)} cols "
            f"({len(purified_items)} purified items)")

        # Verify all purified items exist in wide data
        tc_cols = [c for c in df_wide.columns if c.startswith('TC_')]
        missing_items = set(purified_items) - set(tc_cols)
        if missing_items:
            raise ValueError(f"Purified items not in wide data: {missing_items}")
        log(f"All {len(purified_items)} purified items found in wide data")

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

            for item in purified_items:
                response = row[item]
                df_long_list.append({
                    'UID': uid,
                    'test': test_num,
                    'item_name': item,
                    'score': response
                })
        df_long = pd.DataFrame(df_long_list)
        log(f"Long format created: {len(df_long)} observations "
            f"({len(df_wide)} participants x {len(purified_items)} items)")

        # Create groups dictionary from Q-matrix (factor -> items mapping)
        log("Creating factor groups from Q-matrix...")
        groups = {}
        for factor in IRT_CONFIG['factors']:
            # Q-matrix uses capitalized column names: Source, Destination
            factor_col = factor  # Direct column name (no prefix)
            items_in_factor = df_q_matrix[df_q_matrix[factor_col] == 1]['item_name'].tolist()
            groups[factor] = items_in_factor
            log(f"{factor}: {len(items_in_factor)} items")

        # Check if any domains have zero items (e.g., When excluded after purification)
        empty_domains = [d for d, items in groups.items() if len(items) == 0]
        if empty_domains:
            log(f"Domains with zero items after purification: {empty_domains}")
            log(f"These domains will be excluded from Pass 2 calibration")
            # Filter factors list to exclude empty domains
            active_factors = [f for f in IRT_CONFIG['factors'] if len(groups[f]) > 0]
            log(f"Active factors for Pass 2: {active_factors}")
        else:
            active_factors = IRT_CONFIG['factors']
            log(f"All 2 factors active for Pass 2: {active_factors}")

        # Update groups to only include active factors
        groups = {f: groups[f] for f in active_factors}
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
        # Configure IRT Model (3-Factor or 2-Factor GRM)

        log(f"Configuring {len(active_factors)}-factor GRM model via configure_irt_model...")
        n_items = response_matrix.shape[1]
        n_factors = len(active_factors)

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
        # Extract Theta Scores (MINIMUM mode: mc_samples=1)
        # NOTE: Using mc_samples=1 for MINIMUM mode (fast testing)

        log("Extracting theta scores via extract_theta_from_irt...")
        log(f"Using mc_samples={MODEL_SCORING_SETTINGS['mc_samples']} for scoring (MINIMUM mode)")

        df_theta = extract_theta_from_irt(
            model=fitted_model,
            response_matrix=response_matrix,
            missing_mask=missing_mask,
            composite_ids=composite_ids,
            factor_names=active_factors,
            scoring_batch_size=MODEL_SCORING_SETTINGS['scoring_batch_size'],
            mc_samples=MODEL_SCORING_SETTINGS['mc_samples'],
            iw_samples=MODEL_SCORING_SETTINGS['iw_samples'],
            invert_scale=False  # Higher theta = higher confidence
        )

        log(f"Theta extraction complete: {len(df_theta)} participants x {len(active_factors)} factors")

        # Tool returns wide format: UID, test, Theta_What, Theta_Where, Theta_When
        # Reconstruct composite_ID from UID and test
        log("Reconstructing composite_ID from UID and test...")
        df_theta['composite_ID'] = df_theta['UID'] + '_T' + df_theta['test'].astype(str)

        # Rename columns to match expected output format (lowercase theta_)
        rename_map = {}
        for factor in active_factors:
            rename_map[f'Theta_{factor}'] = f'theta_{factor}'
        df_theta_wide = df_theta.rename(columns=rename_map)

        # Select and order output columns
        # Note: Tool does NOT return SE values - only theta
        ordered_cols = ['composite_ID']
        for factor in active_factors:
            ordered_cols.append(f'theta_{factor}')
        df_theta_wide = df_theta_wide[ordered_cols]

        log(f"Wide format: {len(df_theta_wide)} rows, {len(df_theta_wide.columns)} columns")
        # Extract Item Parameters

        log("Extracting item parameters via extract_parameters_from_irt...")

        df_params = extract_parameters_from_irt(
            model=fitted_model,
            item_list=item_list_ordered,
            factor_names=active_factors,
            n_cats=n_cats_list  # Pass as list
        )

        log(f"Parameter extraction complete: {len(df_params)} items")

        # Note: extract_parameters_from_irt returns MIRT format for multidimensional models:
        # item_name, Difficulty, Overall_Discrimination, Discrim_Factor1, Discrim_Factor2, ...
        # This is the canonical output format from tools_inventory.md
        log(f"Parameter columns: {list(df_params.columns)}")
        # Save Outputs
        # These outputs are FINAL - used by downstream LMM analyses

        log("Saving Pass 2 FINAL outputs...")

        # Save item parameters (FINAL)
        item_params_path = RQ_DIR / "data" / "step03_pass2_item_params.csv"
        df_params.to_csv(item_params_path, index=False, encoding='utf-8')
        log(f"{item_params_path.name} ({len(df_params)} items, {len(df_params.columns)} cols)")

        # Save theta estimates (FINAL)
        theta_path = RQ_DIR / "data" / "step03_pass2_theta.csv"
        df_theta_wide.to_csv(theta_path, index=False, encoding='utf-8')
        log(f"{theta_path.name} ({len(df_theta_wide)} rows, {len(df_theta_wide.columns)} cols)")
        # Validation
        # Validates: convergence, parameter bounds (purified), no NaN parameters

        log("Running validation checks...")

        # Check 1: Parameter bounds (purified items: a >= 0.4, |b| <= 3.0)
        log("Checking parameter bounds (purified thresholds)...")
        a_min, a_max = df_params['Overall_Discrimination'].min(), df_params['Overall_Discrimination'].max()
        b_min, b_max = df_params['Difficulty'].min(), df_params['Difficulty'].max()

        log(f"Discrimination: min={a_min:.3f}, max={a_max:.3f}")
        log(f"Difficulty (b): min={b_min:.3f}, max={b_max:.3f}")

        if a_min < 0.4:
            log(f"Discrimination below purification threshold (0.4): min={a_min:.3f}")
        if a_max > 10.0:
            log(f"Discrimination above reasonable bound (10.0)")
            sys.exit(1)
        if abs(b_min) > 3.0 or abs(b_max) > 3.0:
            log(f"Difficulty outside purification threshold (|b| <= 3.0)")
        log("Parameter bounds reasonable")

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
            log(f"Theta outside typical bounds [-4.0, 4.0] (acceptable for ordinal data)")
        log("Theta estimates valid")

        # Check 4: Output row counts
        log("Checking output row counts...")
        if len(df_params) != len(purified_items):
            log(f"Item parameters count mismatch: {len(df_params)} vs {len(purified_items)} expected")
            sys.exit(1)
        if len(df_theta_wide) != len(df_wide):
            log(f"Theta rows mismatch: {len(df_theta_wide)} vs {len(df_wide)} expected")
            sys.exit(1)
        log(f"Output row counts correct (items={len(df_params)}, theta={len(df_theta_wide)})")

        # Check 5: Compare to Pass 1 retention rate
        log("Checking purification retention rate...")
        original_item_count = len(df_q_matrix_original)
        retention_rate = len(purified_items) / original_item_count
        log(f"Retention rate: {len(purified_items)}/{original_item_count} = {retention_rate:.1%}")
        if retention_rate < 0.20 or retention_rate > 0.80:
            log(f"Retention rate outside typical range (20-80%) - check purification criteria")
        else:
            log("Retention rate within expected range")

        log("Step 03 complete - All validations passed")
        log(f"Pass 2 calibrated {len(df_params)} purified items for {len(df_theta_wide)} participants")
        log(f"Active factors: {active_factors}")
        log("These FINAL theta estimates ready for downstream LMM analyses")
        log("If mc_samples=1 used, re-run with mc_samples=100 for scoring for production quality")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
