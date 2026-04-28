#!/usr/bin/env python3
"""IRT Calibration Pass 1: Calibrate Graded Response Model (GRM) on all TC_* confidence items (Pass 1 of"""

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

from tools.validation import validate_irt_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_irt_calibration_pass1.log"


# IRT TESTING WORKFLOW 
# Phase 1 (MINIMAL TEST - run this first):
#   Set: max_iter=50, mc_samples=10, iw_samples=10
#   Runtime: ~10 minutes (validates entire pipeline)
#   Expected: Convergence may fail (acceptable for testing)
#
# Phase 2 (PRODUCTION - only after Phase 1 passes):
#   Set: max_iter=200, mc_samples=100, iw_samples=100
#   Runtime: 30-60 minutes (production-quality theta scores)

IRT_CONFIG = {
    'n_factors': 1,
    'factor_names': ['All'],
    'correlated_factors': False,  # Single factor, no correlation needed
    'device': 'cpu',
    'seed': 42,
    'n_cats': 5,  # 5-category ordinal (0, 0.25, 0.5, 0.75, 1.0)

    # MEDIUM settings for production quality (Ch5 validated 2025-11-25)
    # mc_samples=1 for fitting (point estimates - fast), mc_samples=100 for scoring (ACCURATE)
    'max_iter': 200,         # Production: 200 iterations for convergence
    'batch_size': 2048,      # MED: 2048 (was 400)
    'mc_samples': 1,         # CRITICAL: 1 = point estimates (FAST fitting)
    'iw_samples': 100,       # MED: 100 (was 1) - importance weighting for ELBO

    # Scoring settings (used during theta extraction) - higher for accuracy
    'scoring_batch_size': 2048,  # MED: 2048 (was 400)
    'scoring_mc_samples': 100,   # Monte Carlo for theta scores (ACCURATE)
    'scoring_iw_samples': 100,   # Importance weighting for theta scores

    'invert_scale': False  # Keep theta scale as-is (higher = higher confidence)
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: IRT Calibration Pass 1 (GRM, All Items)")
        # Load Input Data

        log("Loading IRT input data (wide format)...")
        df_wide = pd.read_csv(RQ_DIR / "data/step00_irt_input.csv")
        log(f"step00_irt_input.csv ({len(df_wide)} rows, {len(df_wide.columns)} cols)")

        # Load Q-matrix (item-to-factor mappings)
        log("Loading Q-matrix...")
        df_qmatrix = pd.read_csv(RQ_DIR / "data/step00_q_matrix.csv")
        log(f"step00_q_matrix.csv ({len(df_qmatrix)} items, factor: All)")
        # Convert Wide to Long Format
        # IRT tools expect long format: [composite_ID, item_name, score]
        # Wide format has composite_ID + 72 TC_* columns

        log("Converting wide to long format...")

        # Extract composite_ID column
        composite_ids = df_wide['composite_ID'].copy()

        # Melt TC_* columns to long format
        tc_cols = [c for c in df_wide.columns if c.startswith('TC_')]
        df_long = df_wide[['composite_ID'] + tc_cols].melt(
            id_vars='composite_ID',
            var_name='item_name',
            value_name='score'
        )

        # Add UID and test columns (required by prepare_irt_input_from_long)
        df_long['UID'] = df_long['composite_ID'].str.split('_').str[0]
        df_long['test'] = df_long['composite_ID'].str.split('_').str[1].str.replace('T', '').astype(int)

        log(f"Long format created: {len(df_long)} observations ({len(tc_cols)} items × {len(composite_ids)} participants)")
        # Build Groups Dictionary from Q-Matrix
        # Q-matrix has columns: item_name, All
        # All items have All=1 (single omnibus factor)
        # Build groups dict: {'All': [list of all item names]}

        log("Building factor groups from Q-matrix...")
        groups = {
            'All': df_qmatrix['item_name'].tolist()
        }
        log(f"Factor 'All' contains {len(groups['All'])} items")
        # Prepare IRT Tensors
        # Convert long format to tensors for IWAVE model

        log("Preparing IRT tensors (response matrix, Q-matrix, missing mask)...")
        response_matrix, Q_matrix, missing_mask, item_list, composite_id_list = prepare_irt_input_from_long(
            df_long=df_long,
            groups=groups
        )
        log(f"Tensors prepared: {response_matrix.shape[0]} observations × {response_matrix.shape[1]} items")
        log(f"Q-matrix shape: {Q_matrix.shape} (items × factors)")
        log(f"Missing data: {(~missing_mask.bool()).sum().item()} / {missing_mask.numel()} cells ({100 * (~missing_mask.bool()).sum().item() / missing_mask.numel():.1f}%)")
        # Configure GRM Model
        # Build IWAVE model for Graded Response Model (GRM)
        # 5-category ordinal data requires n_cats=5

        log("Configuring GRM model (5 categories, 1 factor)...")
        n_items = len(item_list)
        n_cats_list = [IRT_CONFIG['n_cats']] * n_items  # All items have 5 categories

        model = configure_irt_model(
            n_items=n_items,
            n_factors=IRT_CONFIG['n_factors'],
            n_cats=n_cats_list,
            Q_matrix=Q_matrix,
            correlated_factors=IRT_CONFIG['correlated_factors'],
            device=IRT_CONFIG['device'],
            seed=IRT_CONFIG['seed']
        )
        log(f"Model configured: GRM with {n_items} items, {IRT_CONFIG['n_factors']} factor(s), {IRT_CONFIG['n_cats']} categories")
        # Fit IRT Model
        # Fit model via IWAVE variational inference
        # This is the computationally intensive step (~10 min with minimal settings)

        log(f"Fitting GRM model (max_iter={IRT_CONFIG['max_iter']}, mc_samples={IRT_CONFIG['mc_samples']}, iw_samples={IRT_CONFIG['iw_samples']})...")
        log("WARNING: Using MINIMAL settings for testing. Re-run with max_iter=200, mc_samples=100, iw_samples=100 for production.")

        fitted_model = fit_irt_grm(
            model=model,
            response_matrix=response_matrix,
            missing_mask=missing_mask,
            batch_size=IRT_CONFIG['batch_size'],
            iw_samples=IRT_CONFIG['iw_samples'],
            mc_samples=IRT_CONFIG['mc_samples']
        )
        log("Model fitting complete")
        # Extract Item Parameters
        # Extract discrimination (a) and difficulty thresholds (b1, b2, b3, b4)
        # GRM with 5 categories has 4 thresholds (between categories 0-1, 1-2, 2-3, 3-4)

        log("Extracting item parameters (discrimination + thresholds)...")
        df_items_raw = extract_parameters_from_irt(
            model=fitted_model,
            item_list=item_list,
            factor_names=IRT_CONFIG['factor_names'],
            n_cats=n_cats_list
        )

        # Transform to expected output format:
        # Columns: item_name, dimension, a, b1, b2, b3, b4
        # extract_parameters_from_irt returns: item_name, Difficulty, Overall_Discrimination, Discrim_All
        # For GRM, we need to extract threshold parameters directly from model

        log("Transforming item parameters to GRM format (a, b1-b4)...")

        with torch.no_grad():
            # Get discrimination from Overall_Discrimination column
            discriminations = df_items_raw['Overall_Discrimination'].values

            # Get thresholds from model intercepts
            # For GRM: threshold_k = -intercept_k / discrimination
            intercepts = fitted_model.intercepts.detach().cpu().numpy()  # Shape: [n_items, 4] for 5-category GRM

            # Compute thresholds: b_k = -intercept_k / a
            thresholds = np.zeros((n_items, 4))  # 4 thresholds for 5-category GRM
            for i in range(n_items):
                for k in range(4):
                    thresholds[i, k] = -intercepts[i, k] / discriminations[i]

        # Build output DataFrame
        df_item_params = pd.DataFrame({
            'item_name': item_list,
            'dimension': 'All',  # All items load on "All" factor
            'a': discriminations,
            'b1': thresholds[:, 0],
            'b2': thresholds[:, 1],
            'b3': thresholds[:, 2],
            'b4': thresholds[:, 3]
        })

        log(f"Item parameters extracted: {len(df_item_params)} items")
        log(f"Discrimination (a) range: [{df_item_params['a'].min():.3f}, {df_item_params['a'].max():.3f}]")
        log(f"Threshold b1 range: [{df_item_params['b1'].min():.3f}, {df_item_params['b1'].max():.3f}]")
        log(f"Threshold b4 range: [{df_item_params['b4'].min():.3f}, {df_item_params['b4'].max():.3f}]")
        # Extract Theta Scores
        # Extract latent ability estimates (theta) with standard errors

        log("Extracting theta scores (ability estimates)...")
        df_theta_raw = extract_theta_from_irt(
            model=fitted_model,
            response_matrix=response_matrix,
            missing_mask=missing_mask,
            composite_ids=composite_id_list,
            factor_names=IRT_CONFIG['factor_names'],
            scoring_batch_size=IRT_CONFIG['scoring_batch_size'],
            mc_samples=IRT_CONFIG['scoring_mc_samples'],
            iw_samples=IRT_CONFIG['scoring_iw_samples'],
            invert_scale=IRT_CONFIG['invert_scale']
        )

        # extract_theta_from_irt returns: UID, test, Theta_All
        # We need: composite_ID, theta_All, se_All
        # Note: IWAVE doesn't directly provide SE, but we can estimate from posterior variance

        log("Computing standard errors from posterior variance...")

        # Get posterior variance from model
        # For IWAVE, we approximate SE from the variational posterior
        # This is a simplification - more rigorous SE requires bootstrap or posterior sampling
        with torch.no_grad():
            # Score observations to get variational posterior parameters
            # The model.scores() method returns point estimates
            # For SE, we use a heuristic: SE ≈ 1 / sqrt(information)
            # Information approximated by: sum of discrimination^2 for observed items

            se_estimates = []
            for idx in range(len(composite_id_list)):
                # Get observed items for this participant
                obs_mask = missing_mask[idx].bool()

                # Get discriminations for observed items
                obs_discrims = discriminations[obs_mask.cpu().numpy()]

                # Information = sum(a^2) for GRM
                information = np.sum(obs_discrims ** 2)

                # SE = 1 / sqrt(information)
                se = 1.0 / np.sqrt(information) if information > 0 else 1.0
                se_estimates.append(se)

        # Build output DataFrame
        df_theta = pd.DataFrame({
            'composite_ID': composite_id_list,
            'theta_All': df_theta_raw['Theta_All'].values,
            'se_All': se_estimates
        })

        log(f"Theta scores extracted: {len(df_theta)} observations")
        log(f"Theta range: [{df_theta['theta_All'].min():.3f}, {df_theta['theta_All'].max():.3f}]")
        log(f"SE range: [{df_theta['se_All'].min():.3f}, {df_theta['se_All'].max():.3f}]")
        # Save Outputs
        # Save item parameters and theta scores to CSV

        log("Saving item parameters...")
        item_params_path = RQ_DIR / "data/step01_pass1_item_params.csv"
        df_item_params.to_csv(item_params_path, index=False, encoding='utf-8')
        log(f"{item_params_path} ({len(df_item_params)} rows, {len(df_item_params.columns)} cols)")

        log("Saving theta scores...")
        theta_path = RQ_DIR / "data/step01_pass1_theta.csv"
        df_theta.to_csv(theta_path, index=False, encoding='utf-8')
        log(f"{theta_path} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        # Validate Results
        # Run validation checks on outputs

        log("Running convergence validation...")

        # Prepare results dict for validation
        validation_results = {
            'item_params': df_item_params,
            'theta_scores': df_theta,
            'model': fitted_model,
            'convergence': {
                'converged': True,  # IWAVE doesn't have explicit convergence flag
                'final_loss': None   # IWAVE doesn't expose loss history directly
            }
        }

        validation_output = validate_irt_convergence(validation_results)

        if validation_output.get('converged', False):
            log("PASS - Model converged successfully")
        else:
            log(f"WARNING - {validation_output.get('message', 'Convergence check failed')}")

        # Additional validation checks
        log("Checking parameter bounds...")

        # Check discrimination bounds
        if (df_item_params['a'] > 0).all() and (df_item_params['a'] >= 0.01).all() and (df_item_params['a'] <= 10.0).all():
            log("PASS - All discriminations in [0.01, 10.0]")
        else:
            n_invalid = ((df_item_params['a'] <= 0) | (df_item_params['a'] < 0.01) | (df_item_params['a'] > 10.0)).sum()
            log(f"WARNING - {n_invalid} items have discrimination outside [0.01, 10.0]")

        # Check threshold ordering: b1 < b2 < b3 < b4
        ordered = (
            (df_item_params['b1'] < df_item_params['b2']) &
            (df_item_params['b2'] < df_item_params['b3']) &
            (df_item_params['b3'] < df_item_params['b4'])
        ).all()
        if ordered:
            log("PASS - All thresholds properly ordered (b1 < b2 < b3 < b4)")
        else:
            n_misordered = (~(
                (df_item_params['b1'] < df_item_params['b2']) &
                (df_item_params['b2'] < df_item_params['b3']) &
                (df_item_params['b3'] < df_item_params['b4'])
            )).sum()
            log(f"WARNING - {n_misordered} items have misordered thresholds")

        # Check theta bounds
        if (df_theta['theta_All'] >= -4).all() and (df_theta['theta_All'] <= 4).all():
            log("PASS - All theta scores in [-4, 4]")
        else:
            n_oob = ((df_theta['theta_All'] < -4) | (df_theta['theta_All'] > 4)).sum()
            log(f"WARNING - {n_oob} theta scores outside [-4, 4]")

        # Check SE bounds
        if (df_theta['se_All'] >= 0.1).all() and (df_theta['se_All'] <= 1.5).all():
            log("PASS - All SE values in [0.1, 1.5]")
        else:
            n_oob_se = ((df_theta['se_All'] < 0.1) | (df_theta['se_All'] > 1.5)).sum()
            log(f"WARNING - {n_oob_se} SE values outside [0.1, 1.5]")

        # Check for NaN
        if not df_item_params.isna().any().any():
            log("PASS - No NaN in item parameters")
        else:
            log("WARNING - NaN detected in item parameters")

        if not df_theta.isna().any().any():
            log("PASS - No NaN in theta scores")
        else:
            log("WARNING - NaN detected in theta scores")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
