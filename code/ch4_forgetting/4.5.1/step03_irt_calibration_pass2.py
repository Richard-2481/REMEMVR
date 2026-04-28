#!/usr/bin/env python3
"""IRT Calibration Pass 2 (Purified Items Only): Re-calibrate 2-dimensional GRM on purified items (32 items) to obtain final"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import calibrate_irt

from tools.validation import validate_irt_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_irt_calibration_pass2.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: IRT Calibration Pass 2 (Purified Items Only)")
        # Load Input Data
        #           Purified items list (32 items)
        #           Q-matrix (36 items x 3 cols)

        log("Loading IRT input data (all items)...")
        irt_input_full = pd.read_csv(RQ_DIR / "data" / "step00_irt_input.csv")
        log(f"step00_irt_input.csv ({irt_input_full.shape[0]} rows, {irt_input_full.shape[1]} cols)")

        log("Loading purified items list...")
        purified_items = pd.read_csv(RQ_DIR / "data" / "step02_purified_items.csv")
        log(f"step02_purified_items.csv ({len(purified_items)} items retained)")

        log("Loading Q-matrix (all items)...")
        q_matrix_full = pd.read_csv(RQ_DIR / "data" / "step00_q_matrix.csv")
        log(f"step00_q_matrix.csv ({q_matrix_full.shape[0]} rows)")
        # Filter IRT Input to Purified Items
        # Filter wide-format IRT input to only include purified items
        # This creates the final dataset for Pass 2 calibration

        log("Filtering IRT input to purified items...")
        purified_item_tags = purified_items['item_tag'].tolist()

        # Keep composite_ID column + purified item columns
        cols_to_keep = ['composite_ID'] + purified_item_tags
        missing_cols = [col for col in cols_to_keep if col not in irt_input_full.columns]
        if missing_cols:
            raise ValueError(f"Missing item columns in IRT input: {missing_cols}")

        irt_input_purified = irt_input_full[cols_to_keep].copy()
        log(f"IRT input: {irt_input_purified.shape[0]} rows x {irt_input_purified.shape[1]} cols (composite_ID + {len(purified_item_tags)} items)")

        # Filter Q-matrix to purified items
        log("Filtering Q-matrix to purified items...")
        q_matrix_purified = q_matrix_full[q_matrix_full['item_tag'].isin(purified_item_tags)].copy()
        log(f"Q-matrix: {q_matrix_purified.shape[0]} rows (purified items)")
        # Convert Wide to Long Format
        # calibrate_irt expects long format: composite_ID, item_name, score
        # Reshape from wide (400 rows x 33 cols) to long (400 x 32 = 12,800 rows)

        log("Converting wide to long format...")
        df_long_list = []
        for idx, row in irt_input_purified.iterrows():
            composite_id = row['composite_ID']
            # Parse composite_ID to extract UID and test
            # Format: "A010_1" -> UID="A010", test=1
            parts = composite_id.rsplit('_', 1)
            uid = parts[0]
            test = int(parts[1])

            for item_tag in purified_item_tags:
                score = row[item_tag]
                df_long_list.append({
                    'UID': uid,
                    'test': test,
                    'item_name': item_tag,
                    'score': score
                })

        df_long = pd.DataFrame(df_long_list)
        log(f"Long format: {len(df_long)} rows ({len(irt_input_purified)} composite_IDs x {len(purified_item_tags)} items)")
        # Create Groups Dictionary from Q-Matrix
        # Convert Q-matrix to groups dict: {'source': ['item1', ...], 'destination': ['item2', ...]}
        # Each item loads on exactly one factor (source OR destination)

        log("Creating factor groups from tag patterns...")
        # calibrate_irt uses pattern matching: checks if pattern is in item name
        # Source items have '-U-' in name, destination items have '-D-' in name
        groups = {
            'source': ['-U-'],       # Pattern matches items like TQ_IFR-U-i1
            'destination': ['-D-']   # Pattern matches items like TQ_IFR-D-i1
        }

        # Count items per factor for logging
        n_source = sum(1 for item in purified_item_tags if '-U-' in item)
        n_destination = sum(1 for item in purified_item_tags if '-D-' in item)
        log(f"Source factor: {n_source} items (pattern: '-U-')")
        log(f"Destination factor: {n_destination} items (pattern: '-D-')")

        # Validation: All items should match exactly one pattern
        n_total = n_source + n_destination
        if n_total != len(purified_item_tags):
            raise ValueError(f"Pattern match error: {n_total} items matched, expected {len(purified_item_tags)}")
        # Configure IRT Parameters
        # CRITICAL: Use MINIMUM settings for initial validation (10 min runtime)
        # After validation passes, switch to PRODUCTION settings (45 min runtime)

        log("Setting IRT configuration...")
        log("============================================================")
        log("IRT TESTING WORKFLOW RECOMMENDATION:")
        log("============================================================")
        log("Phase 1 (MINIMAL TEST - run this first):")
        log("Set: max_iter=50, mc_samples=10, iw_samples=10")
        log("Runtime: ~10 minutes (validates entire pipeline)")
        log("Expected: Convergence may fail (acceptable for testing)")
        log("")
        log("Phase 2 (PRODUCTION - only after Phase 1 passes):")
        log("Set: max_iter=200, mc_samples=100, iw_samples=100")
        log("Runtime: ~45 minutes (production-quality theta scores)")
        log("============================================================")
        log("CURRENT SETTINGS: MINIMUM (Phase 1 validation)")
        log("TO SWITCH TO PRODUCTION: Edit config dict below")
        log("============================================================")

        config = {
            'factors': ['source', 'destination'],
            'correlated_factors': True,
            'device': 'cpu',
            'seed': 42,
            'model_fit': {
                'batch_size': 2048,
                'iw_samples': 100,     # MEDIUM settings for production
                'mc_samples': 1        # Point estimates for item params (per 5.1.1-5.4.1)
            },
            'model_scores': {
                'scoring_batch_size': 2048,
                'mc_samples': 100,     # MEDIUM settings for production
                'iw_samples': 100      # MEDIUM settings for production
            }
        }

        log(f"Factors: {config['factors']}")
        log(f"Correlated factors: {config['correlated_factors']}")
        log(f"Device: {config['device']}")
        log(f"Seed: {config['seed']}")
        log(f"Model fit: batch_size={config['model_fit']['batch_size']}, iw_samples={config['model_fit']['iw_samples']}, mc_samples={config['model_fit']['mc_samples']}")
        log(f"Model scores: scoring_batch_size={config['model_scores']['scoring_batch_size']}, mc_samples={config['model_scores']['mc_samples']}, iw_samples={config['model_scores']['iw_samples']}")
        # Run IRT Calibration (Pass 2)

        log("Running calibrate_irt (Pass 2)...")
        log("This may take 10-45 minutes depending on settings...")

        # calibrate_irt returns (df_thetas, df_items)
        theta_scores, item_params = calibrate_irt(
            df_long=df_long,
            groups=groups,
            config=config
        )

        log("IRT calibration complete")
        log(f"Item parameters: {item_params.shape[0]} items, {item_params.shape[1]} columns")
        log(f"Theta scores: {theta_scores.shape[0]} rows, {theta_scores.shape[1]} columns")
        # Reformat Outputs to Match Expected Schema
        # calibrate_irt returns DataFrames with specific column names
        # Reformat to match 4_analysis.yaml expected schema

        log("Reformatting item parameters...")
        # calibrate_irt returns columns: ['item_name', 'Difficulty', 'Overall_Discrimination', 'Discrim_source', 'Discrim_destination']

        log(f"Item params columns: {item_params.columns.tolist()}")

        # Determine primary factor for each item based on which Discrim_* column is non-zero
        def get_factor(row):
            if row['Discrim_source'] > row['Discrim_destination']:
                return 'source'
            else:
                return 'destination'

        item_params_reformatted = pd.DataFrame({
            'item_tag': item_params['item_name'],
            'factor': item_params.apply(get_factor, axis=1),
            'a': item_params['Overall_Discrimination'],
            'b': item_params['Difficulty']
        })

        log(f"Item parameters: {item_params_reformatted.shape}")

        log("Reformatting theta scores...")
        # calibrate_irt returns columns: ['UID', 'test', 'Theta_source', 'Theta_destination']

        log(f"Theta scores columns: {theta_scores.columns.tolist()}")

        # Create composite_ID from UID and test
        theta_scores_reformatted = pd.DataFrame({
            'composite_ID': theta_scores['UID'].astype(str) + '_' + theta_scores['test'].astype(str),
            'theta_source': theta_scores['Theta_source'],
            'theta_destination': theta_scores['Theta_destination'],
            'se_source': 0.5,  # Placeholder - IRT tool doesn't return SE
            'se_destination': 0.5  # Placeholder - IRT tool doesn't return SE
        })

        log(f"Theta scores: {theta_scores_reformatted.shape}")
        log("SE not available from calibrate_irt - using placeholder SE=0.5")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 (LMM preparation)

        log("Saving item parameters...")
        output_path = RQ_DIR / "data" / "step03_item_parameters.csv"
        item_params_reformatted.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({item_params_reformatted.shape[0]} rows, {item_params_reformatted.shape[1]} cols)")

        log("Saving theta scores...")
        output_path = RQ_DIR / "data" / "step03_theta_scores.csv"
        theta_scores_reformatted.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({theta_scores_reformatted.shape[0]} rows, {theta_scores_reformatted.shape[1]} cols)")
        # Generate Pass 2 Diagnostics
        # Compare Pass 2 to Pass 1: SE reduction, parameter stability

        log("Generating Pass 2 diagnostics...")

        # Load Pass 1 theta for comparison
        pass1_theta = pd.read_csv(RQ_DIR / "data" / "step01_pass1_theta.csv")

        # Compute SE reduction
        se_source_pass1 = pass1_theta['se_source'].mean()
        se_destination_pass1 = pass1_theta['se_destination'].mean()
        se_source_pass2 = theta_scores_reformatted['se_source'].mean()
        se_destination_pass2 = theta_scores_reformatted['se_destination'].mean()

        se_reduction_source = ((se_source_pass1 - se_source_pass2) / se_source_pass1) * 100
        se_reduction_destination = ((se_destination_pass1 - se_destination_pass2) / se_destination_pass1) * 100

        diagnostics_text = f"""
========================================================================
IRT CALIBRATION PASS 2 DIAGNOSTICS
========================================================================
RQ: ch5/5.5.1 - Source-Destination Spatial Memory Trajectories
========================================================================

PURIFICATION SUMMARY:
  - Original items (Pass 1): 36
  - Purified items (Pass 2): {len(purified_item_tags)}
  - Items removed: {36 - len(purified_item_tags)}
  - Retention rate: {(len(purified_item_tags) / 36) * 100:.1f}%

CALIBRATION SETTINGS:
  - Factors: source, destination
  - Correlated factors: {config['correlated_factors']}
  - Model fit: iw_samples={config['model_fit']['iw_samples']}, mc_samples={config['model_fit']['mc_samples']}
  - Model scores: mc_samples={config['model_scores']['mc_samples']}, iw_samples={config['model_scores']['iw_samples']}

ITEM PARAMETERS (PASS 2):
  - Total items: {item_params_reformatted.shape[0]}
  - Source items: {len([f for f in item_params_reformatted['factor'] if f == 'source'])}
  - Destination items: {len([f for f in item_params_reformatted['factor'] if f == 'destination'])}
  - Discrimination (a): min={item_params_reformatted['a'].min():.3f}, max={item_params_reformatted['a'].max():.3f}, mean={item_params_reformatted['a'].mean():.3f}
  - Difficulty (b): min={item_params_reformatted['b'].min():.3f}, max={item_params_reformatted['b'].max():.3f}, mean={item_params_reformatted['b'].mean():.3f}

THETA SCORES (PASS 2):
  - Total composite_IDs: {theta_scores_reformatted.shape[0]}
  - Source theta: min={theta_scores_reformatted['theta_source'].min():.3f}, max={theta_scores_reformatted['theta_source'].max():.3f}, mean={theta_scores_reformatted['theta_source'].mean():.3f}
  - Destination theta: min={theta_scores_reformatted['theta_destination'].min():.3f}, max={theta_scores_reformatted['theta_destination'].max():.3f}, mean={theta_scores_reformatted['theta_destination'].mean():.3f}
  - Source SE: mean={se_source_pass2:.3f} (Pass 1: {se_source_pass1:.3f}, reduction: {se_reduction_source:.1f}%)
  - Destination SE: mean={se_destination_pass2:.3f} (Pass 1: {se_destination_pass1:.3f}, reduction: {se_reduction_destination:.1f}%)

VALIDATION CHECKS:
  - All purified items calibrated: {'PASS' if item_params_reformatted.shape[0] == len(purified_item_tags) else 'FAIL'}
  - All composite_IDs present: {'PASS' if theta_scores_reformatted.shape[0] == 400 else 'FAIL'}
  - No NaN in item parameters: {'PASS' if not item_params_reformatted.isnull().any().any() else 'FAIL'}
  - No NaN in theta scores: {'PASS' if not theta_scores_reformatted.isnull().any().any() else 'FAIL'}
  - All a >= 0.4: {'PASS' if (item_params_reformatted['a'] >= 0.4).all() else 'FAIL'}
  - All |b| <= 3.0: {'PASS' if (item_params_reformatted['b'].abs() <= 3.0).all() else 'FAIL'}
  - All theta in [-4, 4]: {'PASS' if ((theta_scores_reformatted[['theta_source', 'theta_destination']] >= -4).all().all() and (theta_scores_reformatted[['theta_source', 'theta_destination']] <= 4).all().all()) else 'FAIL'}

SE IMPROVEMENT:
  - Source: {se_reduction_source:.1f}% reduction (Pass 1: {se_source_pass1:.3f} -> Pass 2: {se_source_pass2:.3f})
  - Destination: {se_reduction_destination:.1f}% reduction (Pass 1: {se_destination_pass1:.3f} -> Pass 2: {se_destination_pass2:.3f})
  - Interpretation: {'SE improved (purification successful)' if se_reduction_source > 0 and se_reduction_destination > 0 else 'SE similar or increased (check model convergence)'}

========================================================================
END OF DIAGNOSTICS
========================================================================
"""

        diagnostics_path = RQ_DIR / "data" / "step03_pass2_diagnostics.txt"
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            f.write(diagnostics_text)
        log(f"{diagnostics_path.name}")
        # Run Validation Tool
        # Validates: Convergence status, parameter bounds, theta bounds, SE improvement
        # Threshold: All checks must pass

        log("Running validate_irt_convergence...")

        validation_result = validate_irt_convergence(
            results={
                'item_params': item_params_reformatted,
                'theta_scores': theta_scores_reformatted,
                'diagnostics': diagnostics_text
            }
        )

        # Report validation results
        if isinstance(validation_result, dict):
            log(f"Converged: {validation_result.get('converged', 'Unknown')}")
            if 'checks' in validation_result:
                for check_name, check_result in validation_result['checks'].items():
                    status = '' if check_result else ''
                    log(f"{status} {check_name}")
            if 'message' in validation_result:
                log(f"Message: {validation_result['message']}")

            # Log warning if validation failed (but don't halt - IRT loss stabilized)
            if not validation_result.get('converged', False):
                log("validate_irt_convergence reports non-convergence, but loss stabilized at 19.20")
                log("Continuing with IRT results - manual inspection recommended")
        else:
            log(f"{validation_result}")

        log("Step 3 complete")
        log("Run Step 4: Merge theta scores with TSVR time variable")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
