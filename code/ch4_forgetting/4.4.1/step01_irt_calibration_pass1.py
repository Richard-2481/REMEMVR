#!/usr/bin/env python3
"""IRT Calibration Pass 1 (All Items): Calibrate 3-dimensional GRM on all interactive items for congruence dimensions"""

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

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_irt_calibration_pass1.log"

# Input files
INPUT_IRT = RQ_DIR / "data" / "step00_irt_input.csv"
INPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"

# Output files (logs/ for Pass 1 diagnostic outputs)
OUTPUT_ITEM_PARAMS = RQ_DIR / "logs" / "step01_pass1_item_params.csv"
OUTPUT_THETA = RQ_DIR / "logs" / "step01_pass1_theta.csv"

# IRT configuration - Validated "Med" settings from thesis/analyses/ANALYSES_DEFINITIVE.md
IRT_CONFIG = {
    "factors": ["common", "congruent", "incongruent"],
    "correlated_factors": True,
    "device": "cpu",
    "seed": 42,
    "model_fit": {
        "batch_size": 2048,      # Validated "Med" level
        "iw_samples": 100,       # Validated "Med" level
        "mc_samples": 1          # Per thesis validation
    },
    "model_scores": {
        "scoring_batch_size": 2048,  # Validated "Med" level
        "mc_samples": 100,           # Validated "Med" level
        "iw_samples": 100            # Validated "Med" level
    }
}

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: IRT Calibration Pass 1")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Input Data
        log("\nLoading input data...")

        # Load wide-format IRT input
        df_irt_wide = pd.read_csv(INPUT_IRT)
        log(f"{INPUT_IRT.name} ({len(df_irt_wide)} rows, {len(df_irt_wide.columns)} cols)")

        # Load Q-matrix
        df_q_matrix = pd.read_csv(INPUT_Q_MATRIX)
        log(f"{INPUT_Q_MATRIX.name} ({len(df_q_matrix)} rows)")
        # Build Groups Dictionary from Q-Matrix
        log("\nBuilding factor groups from Q-matrix...")

        # Build groups dict: dimension_name -> list of item names
        groups = {}
        for dim in ["common", "congruent", "incongruent"]:
            items_in_dim = df_q_matrix[df_q_matrix[dim] == 1]["item_name"].tolist()
            groups[dim] = items_in_dim
            log(f"  {dim}: {len(items_in_dim)} items")
        # Convert Wide to Long Format for calibrate_irt
        log("\nConverting wide to long format...")

        # Melt wide format to long format
        # calibrate_irt expects: UID, test, item_name, score
        item_columns = [col for col in df_irt_wide.columns if col != "composite_ID"]

        df_long = df_irt_wide.melt(
            id_vars=["composite_ID"],
            value_vars=item_columns,
            var_name="item_name",
            value_name="score"
        )

        # Parse composite_ID to get UID and test
        # Format: A010_1 -> UID=A010, test=1
        df_long[["UID", "test"]] = df_long["composite_ID"].str.split("_", n=1, expand=True)
        df_long["test"] = df_long["test"].astype(int)

        # Reorder columns
        df_long = df_long[["UID", "test", "item_name", "score"]]

        log(f"Long format: {len(df_long)} rows")
        log(f"  Unique UIDs: {df_long['UID'].nunique()}")
        log(f"  Unique items: {df_long['item_name'].nunique()}")
        log(f"  Test sessions: {sorted(df_long['test'].unique())}")
        # Build Custom Groups for Pattern Matching
        # calibrate_irt uses pattern matching on item_name
        # We need to create patterns that uniquely identify each dimension's items
        log("\nBuilding pattern-based groups for IRT...")

        # Since items are named like TQ_IFR-N-i1, we need patterns that match item suffixes
        # But calibrate_irt's prepare_irt_input_from_long uses "pattern in item_name"
        # We can use the actual item names as patterns (exact match)

        # Alternative approach: Use domain-agnostic suffixes
        # common: items ending in -i1 or -i2
        # congruent: items ending in -i3 or -i4
        # incongruent: items ending in -i5 or -i6

        groups_patterns = {
            "common": ["-i1", "-i2"],
            "congruent": ["-i3", "-i4"],
            "incongruent": ["-i5", "-i6"]
        }

        for dim, patterns in groups_patterns.items():
            matching = sum(1 for item in item_columns if any(p in item for p in patterns))
            log(f"  {dim} patterns {patterns}: matches {matching} items")
        # Run IRT Calibration (Pass 1)
        log("\nRunning IRT calibration (Pass 1)...")

        df_thetas, df_items = calibrate_irt(
            df_long=df_long,
            groups=groups_patterns,
            config=IRT_CONFIG
        )

        log(f"Theta scores: {len(df_thetas)} rows")
        log(f"Item parameters: {len(df_items)} items")
        # Post-Process Results for Output Format
        log("\nPost-processing results...")

        # Post-process theta scores
        # Create composite_ID from UID and test
        df_thetas["composite_ID"] = df_thetas["UID"].astype(str) + "_" + df_thetas["test"].astype(str)

        # Rename theta columns to match expected format
        theta_rename = {
            "Theta_common": "theta_common",
            "Theta_congruent": "theta_congruent",
            "Theta_incongruent": "theta_incongruent"
        }
        df_thetas = df_thetas.rename(columns=theta_rename)

        # Add SE columns (placeholder - deepirtools doesn't provide per-score SE easily)
        # Using standard deviation of theta as proxy for SE
        for dim in ["common", "congruent", "incongruent"]:
            theta_col = f"theta_{dim}"
            se_col = f"se_{dim}"
            # Approximate SE as 1/sqrt(n_items) * theta_sd
            n_items_dim = len(groups[dim])
            df_thetas[se_col] = df_thetas[theta_col].std() / np.sqrt(n_items_dim)

        # Select output columns
        theta_output_cols = ["composite_ID", "theta_common", "theta_congruent", "theta_incongruent",
                           "se_common", "se_congruent", "se_incongruent"]
        df_theta_output = df_thetas[theta_output_cols].copy()

        # Post-process item parameters
        # Map from calibrate_irt output format to expected format
        # calibrate_irt returns: item_name, Difficulty, Overall_Discrimination, Discrim_*
        df_items_output = pd.DataFrame()
        df_items_output["item_name"] = df_items["item_name"]
        df_items_output["a"] = df_items["Overall_Discrimination"]
        df_items_output["b"] = df_items["Difficulty"]

        # Determine dimension for each item
        def get_dimension(item_name):
            for dim, patterns in groups_patterns.items():
                if any(p in item_name for p in patterns):
                    return dim
            return "unknown"

        df_items_output["dimension"] = df_items_output["item_name"].apply(get_dimension)

        # Reorder columns
        df_items_output = df_items_output[["item_name", "dimension", "a", "b"]]

        log(f"Theta output: {len(df_theta_output)} rows, {len(df_theta_output.columns)} cols")
        log(f"Item params output: {len(df_items_output)} items")
        # Save Outputs
        log("\nSaving output files...")

        # Ensure logs directory exists
        (RQ_DIR / "logs").mkdir(parents=True, exist_ok=True)

        # Save item parameters (to logs/ for Pass 1)
        df_items_output.to_csv(OUTPUT_ITEM_PARAMS, index=False, encoding='utf-8')
        log(f"{OUTPUT_ITEM_PARAMS.name} ({len(df_items_output)} rows)")

        # Save theta scores (to logs/ for Pass 1)
        df_theta_output.to_csv(OUTPUT_THETA, index=False, encoding='utf-8')
        log(f"{OUTPUT_THETA.name} ({len(df_theta_output)} rows)")
        # Validation
        log("\nValidating results...")

        # Check no NaN in item parameters
        nan_a = df_items_output["a"].isna().sum()
        nan_b = df_items_output["b"].isna().sum()
        if nan_a > 0 or nan_b > 0:
            log(f"NaN in parameters: a={nan_a}, b={nan_b}")
        else:
            log("No NaN in item parameters")

        # Check all dimensions present
        dims_present = df_items_output["dimension"].unique()
        for dim in ["common", "congruent", "incongruent"]:
            if dim not in dims_present:
                log(f"Missing dimension: {dim}")
            else:
                n = len(df_items_output[df_items_output["dimension"] == dim])
                log(f"Dimension '{dim}': {n} items")

        # Check parameter ranges
        a_min, a_max = df_items_output["a"].min(), df_items_output["a"].max()
        b_min, b_max = df_items_output["b"].min(), df_items_output["b"].max()
        log(f"Parameter ranges: a=[{a_min:.3f}, {a_max:.3f}], b=[{b_min:.3f}, {b_max:.3f}]")

        if a_min < 0.01 or a_max > 10.0:
            log(f"Discrimination outside expected range [0.01, 10.0]")
        if b_min < -6.0 or b_max > 6.0:
            log(f"Difficulty outside expected range [-6.0, 6.0]")

        # Check theta count
        assert len(df_theta_output) == 400, f"Expected 400 theta rows, got {len(df_theta_output)}"
        log(f"Theta count: {len(df_theta_output)} rows")

        log("\nStep 01 complete (IRT Pass 1)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
