#!/usr/bin/env python3
"""Merge Theta Scores with TSVR (Decision D070): Merge theta scores with TSVR time variable and reshape to long format for LMM."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"

# Input files
INPUT_THETA = RQ_DIR / "data" / "step03_theta_scores.csv"
INPUT_TSVR = RQ_DIR / "data" / "step00_tsvr_mapping.csv"

# Output files
OUTPUT_LMM_INPUT = RQ_DIR / "data" / "step04_lmm_input.csv"

# Congruence categories
CONGRUENCE_CATEGORIES = ["common", "congruent", "incongruent"]
REFERENCE_CATEGORY = "common"

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Merge Theta Scores with TSVR")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Input Data
        log("\nLoading input data...")

        # Load theta scores
        df_theta = pd.read_csv(INPUT_THETA)
        log(f"{INPUT_THETA.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")

        # Load TSVR mapping
        df_tsvr = pd.read_csv(INPUT_TSVR)
        log(f"{INPUT_TSVR.name} ({len(df_tsvr)} rows)")
        # Merge Theta with TSVR
        log("\nMerging theta scores with TSVR mapping...")

        # Merge on composite_ID
        df_merged = df_theta.merge(
            df_tsvr[["composite_ID", "UID", "test", "TSVR_hours"]],
            on="composite_ID",
            how="left"
        )

        # Check for missing TSVR
        missing_tsvr = df_merged["TSVR_hours"].isna().sum()
        if missing_tsvr > 0:
            log(f"{missing_tsvr} rows missing TSVR data")
        else:
            log(f"All {len(df_merged)} rows have TSVR data")

        log(f"{len(df_merged)} rows")
        # Melt to Long Format
        log("\nReshaping to long format...")

        # Prepare melt columns
        theta_cols = ["theta_common", "theta_congruent", "theta_incongruent"]
        se_cols = ["se_common", "se_congruent", "se_incongruent"]

        # Melt theta columns
        df_theta_long = df_merged.melt(
            id_vars=["composite_ID", "UID", "test", "TSVR_hours"],
            value_vars=theta_cols,
            var_name="theta_dim",
            value_name="theta"
        )

        # Extract congruence category from column name
        df_theta_long["congruence"] = df_theta_long["theta_dim"].str.replace("theta_", "")

        # Melt SE columns separately
        df_se_long = df_merged.melt(
            id_vars=["composite_ID"],
            value_vars=se_cols,
            var_name="se_dim",
            value_name="se"
        )
        df_se_long["congruence"] = df_se_long["se_dim"].str.replace("se_", "")

        # Merge theta and SE
        df_long = df_theta_long.merge(
            df_se_long[["composite_ID", "congruence", "se"]],
            on=["composite_ID", "congruence"],
            how="left"
        )

        log(f"{len(df_long)} rows (expected ~1200)")
        # Create Time Transformations
        log("\nCreating time transformations...")

        # TSVR squared
        df_long["TSVR_sq"] = df_long["TSVR_hours"] ** 2

        # TSVR log (add 1 to avoid log(0))
        df_long["TSVR_log"] = np.log(df_long["TSVR_hours"] + 1)

        log(f"  TSVR_hours range: [{df_long['TSVR_hours'].min():.2f}, {df_long['TSVR_hours'].max():.2f}]")
        log(f"  TSVR_sq range: [{df_long['TSVR_sq'].min():.2f}, {df_long['TSVR_sq'].max():.2f}]")
        log(f"  TSVR_log range: [{df_long['TSVR_log'].min():.3f}, {df_long['TSVR_log'].max():.3f}]")
        # Set Congruence as Categorical with Reference
        log("\nSetting congruence as categorical...")

        # Create ordered categorical with reference first
        category_order = + [c for c in CONGRUENCE_CATEGORIES if c != REFERENCE_CATEGORY]
        df_long["congruence"] = pd.Categorical(
            df_long["congruence"],
            categories=category_order,
            ordered=True
        )

        log(f"  Congruence categories: {category_order}")
        log(f"  Reference category: {REFERENCE_CATEGORY}")
        # Select and Order Output Columns
        log("\nPreparing output columns...")

        output_cols = [
            "UID", "composite_ID", "test", "congruence",
            "theta", "se", "TSVR_hours", "TSVR_sq", "TSVR_log"
        ]
        df_output = df_long[output_cols].copy()

        # Sort for readability
        df_output = df_output.sort_values(
            ["UID", "test", "congruence"]
        ).reset_index(drop=True)

        log(f"{len(df_output)} rows, {len(df_output.columns)} cols")
        # Save Output
        log("\nSaving output file...")

        df_output.to_csv(OUTPUT_LMM_INPUT, index=False, encoding='utf-8')
        log(f"{OUTPUT_LMM_INPUT.name} ({len(df_output)} rows)")
        # Validation
        log("\nValidating output...")

        # Check row count
        expected_rows = 400 * 3  # 400 composite_IDs x 3 congruence levels
        if len(df_output) != expected_rows:
            log(f"Row count {len(df_output)} != expected {expected_rows}")
        else:
            log(f"Row count: {len(df_output)} (400 x 3)")

        # Check no NaN in TSVR
        tsvr_nan = df_output["TSVR_hours"].isna().sum()
        if tsvr_nan > 0:
            raise ValueError(f"Missing TSVR_hours: {tsvr_nan} rows")
        log("No NaN in TSVR_hours")

        # Check all congruence categories present
        for cat in CONGRUENCE_CATEGORIES:
            n_cat = len(df_output[df_output["congruence"] == cat])
            if n_cat != 400:
                log(f"{cat}: {n_cat} rows (expected 400)")
            else:
                log(f"{cat}: {n_cat} rows")

        # Check TSVR range
        tsvr_min, tsvr_max = df_output["TSVR_hours"].min(), df_output["TSVR_hours"].max()
        if tsvr_min < 0 or tsvr_max > 170:
            log(f"TSVR range [{tsvr_min:.2f}, {tsvr_max:.2f}] outside expected [0, 170]")
        else:
            log(f"TSVR range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        # Check theta range
        theta_min, theta_max = df_output["theta"].min(), df_output["theta"].max()
        log(f"Theta range: [{theta_min:.3f}, {theta_max:.3f}]")

        # Check unique UIDs
        n_uids = df_output["UID"].nunique()
        if n_uids != 100:
            log(f"{n_uids} unique UIDs (expected 100)")
        else:
            log(f"Unique UIDs: {n_uids}")

        log("\nStep 04 complete (Merge Theta + TSVR)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
