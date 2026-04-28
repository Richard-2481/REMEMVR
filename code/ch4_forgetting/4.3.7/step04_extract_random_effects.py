#!/usr/bin/env python3
"""Extract Individual Random Effects Per Paradigm: Extract individual random effects (intercepts and slopes) for all 100 participants"""

import sys
from pathlib import Path
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any
import traceback
from statsmodels.regression.mixed_linear_model import MixedLMResults

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.7 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_extract_random_effects.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Extract Individual Random Effects Per Paradigm")
        # Load Fitted Models from Step 2

        log("Loading fitted models from Step 2...")

        # Model file paths and paradigm labels
        model_files = {
            "free_recall": RQ_DIR / "data" / "step02_lmm_free_recall_model.pkl",
            "cued_recall": RQ_DIR / "data" / "step02_lmm_cued_recall_model.pkl",
            "recognition": RQ_DIR / "data" / "step02_lmm_recognition_model.pkl"
        }

        # Paradigm abbreviations (for output consistency)
        paradigm_abbrev = {
            "free_recall": "IFR",
            "cued_recall": "ICR",
            "recognition": "IRE"
        }

        # Load models
        models = {}
        for paradigm, model_path in model_files.items():
            log(f"Loading {paradigm} model from {model_path.name}...")
            models[paradigm] = MixedLMResults.load(str(model_path))
            log(f"{paradigm} model loaded successfully")
        # Extract Random Effects from Each Model
        #               Format: {UID: [intercept, slope], ...} or {UID: [intercept], ...}

        log("Extracting random effects from models...")

        all_random_effects = []

        for paradigm, model in models.items():
            log(f"Processing {paradigm}...")

            # Extract random effects dictionary
            random_effects_dict = model.random_effects

            # Convert to DataFrame
            paradigm_data = []
            for uid, effects in random_effects_dict.items():
                # effects is a Series or array: [intercept, slope] or just [intercept]
                # Use .iloc for position-based indexing to avoid FutureWarning
                if len(effects) >= 2:
                    # Model has random slopes
                    intercept = float(effects.iloc[0])
                    slope = float(effects.iloc[1])
                else:
                    # Intercept-only model
                    intercept = float(effects.iloc[0])
                    slope = 0.0

                paradigm_data.append({
                    "UID": uid,
                    "paradigm": paradigm_abbrev[paradigm],
                    "Total_Intercept": intercept,
                    "Total_Slope": slope
                })

            # Convert to DataFrame
            paradigm_df = pd.DataFrame(paradigm_data)
            all_random_effects.append(paradigm_df)

            log(f"{paradigm}: {len(paradigm_df)} participants")
            log(f"  - Intercept range: [{paradigm_df['Total_Intercept'].min():.3f}, {paradigm_df['Total_Intercept'].max():.3f}]")
            log(f"  - Slope range: [{paradigm_df['Total_Slope'].min():.3f}, {paradigm_df['Total_Slope'].max():.3f}]")
        # Concatenate All Paradigms
        # These outputs will be used by: RQ 5.3.8 (Paradigm-Based Clustering)

        log("Concatenating all paradigms...")
        random_effects_df = pd.concat(all_random_effects, axis=0, ignore_index=True)

        # Sort by paradigm, then UID for consistent ordering
        random_effects_df = random_effects_df.sort_values(["paradigm", "UID"]).reset_index(drop=True)

        log(f"Total rows: {len(random_effects_df)}")
        log(f"  - Expected: 300 (100 participants × 3 paradigms)")
        log(f"  - IFR rows: {len(random_effects_df[random_effects_df['paradigm'] == 'IFR'])}")
        log(f"  - ICR rows: {len(random_effects_df[random_effects_df['paradigm'] == 'ICR'])}")
        log(f"  - IRE rows: {len(random_effects_df[random_effects_df['paradigm'] == 'IRE'])}")
        # Save Random Effects CSV
        # Output: step04_random_effects.csv
        # Contains: UID, paradigm, Total_Intercept, Total_Slope
        # Columns: 4 (UID, paradigm, Total_Intercept, Total_Slope)

        output_path = RQ_DIR / "data" / "step04_random_effects.csv"
        log(f"Saving {output_path.name}...")
        random_effects_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(random_effects_df)} rows, {len(random_effects_df.columns)} cols)")
        # Generate Descriptive Statistics Per Paradigm
        # Output: step04_random_effects_descriptives.txt
        # Contains: Per-paradigm descriptive statistics (mean, SD, min, max)

        descriptives_path = RQ_DIR / "data" / "step04_random_effects_descriptives.txt"
        log(f"Generating {descriptives_path.name}...")

        with open(descriptives_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RANDOM EFFECTS DESCRIPTIVE STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            for paradigm_code in ["IFR", "ICR", "IRE"]:
                paradigm_data = random_effects_df[random_effects_df['paradigm'] == paradigm_code]

                f.write(f"Paradigm: {paradigm_code}\n")
                f.write("-" * 80 + "\n")
                f.write(f"N participants: {len(paradigm_data)}\n\n")

                # Intercept statistics
                f.write("Total_Intercept:\n")
                f.write(f"  Mean:   {paradigm_data['Total_Intercept'].mean():>10.4f}\n")
                f.write(f"  SD:     {paradigm_data['Total_Intercept'].std():>10.4f}\n")
                f.write(f"  Min:    {paradigm_data['Total_Intercept'].min():>10.4f}\n")
                f.write(f"  Max:    {paradigm_data['Total_Intercept'].max():>10.4f}\n\n")

                # Slope statistics
                f.write("Total_Slope:\n")
                f.write(f"  Mean:   {paradigm_data['Total_Slope'].mean():>10.4f}\n")
                f.write(f"  SD:     {paradigm_data['Total_Slope'].std():>10.4f}\n")
                f.write(f"  Min:    {paradigm_data['Total_Slope'].min():>10.4f}\n")
                f.write(f"  Max:    {paradigm_data['Total_Slope'].max():>10.4f}\n\n")

        log(f"{descriptives_path.name}")
        # Run Validation Tool
        # Validates: Exact row count, column names, no missing data, no duplicates
        # Threshold: 300 rows (100 participants × 3 paradigms)

        log("Running validate_dataframe_structure...")
        validation_result = validate_dataframe_structure(
            df=random_effects_df,
            expected_rows=300,
            expected_columns=["UID", "paradigm", "Total_Intercept", "Total_Slope"]
        )

        # Report validation results
        if validation_result['valid']:
            log("All validation checks passed")
            for check, passed in validation_result['checks'].items():
                status = "" if passed else ""
                log(f"{status} {check}")
        else:
            log("Validation failed")
            for check, passed in validation_result['checks'].items():
                status = "" if passed else ""
                log(f"{status} {check}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation checks specific to this step
        log("Running additional checks...")

        # Check for duplicate UID × paradigm combinations
        duplicates = random_effects_df.duplicated(subset=["UID", "paradigm"]).sum()
        if duplicates > 0:
            log(f"Found {duplicates} duplicate UID × paradigm combinations")
            raise ValueError(f"Duplicate UID × paradigm combinations: {duplicates}")
        else:
            log("No duplicate UID × paradigm combinations")

        # Check all 100 UIDs present per paradigm
        for paradigm_code in ["IFR", "ICR", "IRE"]:
            paradigm_uids = random_effects_df[random_effects_df['paradigm'] == paradigm_code]['UID'].nunique()
            if paradigm_uids != 100:
                log(f"{paradigm_code} has {paradigm_uids} UIDs (expected 100)")
                raise ValueError(f"{paradigm_code} has {paradigm_uids} unique UIDs (expected 100)")
            else:
                log(f"{paradigm_code} has 100 unique UIDs")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
