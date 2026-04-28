#!/usr/bin/env python3
"""extract_and_merge_data: Extract and merge theta scores from Ch5 5.1.1 with strategy data from dfvr.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.3
LOG_FILE = RQ_DIR / "logs" / "step00_extract_and_merge_data.log"

# Ensure output directories exist
(RQ_DIR / "data").mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract and merge data")
        # Load Theta Scores from Ch5 5.1.1

        log("Loading theta scores from Ch5...")
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        theta_df = pd.read_csv(theta_path)
        log(f"Theta scores ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        log(f"Theta columns: {theta_df.columns.tolist()}")
        
        # Adapt column names to expected format
        if 'Theta_All' in theta_df.columns:
            theta_df['theta_all'] = theta_df['Theta_All']
            # Ch5 file doesn't have SE, set to NaN
            theta_df['se_all'] = np.nan
            log("Mapped Theta_All -> theta_all, set se_all=NaN")
        
        # Keep only most recent test per participant (test=4 typically)
        if 'test' in theta_df.columns:
            theta_df = theta_df.groupby('UID').last().reset_index()
            log(f"Kept most recent test per participant ({len(theta_df)} participants)")
        # Load Strategy Data from dfvr.csv

        log("Loading strategy data from dfvr.csv...")
        strategy_path = PROJECT_ROOT / "data" / "dfvr.csv"
        strategy_df = pd.read_csv(strategy_path)
        log(f"Strategy data ({len(strategy_df)} rows, {len(strategy_df.columns)} cols)")
        
        # Check required strategy columns exist
        strategy_cols = ['strategy-8', 'strategy-10', 'strategy-13']
        missing_strategy_cols = [col for col in strategy_cols if col not in strategy_df.columns]
        if missing_strategy_cols:
            raise ValueError(f"Missing strategy columns: {missing_strategy_cols}")
        
        # Aggregate strategy text across test sessions per participant
        log("Aggregating strategy responses across test sessions...")
        
        def combine_strategy_text(group):
            """Combine strategy responses across sessions for a participant."""
            strategy_texts = []
            for col in strategy_cols:
                texts = group[col].dropna().astype(str)
                texts = texts[texts != 'nan']  # Remove string 'nan' values
                if len(texts) > 0:
                    strategy_texts.extend(texts.tolist())
            
            # Combine all strategy text for this participant
            combined_text = ' | '.join(strategy_texts) if strategy_texts else ''
            return combined_text
        
        strategy_aggregated = strategy_df.groupby('UID').apply(combine_strategy_text).reset_index()
        strategy_aggregated.columns = ['UID', 'strategy_text_combined']
        log(f"Strategy text for {len(strategy_aggregated)} participants")
        # Load Demographics from dfnonvr.csv

        log("Loading demographics from dfnonvr.csv...")
        demo_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        demo_df = pd.read_csv(demo_path)
        log(f"Demographics ({len(demo_df)} rows, {len(demo_df.columns)} cols)")
        
        # Select and rename demographic variables
        demo_cols = ['UID', 'age', 'education', 'vr-exposure', 'typical-sleep-hours']
        missing_demo_cols = [col for col in demo_cols if col not in demo_df.columns]
        if missing_demo_cols:
            raise ValueError(f"Missing demographic columns: {missing_demo_cols}")
        
        demo_selected = demo_df[demo_cols].copy()
        demo_selected.rename(columns={
            'vr-exposure': 'vr_exposure',
            'typical-sleep-hours': 'sleep_hours'
        }, inplace=True)
        log(f"Demographics selected: {demo_selected.columns.tolist()}")
        # Merge All Datasets
        # Merge: theta scores + strategy data + demographics

        log("Merging theta scores with strategy data...")
        merged_df = theta_df.merge(strategy_aggregated, on='UID', how='inner')
        log(f"Theta + strategy: {len(merged_df)} participants")
        
        log("Adding demographics...")
        final_df = merged_df.merge(demo_selected, on='UID', how='inner')
        log(f"Complete dataset: {len(final_df)} participants")
        
        # Select final columns in expected order
        final_columns = ['UID', 'theta_all', 'se_all', 'strategy_text_combined', 
                        'age', 'education', 'vr_exposure', 'sleep_hours']
        final_df = final_df[final_columns]
        # Save Merged Dataset
        
        output_path = RQ_DIR / "data" / "step00_merged_data.csv"
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(final_df)} rows, {len(final_df.columns)} cols)")
        
        # Log summary statistics
        log("Dataset summary:")
        log(f"  Participants with complete data: {len(final_df)}")
        log(f"  Participants with strategy text: {(final_df['strategy_text_combined'] != '').sum()}")
        log(f"  Mean theta_all: {final_df['theta_all'].mean():.3f}")
        log(f"  Mean age: {final_df['age'].mean():.1f}")
        # Run Validation
        # Validation tool: validate_data_format with adapted signature

        log("Running validation...")
        
        # Custom validation due to signature mismatch in tools.validation
        expected_columns = ['UID', 'theta_all', 'se_all', 'strategy_text_combined', 
                           'age', 'education', 'vr_exposure', 'sleep_hours']
        
        # Basic validation checks
        validation_result = {
            'columns_match': list(final_df.columns) == expected_columns,
            'expected_rows': len(final_df) >= 80,  # Allow some missing data
            'no_missing_theta': not final_df['theta_all'].isna().any(),
            'no_missing_uid': not final_df['UID'].isna().any(),
            'valid': True
        }
        
        # Check if validation passed
        validation_result['valid'] = all([
            validation_result['columns_match'],
            validation_result['expected_rows'],
            validation_result['no_missing_theta'],
            validation_result['no_missing_uid']
        ])
        
        # Report validation results
        for key, value in validation_result.items():
            status = "" if value else ""
            log(f"{status} {key}: {value}")

        if not validation_result['valid']:
            raise ValueError("Validation failed - see log for details")

        log("Step 00 complete - merged dataset ready for strategy coding")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)