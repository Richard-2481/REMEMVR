#!/usr/bin/env python3
"""
Step 03: Compute Purified CTT Mean Scores

Computes CTT mean scores using ONLY purified items (retained after D039) per congruence level.
Uses wide-format dfData.csv and filters to purified items from RQ 5.4.1.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_compute_ctt_purified.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 03: Compute Purified CTT Mean Scores")

        # Load wide-format data
        data_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        log(f"Reading {data_path}")
        df_data = pd.read_csv(data_path, encoding='utf-8')
        log(f"{len(df_data)} rows, {len(df_data.columns)} columns")

        # Create composite_ID
        df_data['composite_ID'] = df_data['UID'] + '_' + df_data['TEST'].astype(str)
        log(f"composite_ID column (e.g., {df_data['composite_ID'].iloc[0]})")

        # Load purified item list from RQ 5.4.1
        purified_items_path = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step02_purified_items.csv"
        log(f"Reading {purified_items_path}")
        purified_items = pd.read_csv(purified_items_path, encoding='utf-8')
        log(f"{len(purified_items)} purified items")

        # Select only TQ_* columns that are in purified_items
        tq_columns = [col for col in df_data.columns if col.startswith('TQ_') and col in purified_items['item_name'].values]
        log(f"Found {len(tq_columns)} TQ columns in dfData that match purified_items")

        # Create dimension mapping from purified_items
        dimension_map = dict(zip(purified_items['item_name'], purified_items['dimension']))

        # Compute mean scores per dimension
        log("Computing mean scores by dimension")

        results = []
        for idx, row in df_data.iterrows():
            composite_id = row['composite_ID']

            # Compute mean for each dimension
            common_items = [col for col in tq_columns if dimension_map.get(col) == 'common']
            congruent_items = [col for col in tq_columns if dimension_map.get(col) == 'congruent']
            incongruent_items = [col for col in tq_columns if dimension_map.get(col) == 'incongruent']

            ctt_purified_common = row[common_items].mean() if common_items else np.nan
            ctt_purified_congruent = row[congruent_items].mean() if congruent_items else np.nan
            ctt_purified_incongruent = row[incongruent_items].mean() if incongruent_items else np.nan

            results.append({
                'composite_ID': composite_id,
                'ctt_purified_common': ctt_purified_common,
                'ctt_purified_congruent': ctt_purified_congruent,
                'ctt_purified_incongruent': ctt_purified_incongruent
            })

        ctt_purified_scores = pd.DataFrame(results)
        log(f"{len(ctt_purified_scores)} composite_IDs processed")

        # Validation: Check all scores in [0, 1]
        log("Checking all CTT scores in [0, 1]")
        score_cols = ['ctt_purified_common', 'ctt_purified_congruent', 'ctt_purified_incongruent']
        for col in score_cols:
            min_val = ctt_purified_scores[col].min()
            max_val = ctt_purified_scores[col].max()
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(f"{col} out of range: [{min_val:.3f}, {max_val:.3f}]")
            log(f"{col} in [{min_val:.3f}, {max_val:.3f}]")

        # Check for NaN values
        log("Checking for NaN values")
        nan_count = ctt_purified_scores[score_cols].isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in CTT scores")
        log("No NaN values found")

        # Save results
        output_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        log(f"Writing {output_path}")
        ctt_purified_scores.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{len(ctt_purified_scores)} rows, {len(ctt_purified_scores.columns)} columns")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
