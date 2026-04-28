#!/usr/bin/env python3
"""Reliability Assessment (Cronbach's Alpha): Assess internal consistency reliability (Cronbach's alpha) for Full vs Purified CTT scores"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_cronbachs_alpha

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_reliability_assessment.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Reliability Assessment (Cronbach's Alpha)")
        # Load Input Data

        log("Loading item mapping...")
        item_mapping = pd.read_csv(RQ_DIR / "data" / "step01_item_mapping.csv", encoding='utf-8')
        log(f"step01_item_mapping.csv ({len(item_mapping)} rows, {len(item_mapping.columns)} cols)")
        log(f"Location types: {sorted(item_mapping['location_type'].unique())}")
        log(f"Retention counts: {item_mapping['retained'].value_counts().to_dict()}")

        # Load raw binary item responses
        log("Loading raw binary item responses from dfData.csv...")
        dfData = pd.read_csv(PROJECT_ROOT / "data" / "cache" / "dfData.csv", encoding='utf-8')
        log(f"dfData.csv ({len(dfData)} rows, {len(dfData.columns)} cols)")
        # Extract Item Lists per Condition
        # Extract 4 item sets:
        #   1. Source_Full: all 18 TQ_*-U-* items
        #   2. Source_Purified: retained TQ_*-U-* items
        #   3. Destination_Full: all 18 TQ_*-D-* items
        #   4. Destination_Purified: retained TQ_*-D-* items

        log("Building item lists for 4 conditions...")

        # Source items (location_type = 'source')
        source_items_full = item_mapping[item_mapping['location_type'] == 'source']['item_name'].tolist()
        source_items_purified = item_mapping[
            (item_mapping['location_type'] == 'source') & (item_mapping['retained'] == True)
        ]['item_name'].tolist()

        # Destination items (location_type = 'destination')
        dest_items_full = item_mapping[item_mapping['location_type'] == 'destination']['item_name'].tolist()
        dest_items_purified = item_mapping[
            (item_mapping['location_type'] == 'destination') & (item_mapping['retained'] == True)
        ]['item_name'].tolist()

        log(f"Source Full: {len(source_items_full)} items")
        log(f"Source Purified: {len(source_items_purified)} items (retention {len(source_items_purified)/len(source_items_full):.1%})")
        log(f"Destination Full: {len(dest_items_full)} items")
        log(f"Destination Purified: {len(dest_items_purified)} items (retention {len(dest_items_purified)/len(dest_items_full):.1%})")
        # Compute Cronbach's Alpha for Each Condition

        log("Computing Cronbach's alpha with bootstrap CIs (10,000 iterations)...")
        log("This will take ~30 seconds for 4 conditions...")

        results = []
        n_bootstrap = 10000  # Recommended per tools_inventory.md

        # Condition 1: Source Full
        log("Condition 1/4: Source_Full...")
        df_source_full = dfData[source_items_full].copy()
        alpha_source_full = compute_cronbachs_alpha(df_source_full, n_bootstrap=n_bootstrap)
        results.append({
            'location_type': 'source',
            'version': 'full',
            'n_items': alpha_source_full['n_items'],
            'alpha': alpha_source_full['alpha'],
            'CI_lower': alpha_source_full['ci_lower'],
            'CI_upper': alpha_source_full['ci_upper'],
            'alpha_improvement': 0.0  # Computed later
        })
        log(f"Source_Full: alpha={alpha_source_full['alpha']:.3f} [{alpha_source_full['ci_lower']:.3f}, {alpha_source_full['ci_upper']:.3f}]")

        # Condition 2: Source Purified
        log("Condition 2/4: Source_Purified...")
        df_source_purified = dfData[source_items_purified].copy()
        alpha_source_purified = compute_cronbachs_alpha(df_source_purified, n_bootstrap=n_bootstrap)
        results.append({
            'location_type': 'source',
            'version': 'purified',
            'n_items': alpha_source_purified['n_items'],
            'alpha': alpha_source_purified['alpha'],
            'CI_lower': alpha_source_purified['ci_lower'],
            'CI_upper': alpha_source_purified['ci_upper'],
            'alpha_improvement': alpha_source_purified['alpha'] - alpha_source_full['alpha']
        })
        log(f"Source_Purified: alpha={alpha_source_purified['alpha']:.3f} [{alpha_source_purified['ci_lower']:.3f}, {alpha_source_purified['ci_upper']:.3f}]")
        log(f"Source alpha improvement: {alpha_source_purified['alpha'] - alpha_source_full['alpha']:+.3f}")

        # Condition 3: Destination Full
        log("Condition 3/4: Destination_Full...")
        df_dest_full = dfData[dest_items_full].copy()
        alpha_dest_full = compute_cronbachs_alpha(df_dest_full, n_bootstrap=n_bootstrap)
        results.append({
            'location_type': 'destination',
            'version': 'full',
            'n_items': alpha_dest_full['n_items'],
            'alpha': alpha_dest_full['alpha'],
            'CI_lower': alpha_dest_full['ci_lower'],
            'CI_upper': alpha_dest_full['ci_upper'],
            'alpha_improvement': 0.0  # Computed later
        })
        log(f"Destination_Full: alpha={alpha_dest_full['alpha']:.3f} [{alpha_dest_full['ci_lower']:.3f}, {alpha_dest_full['ci_upper']:.3f}]")

        # Condition 4: Destination Purified
        log("Condition 4/4: Destination_Purified...")
        df_dest_purified = dfData[dest_items_purified].copy()
        alpha_dest_purified = compute_cronbachs_alpha(df_dest_purified, n_bootstrap=n_bootstrap)
        results.append({
            'location_type': 'destination',
            'version': 'purified',
            'n_items': alpha_dest_purified['n_items'],
            'alpha': alpha_dest_purified['alpha'],
            'CI_lower': alpha_dest_purified['ci_lower'],
            'CI_upper': alpha_dest_purified['ci_upper'],
            'alpha_improvement': alpha_dest_purified['alpha'] - alpha_dest_full['alpha']
        })
        log(f"Destination_Purified: alpha={alpha_dest_purified['alpha']:.3f} [{alpha_dest_purified['ci_lower']:.3f}, {alpha_dest_purified['ci_upper']:.3f}]")
        log(f"Destination alpha improvement: {alpha_dest_purified['alpha'] - alpha_dest_full['alpha']:+.3f}")
        # Save Reliability Assessment Results
        # Output: data/step04_reliability_assessment.csv
        # Contains: 4 rows (2 location types x 2 versions) with alpha, CIs, improvement

        log("Saving reliability assessment results...")
        df_results = pd.DataFrame(results)
        output_path = RQ_DIR / "data" / "step04_reliability_assessment.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step04_reliability_assessment.csv ({len(df_results)} rows, {len(df_results.columns)} cols)")
        # Run Validation Tool
        # Validates: alpha values in [0.0, 1.0] range (mathematically required)
        # Threshold: alpha in [0.60, 0.95] for acceptable to excellent reliability

        log("Running validate_numeric_range on alpha values...")
        validation_result = validate_numeric_range(
            data=df_results['alpha'].values,
            min_val=0.0,
            max_val=1.0,
            column_name='alpha'
        )

        # Report validation results
        if validation_result['valid']:
            log(f"Alpha range validation: PASS")
            log(f"All alpha values in [0.0, 1.0]")
        else:
            log(f"Alpha range validation: FAIL")
            log(f"{validation_result['message']}")
            log(f"Out-of-range count: {validation_result['out_of_range_count']}")

        # Additional validation: CI_upper > CI_lower
        log("Checking confidence interval validity...")
        ci_valid = all(df_results['CI_upper'] > df_results['CI_lower'])
        if ci_valid:
            log("Confidence intervals: PASS (all CI_upper > CI_lower)")
        else:
            log("Confidence intervals: FAIL (some CI_upper <= CI_lower)")
            invalid_cis = df_results[df_results['CI_upper'] <= df_results['CI_lower']]
            log(f"Invalid CIs:\n{invalid_cis}")

        # Reliability interpretation
        log("Reliability assessment summary:")
        for _, row in df_results.iterrows():
            alpha = row['alpha']
            if alpha >= 0.90:
                interpretation = "Excellent"
            elif alpha >= 0.80:
                interpretation = "Good"
            elif alpha >= 0.70:
                interpretation = "Acceptable"
            elif alpha >= 0.60:
                interpretation = "Questionable"
            else:
                interpretation = "Poor"

            log(f"  {row['location_type'].capitalize()} {row['version'].capitalize()}: "
                f"alpha={alpha:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}] "
                f"({interpretation}, {row['n_items']} items)")

        # Final validation check
        if not validation_result['valid'] or not ci_valid:
            log("Validation failed - see errors above")
            sys.exit(1)

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
