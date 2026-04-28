#!/usr/bin/env python3
"""Compare to Ch5 5.1.2 Accuracy Pattern: Document whether confidence two-phase pattern replicates accuracy two-phase"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step05_compare_to_ch5.log"

# Input paths
INPUT_QUADRATIC = RQ_DIR / "data" / "step02_quadratic_test.csv"
INPUT_PIECEWISE = RQ_DIR / "data" / "step03_piecewise_comparison.csv"
INPUT_RATIO = RQ_DIR / "data" / "step04_slope_ratio.csv"
INPUT_CH5 = PROJECT_ROOT / "results/ch5/5.1.2/data"  # Directory (search for summary file)

# Output path
OUTPUT_FILE = RQ_DIR / "data" / "step05_ch5_comparison.csv"

# Evidence thresholds
BONFERRONI_ALPHA = 0.01  # From Decision D068
DELTA_AIC_THRESHOLD = 2.0
RATIO_THRESHOLD = 0.5

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Compare to Ch5 5.1.2 Accuracy Pattern")
        # Load Test Results (Steps 2-4)

        log(f"Loading test results from Steps 2-4...")

        # Load quadratic test results
        df_quadratic = pd.read_csv(INPUT_QUADRATIC, encoding='utf-8')
        log(f"{INPUT_QUADRATIC.name} ({len(df_quadratic)} rows)")

        # Load piecewise comparison
        df_piecewise = pd.read_csv(INPUT_PIECEWISE, encoding='utf-8')
        log(f"{INPUT_PIECEWISE.name} ({len(df_piecewise)} rows)")

        # Load slope ratio
        df_ratio = pd.read_csv(INPUT_RATIO, encoding='utf-8')
        log(f"{INPUT_RATIO.name} ({len(df_ratio)} rows)")
        # Extract Evidence from Each Test
        # Test 1: Quadratic significant? (p_bonferroni < 0.01 for TSVR_hours_squared)
        # Test 2: Piecewise preferred? (delta_AIC > 2)
        # Test 3: Slope ratio small? (ratio < 0.5)

        log("Extracting evidence from each test...")

        # Test 1: Quadratic
        quadratic_row = df_quadratic[df_quadratic['term'] == 'TSVR_hours_squared']
        if len(quadratic_row) == 0:
            log("Quadratic term not found, trying alternative name...")
            quadratic_row = df_quadratic[df_quadratic['term'].str.contains('squared', case=False, na=False)]

        if len(quadratic_row) > 0:
            quadratic_significant = bool(quadratic_row['significant_bonferroni'].iloc[0])
            log(f"Test 1 - Quadratic: {'SUPPORT' if quadratic_significant else 'NULL'}")
        else:
            log("Cannot find quadratic term in step02 results")
            raise ValueError("Quadratic term missing from step02_quadratic_test.csv")

        # Test 2: Piecewise preference
        comparison_row = df_piecewise[df_piecewise['model'] == 'Comparison']
        if len(comparison_row) > 0:
            piecewise_preferred = bool(comparison_row['piecewise_preferred'].iloc[0])
            log(f"Test 2 - Piecewise: {'SUPPORT' if piecewise_preferred else 'NULL'}")
        else:
            log("Cannot find Comparison row in step03 results")
            raise ValueError("Comparison row missing from step03_piecewise_comparison.csv")

        # Test 3: Slope ratio
        ratio_row = df_ratio[df_ratio['segment'] == 'Ratio']
        if len(ratio_row) > 0:
            slope_ratio_small = bool(ratio_row['two_phase_evidence'].iloc[0])
            log(f"Test 3 - Slope ratio: {'SUPPORT' if slope_ratio_small else 'NULL'}")
        else:
            log("Cannot find Ratio row in step04 results")
            raise ValueError("Ratio row missing from step04_slope_ratio.csv")
        # Count Evidence and Determine Conclusion
        # Evidence count: 0-3 tests supporting two-phase pattern
        # Conclusion: 2-3 = SUPPORT, 0 = NULL, 1 = INCONCLUSIVE

        log("Counting evidence across tests...")

        evidence_count = sum([quadratic_significant, piecewise_preferred, slope_ratio_small])
        log(f"Evidence count: {evidence_count}/3 tests support two-phase pattern")

        if evidence_count >= 2:
            conclusion = "SUPPORT"
            log(f"Conclusion: SUPPORT (>= 2/3 tests positive)")
        elif evidence_count == 0:
            conclusion = "NULL"
            log(f"Conclusion: NULL (0/3 tests positive)")
        else:
            conclusion = "INCONCLUSIVE"
            log(f"Conclusion: INCONCLUSIVE (1/3 tests positive)")
        # Try to Load Ch5 5.1.2 Accuracy Results
        # Optional: If not found, comparison skipped (pattern_match = N/A)

        log("Attempting to load Ch5 5.1.2 accuracy results...")

        ch5_available = False
        ch5_evidence_count = None
        ch5_conclusion = None

        # Search for two-phase summary file in Ch5 5.1.2 directory
        if INPUT_CH5.exists():
            summary_files = list(INPUT_CH5.glob("*two_phase*.csv"))
            if len(summary_files) > 0:
                ch5_file = summary_files[0]
                log(f"Found Ch5 summary file: {ch5_file.name}")
                try:
                    df_ch5 = pd.read_csv(ch5_file, encoding='utf-8')
                    # Extract evidence (file structure may vary)
                    # Assume similar format: columns include evidence booleans or conclusion
                    if 'evidence_count' in df_ch5.columns and 'conclusion' in df_ch5.columns:
                        ch5_row = df_ch5[df_ch5['measure'].str.contains('Accuracy', case=False, na=False)]
                        if len(ch5_row) > 0:
                            ch5_evidence_count = int(ch5_row['evidence_count'].iloc[0])
                            ch5_conclusion = str(ch5_row['conclusion'].iloc[0])
                            ch5_available = True
                            log(f"Ch5 accuracy: {ch5_evidence_count}/3 evidence, conclusion={ch5_conclusion}")
                        else:
                            log("Ch5 file format unexpected (no Accuracy row)")
                    else:
                        log("Ch5 file format unexpected (missing evidence_count/conclusion columns)")
                except Exception as e:
                    log(f"Failed to load Ch5 file: {e}")
            else:
                log("No two-phase summary file found in Ch5 5.1.2 directory")
        else:
            log("Ch5 5.1.2 directory not found (optional comparison skipped)")

        if not ch5_available:
            log("Ch5 comparison not available, pattern_match will be N/A")
        # Determine Pattern Match
        # If Ch5 available: Compare conclusions
        # REPLICATED = both SUPPORT or both NULL
        # DIVERGED = one SUPPORT, one NULL
        # INCONCLUSIVE = either measure INCONCLUSIVE
        # N/A = Ch5 not available

        if ch5_available:
            log("Comparing confidence vs accuracy patterns...")

            if conclusion == "INCONCLUSIVE" or ch5_conclusion == "INCONCLUSIVE":
                pattern_match = "INCONCLUSIVE"
            elif conclusion == ch5_conclusion:
                pattern_match = "REPLICATED"
            else:
                pattern_match = "DIVERGED"

            log(f"Pattern match: {pattern_match}")
        else:
            pattern_match = "N/A"
            log(f"Pattern match: N/A (Ch5 not available)")
        # Create Comparison DataFrame
        # Rows: Confidence (always), Accuracy (if Ch5 available)

        comparison_data = []

        # Confidence row (current RQ)
        comparison_data.append({
            'measure': 'Confidence (Ch6 6.1.2)',
            'quadratic_significant': quadratic_significant,
            'piecewise_preferred': piecewise_preferred,
            'slope_ratio_small': slope_ratio_small,
            'evidence_count': evidence_count,
            'conclusion': conclusion,
            'pattern_match': pattern_match
        })

        # Accuracy row (Ch5 if available)
        if ch5_available:
            # Extract individual test results from Ch5 if possible
            # For now, just use evidence_count and conclusion
            comparison_data.append({
                'measure': 'Accuracy (Ch5 5.1.2)',
                'quadratic_significant': np.nan,  # Not extracted
                'piecewise_preferred': np.nan,
                'slope_ratio_small': np.nan,
                'evidence_count': ch5_evidence_count,
                'conclusion': ch5_conclusion,
                'pattern_match': pattern_match
            })

        comparison_df = pd.DataFrame(comparison_data)
        log("Comparison DataFrame created")
        # Save Comparison Results
        # Output: data/step05_ch5_comparison.csv
        # Contains: Evidence from both measures (if Ch5 available)

        log(f"Saving comparison to {OUTPUT_FILE.name}...")
        comparison_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(comparison_df)} rows)")
        # Run Validation
        # Validates: Row count, column presence, evidence count range

        log("Running custom comparison checks...")

        # Check evidence_count range [0, 3]
        for idx, row in comparison_df.iterrows():
            if pd.notna(row['evidence_count']):
                if row['evidence_count'] < 0 or row['evidence_count'] > 3:
                    log(f"FAIL - evidence_count out of range: {row['evidence_count']}")
                    raise ValueError("evidence_count must be in [0, 3]")
        log("PASS - All evidence_count values in [0, 3]")

        # Check conclusion values
        valid_conclusions = {'SUPPORT', 'NULL', 'INCONCLUSIVE'}
        for idx, row in comparison_df.iterrows():
            if row['conclusion'] not in valid_conclusions:
                log(f"FAIL - Invalid conclusion: {row['conclusion']}")
                raise ValueError(f"Invalid conclusion value: {row['conclusion']}")
        log("PASS - All conclusions valid")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
