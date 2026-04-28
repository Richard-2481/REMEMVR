#!/usr/bin/env python3
"""Merge Data Sources and Compute Baseline Statistics: Merge accuracy theta (from Ch5 5.1.1 2PL IRT), confidence theta (from Ch6 6.1.1 GRM),"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import traceback

# Add project root to path for imports (if needed for future extensions)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.1
LOG_FILE = RQ_DIR / "logs" / "step01_merge_and_baseline.log"

# Input paths (cross-RQ dependencies)
ACCURACY_THETA_PATH = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
CONFIDENCE_THETA_PATH = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv"
TIME_MAPPING_PATH = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step00_tsvr_mapping.csv"

# Output paths
OUTPUT_MERGED = RQ_DIR / "data" / "step01_merged_trajectories.csv"
OUTPUT_BASELINE = RQ_DIR / "data" / "step01_baseline_comparison.csv"
OUTPUT_DIAGNOSTICS = RQ_DIR / "data" / "step01_baseline_diagnostics.txt"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def parse_composite_id(composite_id):
    """
    Parse composite_ID to extract UID and test number.

    Handles two formats:
    - 'UID_test' (e.g., 'A010_1')
    - 'UID_T#' (e.g., 'A010_T1')

    Returns tuple (UID, test_number).
    """
    parts = composite_id.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid composite_ID format: {composite_id}")

    uid = parts[0]
    test_str = parts[1]

    # Handle both '1' and 'T1' formats
    if test_str.startswith('T'):
        test_num = int(test_str[1:])
    else:
        test_num = int(test_str)

    return uid, test_num

def compute_distribution_stats(data, measure_name):
    """
    Compute descriptive statistics for a single measure at baseline (T1).

    Returns dict with: measure, mean, SD, skewness, kurtosis, min, max
    """
    return {
        'measure': measure_name,
        'mean': np.mean(data),
        'SD': np.std(data, ddof=1),  # Sample SD
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def compute_cohens_d(group1, group2):
    """
    Compute Cohen's d for two independent groups.

    Formula: (mean1 - mean2) / pooled_SD
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    cohens_d = (mean1 - mean2) / pooled_sd
    return cohens_d

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: Merge Data Sources and Compute Baseline Statistics")
        # Load Input Data

        log("Loading accuracy theta from Ch5 5.1.1...")
        df_acc = pd.read_csv(ACCURACY_THETA_PATH, encoding='utf-8')
        log(f"Accuracy theta: {len(df_acc)} rows, columns: {list(df_acc.columns)}")

        log("Loading confidence theta from Ch6 6.1.1...")
        df_conf = pd.read_csv(CONFIDENCE_THETA_PATH, encoding='utf-8')
        log(f"Confidence theta: {len(df_conf)} rows, columns: {list(df_conf.columns)}")

        log("Loading time mapping from Ch6 6.1.1...")
        df_time = pd.read_csv(TIME_MAPPING_PATH, encoding='utf-8')
        log(f"Time mapping: {len(df_time)} rows, columns: {list(df_time.columns)}")
        # Rename and Parse Columns
        # Rename for clarity: Theta_All -> theta_acc, theta_All -> theta_conf
        # Parse composite_ID to extract UID and test for merging

        log("Renaming accuracy theta column...")
        df_acc = df_acc.rename(columns={'Theta_All': 'theta_acc'})

        log("Parsing confidence composite_ID and renaming columns...")
        # Parse composite_ID -> UID, test
        df_conf[['UID', 'test']] = df_conf['composite_ID'].apply(
            lambda x: pd.Series(parse_composite_id(x))
        )
        # Rename theta and SE columns
        df_conf = df_conf.rename(columns={
            'theta_All': 'theta_conf',
            'se_All': 'se_conf'
        })
        # Keep only needed columns
        df_conf = df_conf[['UID', 'test', 'theta_conf', 'se_conf']]

        log("Parsing time mapping composite_ID...")
        # Parse composite_ID -> UID, test
        df_time[['UID_parsed', 'test_parsed']] = df_time['composite_ID'].apply(
            lambda x: pd.Series(parse_composite_id(x))
        )
        # Verify parsed test matches existing test column
        test_mismatch = (df_time['test'] != df_time['test_parsed']).sum()
        if test_mismatch > 0:
            log(f"{test_mismatch} rows with test number mismatch in time mapping")
        # Use parsed UID (no UID column in original)
        df_time['UID'] = df_time['UID_parsed']
        # Keep only needed columns
        df_time = df_time[['UID', 'test', 'TSVR_hours']]

        log(f"Accuracy: {len(df_acc)} rows with columns {list(df_acc.columns)}")
        log(f"Confidence: {len(df_conf)} rows with columns {list(df_conf.columns)}")
        log(f"Time: {len(df_time)} rows with columns {list(df_time.columns)}")
        # Three-Way Merge on (UID, test)
        # Merge accuracy + confidence + time mapping

        log("Performing three-way merge on (UID, test)...")

        # First merge: accuracy + confidence
        df_merged = pd.merge(
            df_acc,
            df_conf,
            on=['UID', 'test'],
            how='inner',
            validate='1:1'
        )
        log(f"After acc+conf merge: {len(df_merged)} rows")

        # Second merge: result + time mapping
        df_merged = pd.merge(
            df_merged,
            df_time,
            on=['UID', 'test'],
            how='inner',
            validate='1:1'
        )
        log(f"After adding time mapping: {len(df_merged)} rows")

        # Validate merge result
        if len(df_merged) != 400:
            log(f"Expected 400 rows after merge, got {len(df_merged)}")
            sys.exit(1)

        log("Merge successful: 400 observations (100 participants x 4 tests)")
        # Create Day Labels
        # Map test numbers to nominal day labels for interpretability
        # test: 1 -> Day0, 2 -> Day1, 3 -> Day3, 4 -> Day6

        log("Creating day labels...")
        day_mapping = {1: 'Day0', 2: 'Day1', 3: 'Day3', 4: 'Day6'}
        df_merged['day_label'] = df_merged['test'].map(day_mapping)

        # Reorder columns for clarity
        df_merged = df_merged[['UID', 'test', 'day_label', 'TSVR_hours',
                               'theta_acc', 'theta_conf', 'se_conf']]

        log(f"Day labels created: {df_merged['day_label'].value_counts().to_dict()}")

        # Check for missing data
        missing_pct = {
            col: (df_merged[col].isna().sum() / len(df_merged)) * 100
            for col in ['theta_acc', 'theta_conf', 'TSVR_hours']
        }
        log(f"[QC] Missing data percentages: {missing_pct}")
        for col, pct in missing_pct.items():
            if pct > 5.0:
                log(f"Excessive missing data in {col}: {pct:.2f}%")
                sys.exit(1)
        # Compute Baseline (T1) Statistics - SEPARATE PER MEASURE
        # Extract Day 0 (test=1) subset for baseline distribution analysis
        # Compute statistics SEPARATELY for accuracy and confidence (different IRT scales)

        log("Extracting Day 0 (test=1) subset...")
        df_baseline = df_merged[df_merged['test'] == 1].copy()
        n_baseline = len(df_baseline)

        if n_baseline != 100:
            log(f"Expected 100 baseline observations, got {n_baseline}")
            sys.exit(1)

        log(f"Baseline subset: N={n_baseline}")

        log("Computing accuracy baseline statistics...")
        acc_baseline_stats = compute_distribution_stats(
            df_baseline['theta_acc'].dropna(),
            'accuracy'
        )
        log(f"Accuracy: mean={acc_baseline_stats['mean']:.4f}, "
            f"SD={acc_baseline_stats['SD']:.4f}, "
            f"skew={acc_baseline_stats['skewness']:.4f}")

        log("Computing confidence baseline statistics...")
        conf_baseline_stats = compute_distribution_stats(
            df_baseline['theta_conf'].dropna(),
            'confidence'
        )
        log(f"Confidence: mean={conf_baseline_stats['mean']:.4f}, "
            f"SD={conf_baseline_stats['SD']:.4f}, "
            f"skew={conf_baseline_stats['skewness']:.4f}")

        # Validate baseline SDs (required for standardization in Step 2)
        if acc_baseline_stats['SD'] <= 0:
            log("Accuracy baseline SD = 0, cannot standardize decline rates")
            sys.exit(1)
        if conf_baseline_stats['SD'] <= 0:
            log("Confidence baseline SD = 0, cannot standardize decline rates")
            sys.exit(1)

        log("Baseline SDs valid (both > 0)")
        # IRT Comparability Assessment
        # Compare baseline distributions between accuracy (2PL) and confidence (GRM)

        log("Assessing IRT scale comparability...")

        acc_data = df_baseline['theta_acc'].dropna()
        conf_data = df_baseline['theta_conf'].dropna()

        # Cohen's d (standardized mean difference)
        cohens_d = compute_cohens_d(acc_data, conf_data)
        log(f"Cohen's d (acc vs conf at baseline): {cohens_d:.4f}")

        # SD ratio (relative spread)
        sd_ratio = conf_baseline_stats['SD'] / acc_baseline_stats['SD']
        log(f"SD ratio (conf/acc): {sd_ratio:.4f}")

        # Interpretation flags
        comparability_notes = []
        if abs(cohens_d) > 0.5:
            comparability_notes.append(
                f"|Cohen's d| = {abs(cohens_d):.4f} > 0.5 suggests baseline mean difference (medium effect)"
            )
        else:
            comparability_notes.append(
                f"|Cohen's d| = {abs(cohens_d):.4f} <= 0.5 suggests baseline means similar"
            )

        if sd_ratio < 0.8 or sd_ratio > 1.2:
            comparability_notes.append(
                f"SD ratio = {sd_ratio:.4f} outside [0.8, 1.2] suggests scaling difference"
            )
        else:
            comparability_notes.append(
                f"SD ratio = {sd_ratio:.4f} within [0.8, 1.2] suggests similar spread"
            )

        comparability_notes.append(
            "\nInterpretation: Large differences don't invalidate paired comparison "
            "(within-person design controls for individual scale usage) but flag "
            "interpretation limits for between-measure comparisons."
        )

        log("Assessment complete")
        for note in comparability_notes:
            log(f"{note}")
        # Save Outputs
        # Three outputs: merged trajectories, baseline comparison, diagnostics

        log("Saving merged trajectories...")
        df_merged.to_csv(OUTPUT_MERGED, index=False, encoding='utf-8')
        log(f"{OUTPUT_MERGED} ({len(df_merged)} rows, {len(df_merged.columns)} columns)")

        log("Saving baseline comparison...")
        df_baseline_comparison = pd.DataFrame([acc_baseline_stats, conf_baseline_stats])
        df_baseline_comparison.to_csv(OUTPUT_BASELINE, index=False, encoding='utf-8')
        log(f"{OUTPUT_BASELINE} (2 rows: accuracy, confidence)")

        log("Saving baseline diagnostics...")
        with open(OUTPUT_DIAGNOSTICS, 'w', encoding='utf-8') as f:
            f.write("IRT SCALE COMPARABILITY ASSESSMENT\n")
            f.write("=" * 70 + "\n\n")
            f.write("PURPOSE:\n")
            f.write("Compare baseline (Day 0) distributions between accuracy theta (2PL IRT)\n")
            f.write("and confidence theta (GRM) to assess scale equivalence.\n\n")

            f.write("BASELINE STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy:   mean={acc_baseline_stats['mean']:.4f}, "
                   f"SD={acc_baseline_stats['SD']:.4f}\n")
            f.write(f"Confidence: mean={conf_baseline_stats['mean']:.4f}, "
                   f"SD={conf_baseline_stats['SD']:.4f}\n\n")

            f.write("COMPARABILITY METRICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Cohen's d (accuracy vs confidence): {cohens_d:.4f}\n")
            f.write(f"  Interpretation: ")
            if abs(cohens_d) < 0.2:
                f.write("Negligible baseline mean difference\n")
            elif abs(cohens_d) < 0.5:
                f.write("Small baseline mean difference\n")
            elif abs(cohens_d) < 0.8:
                f.write("Medium baseline mean difference\n")
            else:
                f.write("Large baseline mean difference\n")
            f.write(f"\nSD ratio (conf/acc): {sd_ratio:.4f}\n")
            f.write(f"  Interpretation: ")
            if 0.8 <= sd_ratio <= 1.2:
                f.write("Similar baseline variability\n")
            else:
                f.write("Different baseline variability (scaling difference)\n")

            f.write("\nINTERPRETATION:\n")
            f.write("-" * 70 + "\n")
            for note in comparability_notes:
                f.write(f"{note}\n")

        log(f"{OUTPUT_DIAGNOSTICS}")
        # Final Validation

        log("Verifying outputs...")

        # Check merged trajectories
        assert OUTPUT_MERGED.exists(), "Merged trajectories file missing"
        df_check = pd.read_csv(OUTPUT_MERGED, encoding='utf-8')
        assert len(df_check) == 400, f"Expected 400 rows, got {len(df_check)}"
        assert list(df_check.columns) == ['UID', 'test', 'day_label', 'TSVR_hours',
                                          'theta_acc', 'theta_conf', 'se_conf'], \
            f"Column mismatch: {list(df_check.columns)}"
        log("Merged trajectories validated: 400 rows, 7 columns")

        # Check baseline comparison
        assert OUTPUT_BASELINE.exists(), "Baseline comparison file missing"
        df_baseline_check = pd.read_csv(OUTPUT_BASELINE, encoding='utf-8')
        assert len(df_baseline_check) == 2, f"Expected 2 rows, got {len(df_baseline_check)}"
        assert all(df_baseline_check['SD'] > 0), "Baseline SD must be > 0 for both measures"
        log("Baseline comparison validated: 2 rows, SDs > 0")

        # Check diagnostics file
        assert OUTPUT_DIAGNOSTICS.exists(), "Diagnostics file missing"
        log("Diagnostics file validated")

        log("Step 1 complete")
        log(f"Merged 3 data sources -> 400 observations")
        log(f"Baseline N=100, accuracy SD={acc_baseline_stats['SD']:.4f}, "
            f"confidence SD={conf_baseline_stats['SD']:.4f}")
        log(f"IRT comparability: Cohen's d={cohens_d:.4f}, SD ratio={sd_ratio:.4f}")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
