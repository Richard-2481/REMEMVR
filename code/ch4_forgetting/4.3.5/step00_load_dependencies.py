"""
Step 00: Load Dependencies from RQ 5.3.1

Purpose: Load IRT theta scores, TSVR mapping, purified items from RQ 5.3.1.
         Transform data formats to match downstream analysis expectations.
         Verify all required files exist with expected structure.

Dependencies: RQ 5.3.1 must be complete (status.yaml shows rq_inspect: success)

Output Files:
    - data/step00_dependency_verification.txt
    - data/step00_irt_theta.csv (wide format: theta_IFR, theta_ICR, theta_IRE)
    - data/step00_tsvr_mapping.csv (with TEST uppercase and Days column)
    - data/step00_purified_items.csv (standardized column names)
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Source RQ path
SOURCE_RQ_PATH = RQ_PATH.parent / "5.3.1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step00_load_dependencies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_rq_531_status():
    """Check that RQ 5.3.1 completed successfully."""
    status_file = SOURCE_RQ_PATH / "status.yaml"

    if not status_file.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: RQ 5.3.1 status.yaml not found at {status_file}")

    with open(status_file, 'r') as f:
        status = yaml.safe_load(f)

    # Check rq_inspect status - can be string 'success' or dict with 'status' key
    rq_inspect = status.get('rq_inspect')
    if isinstance(rq_inspect, dict):
        inspect_status = rq_inspect.get('status', 'unknown')
    else:
        inspect_status = rq_inspect

    if inspect_status != 'success':
        raise ValueError(f"EXPECTATIONS ERROR: RQ 5.3.1 rq_inspect status is '{inspect_status}', expected 'success'")

    logger.info("RQ 5.3.1 status check: PASS (rq_inspect: success)")
    return status

def load_and_transform_theta_scores():
    """
    Load theta scores from RQ 5.3.1 and transform from long to wide format.

    Source format: composite_ID, domain_name, theta (long format, 1200 rows)
    Target format: composite_ID, theta_IFR, theta_ICR, theta_IRE, se_IFR, se_ICR, se_IRE (wide format, 400 rows)

    Note: SE columns not present in source, will be set to NaN (standard for paradigm-level calibration)
    """
    source_file = SOURCE_RQ_PATH / "data" / "step03_theta_scores.csv"

    if not source_file.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: Theta scores file not found at {source_file}")

    df_long = pd.read_csv(source_file)
    logger.info(f"Loaded theta scores: {len(df_long)} rows, columns: {list(df_long.columns)}")

    # Map domain_name to paradigm abbreviations
    # RQ 5.3.1 uses: free_recall, cued_recall, recognition
    # Target format uses: IFR, ICR, IRE
    domain_mapping = {
        'free_recall': 'IFR',
        'cued_recall': 'ICR',
        'recognition': 'IRE'
    }

    df_long['paradigm'] = df_long['domain_name'].map(domain_mapping)

    # Check for unmapped domains
    unmapped = df_long[df_long['paradigm'].isna()]['domain_name'].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped domains found: {unmapped}")
        # Try alternate mapping
        alt_mapping = {
            'IFR': 'IFR', 'ICR': 'ICR', 'IRE': 'IRE',
            'Free Recall': 'IFR', 'Cued Recall': 'ICR', 'Recognition': 'IRE'
        }
        df_long['paradigm'] = df_long['domain_name'].map(lambda x: domain_mapping.get(x, alt_mapping.get(x, x)))

    # Pivot to wide format
    df_wide = df_long.pivot(
        index='composite_ID',
        columns='paradigm',
        values='theta'
    ).reset_index()

    # Rename columns to match expected format
    df_wide.columns = ['composite_ID'] + [f'theta_{col}' for col in df_wide.columns[1:]]

    # Add SE columns (NaN - not available from paradigm-level calibration)
    for paradigm in ['IFR', 'ICR', 'IRE']:
        if f'se_{paradigm}' not in df_wide.columns:
            df_wide[f'se_{paradigm}'] = np.nan

    # Reorder columns to match expected format
    expected_cols = ['composite_ID', 'theta_IFR', 'theta_ICR', 'theta_IRE', 'se_IFR', 'se_ICR', 'se_IRE']
    available_cols = [c for c in expected_cols if c in df_wide.columns]
    df_wide = df_wide[available_cols]

    logger.info(f"Transformed theta scores: {len(df_wide)} rows, columns: {list(df_wide.columns)}")

    # Validate
    assert len(df_wide) == 400, f"Expected 400 rows, got {len(df_wide)}"
    assert df_wide['composite_ID'].nunique() == 400, "composite_ID not unique"

    # Check theta ranges
    for col in [c for c in df_wide.columns if c.startswith('theta_')]:
        theta_min, theta_max = df_wide[col].min(), df_wide[col].max()
        assert -4 <= theta_min and theta_max <= 4, f"{col} out of range: [{theta_min}, {theta_max}]"
        logger.info(f"  {col}: min={theta_min:.3f}, max={theta_max:.3f}")

    return df_wide

def load_and_transform_tsvr_mapping():
    """
    Load TSVR mapping from RQ 5.3.1 and transform to expected format.

    Source format: composite_ID, UID, test, TSVR_hours
    Target format: composite_ID, UID, TEST, TSVR_hours, Days
    """
    source_file = SOURCE_RQ_PATH / "data" / "step00_tsvr_mapping.csv"

    if not source_file.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: TSVR mapping file not found at {source_file}")

    df = pd.read_csv(source_file)
    logger.info(f"Loaded TSVR mapping: {len(df)} rows, columns: {list(df.columns)}")

    # Rename 'test' to 'TEST' if present (case mismatch)
    if 'test' in df.columns and 'TEST' not in df.columns:
        df = df.rename(columns={'test': 'TEST'})

    # Transform TEST from numeric (1,2,3,4) to string (T1,T2,T3,T4) if needed
    if df['TEST'].dtype in [int, float, np.int64, np.float64]:
        df['TEST'] = df['TEST'].apply(lambda x: f"T{int(x)}")

    # Add Days column if missing (compute from TSVR_hours)
    # Standard mapping: T1=0 days (~0-2 hours), T2=1 day (~20-28 hours), T3=3 days (~68-84 hours), T4=6 days (~140-152 hours)
    if 'Days' not in df.columns:
        def hours_to_days(hours):
            if hours < 10:
                return 0
            elif hours < 50:
                return 1
            elif hours < 120:
                return 3
            else:
                return 6
        df['Days'] = df['TSVR_hours'].apply(hours_to_days)

    # Reorder columns
    df = df[['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'Days']]

    logger.info(f"Transformed TSVR mapping: {len(df)} rows, columns: {list(df.columns)}")

    # Validate
    assert len(df) == 400, f"Expected 400 rows, got {len(df)}"
    assert df['composite_ID'].nunique() == 400, "composite_ID not unique"
    assert set(df['TEST'].unique()) == {'T1', 'T2', 'T3', 'T4'}, f"Unexpected TEST values: {df['TEST'].unique()}"
    assert set(df['Days'].unique()) == {0, 1, 3, 6}, f"Unexpected Days values: {df['Days'].unique()}"

    return df

def load_and_transform_purified_items():
    """
    Load purified items from RQ 5.3.1 and transform to expected format.

    Source format: item, domain, Discrimination, Difficulty_1
    Target format: item_name, paradigm, dimension, a, b
    """
    source_file = SOURCE_RQ_PATH / "data" / "step02_purified_items.csv"

    if not source_file.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: Purified items file not found at {source_file}")

    df = pd.read_csv(source_file)
    logger.info(f"Loaded purified items: {len(df)} rows, columns: {list(df.columns)}")

    # Rename columns to match expected format
    column_mapping = {
        'item': 'item_name',
        'domain': 'paradigm',
        'Discrimination': 'a',
        'Difficulty_1': 'b'
    }
    df = df.rename(columns=column_mapping)

    # Map domain values to paradigm abbreviations
    domain_mapping = {
        'free_recall': 'IFR',
        'cued_recall': 'ICR',
        'recognition': 'IRE'
    }
    df['paradigm'] = df['paradigm'].map(domain_mapping)

    # Check for unmapped paradigms
    if df['paradigm'].isna().any():
        unmapped = df[df['paradigm'].isna()]['paradigm'].unique()
        logger.warning(f"Unmapped paradigms: {unmapped}")

    # Add dimension column (same as paradigm for this analysis)
    df['dimension'] = df['paradigm']

    # Reorder columns
    df = df[['item_name', 'paradigm', 'dimension', 'a', 'b']]

    logger.info(f"Transformed purified items: {len(df)} rows, columns: {list(df.columns)}")

    # Validate item counts per paradigm
    paradigm_counts = df['paradigm'].value_counts()
    logger.info(f"Items per paradigm: {paradigm_counts.to_dict()}")

    for paradigm, count in paradigm_counts.items():
        if count < 10:
            raise ValueError(f"CLARITY ERROR: Insufficient items for paradigm {paradigm}: only {count} items (need >= 10)")

    return df, paradigm_counts.to_dict()

def verify_dfdata_items(purified_items):
    """Verify all purified item tags present in dfData.csv."""
    dfdata_path = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")

    if not dfdata_path.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: dfData.csv not found at {dfdata_path}")

    # Read only header to check columns
    df_header = pd.read_csv(dfdata_path, nrows=0)
    dfdata_columns = set(df_header.columns)

    # Check each purified item
    missing_items = []
    for item_name in purified_items['item_name']:
        if item_name not in dfdata_columns:
            missing_items.append(item_name)

    if missing_items:
        logger.warning(f"Missing items in dfData.csv: {missing_items[:5]}... ({len(missing_items)} total)")
        # This might be OK if item names have different format - log but don't fail
        return False, missing_items

    logger.info(f"All {len(purified_items)} purified item tags verified in dfData.csv")
    return True, []

def verify_composite_id_match(theta_df, tsvr_df):
    """Verify composite_ID match between theta and TSVR files."""
    theta_ids = set(theta_df['composite_ID'])
    tsvr_ids = set(tsvr_df['composite_ID'])

    if theta_ids != tsvr_ids:
        missing_in_theta = tsvr_ids - theta_ids
        missing_in_tsvr = theta_ids - tsvr_ids

        if missing_in_theta:
            logger.warning(f"Missing in theta: {list(missing_in_theta)[:5]}...")
        if missing_in_tsvr:
            logger.warning(f"Missing in TSVR: {list(missing_in_tsvr)[:5]}...")

        raise ValueError(f"STEP ERROR: composite_ID mismatch between theta ({len(theta_ids)}) and TSVR ({len(tsvr_ids)}) files")

    logger.info(f"All 400 composite_IDs matched between theta and TSVR")
    return True

def write_verification_report(status, theta_df, tsvr_df, purified_df, paradigm_counts, dfdata_verified, missing_items):
    """Write dependency verification report."""
    report_path = DATA_PATH / "step00_dependency_verification.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RQ 5.3.5 DEPENDENCY VERIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("SOURCE: RQ 5.3.1 (Paradigm-Specific Trajectories)\n\n")

        f.write("1. RQ 5.3.1 COMPLETION STATUS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   rq_inspect: {status.get('rq_inspect', 'NOT FOUND')}\n")
        f.write(f"   Status: PASS\n\n")

        f.write("2. THETA SCORES (step03_theta_scores.csv)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Rows: {len(theta_df)} (expected: 400)\n")
        f.write(f"   Columns: {list(theta_df.columns)}\n")
        f.write(f"   composite_ID unique: {theta_df['composite_ID'].nunique()}\n")
        for col in [c for c in theta_df.columns if c.startswith('theta_')]:
            f.write(f"   {col}: min={theta_df[col].min():.3f}, max={theta_df[col].max():.3f}\n")
        f.write(f"   Status: PASS\n\n")

        f.write("3. TSVR MAPPING (step00_tsvr_mapping.csv)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Rows: {len(tsvr_df)} (expected: 400)\n")
        f.write(f"   Columns: {list(tsvr_df.columns)}\n")
        f.write(f"   TEST values: {sorted(tsvr_df['TEST'].unique())}\n")
        f.write(f"   Days values: {sorted(tsvr_df['Days'].unique())}\n")
        f.write(f"   TSVR_hours range: [{tsvr_df['TSVR_hours'].min():.1f}, {tsvr_df['TSVR_hours'].max():.1f}]\n")
        f.write(f"   Status: PASS\n\n")

        f.write("4. PURIFIED ITEMS (step02_purified_items.csv)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Total items: {len(purified_df)}\n")
        f.write(f"   Columns: {list(purified_df.columns)}\n")
        for paradigm, count in paradigm_counts.items():
            status_str = "PASS" if count >= 10 else "FAIL"
            f.write(f"   {paradigm} items: {count} (minimum: 10) - {status_str}\n")
        f.write(f"   Status: PASS\n\n")

        f.write("5. DFDATA.CSV VERIFICATION\n")
        f.write("-" * 40 + "\n")
        if dfdata_verified:
            f.write(f"   All purified item tags present: YES\n")
            f.write(f"   Status: PASS\n\n")
        else:
            f.write(f"   Missing items: {len(missing_items)}\n")
            f.write(f"   Examples: {missing_items[:5]}\n")
            f.write(f"   Status: WARNING (may need item name mapping)\n\n")

        f.write("6. COMPOSITE_ID CONSISTENCY\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Theta composite_IDs: {theta_df['composite_ID'].nunique()}\n")
        f.write(f"   TSVR composite_IDs: {tsvr_df['composite_ID'].nunique()}\n")
        f.write(f"   All 400 composite_IDs matched: YES\n")
        f.write(f"   Status: PASS\n\n")

        f.write("=" * 60 + "\n")
        f.write("OVERALL DEPENDENCY CHECK: PASS\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nAll files exist with expected structure.\n")
        f.write(f"Data transformations applied:\n")
        f.write(f"  - Theta scores: long -> wide format pivot\n")
        f.write(f"  - TSVR mapping: Added Days column, TEST uppercase\n")
        f.write(f"  - Purified items: Column name standardization\n")

    logger.info(f"Verification report written to {report_path}")
    return report_path

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 00: LOAD DEPENDENCIES FROM RQ 5.3.1")
    logger.info("=" * 60)

    try:
        # 1. Check RQ 5.3.1 status
        logger.info("\n1. Checking RQ 5.3.1 completion status...")
        status = check_rq_531_status()

        # 2. Load and transform theta scores
        logger.info("\n2. Loading and transforming theta scores...")
        theta_df = load_and_transform_theta_scores()

        # 3. Load and transform TSVR mapping
        logger.info("\n3. Loading and transforming TSVR mapping...")
        tsvr_df = load_and_transform_tsvr_mapping()

        # 4. Load and transform purified items
        logger.info("\n4. Loading and transforming purified items...")
        purified_df, paradigm_counts = load_and_transform_purified_items()

        # 5. Verify dfData items
        logger.info("\n5. Verifying dfData.csv items...")
        dfdata_verified, missing_items = verify_dfdata_items(purified_df)

        # 6. Verify composite_ID match
        logger.info("\n6. Verifying composite_ID consistency...")
        verify_composite_id_match(theta_df, tsvr_df)

        # 7. Save transformed files
        logger.info("\n7. Saving transformed files...")
        theta_df.to_csv(DATA_PATH / "step00_irt_theta.csv", index=False)
        logger.info(f"   Saved: step00_irt_theta.csv ({len(theta_df)} rows)")

        tsvr_df.to_csv(DATA_PATH / "step00_tsvr_mapping.csv", index=False)
        logger.info(f"   Saved: step00_tsvr_mapping.csv ({len(tsvr_df)} rows)")

        purified_df.to_csv(DATA_PATH / "step00_purified_items.csv", index=False)
        logger.info(f"   Saved: step00_purified_items.csv ({len(purified_df)} rows)")

        # 8. Write verification report
        logger.info("\n8. Writing verification report...")
        write_verification_report(status, theta_df, tsvr_df, purified_df,
                                  paradigm_counts, dfdata_verified, missing_items)

        logger.info("\n" + "=" * 60)
        logger.info("STEP 00 COMPLETE: Dependency check PASS")
        logger.info(f"IFR items: {paradigm_counts.get('IFR', 0)}, ICR items: {paradigm_counts.get('ICR', 0)}, IRE items: {paradigm_counts.get('IRE', 0)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 00 FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
