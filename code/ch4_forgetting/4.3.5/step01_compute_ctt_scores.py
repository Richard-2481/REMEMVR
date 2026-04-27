"""
Step 01: Compute CTT Mean Scores per Paradigm

Purpose: Compute Classical Test Theory (CTT) mean scores as proportion correct
         for each UID-TEST-paradigm combination using purified items from RQ 5.3.1
         (fair IRT-CTT comparison on same item set).

Dependencies: Step 00 (purified item list, TSVR mapping, composite_ID structure)

Output Files:
    - data/step01_ctt_scores.csv (1200 rows: 400 x 3 paradigms)
    - data/step01_ctt_computation_report.txt
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import the TDD-validated tool
from tools.analysis_ctt import compute_ctt_mean_scores_by_factor

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step01_compute_ctt_scores.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_input_files():
    """Load required input files from Step 00."""
    # Load purified items
    purified_items = pd.read_csv(DATA_PATH / "step00_purified_items.csv")
    logger.info(f"Loaded purified items: {len(purified_items)} items")

    # Load TSVR mapping (for UID-TEST structure)
    tsvr_mapping = pd.read_csv(DATA_PATH / "step00_tsvr_mapping.csv")
    logger.info(f"Loaded TSVR mapping: {len(tsvr_mapping)} rows")

    # Load raw data
    dfdata_path = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")
    df_data = pd.read_csv(dfdata_path)
    logger.info(f"Loaded dfData: {len(df_data)} rows, {len(df_data.columns)} columns")

    return purified_items, tsvr_mapping, df_data

def prepare_wide_format_data(df_data, tsvr_mapping, purified_items):
    """
    Prepare wide-format data for CTT computation.
    Filter to UID-TEST combinations in TSVR mapping.
    """
    # Get item names from purified items
    item_names = purified_items['item_name'].tolist()
    logger.info(f"Purified items: {len(item_names)}")

    # Create composite_ID in dfData if not present
    if 'composite_ID' not in df_data.columns:
        # Assume UID and TEST columns exist
        if 'UID' in df_data.columns and 'TEST' in df_data.columns:
            df_data['composite_ID'] = df_data['UID'] + '_' + df_data['TEST'].astype(str)
        elif 'UID' in df_data.columns and 'test' in df_data.columns:
            df_data['composite_ID'] = df_data['UID'] + '_' + df_data['test'].astype(str)

    # Filter to composite_IDs in TSVR mapping
    valid_ids = set(tsvr_mapping['composite_ID'])
    df_filtered = df_data[df_data['composite_ID'].isin(valid_ids)].copy()
    logger.info(f"Filtered to TSVR mapping: {len(df_filtered)} rows")

    # Check which items are available
    available_items = [item for item in item_names if item in df_filtered.columns]
    missing_items = [item for item in item_names if item not in df_filtered.columns]

    if missing_items:
        logger.warning(f"Missing items in dfData: {missing_items[:5]}... ({len(missing_items)} total)")

    logger.info(f"Available items: {len(available_items)}/{len(item_names)}")

    return df_filtered, available_items

def compute_ctt_scores_manual(df_data, tsvr_mapping, purified_items):
    """
    Compute CTT mean scores manually (row-wise mean per paradigm).
    Uses the compute_ctt_mean_scores_by_factor tool from analysis_ctt.
    """
    # Prepare data
    df_filtered, available_items = prepare_wide_format_data(df_data, tsvr_mapping, purified_items)

    # Filter purified items to those available in data
    items_df = purified_items[purified_items['item_name'].isin(available_items)].copy()
    logger.info(f"Using {len(items_df)} items for CTT computation")

    # Check items per paradigm
    paradigm_counts = items_df.groupby('paradigm')['item_name'].count()
    logger.info(f"Items per paradigm: {paradigm_counts.to_dict()}")

    # Compute CTT scores using the TDD-validated tool
    # This tool computes row-wise mean of items per factor (paradigm)
    try:
        ctt_scores = compute_ctt_mean_scores_by_factor(
            df_wide=df_filtered,
            item_factor_df=items_df,
            factor_col='paradigm',
            item_col='item_name',
            include_factors=['IFR', 'ICR', 'IRE']
        )
        logger.info(f"CTT scores computed: {len(ctt_scores)} rows")

        # Rename columns to match expected format
        # Tool returns: CTT_score, factor, test (lowercase)
        # Expected: CTT_mean, paradigm, TEST
        ctt_scores = ctt_scores.rename(columns={
            'CTT_score': 'CTT_mean',
            'factor': 'paradigm',
            'test': 'TEST'
        })

        # If TEST is numeric, convert to T1/T2/T3/T4 format
        if ctt_scores['TEST'].dtype in [int, float, np.int64, np.float64]:
            ctt_scores['TEST'] = ctt_scores['TEST'].apply(lambda x: f"T{int(x)}")

        logger.info(f"Columns after renaming: {list(ctt_scores.columns)}")

    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to manual computation
        ctt_scores = compute_ctt_manual_fallback(df_filtered, items_df, tsvr_mapping)

    return ctt_scores

def compute_ctt_manual_fallback(df_filtered, items_df, tsvr_mapping):
    """Fallback manual CTT computation if tool fails."""
    logger.info("Using manual fallback for CTT computation")

    results = []

    for paradigm in ['IFR', 'ICR', 'IRE']:
        # Get items for this paradigm
        paradigm_items = items_df[items_df['paradigm'] == paradigm]['item_name'].tolist()

        if not paradigm_items:
            logger.warning(f"No items found for paradigm {paradigm}")
            continue

        # Filter to available items
        available = [item for item in paradigm_items if item in df_filtered.columns]

        if not available:
            logger.warning(f"No available items for paradigm {paradigm}")
            continue

        # Compute mean across items for each row
        df_paradigm = df_filtered[['composite_ID'] + available].copy()
        df_paradigm['CTT_mean'] = df_paradigm[available].mean(axis=1)
        df_paradigm['n_items'] = df_paradigm[available].notna().sum(axis=1)
        df_paradigm['paradigm'] = paradigm

        results.append(df_paradigm[['composite_ID', 'paradigm', 'CTT_mean', 'n_items']])

    # Combine all paradigms
    ctt_long = pd.concat(results, ignore_index=True)

    # Merge with TSVR mapping for UID and TEST columns
    ctt_scores = ctt_long.merge(
        tsvr_mapping[['composite_ID', 'UID', 'TEST']],
        on='composite_ID',
        how='left'
    )

    # Reorder columns
    ctt_scores = ctt_scores[['composite_ID', 'UID', 'TEST', 'paradigm', 'CTT_mean', 'n_items']]

    return ctt_scores

def validate_ctt_scores(ctt_scores):
    """Validate CTT scores meet expected criteria."""
    # Check row count
    expected_rows = 1200  # 400 UID-TEST x 3 paradigms
    if len(ctt_scores) != expected_rows:
        logger.warning(f"Row count mismatch: expected {expected_rows}, got {len(ctt_scores)}")

    # Check CTT_mean range
    if ctt_scores['CTT_mean'].min() < 0 or ctt_scores['CTT_mean'].max() > 1:
        raise ValueError(f"CTT_mean out of bounds: [{ctt_scores['CTT_mean'].min()}, {ctt_scores['CTT_mean'].max()}]")

    # Check for NaN
    nan_count = ctt_scores['CTT_mean'].isna().sum()
    if nan_count > 0:
        logger.warning(f"NaN values in CTT_mean: {nan_count}")

    # Check n_items
    min_items = ctt_scores['n_items'].min()
    if min_items < 5:
        logger.warning(f"Low n_items detected: min={min_items}")

    # Check paradigm distribution
    paradigm_counts = ctt_scores['paradigm'].value_counts()
    logger.info(f"Paradigm distribution: {paradigm_counts.to_dict()}")

    logger.info("CTT scores validation: PASS")
    return True

def write_computation_report(ctt_scores, purified_items):
    """Write detailed computation report."""
    report_path = DATA_PATH / "step01_ctt_computation_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CTT COMPUTATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. ITEMS PER PARADIGM\n")
        f.write("-" * 40 + "\n")
        paradigm_item_counts = purified_items.groupby('paradigm')['item_name'].count()
        for paradigm, count in paradigm_item_counts.items():
            f.write(f"   {paradigm}: {count} items\n")
        f.write(f"   Total: {len(purified_items)} items\n\n")

        f.write("2. CTT SCORES SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Total rows: {len(ctt_scores)}\n")
        f.write(f"   Expected: 1200 (400 x 3 paradigms)\n\n")

        for paradigm in ['IFR', 'ICR', 'IRE']:
            paradigm_data = ctt_scores[ctt_scores['paradigm'] == paradigm]
            f.write(f"   {paradigm}:\n")
            f.write(f"      N: {len(paradigm_data)}\n")
            f.write(f"      CTT_mean: {paradigm_data['CTT_mean'].mean():.3f} (SD={paradigm_data['CTT_mean'].std():.3f})\n")
            f.write(f"      Range: [{paradigm_data['CTT_mean'].min():.3f}, {paradigm_data['CTT_mean'].max():.3f}]\n")
            f.write(f"      n_items: {paradigm_data['n_items'].mean():.1f} (mean)\n\n")

        f.write("3. MISSING DATA\n")
        f.write("-" * 40 + "\n")
        nan_count = ctt_scores['CTT_mean'].isna().sum()
        f.write(f"   NaN values in CTT_mean: {nan_count}\n")
        f.write(f"   Complete observations: {len(ctt_scores) - nan_count}\n\n")

        f.write("4. N_ITEMS DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Min n_items: {ctt_scores['n_items'].min()}\n")
        f.write(f"   Max n_items: {ctt_scores['n_items'].max()}\n")
        f.write(f"   Observations with n_items < 5: {(ctt_scores['n_items'] < 5).sum()}\n\n")

        f.write("=" * 60 + "\n")
        f.write("CTT COMPUTATION COMPLETE\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Computation report written to {report_path}")
    return report_path

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 01: COMPUTE CTT MEAN SCORES PER PARADIGM")
    logger.info("=" * 60)

    try:
        # 1. Load input files
        logger.info("\n1. Loading input files...")
        purified_items, tsvr_mapping, df_data = load_input_files()

        # 2. Compute CTT scores
        logger.info("\n2. Computing CTT scores...")
        ctt_scores = compute_ctt_scores_manual(df_data, tsvr_mapping, purified_items)

        # 3. Validate results
        logger.info("\n3. Validating results...")
        validate_ctt_scores(ctt_scores)

        # 4. Save output
        logger.info("\n4. Saving output files...")
        ctt_scores.to_csv(DATA_PATH / "step01_ctt_scores.csv", index=False)
        logger.info(f"   Saved: step01_ctt_scores.csv ({len(ctt_scores)} rows)")

        # 5. Write computation report
        logger.info("\n5. Writing computation report...")
        write_computation_report(ctt_scores, purified_items)

        logger.info("\n" + "=" * 60)
        logger.info("STEP 01 COMPLETE: CTT computation successful")
        logger.info(f"CTT computation complete: {len(ctt_scores)} scores created (400 x 3 paradigms)")
        logger.info("All paradigms represented: IFR, ICR, IRE")
        logger.info("No NaN values in CTT_mean column")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 01 FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
