"""
Step 01: Compute CTT Mean Scores per Congruence Level
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting

Purpose: Compute Classical Test Theory mean scores (proportion correct) for each
         UID x TEST x congruence level combination using purified items from RQ 5.4.1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_ctt_mean_scores_by_factor

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "step01_compute_ctt_scores.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 01: Compute CTT Mean Scores per Congruence Level")
    logger.info("=" * 60)

    # 1. Load purified items
    logger.info("Loading purified items...")
    purified_df = pd.read_csv(DATA_DIR / "step00_purified_items.csv")
    logger.info(f"Loaded {len(purified_df)} purified items")

    # 2. Load TSVR mapping to get UID-TEST combinations
    logger.info("Loading TSVR mapping...")
    tsvr_df = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv")
    logger.info(f"Loaded {len(tsvr_df)} TSVR mappings")

    # 3. Load raw data
    logger.info("Loading raw response data...")
    df_data = pd.read_csv(PROJECT_ROOT / "data" / "cache" / "dfData.csv")
    logger.info(f"Loaded raw data: {len(df_data)} rows")

    # 4. Get list of purified item names
    purified_items = purified_df['item_name'].tolist()
    logger.info(f"Purified items: {len(purified_items)}")

    # Check which items exist in dfData columns
    available_items = [item for item in purified_items if item in df_data.columns]
    missing_items = [item for item in purified_items if item not in df_data.columns]

    if missing_items:
        logger.warning(f"Missing {len(missing_items)} items from raw data: {missing_items[:5]}...")

    logger.info(f"Available items in raw data: {len(available_items)}")

    # 5. Prepare wide-format data with composite_ID
    # Create composite_ID from UID and TEST columns
    if 'composite_ID' not in df_data.columns:
        # Find UID and TEST columns (may have different names)
        uid_col = 'UID' if 'UID' in df_data.columns else df_data.columns[df_data.columns.str.contains('UID', case=False)].tolist()[0]
        test_col = 'TEST' if 'TEST' in df_data.columns else df_data.columns[df_data.columns.str.contains('TEST', case=False)].tolist()[0]
        df_data['composite_ID'] = df_data[uid_col].astype(str) + '_' + df_data[test_col].astype(str)

    # Filter to only the composite_IDs in our analysis
    valid_ids = set(tsvr_df['composite_ID'])
    df_filtered = df_data[df_data['composite_ID'].isin(valid_ids)].copy()
    logger.info(f"Filtered to {len(df_filtered)} rows matching TSVR mapping")

    # 6. Compute CTT scores using the tool function
    logger.info("Computing CTT mean scores by congruence level...")

    # Prepare item-factor mapping
    item_factor_df = purified_df[['item_name', 'dimension']].copy()
    item_factor_df = item_factor_df[item_factor_df['item_name'].isin(available_items)]

    # Use the catalogued tool
    ctt_scores = compute_ctt_mean_scores_by_factor(
        df_wide=df_filtered,
        item_factor_df=item_factor_df,
        factor_col='dimension',
        item_col='item_name',
        include_factors=['common', 'congruent', 'incongruent']
    )

    logger.info(f"CTT scores computed: {len(ctt_scores)} rows")
    logger.info(f"CTT columns: {list(ctt_scores.columns)}")

    # 7. Merge with TSVR to add UID and TEST columns
    if 'UID' not in ctt_scores.columns:
        ctt_scores['UID'] = ctt_scores['composite_ID'].str.split('_').str[0]
    if 'TEST' not in ctt_scores.columns:
        ctt_scores['TEST'] = ctt_scores['composite_ID'].str.split('_').str[1]

    # Rename factor column to 'congruence' for clarity
    if 'factor' in ctt_scores.columns:
        ctt_scores = ctt_scores.rename(columns={'factor': 'congruence'})

    # 8. Validate CTT scores
    logger.info("Validating CTT scores...")

    if 'CTT_mean' in ctt_scores.columns:
        ctt_col = 'CTT_mean'
    elif 'CTT_score' in ctt_scores.columns:
        ctt_col = 'CTT_score'
        ctt_scores = ctt_scores.rename(columns={'CTT_score': 'CTT_mean'})
        ctt_col = 'CTT_mean'
    else:
        raise ValueError(f"CTT score column not found. Available: {list(ctt_scores.columns)}")

    # Check range [0, 1]
    min_ctt = ctt_scores[ctt_col].min()
    max_ctt = ctt_scores[ctt_col].max()
    logger.info(f"CTT score range: [{min_ctt:.4f}, {max_ctt:.4f}]")

    if min_ctt < 0 or max_ctt > 1:
        raise ValueError(f"CTT scores out of [0,1] range: [{min_ctt}, {max_ctt}]")

    # Check for NaN
    nan_count = ctt_scores[ctt_col].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in CTT scores")

    # Verify row count
    expected_rows = 400 * 3  # 400 composite_IDs x 3 congruence levels
    if len(ctt_scores) != expected_rows:
        logger.warning(f"Expected {expected_rows} rows, found {len(ctt_scores)}")

    # 9. Generate computation report
    report_lines = [
        "CTT Computation Report for RQ 5.4.4",
        "=" * 50,
        f"",
        f"Items per congruence level:",
    ]

    for level in ['common', 'congruent', 'incongruent']:
        count = len(item_factor_df[item_factor_df['dimension'] == level])
        report_lines.append(f"  {level}: {count} items")

    report_lines.extend([
        f"",
        f"Total purified items used: {len(item_factor_df)}",
        f"",
        f"CTT Score Descriptives by Congruence:",
    ])

    for level in ['common', 'congruent', 'incongruent']:
        subset = ctt_scores[ctt_scores['congruence'] == level][ctt_col]
        report_lines.append(f"  {level}: mean={subset.mean():.4f}, sd={subset.std():.4f}, n={len(subset)}")

    report_lines.extend([
        f"",
        f"Overall CTT: mean={ctt_scores[ctt_col].mean():.4f}, sd={ctt_scores[ctt_col].std():.4f}",
        f"Total observations: {len(ctt_scores)}",
        f"Missing values: {nan_count}",
    ])

    # 10. Save outputs
    logger.info("Saving outputs...")

    ctt_scores.to_csv(DATA_DIR / "step01_ctt_scores.csv", index=False)
    logger.info(f"Saved: data/step01_ctt_scores.csv ({len(ctt_scores)} rows)")

    with open(DATA_DIR / "step01_ctt_computation_report.txt", 'w') as f:
        f.write("\n".join(report_lines))
    logger.info("Saved: data/step01_ctt_computation_report.txt")

    logger.info("=" * 60)
    logger.info("Step 01 COMPLETE: CTT scores computed successfully")
    logger.info("=" * 60)

    return ctt_scores


if __name__ == "__main__":
    main()
