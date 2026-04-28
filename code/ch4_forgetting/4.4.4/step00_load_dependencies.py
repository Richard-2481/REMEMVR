"""
Step 00: Load Dependencies from RQ 5.4.1
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting

Purpose: Load IRT theta scores, TSVR mapping, and purified items from RQ 5.4.1.
         Verify dependency completion and copy files for lineage tracking.
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"
PARENT_RQ_DIR = RQ_DIR.parent / "5.4.1"
PROJECT_ROOT = RQ_DIR.parent.parent.parent

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "step00_load_dependencies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 00: Load Dependencies from RQ 5.4.1")
    logger.info("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    verification_report = []

    # 1. Check RQ 5.4.1 completion status
    logger.info("Checking RQ 5.4.1 completion status...")
    status_path = PARENT_RQ_DIR / "status.yaml"

    if not status_path.exists():
        raise FileNotFoundError(f"RQ 5.4.1 status.yaml not found: {status_path}")

    with open(status_path, 'r') as f:
        status = yaml.safe_load(f)

    # Check that results analysis is success (full completion)
    results analysis_status = status.get('results analysis', {}).get('status', 'unknown')
    if results analysis_status != 'success':
        raise ValueError(f"RQ 5.4.1 not complete: results analysis status = {results analysis_status}")

    verification_report.append(f"RQ 5.4.1 status: {results analysis_status}")
    logger.info(f"RQ 5.4.1 results analysis status: {results analysis_status}")

    # 2. Load theta scores
    logger.info("Loading theta scores from RQ 5.4.1...")
    theta_path = PARENT_RQ_DIR / "data" / "step03_theta_scores.csv"

    if not theta_path.exists():
        raise FileNotFoundError(f"Theta scores not found: {theta_path}")

    theta_df = pd.read_csv(theta_path)
    logger.info(f"Loaded theta scores: {len(theta_df)} rows x {len(theta_df.columns)} columns")
    verification_report.append(f"Theta scores: {len(theta_df)} rows x {len(theta_df.columns)} columns")

    # Verify expected structure
    expected_theta_cols = ['composite_ID', 'theta_common', 'theta_congruent', 'theta_incongruent',
                          'se_common', 'se_congruent', 'se_incongruent']
    missing_cols = [c for c in expected_theta_cols if c not in theta_df.columns]
    if missing_cols:
        raise ValueError(f"Missing theta columns: {missing_cols}")

    if len(theta_df) != 400:
        logger.warning(f"Expected 400 theta rows, found {len(theta_df)}")

    verification_report.append(f"Theta columns verified: {expected_theta_cols}")

    # 3. Load TSVR mapping
    logger.info("Loading TSVR mapping from RQ 5.4.1...")
    tsvr_path = PARENT_RQ_DIR / "data" / "step00_tsvr_mapping.csv"

    if not tsvr_path.exists():
        raise FileNotFoundError(f"TSVR mapping not found: {tsvr_path}")

    tsvr_df = pd.read_csv(tsvr_path)
    logger.info(f"Loaded TSVR mapping: {len(tsvr_df)} rows x {len(tsvr_df.columns)} columns")
    verification_report.append(f"TSVR mapping: {len(tsvr_df)} rows x {len(tsvr_df.columns)} columns")

    # Verify expected columns (flexible - check for key columns)
    required_tsvr_cols = ['composite_ID', 'TSVR_hours']
    missing_tsvr = [c for c in required_tsvr_cols if c not in tsvr_df.columns]
    if missing_tsvr:
        raise ValueError(f"Missing TSVR columns: {missing_tsvr}")

    # Extract UID and TEST from composite_ID if not present
    if 'UID' not in tsvr_df.columns:
        tsvr_df['UID'] = tsvr_df['composite_ID'].str.split('_').str[0]
    if 'TEST' not in tsvr_df.columns:
        tsvr_df['TEST'] = tsvr_df['composite_ID'].str.split('_').str[1]

    # Add Days column if not present
    if 'Days' not in tsvr_df.columns:
        tsvr_df['Days'] = tsvr_df['TSVR_hours'] / 24.0

    verification_report.append(f"TSVR columns: {list(tsvr_df.columns)}")

    # 4. Load purified items
    logger.info("Loading purified items from RQ 5.4.1...")
    purified_path = PARENT_RQ_DIR / "data" / "step02_purified_items.csv"

    if not purified_path.exists():
        raise FileNotFoundError(f"Purified items not found: {purified_path}")

    purified_df = pd.read_csv(purified_path)
    logger.info(f"Loaded purified items: {len(purified_df)} rows x {len(purified_df.columns)} columns")
    verification_report.append(f"Purified items: {len(purified_df)} rows x {len(purified_df.columns)} columns")

    # Count items per congruence level
    if 'dimension' in purified_df.columns:
        counts = purified_df['dimension'].value_counts()
        logger.info(f"Items per congruence level:\n{counts}")
        verification_report.append(f"Items per congruence: common={counts.get('common', 0)}, congruent={counts.get('congruent', 0)}, incongruent={counts.get('incongruent', 0)}")

        # Verify minimum items per level
        for level in ['common', 'congruent', 'incongruent']:
            if counts.get(level, 0) < 10:
                logger.warning(f"Low item count for {level}: {counts.get(level, 0)} < 10")

    # 5. Verify composite_ID consistency
    logger.info("Verifying composite_ID consistency...")
    theta_ids = set(theta_df['composite_ID'])
    tsvr_ids = set(tsvr_df['composite_ID'])

    common_ids = theta_ids.intersection(tsvr_ids)
    logger.info(f"Common composite_IDs: {len(common_ids)}")
    verification_report.append(f"Common composite_IDs between theta and TSVR: {len(common_ids)}")

    if len(common_ids) != len(theta_ids):
        logger.warning(f"ID mismatch: {len(theta_ids)} theta IDs, {len(tsvr_ids)} TSVR IDs, {len(common_ids)} common")

    # 6. Copy files to this RQ's data folder for lineage tracking
    logger.info("Copying files to RQ 5.4.4 data folder...")

    theta_df.to_csv(DATA_DIR / "step00_irt_theta.csv", index=False)
    logger.info(f"Saved: data/step00_irt_theta.csv ({len(theta_df)} rows)")

    tsvr_df.to_csv(DATA_DIR / "step00_tsvr_mapping.csv", index=False)
    logger.info(f"Saved: data/step00_tsvr_mapping.csv ({len(tsvr_df)} rows)")

    purified_df.to_csv(DATA_DIR / "step00_purified_items.csv", index=False)
    logger.info(f"Saved: data/step00_purified_items.csv ({len(purified_df)} rows)")

    # 7. Save verification report
    verification_text = "\n".join(verification_report)
    verification_path = DATA_DIR / "step00_dependency_verification.txt"
    with open(verification_path, 'w') as f:
        f.write("RQ 5.4.4 Dependency Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Parent RQ: 5.4.1 (Schema Congruence Trajectories)\n")
        f.write("=" * 50 + "\n\n")
        f.write(verification_text)

    logger.info(f"Saved verification report: {verification_path}")

    logger.info("=" * 60)
    logger.info("Step 00 COMPLETE: All dependencies loaded successfully")
    logger.info("=" * 60)

    return theta_df, tsvr_df, purified_df


if __name__ == "__main__":
    main()
