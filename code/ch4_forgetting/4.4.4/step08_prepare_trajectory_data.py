"""
Step 08: Prepare Trajectory Comparison Data (IRT vs CTT)
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "step08_prepare_trajectory_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 08: Prepare Trajectory Comparison Data")
    logger.info("=" * 60)

    # Load LMM input data
    irt_input = pd.read_csv(DATA_DIR / "step03_irt_lmm_input.csv")
    ctt_input = pd.read_csv(DATA_DIR / "step03_ctt_lmm_input.csv")
    logger.info(f"IRT input: {len(irt_input)} rows, CTT input: {len(ctt_input)} rows")

    # Aggregate IRT by congruence x TEST
    irt_agg = irt_input.groupby(['congruence', 'TEST']).agg(
        observed_mean=('theta', 'mean'),
        observed_sd=('theta', 'std'),
        n=('theta', 'count'),
        TSVR_hours=('TSVR_hours', 'mean')
    ).reset_index()
    irt_agg['CI_lower'] = irt_agg['observed_mean'] - 1.96 * irt_agg['observed_sd'] / np.sqrt(irt_agg['n'])
    irt_agg['CI_upper'] = irt_agg['observed_mean'] + 1.96 * irt_agg['observed_sd'] / np.sqrt(irt_agg['n'])
    irt_agg['measurement_type'] = 'IRT'

    # Aggregate CTT by congruence x TEST
    ctt_agg = ctt_input.groupby(['congruence', 'TEST']).agg(
        observed_mean=('CTT_mean', 'mean'),
        observed_sd=('CTT_mean', 'std'),
        n=('CTT_mean', 'count'),
        TSVR_hours=('TSVR_hours', 'mean')
    ).reset_index()
    ctt_agg['CI_lower'] = ctt_agg['observed_mean'] - 1.96 * ctt_agg['observed_sd'] / np.sqrt(ctt_agg['n'])
    ctt_agg['CI_upper'] = ctt_agg['observed_mean'] + 1.96 * ctt_agg['observed_sd'] / np.sqrt(ctt_agg['n'])
    ctt_agg['measurement_type'] = 'CTT'

    # Stack
    trajectory_data = pd.concat([irt_agg, ctt_agg], ignore_index=True)
    trajectory_data = trajectory_data[['congruence', 'TEST', 'TSVR_hours', 'measurement_type', 'observed_mean', 'CI_lower', 'CI_upper']]
    trajectory_data = trajectory_data.sort_values(['measurement_type', 'congruence', 'TEST']).reset_index(drop=True)

    logger.info(f"Trajectory data: {len(trajectory_data)} rows (expected 24)")

    # Validate
    assert len(trajectory_data) == 24, f"Expected 24 rows, got {len(trajectory_data)}"
    assert set(trajectory_data['congruence']) == {'common', 'congruent', 'incongruent'}
    assert set(trajectory_data['measurement_type']) == {'IRT', 'CTT'}

    # Save
    trajectory_data.to_csv(DATA_DIR / "step08_trajectory_data.csv", index=False)
    logger.info(f"Saved: step08_trajectory_data.csv ({len(trajectory_data)} rows)")

    logger.info("=" * 60)
    logger.info("Step 08 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
