"""
Step 07: Prepare Scatterplot Data (IRT vs CTT Convergence)
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
        logging.FileHandler(LOGS_DIR / "step07_prepare_scatterplot_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 07: Prepare Scatterplot Data")
    logger.info("=" * 60)

    # Load merged data
    merged_df = pd.read_csv(DATA_DIR / "step02_merged_irt_ctt.csv")
    tsvr_df = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv")
    logger.info(f"Loaded merged data: {len(merged_df)} rows")

    # Merge with TSVR
    merged_with_tsvr = merged_df.merge(tsvr_df[['composite_ID', 'TSVR_hours']], on='composite_ID')

    # Reshape to long format
    theta_cols = ['theta_common', 'theta_congruent', 'theta_incongruent']
    ctt_cols = ['CTT_common', 'CTT_congruent', 'CTT_incongruent']

    # Melt theta
    theta_long = merged_with_tsvr[['composite_ID', 'UID', 'TEST', 'TSVR_hours'] + theta_cols].melt(
        id_vars=['composite_ID', 'UID', 'TEST', 'TSVR_hours'],
        var_name='congruence_var',
        value_name='IRT_theta'
    )
    theta_long['congruence'] = theta_long['congruence_var'].str.replace('theta_', '')

    # Melt CTT
    ctt_long = merged_with_tsvr[['composite_ID'] + ctt_cols].melt(
        id_vars=['composite_ID'],
        var_name='congruence_var',
        value_name='CTT_mean'
    )
    ctt_long['congruence'] = ctt_long['congruence_var'].str.replace('CTT_', '')

    # Merge
    scatter_data = theta_long.merge(
        ctt_long[['composite_ID', 'congruence', 'CTT_mean']],
        on=['composite_ID', 'congruence']
    )

    # Clean up
    scatter_data = scatter_data[['UID', 'TEST', 'congruence', 'IRT_theta', 'CTT_mean', 'TSVR_hours']]
    scatter_data = scatter_data.sort_values(['congruence', 'UID', 'TEST']).reset_index(drop=True)

    logger.info(f"Scatterplot data: {len(scatter_data)} rows")
    logger.info(f"IRT_theta range: [{scatter_data['IRT_theta'].min():.2f}, {scatter_data['IRT_theta'].max():.2f}]")
    logger.info(f"CTT_mean range: [{scatter_data['CTT_mean'].min():.2f}, {scatter_data['CTT_mean'].max():.2f}]")

    # Save
    scatter_data.to_csv(DATA_DIR / "step07_scatterplot_data.csv", index=False)
    logger.info(f"Saved: step07_scatterplot_data.csv ({len(scatter_data)} rows)")

    logger.info("=" * 60)
    logger.info("Step 07 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
