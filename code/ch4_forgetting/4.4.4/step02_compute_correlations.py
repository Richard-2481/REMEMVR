"""
Step 02: Compute Pearson Correlations (IRT vs CTT per Congruence)
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting

Purpose: Compute Pearson correlations between IRT theta and CTT mean scores
         for each congruence level with Holm-Bonferroni correction (Decision D068).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_pearson_correlations_with_correction

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
        logging.FileHandler(LOGS_DIR / "step02_compute_correlations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 02: Compute Pearson Correlations (IRT vs CTT)")
    logger.info("=" * 60)

    # 1. Load IRT theta scores (wide format)
    logger.info("Loading IRT theta scores...")
    theta_df = pd.read_csv(DATA_DIR / "step00_irt_theta.csv")
    logger.info(f"Loaded theta: {len(theta_df)} rows x {len(theta_df.columns)} columns")

    # 2. Load CTT scores (long format)
    logger.info("Loading CTT scores...")
    ctt_df = pd.read_csv(DATA_DIR / "step01_ctt_scores.csv")
    logger.info(f"Loaded CTT: {len(ctt_df)} rows")

    # Standardize column names
    if 'factor' in ctt_df.columns:
        ctt_df = ctt_df.rename(columns={'factor': 'congruence'})
    if 'CTT_score' in ctt_df.columns:
        ctt_df = ctt_df.rename(columns={'CTT_score': 'CTT_mean'})

    # 3. Pivot CTT to wide format to merge with theta
    logger.info("Pivoting CTT to wide format...")
    ctt_wide = ctt_df.pivot(index='composite_ID', columns='congruence', values='CTT_mean').reset_index()
    ctt_wide.columns = ['composite_ID', 'CTT_common', 'CTT_congruent', 'CTT_incongruent']
    logger.info(f"CTT wide format: {len(ctt_wide)} rows")

    # 4. Merge theta and CTT
    logger.info("Merging IRT theta and CTT scores...")
    merged_df = theta_df.merge(ctt_wide, on='composite_ID', how='inner')
    logger.info(f"Merged: {len(merged_df)} rows")

    # Extract UID and TEST if not present
    if 'UID' not in merged_df.columns:
        merged_df['UID'] = merged_df['composite_ID'].str.split('_').str[0]
    if 'TEST' not in merged_df.columns:
        merged_df['TEST'] = merged_df['composite_ID'].str.split('_').str[1]

    # 5. Create long format for correlation function
    logger.info("Creating long format for correlation computation...")

    # Melt theta columns
    theta_long = merged_df[['composite_ID', 'UID', 'TEST', 'theta_common', 'theta_congruent', 'theta_incongruent']].melt(
        id_vars=['composite_ID', 'UID', 'TEST'],
        var_name='congruence_theta',
        value_name='theta'
    )
    theta_long['congruence'] = theta_long['congruence_theta'].str.replace('theta_', '')

    # Melt CTT columns
    ctt_long = merged_df[['composite_ID', 'CTT_common', 'CTT_congruent', 'CTT_incongruent']].melt(
        id_vars=['composite_ID'],
        var_name='congruence_ctt',
        value_name='CTT_mean'
    )
    ctt_long['congruence'] = ctt_long['congruence_ctt'].str.replace('CTT_', '')

    # Merge theta and CTT long formats
    merged_long = theta_long.merge(ctt_long[['composite_ID', 'congruence', 'CTT_mean']],
                                    on=['composite_ID', 'congruence'])
    logger.info(f"Merged long format: {len(merged_long)} rows")

    # 6. Compute correlations using the catalogued tool
    logger.info("Computing Pearson correlations with Holm-Bonferroni correction...")

    correlations = compute_pearson_correlations_with_correction(
        df=merged_long,
        irt_col='theta',
        ctt_col='CTT_mean',
        factor_col='congruence',
        thresholds=[0.70, 0.90]
    )

    logger.info(f"Correlations computed:")
    print(correlations.to_string())

    # 7. Validate D068 compliance
    logger.info("Validating D068 dual p-value compliance...")

    has_p_uncorrected = 'p_uncorrected' in correlations.columns
    has_p_holm = 'p_holm' in correlations.columns

    if not has_p_uncorrected:
        logger.error("D068 VIOLATION: p_uncorrected column missing")
    if not has_p_holm:
        logger.error("D068 VIOLATION: p_holm column missing")

    if has_p_uncorrected and has_p_holm:
        # Verify p_holm >= p_uncorrected
        violations = correlations['p_holm'] < correlations['p_uncorrected']
        if violations.any():
            logger.error(f"D068 VIOLATION: p_holm < p_uncorrected for {violations.sum()} rows")
        else:
            logger.info("D068 VALIDATION PASSED: Dual p-values present and valid")

    # 8. Check convergence thresholds
    logger.info("Checking convergence thresholds...")

    # Column may be 'factor' or 'congruence' depending on tool output
    factor_col = 'congruence' if 'congruence' in correlations.columns else 'factor'
    correlations = correlations.rename(columns={'factor': 'congruence'}) if 'factor' in correlations.columns else correlations

    r_values = correlations[correlations['congruence'] != 'Overall']['r']
    all_above_070 = (r_values > 0.70).all()
    any_above_090 = (r_values > 0.90).any()

    logger.info(f"All r > 0.70 (strong threshold): {all_above_070}")
    logger.info(f"Any r > 0.90 (exceptional threshold): {any_above_090}")

    # 9. Save outputs
    logger.info("Saving outputs...")

    correlations.to_csv(DATA_DIR / "step02_correlations.csv", index=False)
    logger.info(f"Saved: data/step02_correlations.csv ({len(correlations)} rows)")

    merged_df.to_csv(DATA_DIR / "step02_merged_irt_ctt.csv", index=False)
    logger.info(f"Saved: data/step02_merged_irt_ctt.csv ({len(merged_df)} rows)")

    logger.info("=" * 60)
    logger.info("Step 02 COMPLETE: Correlations computed successfully")
    logger.info("=" * 60)

    return correlations, merged_df


if __name__ == "__main__":
    main()
