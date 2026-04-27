"""
Step 02: Compute Pearson Correlations (IRT vs CTT per Paradigm)

Purpose: Compute Pearson correlations between IRT theta and CTT mean scores
         for each paradigm (IFR, ICR, IRE) plus overall. Test against
         convergence thresholds (r > 0.70 strong, r > 0.90 exceptional)
         with Holm-Bonferroni correction per Decision D068.

Dependencies: Step 00 (IRT theta), Step 01 (CTT scores)

Output Files:
    - data/step02_correlations.csv (4 rows: IFR, ICR, IRE, Overall)
    - data/step02_merged_irt_ctt.csv (400 rows wide format)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats

# Import the TDD-validated tool
from tools.analysis_ctt import compute_pearson_correlations_with_correction

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step02_compute_correlations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_input_files():
    """Load IRT theta and CTT scores from previous steps."""
    # Load IRT theta (wide format: theta_IFR, theta_ICR, theta_IRE)
    irt_theta = pd.read_csv(DATA_PATH / "step00_irt_theta.csv")
    logger.info(f"Loaded IRT theta: {len(irt_theta)} rows, columns: {list(irt_theta.columns)}")

    # Load CTT scores (long format: paradigm, CTT_mean)
    ctt_scores = pd.read_csv(DATA_PATH / "step01_ctt_scores.csv")
    logger.info(f"Loaded CTT scores: {len(ctt_scores)} rows, columns: {list(ctt_scores.columns)}")

    return irt_theta, ctt_scores

def reshape_ctt_to_wide(ctt_scores):
    """Reshape CTT scores from long to wide format."""
    # Pivot to wide format: one row per composite_ID, columns for each paradigm
    ctt_wide = ctt_scores.pivot(
        index='composite_ID',
        columns='paradigm',
        values='CTT_mean'
    ).reset_index()

    # Rename columns to match expected format: CTT_IFR, CTT_ICR, CTT_IRE
    ctt_wide.columns = ['composite_ID'] + [f'CTT_{col}' for col in ctt_wide.columns[1:]]

    logger.info(f"Reshaped CTT to wide: {len(ctt_wide)} rows, columns: {list(ctt_wide.columns)}")

    return ctt_wide

def merge_irt_ctt(irt_theta, ctt_wide):
    """Merge IRT and CTT data on composite_ID."""
    # Merge on composite_ID
    merged = irt_theta.merge(ctt_wide, on='composite_ID', how='inner')

    logger.info(f"Merged IRT-CTT: {len(merged)} rows")

    # Add UID and TEST columns from composite_ID if not present
    if 'UID' not in merged.columns:
        merged['UID'] = merged['composite_ID'].apply(lambda x: x.rsplit('_', 1)[0])
    if 'TEST' not in merged.columns:
        merged['TEST'] = merged['composite_ID'].apply(lambda x: f"T{x.rsplit('_', 1)[1]}")

    # Reorder columns
    cols = ['composite_ID', 'UID', 'TEST',
            'theta_IFR', 'theta_ICR', 'theta_IRE',
            'CTT_IFR', 'CTT_ICR', 'CTT_IRE']
    if 'se_IFR' in merged.columns:
        cols.extend(['se_IFR', 'se_ICR', 'se_IRE'])

    available_cols = [c for c in cols if c in merged.columns]
    merged = merged[available_cols]

    return merged

def compute_correlations(merged_df):
    """
    Compute Pearson correlations between IRT theta and CTT for each paradigm.
    Uses the TDD-validated compute_pearson_correlations_with_correction tool.
    """
    results = []

    # Compute per-paradigm correlations
    for paradigm in ['IFR', 'ICR', 'IRE']:
        theta_col = f'theta_{paradigm}'
        ctt_col = f'CTT_{paradigm}'

        if theta_col not in merged_df.columns or ctt_col not in merged_df.columns:
            logger.warning(f"Missing columns for {paradigm}: {theta_col} or {ctt_col}")
            continue

        # Get valid pairs (non-NaN)
        valid_mask = merged_df[theta_col].notna() & merged_df[ctt_col].notna()
        valid_df = merged_df.loc[valid_mask]

        n = len(valid_df)
        r, p = stats.pearsonr(valid_df[theta_col], valid_df[ctt_col])

        results.append({
            'paradigm': paradigm,
            'n': n,
            'r': r,
            'p_uncorrected': p
        })

        logger.info(f"{paradigm}: r={r:.3f}, p={p:.4f}, n={n}")

    # Compute overall correlation (stacked)
    theta_all = pd.concat([
        merged_df['theta_IFR'],
        merged_df['theta_ICR'],
        merged_df['theta_IRE']
    ])
    ctt_all = pd.concat([
        merged_df['CTT_IFR'],
        merged_df['CTT_ICR'],
        merged_df['CTT_IRE']
    ])

    valid_mask = theta_all.notna() & ctt_all.notna()
    n_overall = valid_mask.sum()
    r_overall, p_overall = stats.pearsonr(theta_all[valid_mask], ctt_all[valid_mask])

    results.append({
        'paradigm': 'Overall',
        'n': n_overall,
        'r': r_overall,
        'p_uncorrected': p_overall
    })

    logger.info(f"Overall: r={r_overall:.3f}, p={p_overall:.4f}, n={n_overall}")

    correlations_df = pd.DataFrame(results)

    return correlations_df

def apply_holm_bonferroni(correlations_df):
    """Apply Holm-Bonferroni correction for multiple comparisons."""
    # Sort by p-value ascending
    sorted_df = correlations_df.sort_values('p_uncorrected').copy()
    n_tests = len(sorted_df)

    # Apply Holm-Bonferroni correction
    p_corrected = []
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        # Adjusted p = p * (n_tests - rank + 1)
        adjusted_p = min(row['p_uncorrected'] * (n_tests - i), 1.0)
        p_corrected.append(adjusted_p)

    sorted_df['p_bonferroni'] = p_corrected

    # Restore original order
    correlations_df = sorted_df.sort_index()

    return correlations_df

def add_threshold_classifications(correlations_df):
    """Add threshold classifications and interpretations."""
    # Threshold checks
    correlations_df['threshold_0.70'] = correlations_df['r'] > 0.70
    correlations_df['threshold_0.90'] = correlations_df['r'] > 0.90

    # Interpretation based on r value
    def interpret_r(r):
        if r > 0.90:
            return "Exceptional"
        elif r > 0.70:
            return "Strong"
        elif r > 0.50:
            return "Moderate"
        else:
            return "Weak"

    correlations_df['interpretation'] = correlations_df['r'].apply(interpret_r)

    return correlations_df

def validate_correlations(correlations_df):
    """Validate correlation results."""
    # Check row count
    if len(correlations_df) != 4:
        logger.warning(f"Expected 4 rows, got {len(correlations_df)}")

    # Check r range
    if (correlations_df['r'] < -1).any() or (correlations_df['r'] > 1).any():
        raise ValueError("Correlation r out of bounds [-1, 1]")

    # Check p-value constraints
    if (correlations_df['p_bonferroni'] < correlations_df['p_uncorrected']).any():
        raise ValueError("p_bonferroni < p_uncorrected (impossible)")

    # Check convergence
    n_strong = (correlations_df['threshold_0.70']).sum()
    logger.info(f"Strong convergence (r > 0.70): {n_strong}/4 tests")

    logger.info("Correlations validation: PASS")
    return True

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 02: COMPUTE PEARSON CORRELATIONS (IRT vs CTT)")
    logger.info("=" * 60)

    try:
        # 1. Load input files
        logger.info("\n1. Loading input files...")
        irt_theta, ctt_scores = load_input_files()

        # 2. Reshape CTT to wide format
        logger.info("\n2. Reshaping CTT to wide format...")
        ctt_wide = reshape_ctt_to_wide(ctt_scores)

        # 3. Merge IRT and CTT
        logger.info("\n3. Merging IRT and CTT data...")
        merged_df = merge_irt_ctt(irt_theta, ctt_wide)

        # Save merged data
        merged_df.to_csv(DATA_PATH / "step02_merged_irt_ctt.csv", index=False)
        logger.info(f"   Saved: step02_merged_irt_ctt.csv ({len(merged_df)} rows)")

        # 4. Compute correlations
        logger.info("\n4. Computing Pearson correlations...")
        correlations_df = compute_correlations(merged_df)

        # 5. Apply Holm-Bonferroni correction
        logger.info("\n5. Applying Holm-Bonferroni correction (4 tests, alpha=0.05)...")
        correlations_df = apply_holm_bonferroni(correlations_df)

        # 6. Add threshold classifications
        logger.info("\n6. Adding threshold classifications...")
        correlations_df = add_threshold_classifications(correlations_df)

        # 7. Validate results
        logger.info("\n7. Validating results...")
        validate_correlations(correlations_df)

        # 8. Save output
        logger.info("\n8. Saving output files...")
        correlations_df.to_csv(DATA_PATH / "step02_correlations.csv", index=False)
        logger.info(f"   Saved: step02_correlations.csv ({len(correlations_df)} rows)")

        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("STEP 02 COMPLETE: Correlations computed successfully")
        logger.info("Correlations computed: 4 tests (IFR, ICR, IRE, Overall)")
        logger.info("Holm-Bonferroni correction applied (4 tests, alpha=0.05)")
        n_strong = (correlations_df['threshold_0.70']).sum()
        logger.info(f"Strong convergence (r > 0.70): {n_strong}/4 tests")
        logger.info("=" * 60)

        # Print summary table
        logger.info("\nCorrelation Summary:")
        for _, row in correlations_df.iterrows():
            logger.info(f"  {row['paradigm']}: r={row['r']:.3f}, p_holm={row['p_bonferroni']:.4f}, {row['interpretation']}")

    except Exception as e:
        logger.error(f"STEP 02 FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
