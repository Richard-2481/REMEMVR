#!/usr/bin/env python3
"""
RQ 5.5.7 Step 05: Characterize Clusters

Purpose: Characterize clusters by computing mean/SD per feature per cluster
(original scale, not z-scores), assign interpretive labels, create
human-readable descriptions.

Input:
- data/step00_random_effects_from_rq556.csv (100 rows, original scale)
- data/step04_cluster_assignments.csv (100 rows: UID, cluster)

Output:
- data/step05_cluster_characterization.csv (K rows with means, SDs, labels)
- data/step05_cluster_descriptions.txt (human-readable paragraphs)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "step05_characterize_clusters.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def assign_cluster_label(row, grand_means):
    """
    Assign interpretive label based on cluster feature patterns.

    Classification based on:
    - High/Low intercepts (relative to grand mean, 0 for random effects)
    - Pattern of slopes (positive = forgetting faster, negative = maintaining)
    """
    s_int = row['Source_intercept_mean']
    s_slope = row['Source_slope_mean']
    d_int = row['Destination_intercept_mean']
    d_slope = row['Destination_slope_mean']

    # Classify intercepts
    s_int_level = "High" if s_int > 0 else "Low"
    d_int_level = "High" if d_int > 0 else "Low"

    # Combined pattern
    if s_int > 0 and d_int > 0:
        baseline = "Dual High"
    elif s_int < 0 and d_int < 0:
        baseline = "Dual Low"
    elif s_int > 0 and d_int < 0:
        baseline = "Source > Destination"
    else:  # s_int < 0 and d_int > 0
        baseline = "Destination > Source"

    # Slope pattern (key finding from RQ 5.5.6)
    # Positive slope = faster forgetting, negative = maintaining advantage
    if s_slope > 0 and d_slope < 0:
        trajectory = "Source declines, Destination maintains"
    elif s_slope < 0 and d_slope > 0:
        trajectory = "Source maintains, Destination declines"
    elif s_slope > 0 and d_slope > 0:
        trajectory = "Both decline"
    else:
        trajectory = "Both maintain"

    # Create label
    return f"{baseline}: {trajectory}"


def main():
    """Characterize clusters with interpretive labels."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 05: Characterize Clusters")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load inputs
    # -------------------------------------------------------------------------
    original_path = DATA_DIR / "step00_random_effects_from_rq556.csv"
    assignments_path = DATA_DIR / "step04_cluster_assignments.csv"

    if not original_path.exists():
        logger.error(f"Input file not found: {original_path}")
        sys.exit(1)

    if not assignments_path.exists():
        logger.error(f"Input file not found: {assignments_path}")
        sys.exit(1)

    df_original = pd.read_csv(original_path)
    df_assignments = pd.read_csv(assignments_path)

    logger.info(f"Loaded {len(df_original)} rows (original scale)")
    logger.info(f"Loaded {len(df_assignments)} cluster assignments")

    # -------------------------------------------------------------------------
    # 2. Merge data
    # -------------------------------------------------------------------------
    df = df_original.merge(df_assignments, on='UID')
    logger.info(f"Merged data: {len(df)} rows")

    if len(df) != 100:
        logger.error(f"Expected 100 rows after merge, got {len(df)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Compute grand means (for labeling)
    # -------------------------------------------------------------------------
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']
    grand_means = {col: df[col].mean() for col in feature_cols}
    logger.info(f"Grand means (all near 0 for random effects):")
    for col, val in grand_means.items():
        logger.info(f"  {col}: {val:.4f}")

    # -------------------------------------------------------------------------
    # 4. Compute cluster-level statistics
    # -------------------------------------------------------------------------
    logger.info("\nComputing cluster statistics...")

    results = []
    K = df['cluster'].nunique()

    for cluster_id in range(K):
        cluster_df = df[df['cluster'] == cluster_id]
        N = len(cluster_df)

        row = {
            'cluster': cluster_id,
            'N': N
        }

        for col in feature_cols:
            row[f'{col}_mean'] = cluster_df[col].mean()
            row[f'{col}_sd'] = cluster_df[col].std(ddof=1)  # Sample SD

        results.append(row)

    df_char = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 5. Assign interpretive labels
    # -------------------------------------------------------------------------
    logger.info("\nAssigning interpretive labels...")

    labels = []
    for _, row in df_char.iterrows():
        label = assign_cluster_label(row, grand_means)
        labels.append(label)
        logger.info(f"  Cluster {int(row['cluster'])}: {label}")

    df_char['label'] = labels

    # -------------------------------------------------------------------------
    # 6. Validate results
    # -------------------------------------------------------------------------
    logger.info("\nValidating cluster characterization...")

    # Check all SDs >= 0
    sd_cols = [c for c in df_char.columns if c.endswith('_sd')]
    for col in sd_cols:
        if (df_char[col] < 0).any():
            logger.error(f"Negative SD found in {col}")
            sys.exit(1)
    logger.info("All SDs >= 0: PASS")

    # Check all N > 0
    if (df_char['N'] <= 0).any():
        logger.error("Found cluster with N <= 0")
        sys.exit(1)
    logger.info("All N > 0: PASS")

    # Check N sums to 100
    total_n = df_char['N'].sum()
    if total_n != 100:
        logger.error(f"N sums to {total_n}, expected 100")
        sys.exit(1)
    logger.info(f"Sum of N = 100: PASS")

    # Check all labels non-empty
    if df_char['label'].isna().any() or (df_char['label'] == '').any():
        logger.error("Found cluster with empty label")
        sys.exit(1)
    logger.info("All clusters labeled: PASS")

    # -------------------------------------------------------------------------
    # 7. Create human-readable descriptions
    # -------------------------------------------------------------------------
    logger.info("\nCreating human-readable descriptions...")

    descriptions = []
    descriptions.append("=" * 70)
    descriptions.append("RQ 5.5.7: Source-Destination Memory Clustering - Cluster Descriptions")
    descriptions.append("=" * 70)
    descriptions.append("")

    for _, row in df_char.iterrows():
        cluster_id = int(row['cluster'])
        N = int(row['N'])
        label = row['label']

        s_int = row['Source_intercept_mean']
        s_slope = row['Source_slope_mean']
        d_int = row['Destination_intercept_mean']
        d_slope = row['Destination_slope_mean']

        desc = f"""
Cluster {cluster_id} (N={N}, '{label}'):
  This cluster contains {N} participants ({N}% of sample).

  Source Memory:
    - Baseline (intercept): {s_int:.3f} theta ({'above' if s_int > 0 else 'below'} average)
    - Trajectory (slope): {s_slope:.4f} theta/day ({'faster' if s_slope > 0 else 'slower'} forgetting)

  Destination Memory:
    - Baseline (intercept): {d_int:.3f} theta ({'above' if d_int > 0 else 'below'} average)
    - Trajectory (slope): {d_slope:.4f} theta/day ({'faster' if d_slope > 0 else 'slower'} forgetting)

  Interpretation: Participants in this cluster show {label.lower()}.
"""
        descriptions.append(desc)

    # Add note about opposite correlations
    descriptions.append("-" * 70)
    descriptions.append("""
NOTE ON INTERCEPT-SLOPE PATTERNS:

RQ 5.5.6 discovered that source and destination memory show OPPOSITE
intercept-slope correlations:
  - Source: r = +0.99 (high baseline = faster forgetting = regression to mean)
  - Destination: r = -0.90 (high baseline = slower forgetting = advantage maintenance)

These opposite patterns are reflected in the cluster profiles above.
""")

    descriptions_text = "\n".join(descriptions)

    # -------------------------------------------------------------------------
    # 8. Save outputs
    # -------------------------------------------------------------------------
    char_path = DATA_DIR / "step05_cluster_characterization.csv"
    df_char.to_csv(char_path, index=False)
    logger.info(f"\nSaved characterization to {char_path}")

    desc_path = DATA_DIR / "step05_cluster_descriptions.txt"
    with open(desc_path, 'w') as f:
        f.write(descriptions_text)
    logger.info(f"Saved descriptions to {desc_path}")

    # -------------------------------------------------------------------------
    # 9. Print summary table
    # -------------------------------------------------------------------------
    logger.info("\nCluster Characterization Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Cluster':<8} {'N':<5} {'S_int':<10} {'S_slope':<10} "
               f"{'D_int':<10} {'D_slope':<10} {'Label'}")
    logger.info("-" * 80)

    for _, row in df_char.iterrows():
        logger.info(f"{int(row['cluster']):<8} {int(row['N']):<5} "
                   f"{row['Source_intercept_mean']:<10.4f} "
                   f"{row['Source_slope_mean']:<10.4f} "
                   f"{row['Destination_intercept_mean']:<10.4f} "
                   f"{row['Destination_slope_mean']:<10.4f} "
                   f"{row['label'][:30]}")

    logger.info("-" * 80)

    # -------------------------------------------------------------------------
    # 10. Final summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 05 COMPLETE")
    logger.info(f"  Characterized {K} clusters")
    logger.info(f"  Cluster sizes: {list(df_char['N'])}")
    logger.info("  All clusters labeled")
    logger.info(f"  Output: {char_path}")
    logger.info(f"  Output: {desc_path}")
    logger.info("=" * 60)

    return df_char

if __name__ == "__main__":
    main()
