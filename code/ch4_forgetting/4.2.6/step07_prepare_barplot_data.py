#!/usr/bin/env python3
"""
RQ 5.2.6 Step 07: Prepare Domain ICC Barplot Data

Purpose:
    Prepare plot source CSV for visualizing ICC_slope_conditional across domains.
    This enables comparison of between-person variance proportions.

Input:
    - data/step06_domain_icc_comparison.csv (2 rows: one per domain)

Output:
    - data/step07_domain_icc_barplot_data.csv (2 rows: plot-ready data)

Plot Description:
    Barplot comparing ICC_slope_conditional across two domains (What, Where)
    with threshold line at 0.40 (substantial reliability cutoff).
    Y-axis: ICC (0-1 scale)
    X-axis: Domain
    Colored bars indicate interpretation category (Low/Moderate/Substantial)

Note: Actual PNG plot will be generated later by rq_plots agent.

Author: Claude (g_code agent)
Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step07_prepare_barplot_data.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Prepare barplot data for domain ICC comparison."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 07: Prepare Domain ICC Barplot Data")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load domain comparison from Step 06
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.2.6/data/step06_domain_icc_comparison.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Comparison file not found: {input_path}")
        logger.error("Step 06 must complete before Step 07")
        sys.exit(1)

    comparison_df = pd.read_csv(input_path)
    logger.info(f"Loaded domain comparison: {len(comparison_df)} rows from {input_path}")

    # ---------------------------------------------------------------------
    # 2. Add plot-specific columns
    # ---------------------------------------------------------------------
    logger.info("\nAdding plot-specific columns...")

    # Plot order (alphabetical: 1=What, 2=Where)
    domain_order = {'What': 1, 'Where': 2}
    comparison_df['plot_order'] = comparison_df['domain'].map(domain_order)
    logger.info(f"  plot_order: {dict(zip(comparison_df['domain'], comparison_df['plot_order']))}")

    # Color category (matches interpretation)
    comparison_df['color_category'] = comparison_df['interpretation']
    logger.info(f"  color_category: {dict(zip(comparison_df['domain'], comparison_df['color_category']))}")

    # Threshold line (constant at 0.40)
    comparison_df['threshold_line'] = 0.40
    logger.info(f"  threshold_line: 0.40 for all rows")

    # ---------------------------------------------------------------------
    # 3. Select and order output columns
    # ---------------------------------------------------------------------
    output_cols = [
        'domain',
        'icc_slope_conditional',
        'interpretation',
        'plot_order',
        'color_category',
        'threshold_line'
    ]

    barplot_data = comparison_df[output_cols].copy()

    # Sort by plot_order
    barplot_data = barplot_data.sort_values('plot_order').reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking barplot data")
    logger.info("=" * 60)

    # Check row count
    if len(barplot_data) != 2:  # 2 domains (What, Where)
        logger.error(f"Expected 2 rows (2 domains), got {len(barplot_data)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(barplot_data)}: PASS")

    # Check all domains present
    domains = set(barplot_data['domain'])
    expected_domains = {'What', 'Where'}
    if domains != expected_domains:
        logger.error(f"Expected domains {expected_domains}, got {domains}")
        sys.exit(1)
    logger.info(f"  All domains present: PASS")

    # Check unique plot_order
    plot_orders = barplot_data['plot_order'].tolist()
    if len(set(plot_orders)) != len(plot_orders):
        logger.error(f"Duplicate plot_order values: {plot_orders}")
        sys.exit(1)
    logger.info(f"  Unique plot_order: PASS")

    # Check threshold_line constant
    if not all(barplot_data['threshold_line'] == 0.40):
        logger.error("threshold_line not constant at 0.40")
        sys.exit(1)
    logger.info(f"  threshold_line constant at 0.40: PASS")

    # Check no NaN values
    if barplot_data.isna().any().any():
        logger.error("NaN values found in barplot data")
        sys.exit(1)
    logger.info(f"  No NaN values: PASS")

    # Check color_category matches interpretation
    if not all(barplot_data['color_category'] == barplot_data['interpretation']):
        logger.error("color_category does not match interpretation")
        sys.exit(1)
    logger.info(f"  color_category matches interpretation: PASS")

    # ---------------------------------------------------------------------
    # 5. Save barplot data
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step07_domain_icc_barplot_data.csv")
    barplot_data.to_csv(output_path, index=False)
    logger.info(f"\nSaved barplot data to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Preview for rq_plots agent
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("BARPLOT DATA PREVIEW (for rq_plots)")
    logger.info("=" * 60)

    logger.info(f"\n{'Domain':<10} {'ICC':<10} {'Interp':<15} {'Order':<6} {'Color':<15}")
    logger.info("-" * 60)
    for _, row in barplot_data.iterrows():
        logger.info(f"{row['domain']:<10} {row['icc_slope_conditional']:<10.4f} {row['interpretation']:<15} {int(row['plot_order']):<6} {row['color_category']:<15}")

    logger.info(f"\nThreshold line: 0.40 (horizontal reference)")

    # Plot instructions for rq_plots
    logger.info("\n" + "-" * 60)
    logger.info("PLOT INSTRUCTIONS (for rq_plots agent):")
    logger.info("-" * 60)
    logger.info("Type: Grouped barplot")
    logger.info("X-axis: domain (What, Where)")
    logger.info("Y-axis: icc_slope_conditional (0-1 scale)")
    logger.info("Colors: By color_category (Low=red, Moderate=yellow, Substantial=green)")
    logger.info("Threshold: Horizontal line at 0.40")
    logger.info("Title: 'Domain-Specific ICC (Slope Conditional at Day 6)'")
    logger.info("Note: When domain excluded due to floor effect")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 07 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Prepared barplot data for {len(barplot_data)} domains")
    logger.info("Plot source CSV ready for rq_plots agent")
    logger.info("\nAll analysis steps complete for RQ 5.2.6")


if __name__ == "__main__":
    main()
