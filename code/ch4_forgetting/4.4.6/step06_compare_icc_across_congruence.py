#!/usr/bin/env python3
"""
Step 06: Compare ICC Across Congruence Levels

PURPOSE:
Compare ICC estimates across the three congruence levels (Common, Congruent,
Incongruent) to assess differential stability of schema-based memory. Rank
congruence levels by ICC_slope magnitude. Create bar plot for visualization.

EXPECTED INPUTS:
- data/step03_icc_estimates.csv: ICC estimates (9 rows: 3 types x 3 congruence)

EXPECTED OUTPUTS:
- data/step06_congruence_icc_comparison.csv: ICC comparison table (3 rows x 5 columns)
- data/step06_icc_comparison_interpretation.txt: Text report with ranking
- data/step06_congruence_icc_barplot.png: Bar plot (800x600 @ 300 DPI)

VALIDATION CRITERIA:
- Exactly 3 rows in comparison table
- All 3 congruence levels present
- All ICC values in [0, 1]
- rank_by_slope contains {1, 2, 3} with no duplicates
- PNG file > 10KB
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_compare_icc_across_congruence.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 06: Compare ICC Across Congruence Levels")

        # =====================================================================
        # STEP 1: Load ICC Estimates
        # =====================================================================
        log("[LOAD] Loading ICC estimates...")

        icc_file = RQ_DIR / "data" / "step03_icc_estimates.csv"
        df_icc = pd.read_csv(icc_file, encoding='utf-8')

        log(f"[LOADED] {icc_file.name} ({len(df_icc)} rows)")

        # =====================================================================
        # STEP 2: Pivot to Wide Format
        # =====================================================================
        log("[TRANSFORM] Pivoting ICC estimates to wide format...")

        # Pivot: rows=congruence, columns=icc_type
        df_wide = df_icc.pivot(index='congruence', columns='icc_type', values='value')

        # Reset index to make congruence a column
        df_wide = df_wide.reset_index()

        log(f"[PIVOTED] {len(df_wide)} rows (one per congruence)")
        log(f"  Columns: {list(df_wide.columns)}")

        # =====================================================================
        # STEP 3: Rank by ICC_slope_simple
        # =====================================================================
        log("[RANK] Ranking congruence levels by ICC slope (descending)...")

        # Rank by icc_slope_simple (1 = highest, 3 = lowest)
        df_wide['rank_by_slope'] = df_wide['slope_simple'].rank(ascending=False).astype(int)

        # Sort by rank
        df_wide = df_wide.sort_values('rank_by_slope')

        log("\n[RANKING]")
        for _, row in df_wide.iterrows():
            log(f"  Rank {row['rank_by_slope']}: {row['congruence']:12s} (ICC_slope={row['slope_simple']:.4f})")

        # =====================================================================
        # STEP 4: Save Comparison Table
        # =====================================================================
        log("\n[SAVE] Saving ICC comparison table...")

        output_file = RQ_DIR / "data" / "step06_congruence_icc_comparison.csv"
        df_wide.to_csv(output_file, index=False, encoding='utf-8')

        log(f"[SAVED] {output_file.name} ({len(df_wide)} rows)")

        # =====================================================================
        # STEP 5: Create Bar Plot
        # =====================================================================
        log("[PLOT] Creating ICC comparison bar plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for grouped bar chart
        congruence_labels = df_wide['congruence'].values
        icc_intercept = df_wide['intercept'].values
        icc_slope_simple = df_wide['slope_simple'].values
        icc_slope_conditional = df_wide['slope_conditional'].values

        # Bar positions
        x = np.arange(len(congruence_labels))
        width = 0.25

        # Create bars
        ax.bar(x - width, icc_intercept, width, label='ICC Intercept',
              color='steelblue', edgecolor='black')
        ax.bar(x, icc_slope_simple, width, label='ICC Slope (Simple)',
              color='coral', edgecolor='black')
        ax.bar(x + width, icc_slope_conditional, width, label='ICC Slope (Conditional)',
              color='lightgreen', edgecolor='black')

        # Add reference lines
        ax.axhline(y=0.20, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                  label='Moderate threshold (0.20)')
        ax.axhline(y=0.40, color='black', linestyle='--', linewidth=1, alpha=0.7,
                  label='Substantial threshold (0.40)')

        # Formatting
        ax.set_xlabel('Congruence Level', fontsize=12)
        ax.set_ylabel('ICC Value', fontsize=12)
        ax.set_title('ICC Comparison Across Congruence Levels', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(congruence_labels)
        ax.set_ylim(0, max(1.0, max(icc_intercept.max(), icc_slope_simple.max(),
                                   icc_slope_conditional.max()) + 0.1))
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        plot_file = RQ_DIR / "data" / "step06_congruence_icc_barplot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"[SAVED] {plot_file.name}")

        # =====================================================================
        # STEP 6: Create Interpretation Report
        # =====================================================================
        log("[REPORT] Creating interpretation report...")

        report_path = RQ_DIR / "data" / "step06_icc_comparison_interpretation.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ICC COMPARISON ACROSS CONGRUENCE LEVELS\n")
            f.write("=" * 80 + "\n\n")

            f.write("RANKING BY ICC_SLOPE_SIMPLE (Forgetting Rate Stability):\n")
            f.write("-" * 80 + "\n")

            for _, row in df_wide.iterrows():
                f.write(f"\nRank {row['rank_by_slope']}: {row['congruence']}\n")
                f.write(f"  ICC Intercept:         {row['intercept']:.4f}\n")
                f.write(f"  ICC Slope (Simple):    {row['slope_simple']:.4f}\n")
                f.write(f"  ICC Slope (Cond.):     {row['slope_conditional']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n\n")

            f.write("INTERPRETATION:\n")
            f.write("-" * 80 + "\n\n")

            # Identify highest stability congruence
            highest_rank = df_wide[df_wide['rank_by_slope'] == 1].iloc[0]

            f.write(f"HIGHEST STABILITY: {highest_rank['congruence']} congruence\n")
            f.write(f"  ICC_slope = {highest_rank['slope_simple']:.4f}\n")

            if highest_rank['slope_simple'] >= 0.40:
                f.write("  -> SUBSTANTIAL between-person differences in forgetting rates\n")
            elif highest_rank['slope_simple'] >= 0.20:
                f.write("  -> MODERATE between-person differences in forgetting rates\n")
            else:
                f.write("  -> LOW between-person differences in forgetting rates\n")

            f.write("\n")

            # Compare across congruence levels
            f.write("THEORETICAL IMPLICATIONS:\n\n")

            # Get ICC slope values for each congruence
            icc_common = df_wide[df_wide['congruence'] == 'Common']['slope_simple'].values[0]
            icc_congruent = df_wide[df_wide['congruence'] == 'Congruent']['slope_simple'].values[0]
            icc_incongruent = df_wide[df_wide['congruence'] == 'Incongruent']['slope_simple'].values[0]

            # Identify pattern
            if icc_congruent > icc_incongruent:
                f.write("PATTERN: Congruent > Incongruent ICC\n")
                f.write("  Schema congruence predicts MORE stable individual differences\n")
                f.write("  in forgetting rates (stronger trait-like stability).\n\n")
            elif icc_incongruent > icc_congruent:
                f.write("PATTERN: Incongruent > Congruent ICC\n")
                f.write("  Incongruent information paradoxically shows MORE stable\n")
                f.write("  individual differences in forgetting rates.\n\n")
            else:
                f.write("PATTERN: Congruent ~ Incongruent ICC\n")
                f.write("  No differential stability by schema congruence.\n\n")

            # Compare Common to schema-specific
            if icc_common > max(icc_congruent, icc_incongruent):
                f.write("Common items show HIGHEST stability (most trait-like).\n")
            elif icc_common < min(icc_congruent, icc_incongruent):
                f.write("Common items show LOWEST stability (most situation-dependent).\n")
            else:
                f.write("Common items show intermediate stability.\n")

        log(f"[SAVED] {report_path.name}")

        # =====================================================================
        # STEP 7: Validate Comparison Table
        # =====================================================================
        log("\n[VALIDATION] Validating comparison table...")

        # Check row count
        if len(df_wide) != 3:
            raise ValueError(f"Expected 3 rows (one per congruence), got {len(df_wide)}")

        log("[PASS] Comparison table has 3 rows")

        # Check congruence levels
        expected_congruence = {'Common', 'Congruent', 'Incongruent'}
        actual_congruence = set(df_wide['congruence'].values)

        if actual_congruence != expected_congruence:
            raise ValueError(f"Expected congruence levels {expected_congruence}, got {actual_congruence}")

        log("[PASS] All 3 congruence levels present")

        # Check ICC values in [0, 1]
        icc_cols = ['intercept', 'slope_simple', 'slope_conditional']
        for col in icc_cols:
            if (df_wide[col] < 0).any() or (df_wide[col] > 1).any():
                raise ValueError(f"ICC values in {col} outside [0, 1] range")

        log("[PASS] All ICC values in [0, 1] range")

        # Check ranking
        ranks = set(df_wide['rank_by_slope'].values)
        if ranks != {1, 2, 3}:
            raise ValueError(f"Expected ranks {{1, 2, 3}}, got {ranks}")

        log("[PASS] Ranking valid (1, 2, 3 with no duplicates)")

        # Check plot file
        if not plot_file.exists():
            raise FileNotFoundError(f"Plot file not found: {plot_file}")

        if plot_file.stat().st_size < 10000:
            raise ValueError(f"Plot file too small (< 10KB): {plot_file.stat().st_size} bytes")

        log(f"[PASS] Plot file validated ({plot_file.stat().st_size} bytes)")

        log("\n[SUCCESS] Step 06 complete - ICC comparison across congruence complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
