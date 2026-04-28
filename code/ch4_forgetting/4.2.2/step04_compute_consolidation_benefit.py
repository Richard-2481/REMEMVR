#!/usr/bin/env python3
"""Compute Consolidation Benefit Indices: Compute consolidation benefit index per domain (Early slope - Late slope)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_compute_consolidation_benefit.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute Consolidation Benefit Indices")
        log("")
        # Load Input Data

        log("Loading segment-domain slopes from Step 2...")
        input_path = RQ_DIR / "results" / "step02_segment_domain_slopes.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_slopes = pd.read_csv(input_path)
        log(f"step02_segment_domain_slopes.csv ({len(df_slopes)} rows, {len(df_slopes.columns)} cols)")

        # Validate input columns
        expected_cols = ["segment", "domain", "slope", "se", "CI_lower", "CI_upper"]
        actual_cols = list(df_slopes.columns)
        if actual_cols != expected_cols:
            raise ValueError(f"Column mismatch. Expected: {expected_cols}, Got: {actual_cols}")

        log(f"Input columns match expected: {expected_cols}")
        log("")
        # Pivot Data by Domain and Segment
        # Pivot to get Early and Late slopes side by side per domain
        # This enables vectorized benefit computation

        log("Organizing slopes by domain and segment...")

        # Create pivot table: rows = domain, columns = segment
        df_early = df_slopes[df_slopes["segment"] == "Early"].set_index("domain")
        df_late = df_slopes[df_slopes["segment"] == "Late"].set_index("domain")

        # Verify both segments exist for all domains (When excluded due to floor effect)
        domains = ["what", "where"]  # When excluded
        for domain in domains:
            if domain not in df_early.index:
                raise ValueError(f"Missing Early segment for domain: {domain}")
            if domain not in df_late.index:
                raise ValueError(f"Missing Late segment for domain: {domain}")

        log(f"All 2 domains present in both segments (When excluded)")
        log("")
        # Compute Consolidation Benefit Per Domain
        # Formula: consolidation_benefit = early_slope - late_slope
        # Interpretation:
        #   - Positive benefit = less forgetting in Early (consolidation protected memory)
        #   - Negative benefit = more forgetting in Early (no consolidation benefit)
        #   - Larger positive values = greater consolidation benefit
        #
        # SE computation via delta method: SE = sqrt(SE_early^2 + SE_late^2)
        # (assumes independence between segments, conservative estimate)

        log("Computing consolidation benefit indices...")

        benefit_records = []
        for domain in domains:
            early_slope = df_early.loc[domain, "slope"]
            early_se = df_early.loc[domain, "se"]
            late_slope = df_late.loc[domain, "slope"]
            late_se = df_late.loc[domain, "se"]

            # Consolidation benefit formula
            consolidation_benefit = early_slope - late_slope

            # SE via delta method (propagation of uncertainty)
            benefit_se = np.sqrt(early_se**2 + late_se**2)

            # 95% CI (z = 1.96 for normal approximation)
            z_critical = 1.96
            benefit_CI_lower = consolidation_benefit - z_critical * benefit_se
            benefit_CI_upper = consolidation_benefit + z_critical * benefit_se

            benefit_records.append({
                "domain": domain,
                "early_slope": early_slope,
                "late_slope": late_slope,
                "consolidation_benefit": consolidation_benefit,
                "benefit_se": benefit_se,
                "benefit_CI_lower": benefit_CI_lower,
                "benefit_CI_upper": benefit_CI_upper
            })

            log(f"  {domain}: Early={early_slope:.4f}, Late={late_slope:.4f}, Benefit={consolidation_benefit:.4f} (SE={benefit_se:.4f})")

        df_benefit = pd.DataFrame(benefit_records)
        log("")
        # Rank Domains by Consolidation Benefit
        # Rank by consolidation benefit magnitude (1 = most protected by consolidation)
        # Note: More negative benefit (more negative Early slope) means MORE forgetting in Early
        # We want to rank by PROTECTION, so larger (less negative) benefit = better = rank 1

        log("Ranking domains by consolidation benefit...")

        # Sort by consolidation benefit descending (larger = more protected = rank 1)
        # Note: All benefits are negative (forgetting slopes are negative)
        # Less negative benefit = less forgetting difference = better consolidation
        # Actually: early_slope - late_slope, where slopes are negative
        # More negative early_slope means MORE forgetting in early
        # So positive benefit would mean early forgetting > late forgetting (unusual)
        # Negative benefit means late forgetting > early forgetting (expected for consolidation)
        # WAIT: That's backwards for interpretation!
        #
        # Let me re-interpret:
        # - Slopes are already NEGATIVE (forgetting)
        # - Early slope = -0.50 means losing 0.50 theta per day in Early
        # - Late slope = -0.03 means losing 0.03 theta per day in Late
        # - Benefit = Early - Late = -0.50 - (-0.03) = -0.47
        #
        # So NEGATIVE benefit means more forgetting in Early than Late
        # This suggests consolidation DIDN'T protect memory (unusual result)
        #
        # For ranking:
        # - Rank 1 = BEST consolidation = most protected = LEAST negative benefit (closest to 0)
        # - Rank 2 = WORST consolidation = least protected = MOST negative benefit

        df_benefit = df_benefit.sort_values("consolidation_benefit", ascending=False)
        df_benefit["rank"] = range(1, len(df_benefit) + 1)

        # Log ranking interpretation
        log("  Ranking interpretation:")
        log("    Rank 1 = Best consolidation benefit (least negative = least excess early forgetting)")
        log("    Rank 2 = Worst consolidation benefit (most negative = most excess early forgetting)")
        log("")

        for _, row in df_benefit.iterrows():
            log(f"  Rank {int(row['rank'])}: {row['domain']} (benefit={row['consolidation_benefit']:.4f})")
        log("")
        # Save Consolidation Benefit Results
        # Output goes to results/ folder as this is a final report table
        # (consolidation benefit indices are a key thesis result)

        output_path = RQ_DIR / "results" / "step04_consolidation_benefit.csv"

        log(f"Saving consolidation benefit indices to {output_path}...")

        # Ensure column order matches specification
        df_benefit = df_benefit[["domain", "early_slope", "late_slope", "consolidation_benefit",
                                  "benefit_se", "benefit_CI_lower", "benefit_CI_upper", "rank"]]

        df_benefit.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step04_consolidation_benefit.csv ({len(df_benefit)} rows, {len(df_benefit.columns)} cols)")
        log("")
        # Inline Validation
        # Validate all criteria from 4_analysis.yaml

        log("Running inline validation checks...")
        validation_passed = True

        # Criterion 1: All domains computed (2 rows - When excluded)
        if len(df_benefit) != 2:
            log(f"Row count: Expected 2 (When excluded), got {len(df_benefit)}")
            validation_passed = False
        else:
            log("Row count: 2 rows (what, where - When excluded)")

        # Criterion 2: Ranks unique {1, 2}
        rank_set = set(df_benefit["rank"].astype(int).tolist())
        expected_ranks = {1, 2}
        if rank_set != expected_ranks:
            log(f"Ranks: Expected {expected_ranks}, got {rank_set}")
            validation_passed = False
        else:
            log("Ranks: {1, 2} (all unique)")

        # Criterion 3: Benefit values are numeric (not NaN)
        if df_benefit["consolidation_benefit"].isna().any():
            log("Benefit values: Contains NaN")
            validation_passed = False
        elif df_benefit["benefit_se"].isna().any():
            log("Benefit SE: Contains NaN")
            validation_passed = False
        else:
            log("Benefit values: All numeric (no NaN)")

        # Criterion 4: CI structure valid (CI_lower < benefit < CI_upper)
        ci_valid = True
        for _, row in df_benefit.iterrows():
            if not (row["benefit_CI_lower"] < row["consolidation_benefit"] < row["benefit_CI_upper"]):
                ci_valid = False
                log(f"CI structure for {row['domain']}: {row['benefit_CI_lower']:.4f} < {row['consolidation_benefit']:.4f} < {row['benefit_CI_upper']:.4f}")
                break
        if ci_valid:
            log("CI structure: benefit_CI_lower < consolidation_benefit < benefit_CI_upper (all rows)")
        else:
            validation_passed = False

        log("")

        if not validation_passed:
            raise ValueError("Validation failed - see log for details")

        log("Step 04 complete - Consolidation benefit indices computed")
        log("")

        # Final summary
        log("=" * 60)
        log("CONSOLIDATION BENEFIT SUMMARY")
        log("=" * 60)
        log("")
        log("Interpretation:")
        log("  - Early slope: forgetting rate per day during consolidation phase (test 0-1)")
        log("  - Late slope: forgetting rate per day during decay phase (test 3-6)")
        log("  - Consolidation benefit: Early - Late (negative = more Early forgetting)")
        log("")
        log("Results:")
        for _, row in df_benefit.iterrows():
            sig = "significant" if not (row["benefit_CI_lower"] <= 0 <= row["benefit_CI_upper"]) else "non-significant"
            log(f"  {row['domain']}: benefit={row['consolidation_benefit']:.4f} [{row['benefit_CI_lower']:.4f}, {row['benefit_CI_upper']:.4f}] ({sig})")
        log("")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
