#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.2.6 random effects data exists and contains required domain slope data."""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.3
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Ch5 5.2.6 random effects file
CH5_RANDOM_EFFECTS = PROJECT_ROOT / "results" / "ch5" / "5.2.6" / "data" / "step04_random_effects.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")
        # Validate Ch5 5.2.6 Random Effects File

        log(f"Checking Ch5 5.2.6 random effects file...")
        log(f"{CH5_RANDOM_EFFECTS}")

        validation_result = check_file_exists(
            file_path=str(CH5_RANDOM_EFFECTS),
            min_size_bytes=500  # Ensures file has meaningful content
        )

        if not validation_result.get('valid', False):
            log(f"Ch5 5.2.6 random effects file validation failed")
            log(f"{validation_result}")
            sys.exit(1)

        log(f"File exists: {CH5_RANDOM_EFFECTS}")
        log(f"{validation_result.get('size_bytes', 0)} bytes")
        # Load and Validate Domain Data
        # Critical: When domain should be ABSENT (floor effects)

        log(f"Reading random effects data...")
        df = pd.read_csv(CH5_RANDOM_EFFECTS)
        log(f"{len(df)} rows, {len(df.columns)} columns")

        # Check columns
        expected_cols = ['UID', 'domain', 'Total_Intercept', 'Total_Slope', 'intercept_se', 'slope_se']
        actual_cols = df.columns.tolist()

        if actual_cols != expected_cols:
            log(f"Column mismatch")
            log(f"{expected_cols}")
            log(f"{actual_cols}")
            sys.exit(1)

        log(f"Columns match expected format")

        # Check domain counts
        domain_counts = df['domain'].value_counts().to_dict()
        log(f"[DOMAIN COUNTS] {domain_counts}")

        # Validate What domain
        if 'What' not in domain_counts:
            log(f"What domain missing from data")
            sys.exit(1)
        elif domain_counts['What'] != 100:
            log(f"What domain has {domain_counts['What']} participants, expected 100")
            sys.exit(1)
        else:
            log(f"What domain: 100 participants")

        # Validate Where domain
        if 'Where' not in domain_counts:
            log(f"Where domain missing from data")
            sys.exit(1)
        elif domain_counts['Where'] != 100:
            log(f"Where domain has {domain_counts['Where']} participants, expected 100")
            sys.exit(1)
        else:
            log(f"Where domain: 100 participants")

        # Check When domain absence (EXPECTED)
        if 'When' in domain_counts:
            log(f"When domain present with {domain_counts['When']} participants")
            log(f"This RQ analysis plan excludes When domain - unexpected data")
        else:
            log(f"When domain not present")
            log(f"When domain excluded from Ch5 5.2.6 analysis due to floor effects")
            log(f"Ch5 5.2.3 purification excluded 77% of When items (poor quality)")
            log(f"Insufficient valid items for reliable slope estimation")
        # Save Validation Report
        # Content: File paths, domain counts, When exclusion reason

        log(f"Writing validation report...")

        report_lines = [
            "=" * 80,
            "RQ 7.6.3 - DEPENDENCY VALIDATION REPORT",
            "=" * 80,
            "",
            "CRITICAL MODIFICATION:",
            "  - Original plan: 3 domains (What, Where, When)",
            "  - Actual analysis: 2 domains (What, Where ONLY)",
            "  - Reason: When domain excluded due to floor effects in Ch5 5.2.3",
            "",
            "WHEN DOMAIN EXCLUSION RATIONALE:",
            "  - Ch5 5.2.3 IRT purification excluded 77% of When items",
            "  - Insufficient valid items for reliable slope estimation",
            "  - Would compromise ICC calculation validity",
            "  - Decision: Proceed with What + Where only",
            "",
            "DATA SOURCE VALIDATION:",
            f"  File: {CH5_RANDOM_EFFECTS}",
            f"  Size: {validation_result.get('size_bytes', 0)} bytes",
            f"  Status: PASS",
            "",
            "DOMAIN AVAILABILITY:",
            f"  What domain: {domain_counts.get('What', 0)} participants - VALID",
            f"  Where domain: {domain_counts.get('Where', 0)} participants - VALID",
            f"  When domain: {'PRESENT (unexpected)' if 'When' in domain_counts else 'ABSENT (expected)'}",
            "",
            "ANALYSIS MODIFICATIONS:",
            "  - step02: Expected 3 rows -> 2 rows (2 domains)",
            "  - step03: Expected 3000 rows -> 2000 rows (1000 bootstrap x 2 domains)",
            "  - step04: Expected 3 pairwise tests -> 1 test (What vs Where only)",
            "  - step04: Bonferroni correction factor = 1 (single test)",
            "",
            "VALIDATION OUTCOME: PASS",
            "  - All required data available for 2-domain analysis",
            "  - Proceed to step01 (extract and merge domain slopes)",
            "",
            "=" * 80,
        ]

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        log(f"{OUTPUT_FILE}")
        log(f"Step 00 complete - Dependencies validated")
        log(f"Proceed to step01 with 2-domain analysis (What + Where)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
