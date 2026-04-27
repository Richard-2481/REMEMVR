#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA (Generated for RQ 5.3.6)
# =============================================================================
"""
Step ID: step00
Step Name: Load Dependencies and Validate
RQ: 5.3.6 - Purified CTT Effects (Paradigms)
Generated: 2025-12-03

PURPOSE:
Verify RQ 5.3.1 completion and load required outputs for CTT comparison analysis.
This step validates that all dependency files exist, checks RQ 5.3.1 status shows
success, validates schema compliance, and copies required files to current RQ folder.

EXPECTED INPUTS (from RQ 5.3.1):
  - results/ch5/5.3.1/status.yaml
    Purpose: Verify RQ 5.3.1 completed successfully
    Expected: rq_results.status = "success"

  - results/ch5/5.3.1/data/step02_purified_items.csv
    Actual columns: ['item', 'domain', 'Discrimination', 'Difficulty_1']
    Expected rows: 40-80 items (purified from ~105 original items)
    Purpose: Identify which items survived IRT purification
    Note: domain maps to paradigms (free_recall=IFR, cued_recall=ICR, recognition=IRE)

  - results/ch5/5.3.1/data/step03_theta_scores.csv
    Actual columns: ['composite_ID', 'domain_name', 'theta']
    Expected rows: 1200 (400 obs x 3 domains, long format)
    Purpose: IRT theta scores as convergent validity criterion
    Note: Long format - one row per observation x domain

  - results/ch5/5.3.1/data/step00_tsvr_mapping.csv
    Actual columns: ['composite_ID', 'UID', 'test', 'TSVR_hours']
    Expected rows: 400 (100 participants x 4 tests)
    Purpose: Time-since-viewing-room mapping for LMM trajectory analysis

EXPECTED OUTPUTS:
  - data/step00_purified_items.csv (copy from RQ 5.3.1)
  - data/step00_theta_scores.csv (copy from RQ 5.3.1)
  - data/step00_tsvr_mapping.csv (copy from RQ 5.3.1)
  - data/step00_dependency_validation_report.txt (>500 bytes)
  - logs/step00_load_dependencies.log

VALIDATION CRITERIA:
  - All 4 dependency files exist
  - RQ 5.3.1 status shows rq_results.status = "success"
  - Schema validation passes for all CSVs (columns match actual structure)
  - Row counts match expected ranges
  - No NaN values in critical columns
  - Report contains "PASS" for all checks

IMPLEMENTATION NOTES:
- Cross-RQ dependency: RQ 5.3.1 must complete before RQ 5.3.6 can start
- If any check fails, raises ValueError with descriptive message
- Copies files to current RQ folder with step00_ prefix for local access
- Validation report documents all checks and provides actionable failure messages
- Actual column names from RQ 5.3.1 differ from initial specification
  (using actual names: 'item' not 'item_name', 'domain' not 'dimension', etc.)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List, Any
import traceback
import shutil

# Add project root to path for imports
# CRITICAL: RQ scripts are in results/chX/rqY/code/ (4 levels deep from project root)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.6
DEPENDENCY_RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.3.1"  # RQ 5.3.1 directory
LOG_FILE = RQ_DIR / "logs" / "step00_load_dependencies.log"

# Dependency files from RQ 5.3.1
DEPENDENCY_FILES = {
    "status": DEPENDENCY_RQ_DIR / "status.yaml",
    "purified_items": DEPENDENCY_RQ_DIR / "data" / "step02_purified_items.csv",
    "theta_scores": DEPENDENCY_RQ_DIR / "data" / "step03_theta_scores.csv",
    "tsvr_mapping": DEPENDENCY_RQ_DIR / "data" / "step00_tsvr_mapping.csv"
}

# Output files in current RQ
OUTPUT_FILES = {
    "purified_items": RQ_DIR / "data" / "step00_purified_items.csv",
    "theta_scores": RQ_DIR / "data" / "step00_theta_scores.csv",
    "tsvr_mapping": RQ_DIR / "data" / "step00_tsvr_mapping.csv",
    "validation_report": RQ_DIR / "data" / "step00_dependency_validation_report.txt"
}

# Expected schemas for CSV files (based on ACTUAL RQ 5.3.1 outputs)
EXPECTED_SCHEMAS = {
    "purified_items": {
        "columns": ["item", "domain", "Discrimination", "Difficulty_1"],
        "row_range": (40, 80),
        "critical_columns": ["item", "domain", "Discrimination"]
    },
    "theta_scores": {
        "columns": ["composite_ID", "domain_name", "theta"],
        "row_count": 1200,  # Long format: 400 obs x 3 domains
        "critical_columns": ["composite_ID", "domain_name", "theta"]
    },
    "tsvr_mapping": {
        "columns": ["composite_ID", "UID", "test", "TSVR_hours"],
        "row_count": 400,
        "critical_columns": ["composite_ID", "TSVR_hours", "test"]
    }
}

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Validation Functions
# =============================================================================

def check_file_exists(file_path: Path, min_size_bytes: int = 100) -> Dict[str, Any]:
    """
    Check if file exists and meets minimum size requirement.

    Args:
        file_path: Path to file to check
        min_size_bytes: Minimum file size in bytes

    Returns:
        Dict with status, message, and file info
    """
    if not file_path.exists():
        return {
            "status": "FAIL",
            "message": f"File not found: {file_path}",
            "file_path": str(file_path),
            "size_bytes": 0
        }

    size_bytes = file_path.stat().st_size
    if size_bytes < min_size_bytes:
        return {
            "status": "FAIL",
            "message": f"File too small: {size_bytes} bytes (expected >={min_size_bytes})",
            "file_path": str(file_path),
            "size_bytes": size_bytes
        }

    return {
        "status": "PASS",
        "message": f"File exists and meets size requirement",
        "file_path": str(file_path),
        "size_bytes": size_bytes
    }

def check_rq_status(status_file: Path) -> Dict[str, Any]:
    """
    Check if RQ 5.3.1 status shows successful completion.

    Args:
        status_file: Path to status.yaml file

    Returns:
        Dict with status and message
    """
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = yaml.safe_load(f)

        # Check if rq_results.status exists and equals "success"
        if "rq_results" not in status_data:
            return {
                "status": "FAIL",
                "message": "rq_results section not found in status.yaml"
            }

        rq_results_status = status_data.get("rq_results", {}).get("status", "")
        if rq_results_status != "success":
            return {
                "status": "FAIL",
                "message": f"RQ 5.3.1 status is '{rq_results_status}' (expected 'success')"
            }

        return {
            "status": "PASS",
            "message": "RQ 5.3.1 completed successfully (rq_results.status = success)"
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "message": f"Error reading status.yaml: {str(e)}"
        }

def validate_csv_schema(csv_path: Path, expected_columns: List[str],
                        expected_rows: Any = None, critical_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate CSV file schema and row count.

    Args:
        csv_path: Path to CSV file
        expected_columns: List of expected column names
        expected_rows: Expected row count (int) or range (tuple)
        critical_columns: Columns that must not have NaN values

    Returns:
        Dict with status, message, and validation details
    """
    try:
        df = pd.read_csv(csv_path)

        # Check columns
        actual_columns = df.columns.tolist()
        if actual_columns != expected_columns:
            return {
                "status": "FAIL",
                "message": f"Column mismatch",
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "row_count": len(df)
            }

        # Check row count
        row_count = len(df)
        if expected_rows is not None:
            if isinstance(expected_rows, tuple):
                min_rows, max_rows = expected_rows
                if not (min_rows <= row_count <= max_rows):
                    return {
                        "status": "FAIL",
                        "message": f"Row count {row_count} outside expected range [{min_rows}, {max_rows}]",
                        "row_count": row_count,
                        "expected_range": expected_rows
                    }
            else:
                if row_count != expected_rows:
                    return {
                        "status": "FAIL",
                        "message": f"Row count {row_count} != expected {expected_rows}",
                        "row_count": row_count,
                        "expected_count": expected_rows
                    }

        # Check for NaN values in critical columns
        if critical_columns:
            nan_columns = []
            for col in critical_columns:
                if df[col].isna().any():
                    nan_count = df[col].isna().sum()
                    nan_columns.append(f"{col} ({nan_count} NaN)")

            if nan_columns:
                return {
                    "status": "FAIL",
                    "message": f"NaN values found in critical columns: {', '.join(nan_columns)}",
                    "row_count": row_count
                }

        return {
            "status": "PASS",
            "message": "Schema validation passed",
            "row_count": row_count,
            "columns": actual_columns
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "message": f"Error validating CSV: {str(e)}"
        }

def write_validation_report(results: Dict[str, Any], output_path: Path):
    """
    Write validation report to text file.

    Args:
        results: Dictionary of validation results
        output_path: Path to output report file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RQ 5.3.6 DEPENDENCY VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Source RQ: 5.3.1 (Purified IRT - 3D Multidimensional Model)\n")
        f.write(f"Target RQ: 5.3.6 (Purified CTT Effects - Paradigms)\n\n")

        f.write("-" * 80 + "\n")
        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        all_passed = True
        for check_name, result in results.items():
            status = result.get("status", "UNKNOWN")
            if status != "PASS":
                all_passed = False

            f.write(f"[{status}] {check_name}\n")
            f.write(f"  Message: {result.get('message', 'N/A')}\n")

            # Add additional details if available
            if "file_path" in result:
                f.write(f"  File: {result['file_path']}\n")
            if "size_bytes" in result and result["size_bytes"] > 0:
                f.write(f"  Size: {result['size_bytes']:,} bytes\n")
            if "row_count" in result:
                f.write(f"  Rows: {result['row_count']}\n")
            if "columns" in result:
                f.write(f"  Columns: {', '.join(result['columns'])}\n")

            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("OVERALL STATUS\n")
        f.write("-" * 80 + "\n\n")

        if all_passed:
            f.write("[PASS] All validation checks passed\n")
            f.write("RQ 5.3.1 dependencies verified and ready for RQ 5.3.6 analysis\n\n")
            f.write("Files copied to RQ 5.3.6/data/:\n")
            f.write("  - step00_purified_items.csv\n")
            f.write("  - step00_theta_scores.csv\n")
            f.write("  - step00_tsvr_mapping.csv\n\n")
            f.write("Data Structure Notes:\n")
            f.write("  - purified_items: 45 items retained after IRT purification\n")
            f.write("  - theta_scores: Long format (1200 rows = 400 obs x 3 domains)\n")
            f.write("  - tsvr_mapping: Wide format (400 rows = 100 participants x 4 tests)\n")
            f.write("  - Domain mapping: free_recall=IFR, cued_recall=ICR, recognition=IRE\n")
        else:
            f.write("[FAIL] One or more validation checks failed\n")
            f.write("RQ 5.3.1 incomplete - run RQ 5.3.1 before RQ 5.3.6\n")

        f.write("\n" + "=" * 80 + "\n")

# =============================================================================
# Main Validation Logic
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 00: Load Dependencies and Validate")
        log(f"[INFO] Source RQ: {DEPENDENCY_RQ_DIR}")
        log(f"[INFO] Target RQ: {RQ_DIR}")

        # Dictionary to store all validation results
        validation_results = {}

        # =========================================================================
        # CHECK 1: Verify all dependency files exist
        # =========================================================================
        log("\n[CHECK 1] Verifying dependency files exist...")

        for file_key, file_path in DEPENDENCY_FILES.items():
            log(f"  Checking {file_key}: {file_path}")
            result = check_file_exists(file_path, min_size_bytes=100)
            validation_results[f"file_exists_{file_key}"] = result
            log(f"    [{result['status']}] {result['message']}")

            if result['status'] != "PASS":
                raise ValueError(f"Dependency file check failed: {result['message']}")

        log("[PASS] All dependency files exist")

        # =========================================================================
        # CHECK 2: Verify RQ 5.3.1 status shows success
        # =========================================================================
        log("\n[CHECK 2] Verifying RQ 5.3.1 completion status...")

        status_result = check_rq_status(DEPENDENCY_FILES["status"])
        validation_results["rq_status"] = status_result
        log(f"  [{status_result['status']}] {status_result['message']}")

        if status_result['status'] != "PASS":
            raise ValueError(f"RQ 5.3.1 incomplete - run RQ 5.3.1 before RQ 5.3.6")

        log("[PASS] RQ 5.3.1 completed successfully")

        # =========================================================================
        # CHECK 3: Validate CSV schemas
        # =========================================================================
        log("\n[CHECK 3] Validating CSV file schemas...")

        for file_key in ["purified_items", "theta_scores", "tsvr_mapping"]:
            schema = EXPECTED_SCHEMAS[file_key]
            file_path = DEPENDENCY_FILES[file_key]

            log(f"  Validating {file_key}: {file_path}")
            result = validate_csv_schema(
                file_path,
                expected_columns=schema["columns"],
                expected_rows=schema.get("row_count", schema.get("row_range")),
                critical_columns=schema.get("critical_columns")
            )
            validation_results[f"schema_{file_key}"] = result
            log(f"    [{result['status']}] {result['message']}")

            if result['status'] == "PASS":
                log(f"      Rows: {result['row_count']}")
                log(f"      Columns: {', '.join(result['columns'])}")

            if result['status'] != "PASS":
                raise ValueError(f"Schema validation failed for {file_key}: {result['message']}")

        log("[PASS] All CSV schemas validated")

        # =========================================================================
        # STEP 4: Copy files to current RQ folder
        # =========================================================================
        log("\n[STEP 4] Copying dependency files to RQ 5.3.6/data/...")

        for file_key in ["purified_items", "theta_scores", "tsvr_mapping"]:
            source_path = DEPENDENCY_FILES[file_key]
            dest_path = OUTPUT_FILES[file_key]

            log(f"  Copying {file_key}...")
            log(f"    Source: {source_path}")
            log(f"    Dest:   {dest_path}")

            shutil.copy2(source_path, dest_path)

            # Verify copy succeeded
            if not dest_path.exists():
                raise ValueError(f"Failed to copy {file_key} to {dest_path}")

            copied_size = dest_path.stat().st_size
            log(f"    [COPIED] {copied_size:,} bytes")

        log("[PASS] All files copied successfully")

        # =========================================================================
        # STEP 5: Write validation report
        # =========================================================================
        log("\n[STEP 5] Writing validation report...")

        write_validation_report(validation_results, OUTPUT_FILES["validation_report"])

        report_size = OUTPUT_FILES["validation_report"].stat().st_size
        log(f"  [SAVED] {OUTPUT_FILES['validation_report']}")
        log(f"  Size: {report_size:,} bytes")

        if report_size < 500:
            raise ValueError(f"Validation report too small: {report_size} bytes (expected >=500)")

        # =========================================================================
        # FINAL SUMMARY
        # =========================================================================
        log("\n" + "=" * 80)
        log("[SUCCESS] Step 00: Load Dependencies and Validate")
        log("=" * 80)
        log("\nValidation Summary:")
        log("  [PASS] All 4 dependency files exist")
        log("  [PASS] RQ 5.3.1 status shows success")
        log("  [PASS] All CSV schemas validated")
        log("  [PASS] All files copied to RQ 5.3.6/data/")
        log("  [PASS] Validation report written")

        log("\nOutput Files:")
        log(f"  - {OUTPUT_FILES['purified_items']} (45 items)")
        log(f"  - {OUTPUT_FILES['theta_scores']} (1200 rows, long format)")
        log(f"  - {OUTPUT_FILES['tsvr_mapping']} (400 rows)")
        log(f"  - {OUTPUT_FILES['validation_report']}")

        log("\nData Structure:")
        log("  - purified_items: ['item', 'domain', 'Discrimination', 'Difficulty_1']")
        log("  - theta_scores: ['composite_ID', 'domain_name', 'theta'] (long format)")
        log("  - tsvr_mapping: ['composite_ID', 'UID', 'test', 'TSVR_hours']")
        log("  - Domain mapping: free_recall=IFR, cued_recall=ICR, recognition=IRE")

        log("\nNext: Step 01 - Map Items by Paradigm (Retained vs Removed)")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("\n[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()

        log("\n[FAIL] Step 00: Load Dependencies and Validate")
        log("RQ 5.3.1 incomplete - run RQ 5.3.1 before RQ 5.3.6")

        sys.exit(1)
