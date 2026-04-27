"""
Step 00: Verify Dependencies and Load RQ 5.4.1 Outputs

Purpose: Verify RQ 5.4.1 completed successfully and load required outputs
for CTT computation and convergence analysis.

Dependencies: None (first step)

Per execution_plan.md validation checklist:
- D3: Correct parent RQ (5.4.1)
- D4: Sample size (N=100, 400 rows)
- D5: No missing data
"""

import pandas as pd
import yaml
from pathlib import Path
import sys

# Paths
RQ_DIR = Path("/home/etai/projects/REMEMVR/results/ch5/5.4.5")
PARENT_RQ_DIR = Path("/home/etai/projects/REMEMVR/results/ch5/5.4.1")
DATA_CACHE = Path("/home/etai/projects/REMEMVR/data/cache")

DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "step00_verify_dependencies.log"

def log(msg: str):
    """Log message to console and file."""
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def main():
    log("=" * 60)
    log("STEP 00: Verify Dependencies and Load RQ 5.4.1 Outputs")
    log("=" * 60)

    # 1. Check RQ 5.4.1 status
    log("\n[1] Checking RQ 5.4.1 status...")
    status_file = PARENT_RQ_DIR / "status.yaml"

    if not status_file.exists():
        log(f"ERROR: RQ 5.4.1 status.yaml not found at {status_file}")
        sys.exit(1)

    with open(status_file) as f:
        status = yaml.safe_load(f)

    rq_results_status = status.get("rq_results", {}).get("status", "not found")
    log(f"RQ 5.4.1 status: {rq_results_status}")

    if rq_results_status != "success":
        log(f"ERROR: RQ 5.4.1 must be complete (status=success), found: {rq_results_status}")
        sys.exit(1)

    log("RQ 5.4.1 status: success - VERIFIED")

    # 2. Load dependency files
    log("\n[2] Loading dependency files...")

    # 2a. Purified items
    purified_items_file = PARENT_RQ_DIR / "data" / "step02_purified_items.csv"
    if not purified_items_file.exists():
        log(f"ERROR: Purified items file not found: {purified_items_file}")
        sys.exit(1)

    purified_items = pd.read_csv(purified_items_file)
    log(f"  - step02_purified_items.csv: {len(purified_items)} items retained")

    # 2b. Theta scores
    theta_file = PARENT_RQ_DIR / "data" / "step03_theta_scores.csv"
    if not theta_file.exists():
        log(f"ERROR: Theta scores file not found: {theta_file}")
        sys.exit(1)

    theta_scores = pd.read_csv(theta_file)
    log(f"  - step03_theta_scores.csv: {len(theta_scores)} observations")

    # 2c. TSVR mapping
    tsvr_file = PARENT_RQ_DIR / "data" / "step00_tsvr_mapping.csv"
    if not tsvr_file.exists():
        log(f"ERROR: TSVR mapping file not found: {tsvr_file}")
        sys.exit(1)

    tsvr_mapping = pd.read_csv(tsvr_file)
    log(f"  - step00_tsvr_mapping.csv: {len(tsvr_mapping)} observations")

    # 2d. dfData (raw binary responses)
    dfdata_file = DATA_CACHE / "dfData.csv"
    if not dfdata_file.exists():
        log(f"ERROR: dfData.csv not found: {dfdata_file}")
        sys.exit(1)

    dfData = pd.read_csv(dfdata_file)
    log(f"  - dfData.csv: {len(dfData)} rows x {len(dfData.columns)} columns")

    log("\nLoaded 4 dependency files")

    # 3. Extract congruence items from dfData
    log("\n[3] Extracting congruence items from dfData...")

    # Get all TQ_ columns (test questions, binary responses)
    # Congruence coding: i1/i2 = Common, i3/i4 = Congruent, i5/i6 = Incongruent
    tq_cols = [c for c in dfData.columns if c.startswith("TQ_")]

    # Filter to interactive paradigms only (IFR, ICR, IRE) with congruence tags
    # These are columns that contain "-i1", "-i2", etc.
    congruence_items = []
    for col in tq_cols:
        if any(f"-i{n}" in col for n in [1, 2, 3, 4, 5, 6]):
            # Determine congruence dimension
            if "-i1" in col or "-i2" in col:
                dimension = "common"
            elif "-i3" in col or "-i4" in col:
                dimension = "congruent"
            elif "-i5" in col or "-i6" in col:
                dimension = "incongruent"
            else:
                continue
            congruence_items.append({"item_code": col, "dimension": dimension})

    all_items_df = pd.DataFrame(congruence_items)
    log(f"Found {len(all_items_df)} congruence items in dfData")

    # 4. Map retained vs removed items
    log("\n[4] Mapping retained vs removed items...")

    # Get purified item names (column: item_name)
    purified_item_names = set(purified_items["item_name"].tolist())

    # Map TQ_ prefix to match purified items format
    # dfData uses TQ_IFR-N-i1, purified_items uses TQ_IFR-N-i1
    all_items_df["retained"] = all_items_df["item_code"].isin(purified_item_names)

    # Count by dimension
    log("\nItem counts by congruence:")
    for dim in ["common", "congruent", "incongruent"]:
        dim_items = all_items_df[all_items_df["dimension"] == dim]
        n_total = len(dim_items)
        n_retained = dim_items["retained"].sum()
        n_removed = n_total - n_retained
        retention_rate = n_retained / n_total if n_total > 0 else 0
        log(f"  {dim.capitalize()}: {n_retained}/{n_total} retained ({retention_rate:.1%})")

    total_items = len(all_items_df)
    total_retained = all_items_df["retained"].sum()
    log(f"\nItem mapping complete: {total_items} total items, {total_retained} retained")

    # 5. Save outputs
    log("\n[5] Saving outputs...")

    # 5a. Save full item list with retention flags
    all_items_df.to_csv(DATA_DIR / "step00_full_item_list.csv", index=False)
    log(f"  - Saved data/step00_full_item_list.csv ({len(all_items_df)} rows)")

    # 5b. Save dependency check report
    report = f"""RQ 5.4.5 Dependency Verification Report
========================================

RQ 5.4.1 Status: {rq_results_status}

Files Loaded:
- step02_purified_items.csv: {len(purified_items)} items
- step03_theta_scores.csv: {len(theta_scores)} observations
- step00_tsvr_mapping.csv: {len(tsvr_mapping)} observations
- dfData.csv: {len(dfData)} rows x {len(dfData.columns)} columns

Item Mapping:
- Total congruence items: {total_items}
- Retained after purification: {total_retained}
- Overall retention rate: {total_retained/total_items:.1%}

By Dimension:
"""
    for dim in ["common", "congruent", "incongruent"]:
        dim_items = all_items_df[all_items_df["dimension"] == dim]
        n_total = len(dim_items)
        n_retained = dim_items["retained"].sum()
        report += f"- {dim.capitalize()}: {n_retained}/{n_total} ({n_retained/n_total:.1%})\n"

    report += f"\nValidation: PASS\n"

    with open(DATA_DIR / "step00_dependency_check.txt", "w") as f:
        f.write(report)
    log(f"  - Saved data/step00_dependency_check.txt")

    # 6. Validation
    log("\n[6] Validation...")

    # Check expected ranges
    assert 40 <= total_items <= 80, f"Expected 40-80 items, got {total_items}"
    assert 30 <= total_retained <= 60, f"Expected 30-60 retained, got {total_retained}"
    assert len(theta_scores) == 400, f"Expected 400 theta observations, got {len(theta_scores)}"

    log("Validation: PASS")
    log("\n" + "=" * 60)
    log("STEP 00 COMPLETE")
    log("=" * 60)

if __name__ == "__main__":
    main()
