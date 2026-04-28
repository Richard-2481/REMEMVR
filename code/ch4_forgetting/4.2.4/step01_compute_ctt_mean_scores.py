#!/usr/bin/env python3
"""step01_compute_ctt: Calculate CTT (Classical Test Theory) mean scores per UID × test × domain using"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_compute_ctt_mean_scores.log"


# Domain Tag Patterns (from RQ 5.2.1 IRT analysis - When EXCLUDED)

DOMAIN_PATTERNS = {
    'what': '-N-',      # What items (narrative/object content)
    'where': ['-L-', '-U-', '-D-'],  # Where items (left/up/down spatial)
    # 'when': '-O-'     # When items (temporal order) - EXCLUDED due to floor effects
}

# Domains to include (When excluded)
INCLUDED_DOMAINS = {'what', 'where'}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def parse_domain_from_item_tag(item_name: str) -> str:
    """
    Parse domain from item tag pattern.

    Rules (When EXCLUDED):
      - What items: contain '-N-'
      - Where items: contain '-L-', '-U-', or '-D-'

    Args:
        item_name: Item tag (e.g., 'TQ_ICR-N-i1')

    Returns:
        Domain name ('what', 'where') or None if excluded (When)

    Raises:
        ValueError: If item doesn't match any known pattern
    """
    item_lower = item_name.lower()

    if DOMAIN_PATTERNS['what'].lower() in item_lower:
        return 'what'
    elif any(pattern.lower() in item_lower for pattern in DOMAIN_PATTERNS['where']):
        return 'where'
    elif '-o-' in item_lower:
        # When items - excluded but recognized
        return 'when'  # Will be filtered out later
    else:
        raise ValueError(f"Item '{item_name}' doesn't match any known domain pattern")

def compute_ctt_mean_scores(df_wide: pd.DataFrame, item_list: List[str],
                            domain_map: Dict[str, str]) -> pd.DataFrame:
    """
    Compute CTT mean scores per UID × TEST × domain.

    CTT score = mean(item responses) for items in domain
    This is simply proportion correct (since items are dichotomized 0/1).

    **CRITICAL: Only computes for What and Where domains (When excluded)**

    Args:
        df_wide: Wide-format data with UID, TEST, and item columns
        item_list: List of item names to include
        domain_map: Dict mapping item_name -> domain

    Returns:
        Long-format DataFrame with columns:
          - composite_ID: {UID}_{test} format
          - UID: Participant ID
          - test: Test session (1, 2, 3, 4)
          - domain: Domain name (what, where - NO when)
          - CTT_score: Mean score (proportion correct)
          - n_items: Number of items in domain
    """
    log("Computing domain-wise CTT mean scores...")
    log("NOTE: When domain EXCLUDED due to floor effects")

    # Group items by domain (only What and Where)
    items_by_domain = {'what': [], 'where': []}
    excluded_count = 0

    for item in item_list:
        if item not in domain_map:
            log(f"Item '{item}' not in domain map, skipping")
            continue
        domain = domain_map[item]
        if domain in INCLUDED_DOMAINS:
            items_by_domain[domain].append(item)
        else:
            excluded_count += 1  # When items

    log(f"Domain item counts (When excluded):")
    for domain, items in items_by_domain.items():
        log(f"  - {domain}: {len(items)} items")
    if excluded_count > 0:
        log(f"  - when: {excluded_count} items (EXCLUDED)")

    # Compute mean scores per domain (What and Where only)
    ctt_rows = []

    for idx, row in df_wide.iterrows():
        uid = row['UID']
        test = row['TEST']
        composite_id = f"{uid}_{test}"

        for domain, items in items_by_domain.items():
            # Extract scores for items in this domain
            domain_scores = [row[item] for item in items if item in df_wide.columns]

            # Compute mean (handling NaN)
            ctt_score = np.nanmean(domain_scores) if domain_scores else np.nan
            n_items = len(domain_scores)

            ctt_rows.append({
                'composite_ID': composite_id,
                'UID': uid,
                'test': test,
                'domain': domain,
                'CTT_score': ctt_score,
                'n_items': n_items
            })

    df_ctt = pd.DataFrame(ctt_rows)

    log(f"Computed CTT scores: {len(df_ctt)} rows (UID x test x 2 domains)")
    log(f"Score range: [{df_ctt['CTT_score'].min():.3f}, {df_ctt['CTT_score'].max():.3f}]")

    return df_ctt

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Compute CTT Mean Scores")
        # Load Input Data
        #           purified items from RQ 5.1 Pass 2
        #          convergent validity comparison with IRT theta

        log("Loading input data...")

        # Load raw data (wide format, dichotomized)
        raw_data_path = RQ_DIR / "data/step00_raw_data_filtered.csv"
        df_raw = pd.read_csv(raw_data_path)
        log(f"{raw_data_path.name} ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Load purified item list (to know which items to use and their domains)
        purified_items_path = RQ_DIR / "data/step00_purified_items.csv"
        df_items = pd.read_csv(purified_items_path)
        log(f"{purified_items_path.name} ({len(df_items)} items)")

        # NOTE: 4_analysis.yaml says column name is 'dimension' but actual data uses 'factor'
        # This is verified from RQ 5.1 source data. Using 'factor' to match reality.
        if 'factor' not in df_items.columns:
            raise ValueError(f"Expected column 'factor' in purified items, found: {df_items.columns.tolist()}")
        # Parse Domain Assignments
        # Map each item to its domain using tag patterns
        # Domain assignment logic matches RQ 5.1 IRT analysis for consistency

        log("Parsing domain assignments from item tags...")

        item_domain_map = {}
        for _, row in df_items.iterrows():
            item_name = row['item_name']
            try:
                domain = parse_domain_from_item_tag(item_name)
                item_domain_map[item_name] = domain
            except ValueError as e:
                log(f"{e}")

        # Verify domain assignments match 'factor' column from RQ 5.1
        mismatches = []
        for _, row in df_items.iterrows():
            item_name = row['item_name']
            expected_domain = row['factor'].lower()  # RQ 5.1 uses lowercase
            parsed_domain = item_domain_map.get(item_name)

            if parsed_domain != expected_domain:
                mismatches.append(f"{item_name}: parsed={parsed_domain}, expected={expected_domain}")

        if mismatches:
            log(f"Domain assignment mismatches ({len(mismatches)} items):")
            for mismatch in mismatches[:5]:  # Show first 5
                log(f"  - {mismatch}")
            log("Using 'factor' column from RQ 5.1 as authoritative source")
            # Override parsed domains with RQ 5.1 'factor' column
            item_domain_map = dict(zip(df_items['item_name'], df_items['factor'].str.lower()))
        else:
            log(f"Domain assignments match RQ 5.1 'factor' column ({len(item_domain_map)} items)")
        # Compute CTT Mean Scores
        # For each UID × test × domain combination:
        #   CTT_score = mean(item_responses) where items belong to domain
        # This is proportion correct since items are dichotomized (0/1)

        item_list = df_items['item_name'].tolist()
        df_ctt = compute_ctt_mean_scores(df_raw, item_list, item_domain_map)
        # Save CTT Scores
        # Output: Long-format CTT scores for merging with IRT theta in step 2

        output_path = RQ_DIR / "data/step01_ctt_scores.csv"
        df_ctt.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_ctt)} rows, {len(df_ctt.columns)} cols)")
        # Validate Output
        # Validation criteria (When EXCLUDED):
        #   - Exactly 800 rows (400 UID × test × 2 domains)
        #   - All required columns present
        #   - Both domains present (What, Where - NO When)
        #   - CTT_score in [0, 1]
        #   - n_items > 0

        log("Running validate_data_format...")

        # Required columns check
        required_columns = ['composite_ID', 'UID', 'test', 'domain', 'CTT_score', 'n_items']
        validation_result = validate_data_format(df_ctt, required_columns)

        if not validation_result['valid']:
            raise ValueError(f"Column validation failed: {validation_result['message']}")

        log(f"Column check: {validation_result['message']}")

        # Row count check (800 = 400 × 2 domains, When excluded)
        expected_rows = 800  # 400 UID × test × 2 domains (When excluded)
        if len(df_ctt) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {len(df_ctt)}")
        log(f"Row count check: {len(df_ctt)} rows (expected {expected_rows})")

        # Domain completeness check (What, Where only - NO When)
        unique_domains = df_ctt['domain'].unique()
        expected_domains = {'what', 'where'}  # When excluded
        if set(unique_domains) != expected_domains:
            raise ValueError(f"Expected domains {expected_domains}, got {set(unique_domains)}")
        log(f"Domain check: {sorted(unique_domains)} (expected {sorted(expected_domains)} - When excluded)")

        # CTT score range check
        ctt_min = df_ctt['CTT_score'].min()
        ctt_max = df_ctt['CTT_score'].max()
        if ctt_min < 0 or ctt_max > 1:
            raise ValueError(f"CTT_score out of [0,1] range: [{ctt_min:.3f}, {ctt_max:.3f}]")
        log(f"CTT_score range: [{ctt_min:.3f}, {ctt_max:.3f}] (expected [0, 1])")

        # Item count check
        n_items_min = df_ctt['n_items'].min()
        if n_items_min <= 0:
            raise ValueError(f"Some domains have 0 items (min={n_items_min})")
        log(f"n_items range: [{n_items_min}, {df_ctt['n_items'].max()}] (all > 0)")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
