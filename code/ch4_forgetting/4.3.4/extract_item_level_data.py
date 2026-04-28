"""
Extract Item-Level Data for GLMM Validation

Purpose: Convert RQ 5.3.1 wide-format IRT data to long-format item-level responses
Input: results/ch5/5.3.1/data/step00_irt_input.csv (wide format)
Output: data/item_level_responses.csv (long format for GLMM)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("Extracting Item-Level Data for GLMM Validation")
print("="*80)

# Load wide-format data from RQ 5.3.1
irt_input_path = Path("../../5.3.1/data/step00_irt_input.csv")
print(f"\n[STEP 1] Loading IRT input: {irt_input_path}")

irt_data = pd.read_csv(irt_input_path)
print(f"Loaded {irt_data.shape[0]} rows x {irt_data.shape[1]} columns")

# Parse composite_ID to extract UID and test
print("\n[STEP 2] Parsing composite_ID...")
irt_data[['UID', 'test']] = irt_data['composite_ID'].str.split('_', expand=True)
irt_data['test'] = irt_data['test'].astype(int)
print(f"Parsed UID and test from composite_ID")
print(f"  Unique UIDs: {irt_data['UID'].nunique()}")
print(f"  Test sessions: {sorted(irt_data['test'].unique())}")

# Melt to long format
print("\n[STEP 3] Converting to long format...")

# Item columns start with "TQ_" (paradigm-item format: TQ_IFR-N-i1, etc.)
item_cols = [col for col in irt_data.columns if col.startswith('TQ_')]
print(f"Found {len(item_cols)} item columns")

# Melt
long_data = irt_data.melt(
    id_vars=['composite_ID', 'UID', 'test'],
    value_vars=item_cols,
    var_name='item',
    value_name='Correct'
)

print(f"Melted to long format: {len(long_data):,} observations")

# Parse item name to extract paradigm
# Format: TQ_IFR-N-i1 → Paradigm=IFR, Domain=N, Item=i1
print("\n[STEP 4] Parsing item metadata...")

def parse_item(item_str):
    """Parse TQ_IFR-N-i1 format"""
    parts = item_str.replace('TQ_', '').split('-')
    return {
        'Paradigm': parts[0],
        'Domain': parts[1],
        'Item_Num': parts[2]
    }

item_info = long_data['item'].apply(parse_item).apply(pd.Series)
long_data = pd.concat([long_data, item_info], axis=1)

print(f"Parsed item metadata")
print(f"  Paradigms: {sorted(long_data['Paradigm'].unique())}")
print(f"  Domains: {sorted(long_data['Domain'].unique())}")

# Filter to paradigms of interest (IFR, ICR, IRE)
paradigms = ['IFR', 'ICR', 'IRE']
long_data = long_data[long_data['Paradigm'].isin(paradigms)].copy()
print(f"\nFiltered to paradigms {paradigms}: {len(long_data):,} observations")

# Create Item identifier
long_data['Item'] = long_data['Paradigm'] + '_' + long_data['Domain'] + '_' + long_data['Item_Num']

# Drop rows with missing responses (NaN)
n_before = len(long_data)
long_data = long_data.dropna(subset=['Correct'])
n_after = len(long_data)
if n_before > n_after:
    print(f"Dropped {n_before - n_after:,} rows with missing responses")

# Save
output_path = Path("../data/item_level_responses.csv")
long_data[['composite_ID', 'UID', 'test', 'Paradigm', 'Item', 'Correct']].to_csv(output_path, index=False)

print(f"\nSaved item-level data: {output_path}")
print(f"  Total observations: {len(long_data):,}")
print(f"  Participants: {long_data['UID'].nunique()}")
print(f"  Items: {long_data['Item'].nunique()}")
print(f"  Paradigms: {long_data['Paradigm'].nunique()}")

print("\nSample data:")
print(long_data[['UID', 'test', 'Paradigm', 'Item', 'Correct']].head(10))

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
