"""
Check if switching to intercepts-only model would change fixed effects conclusions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

BASE = Path("/home/etai/projects/REMEMVR/results/ch5/5.2.2")
DATA_DIR = BASE / "data"

# Load both models
with open(DATA_DIR / "step01_piecewise_lmm_model.pkl", 'rb') as f:
    model_slopes = pickle.load(f)
    
with open(DATA_DIR / "step01_piecewise_lmm_model_intercepts_only.pkl", 'rb') as f:
    model_intercepts = pickle.load(f)

print("=" * 70)
print("FIXED EFFECTS COMPARISON")
print("=" * 70)
print("")

# Extract fixed effects
fe_slopes = model_slopes.params
fe_intercepts = model_intercepts.params

# Compare
comparison = pd.DataFrame({
    'Parameter': fe_slopes.index,
    'Slopes_Model': fe_slopes.values,
    'Intercepts_Model': fe_intercepts.values,
    'Difference': fe_slopes.values - fe_intercepts.values,
    'Pct_Change': 100 * (fe_slopes.values - fe_intercepts.values) / fe_slopes.values
})

print(comparison.to_string(index=False))
print("")

# Check p-values for key effects
print("=" * 70)
print("P-VALUE COMPARISON (Key Effects)")
print("=" * 70)
print("")

key_effects = [
    'Days_within',
    'Days_within:C(Segment, Treatment(\'Early\'))[T.Late]',
    'Days_within:C(domain, Treatment(\'what\'))[T.where]',
    'Days_within:C(Segment, Treatment(\'Early\'))[T.Late]:C(domain, Treatment(\'what\'))[T.where]'
]

for effect in key_effects:
    p_slopes = model_slopes.pvalues[effect]
    p_intercepts = model_intercepts.pvalues[effect]
    
    print(f"{effect}:")
    print(f"  Slopes model:     p = {p_slopes:.4f}")
    print(f"  Intercepts model: p = {p_intercepts:.4f}")
    print(f"  Change: {abs(p_slopes - p_intercepts):.4f}")
    print("")

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("")

# Check if any conclusions would change
max_coef_change = comparison['Difference'].abs().max()
max_pct_change = comparison['Pct_Change'].abs().max()

print(f"Maximum coefficient change: {max_coef_change:.4f}")
print(f"Maximum percent change: {max_pct_change:.2f}%")
print("")

if max_pct_change < 5:
    print("✓ Fixed effects nearly identical (<5% change)")
    print("✓ Conclusions would NOT change")
    print("  → Can document intercepts-only as preferred")
    print("  → Do NOT need to re-run Steps 2-5")
else:
    print("⚠ Fixed effects differ substantially (>5% change)")
    print("⚠ Conclusions MAY change")
    print("  → MUST re-run Steps 2-5 with intercepts-only model")
