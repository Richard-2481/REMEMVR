"""
PLATINUM TASK 3: TOST Equivalence Testing for NULL Domain-Specific Consolidation

Uses contrasts (beta coefficients) and converts to standardized effect sizes for TOST
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path("/home/etai/projects/REMEMVR/results/ch5/5.2.2")
RESULTS_DIR = BASE / "results"
LOGS_DIR = BASE / "logs"
DATA_DIR = BASE / "data"

log_file = LOGS_DIR / "platinum_task03_tost_equivalence.log"
log = open(log_file, 'w')

def logprint(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

logprint("=" * 70)
logprint("PLATINUM TASK 3: TOST Equivalence Testing")
logprint("=" * 70)
logprint("")

# Load contrasts
logprint("[LOAD] Loading planned contrasts from step03...")
contrasts_path = RESULTS_DIR / "step03_planned_contrasts.csv"
contrasts = pd.read_csv(contrasts_path)

logprint(f"[LOADED] {len(contrasts)} contrasts")
logprint("")

# Load data to get pooled SD for Cohen's d conversion
logprint("[LOAD] Loading piecewise LMM input for SD calculation...")
data_path = DATA_DIR / "step00_piecewise_lmm_input.csv"
data = pd.read_csv(data_path)

# Pooled SD of theta (outcome variable)
pooled_sd = data['theta'].std()
logprint(f"[CALCULATED] Pooled SD of theta: {pooled_sd:.4f}")
logprint("")

# Equivalence bound
equiv_bound_d = 0.20  # Cohen's d
equiv_bound_beta = equiv_bound_d * pooled_sd  # Convert to beta scale

logprint(f"[PARAMETER] Equivalence bound:")
logprint(f"            Cohen's d < {equiv_bound_d}")
logprint(f"            Beta < {equiv_bound_beta:.4f} (theta units)")
logprint(f"            Rationale: d < 0.20 considered 'negligible' per Cohen (1988)")
logprint("")

# TOST for each contrast
logprint("=" * 70)
logprint("TOST EQUIVALENCE TESTS")
logprint("=" * 70)
logprint("")

tost_results = []

for idx, row in contrasts.iterrows():
    comparison = row['comparison']
    description = row['description']
    beta_obs = row['beta']
    se_beta = row['se']
    
    # Convert beta to Cohen's d
    d_obs = beta_obs / pooled_sd
    se_d = se_beta / pooled_sd
    
    logprint(f"[TEST {idx+1}] {comparison}")
    logprint(f"  Description: {description}")
    logprint(f"  Beta: {beta_obs:.4f} ± {se_beta:.4f}")
    logprint(f"  Cohen's d: {d_obs:.4f} ± {se_d:.4f}")
    
    # TOST: Two one-sided tests
    # H0: |d| >= equiv_bound
    # H1: |d| < equiv_bound
    
    # df = N - k (N=100 participants, k≈10 parameters)
    df = 90
    
    # Test 1: d < upper_bound (+equiv_bound)
    t1 = (d_obs - equiv_bound_d) / se_d
    p1 = stats.t.cdf(t1, df)
    
    # Test 2: d > lower_bound (-equiv_bound)
    t2 = (d_obs - (-equiv_bound_d)) / se_d
    p2 = 1 - stats.t.cdf(t2, df)
    
    # TOST p-value = max(p1, p2)
    tost_p = max(p1, p2)
    
    logprint(f"  Test 1 (d < +{equiv_bound_d}): t = {t1:.4f}, p = {p1:.4f}")
    logprint(f"  Test 2 (d > -{equiv_bound_d}): t = {t2:.4f}, p = {p2:.4f}")
    logprint(f"  TOST p-value: {tost_p:.4f}")
    
    # Interpretation
    if tost_p < 0.05:
        decision = "EQUIVALENT"
        interpretation = "True negligible effect (d < 0.20)"
        logprint(f"  ✓ EQUIVALENCE ESTABLISHED (p = {tost_p:.4f} < 0.05)")
        logprint(f"    Effect IS negligible (d < {equiv_bound_d})")
    else:
        decision = "INCONCLUSIVE"
        interpretation = "Cannot confirm negligible effect"
        logprint(f"  ? INCONCLUSIVE (p = {tost_p:.4f} >= 0.05)")
        logprint(f"    Cannot rule out small but meaningful effect")
    
    logprint("")
    
    tost_results.append({
        'comparison': comparison,
        'description': description,
        'beta': beta_obs,
        'se_beta': se_beta,
        'cohens_d': d_obs,
        'se_d': se_d,
        'equiv_bound_d': equiv_bound_d,
        't_upper': t1,
        'p_upper': p1,
        't_lower': t2,
        'p_lower': p2,
        'tost_p_value': tost_p,
        'decision': decision,
        'interpretation': interpretation
    })

# Save results
tost_df = pd.DataFrame(tost_results)
output_path = RESULTS_DIR / "platinum_task03_tost_equivalence.csv"
tost_df.to_csv(output_path, index=False)
logprint(f"[SAVED] {output_path}")
logprint("")

# Summary
logprint("=" * 70)
logprint("SUMMARY")
logprint("=" * 70)
logprint("")

n_equivalent = (tost_df['decision'] == "EQUIVALENT").sum()
n_inconclusive = (tost_df['decision'] == "INCONCLUSIVE").sum()

logprint(f"Total contrasts tested: {len(tost_df)}")
logprint(f"  - Equivalence established: {n_equivalent}")
logprint(f"  - Inconclusive: {n_inconclusive}")
logprint("")

if n_equivalent == len(tost_df):
    logprint("[CONCLUSION] ALL domain-specific consolidation effects NEGLIGIBLE")
    logprint("             NULL hypothesis is TRUE NULL (not underpowered)")
    logprint("             Domain-specific consolidation absent in VR")
elif n_inconclusive == len(tost_df):
    logprint("[CONCLUSION] Equivalence NOT established")
    logprint("             NULL findings may be underpowered")
    logprint("             Larger N needed to establish true null")
else:
    logprint("[CONCLUSION] MIXED results")
    logprint("             Interpret null findings cautiously")

logprint("")
logprint("=" * 70)
logprint("TASK 3 COMPLETE")
logprint("=" * 70)

log.close()

print("")
print(f"✓ TOST Complete: {n_equivalent} equivalent, {n_inconclusive} inconclusive")
