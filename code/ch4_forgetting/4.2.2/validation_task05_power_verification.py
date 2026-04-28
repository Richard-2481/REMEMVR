"""
validation TASK 5: Post-Hoc Power Verification

Verify documented power calculation and compute N required for 80% power
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path("/home/etai/projects/REMEMVR/results/ch5/5.2.2")
RESULTS_DIR = BASE / "results"
LOGS_DIR = BASE / "logs"

log_file = LOGS_DIR / "platinum_task05_power_verification.log"
log = open(log_file, 'w')

def logprint(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

logprint("=" * 70)
logprint("validation TASK 5: Post-Hoc Power Verification")
logprint("=" * 70)
logprint("")

# Load TOST results (has observed Cohen's d)
logprint("Loading TOST results for observed effect sizes...")
tost_path = RESULTS_DIR / "platinum_task03_tost_equivalence.csv"
tost = pd.read_csv(tost_path)

logprint(f"{len(tost)} contrasts")
logprint("")

# Power calculation parameters
current_n = 100  # Current sample size
alpha = 0.0167  # Bonferroni-corrected alpha (0.05/3)
target_power = 0.80

logprint(f"")
logprint(f"  Current N: {current_n}")
logprint(f"  Alpha: {alpha} (Bonferroni-corrected)")
logprint(f"  Target power: {target_power}")
logprint("")

# Compute power for each contrast
logprint("=" * 70)
logprint("POST-HOC POWER ANALYSIS")
logprint("=" * 70)
logprint("")

power_results = []

for idx, row in tost.iterrows():
    comparison = row['comparison']
    d_obs = row['cohens_d']
    se_d = row['se_d']
    
    logprint(f"[CONTRAST {idx+1}] {comparison}")
    logprint(f"  Observed Cohen's d: {d_obs:.4f}")
    
    # Post-hoc power for observed effect
    # For independent t-test with equal groups: n per group = N/2
    n_per_group = current_n / 2
    
    # Noncentrality parameter
    ncp = d_obs * np.sqrt(n_per_group / 2)
    df = current_n - 2
    
    # Critical t-value (two-tailed)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Power = P(reject H0 | H1 true)
    if d_obs >= 0:
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    else:
        power = 1 - stats.nct.cdf(t_crit, df, -ncp) + stats.nct.cdf(-t_crit, df, -ncp)
    
    logprint(f"  Post-hoc power: {power:.4f} ({power*100:.1f}%)")
    
    # N required for 80% power
    # Iterative search for N
    target_ncp = None
    for n_test in range(50, 10000, 10):
        ncp_test = abs(d_obs) * np.sqrt((n_test/2) / 2)
        df_test = n_test - 2
        t_crit_test = stats.t.ppf(1 - alpha/2, df_test)
        
        power_test = 1 - stats.nct.cdf(t_crit_test, df_test, ncp_test) + stats.nct.cdf(-t_crit_test, df_test, ncp_test)
        
        if power_test >= target_power:
            n_required = n_test
            break
    else:
        n_required = ">10000"
    
    logprint(f"  N for 80% power: {n_required}")
    logprint("")
    
    power_results.append({
        'comparison': comparison,
        'cohens_d': d_obs,
        'current_n': current_n,
        'post_hoc_power': power,
        'n_for_80pct_power': n_required
    })

# Save results
power_df = pd.DataFrame(power_results)
output_path = RESULTS_DIR / "platinum_task05_power_analysis.csv"
power_df.to_csv(output_path, index=False)
logprint(f"{output_path}")
logprint("")

# Summary
logprint("=" * 70)
logprint("SUMMARY")
logprint("=" * 70)
logprint("")

mean_power = power_df['post_hoc_power'].mean()
logprint(f"Mean post-hoc power: {mean_power:.4f} ({mean_power*100:.1f}%)")

all_low_power = all(power_df['post_hoc_power'] < 0.50)
if all_low_power:
    logprint("Study SEVERELY UNDERPOWERED (all contrasts < 50% power)")
    logprint("             NULL findings likely due to insufficient power")
    logprint("             Larger N needed to detect small effects")
else:
    logprint("Study adequately powered for some contrasts")

logprint("")

# Compare to documented claim in summary.md
documented_power = 0.20  # From summary.md Section 4
logprint(f"Summary.md claims ~20% power")
logprint(f"               Actual mean power: {mean_power*100:.1f}%")

if abs(mean_power - documented_power) < 0.10:
    logprint(f"               ✓ Claim VERIFIED (within 10%)")
else:
    logprint(f"               ⚠ Claim differs from calculated power")

logprint("")
logprint("=" * 70)
logprint("TASK 5 COMPLETE")
logprint("=" * 70)

log.close()

print("")
print(f"✓ Power Verification Complete")
print(f"  Mean post-hoc power: {mean_power*100:.1f}%")
