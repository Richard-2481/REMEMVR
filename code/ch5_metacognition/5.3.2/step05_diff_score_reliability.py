"""
Step 05: Compute Difference Score Reliability for Calibration RQ 6.3.2

MANDATORY analysis per improvement_taxonomy.md Section 6.2 and finalization TIER 1 BLOCKER.

Calibration is computed as difference between confidence and accuracy theta.
Difference score reliability must be ≥ 0.70 for valid interpretation.

Formula: r_diff = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)
where:
- r_xx = reliability of accuracy measure (IRT person separation reliability)
- r_yy = reliability of confidence measure (IRT person separation reliability)  
- r_xy = correlation between accuracy and confidence

Circuit breaker: If r_diff < 0.70 → STOP, report need for SEM approach
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sys

print("=" * 80)
print("STEP 05: DIFFERENCE SCORE RELIABILITY ANALYSIS")
print("=" * 80)

# Load calibration data
print("\n[1] Loading calibration data...")
data = pd.read_csv('/home/etai/projects/REMEMVR/results/ch6/6.3.2/data/step00_calibration_by_domain.csv')
print(f"   Loaded {len(data)} observations")
print(f"   Columns: {list(data.columns)}")

# Compute correlations
print("\n[2] Computing correlations between accuracy and confidence theta...")

# Overall correlation
r_xy_overall, p_overall = pearsonr(data['theta_accuracy'], data['theta_confidence'])
print(f"\n   OVERALL:")
print(f"   r(theta_accuracy, theta_confidence) = {r_xy_overall:.3f}, p = {p_overall:.6f}")

# By domain
domains = ['What', 'Where', 'When']
r_xy_by_domain = {}

print(f"\n   BY DOMAIN:")
for domain in domains:
    domain_data = data[data['Domain'] == domain]
    if len(domain_data) > 0:
        r_xy, p_val = pearsonr(domain_data['theta_accuracy'], domain_data['theta_confidence'])
        r_xy_by_domain[domain] = r_xy
        print(f"   {domain:8s}: r = {r_xy:.3f}, p = {p_val:.6f}, N = {len(domain_data)}")

# Estimate IRT reliabilities from empirical data
print("\n[3] Estimating IRT reliabilities (person separation)...")
print("   NOTE: Using empirical approach since source RQs don't report formal reliability.")
print("   Method: Variance of true scores / Total variance approximation")

# For accuracy: Use variance across participants as proxy
accuracy_by_person = data.groupby('UID')['theta_accuracy'].mean()
r_xx = accuracy_by_person.var() / data['theta_accuracy'].var()
print(f"\n   Accuracy reliability estimate (r_xx): {r_xx:.3f}")
print(f"   (Variance between persons / Total variance)")

# For confidence: Same approach
confidence_by_person = data.groupby('UID')['theta_confidence'].mean()
r_yy = confidence_by_person.var() / data['theta_confidence'].var()
print(f"\n   Confidence reliability estimate (r_yy): {r_yy:.3f}")
print(f"   (Variance between persons / Total variance)")

# Compute difference score reliability
print("\n[4] Computing difference score reliability...")
print(f"\n   Formula: r_diff = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)")

def compute_diff_reliability(r_xx, r_yy, r_xy):
    """Compute reliability of difference score."""
    numerator = r_xx + r_yy - 2*r_xy
    denominator = 2 - 2*r_xy
    if denominator == 0:
        return np.nan
    return numerator / denominator

# Overall
r_diff_overall = compute_diff_reliability(r_xx, r_yy, r_xy_overall)
print(f"\n   OVERALL DIFFERENCE SCORE RELIABILITY:")
print(f"   r_xx (accuracy) = {r_xx:.3f}")
print(f"   r_yy (confidence) = {r_yy:.3f}")
print(f"   r_xy (correlation) = {r_xy_overall:.3f}")
print(f"   r_diff = {r_diff_overall:.3f}")

# By domain
print(f"\n   BY DOMAIN:")
r_diff_by_domain = {}
for domain in domains:
    domain_data = data[data['Domain'] == domain]
    if len(domain_data) > 0:
        # Domain-specific reliabilities
        acc_by_person_domain = domain_data.groupby('UID')['theta_accuracy'].mean()
        r_xx_domain = acc_by_person_domain.var() / domain_data['theta_accuracy'].var()
        
        conf_by_person_domain = domain_data.groupby('UID')['theta_confidence'].mean()
        r_yy_domain = conf_by_person_domain.var() / domain_data['theta_confidence'].var()
        
        r_xy_domain = r_xy_by_domain[domain]
        r_diff_domain = compute_diff_reliability(r_xx_domain, r_yy_domain, r_xy_domain)
        r_diff_by_domain[domain] = r_diff_domain
        
        print(f"   {domain:8s}: r_xx={r_xx_domain:.3f}, r_yy={r_yy_domain:.3f}, r_xy={r_xy_domain:.3f} → r_diff={r_diff_domain:.3f}")

# CIRCUIT BREAKER CHECK
print("\n" + "=" * 80)
print("RELIABILITY ASSESSMENT")
print("=" * 80)

threshold = 0.70
overall_status = "PASS" if r_diff_overall >= threshold else "FAIL"
print(f"\n   Overall r_diff = {r_diff_overall:.3f} {'≥' if r_diff_overall >= threshold else '<'} {threshold} ... {overall_status}")

domain_status = {}
for domain in domains:
    r_diff = r_diff_by_domain[domain]
    status = "PASS" if r_diff >= threshold else "FAIL"
    domain_status[domain] = status
    print(f"   {domain} r_diff = {r_diff:.3f} {'≥' if r_diff >= threshold else '<'} {threshold} ... {status}")

# Save results
print("\n[5] Saving results...")
output_file = '/home/etai/projects/REMEMVR/results/ch6/6.3.2/data/step05_diff_score_reliability.csv'

results_df = pd.DataFrame({
    'Domain': ['Overall'] + domains,
    'r_xx': [r_xx] + [domain_data.groupby('UID')['theta_accuracy'].mean().var() / data[data['Domain']==d]['theta_accuracy'].var() for d in domains],
    'r_yy': [r_yy] + [domain_data.groupby('UID')['theta_confidence'].mean().var() / data[data['Domain']==d]['theta_confidence'].var() for d in domains],
    'r_xy': [r_xy_overall] + [r_xy_by_domain[d] for d in domains],
    'r_diff': [r_diff_overall] + [r_diff_by_domain[d] for d in domains],
    'threshold': threshold,
    'status': [overall_status] + [domain_status[d] for d in domains]
})

results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# CIRCUIT BREAKER
print("\n" + "=" * 80)
print("CIRCUIT BREAKER CHECK")
print("=" * 80)

if overall_status == "FAIL":
    print(f"\n🔴 BLOCKER: Overall difference score reliability ({r_diff_overall:.3f}) < {threshold}")
    print(f"   Calibration difference scores are UNRELIABLE.")
    print(f"   REQUIRED ACTION: Switch to latent variable approach (SEM)")
    print(f"   Estimated additional time: 4-6 hours per calibration RQ")
    print(f"\n   Recommend consulting with user before proceeding.")
    sys.exit(1)
elif any(domain_status[d] == "FAIL" for d in domains):
    failed_domains = [d for d in domains if domain_status[d] == "FAIL"]
    print(f"\n⚠️  WARNING: Difference score reliability < {threshold} for domains: {', '.join(failed_domains)}")
    for domain in failed_domains:
        print(f"   {domain}: r_diff = {r_diff_by_domain[domain]:.3f}")
    print(f"\n   Overall reliability ({r_diff_overall:.3f}) is acceptable, but domain-specific findings should be interpreted cautiously.")
    print(f"   Document this limitation in summary.md Section 4 (Limitations).")
else:
    print(f"\n✅ PASS: All difference score reliabilities ≥ {threshold}")
    print(f"   Calibration difference scores are RELIABLE.")
    print(f"   Overall: r_diff = {r_diff_overall:.3f}")
    print(f"   Domain range: {min(r_diff_by_domain.values()):.3f} - {max(r_diff_by_domain.values()):.3f}")
    print(f"\n   Proceeding with existing calibration methodology is valid.")

print("\n" + "=" * 80)
print("STEP 05 COMPLETE")
print("=" * 80)
