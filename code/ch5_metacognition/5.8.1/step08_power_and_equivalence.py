"""
Step 8: Power Analysis and Equivalence Testing for NULL Finding

Purpose: Mandatory analyses for NULL interaction finding per taxonomy Section 3.
         - Post-hoc power analysis
         - TOST (Two One-Sided Tests) equivalence testing

CRITICAL: NULL findings MUST demonstrate either:
         1. Adequately powered to detect meaningful effects (power ≥ 0.80), OR
         2. Effect is significantly smaller than meaningful threshold (TOST)

Date: 2025-12-27
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_post_hoc_power(observed_effect, se, n_obs, alpha=0.05):
    """
    Compute post-hoc power for observed effect.

    Power = P(reject H0 | true effect = observed_effect)
    """
    print("\n" + "="*80)
    print("POST-HOC POWER ANALYSIS")
    print("="*80)

    print(f"\nObserved effect: β = {observed_effect:.4f}")
    print(f"Standard error: SE = {se:.4f}")
    print(f"Sample size: N = {n_obs}")

    # Critical z-value for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha/2)

    # Non-centrality parameter (effect / SE)
    ncp = abs(observed_effect) / se

    # Power = P(|Z| > z_crit | Z ~ N(ncp, 1))
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    print(f"\n📊 POST-HOC POWER:")
    print(f"   Power to detect observed effect (β = {observed_effect:.4f}): {power:.3f}")

    if power < 0.80:
        print(f"   ⚠️  Underpowered (power < 0.80)")
        print(f"   → Study has low sensitivity to detect this effect size")
    else:
        print(f"   ✅ Adequately powered (power ≥ 0.80)")

    # But this is circular for NULL findings!
    print(f"\n⚠️  NOTE: Post-hoc power for NULL findings is CIRCULAR")
    print(f"   When observed effect ≈ 0, post-hoc power ≈ α (Type I error rate)")
    print(f"   More informative: Power to detect MEANINGFUL effects (see below)")

    return power

def compute_power_for_meaningful_effects(se, n_obs, alpha=0.05):
    """
    Compute power to detect meaningful effects (Cohen's benchmarks).

    Small effect: d = 0.20 → β ≈ 0.05 on theta scale
    Medium effect: d = 0.50 → β ≈ 0.12 on theta scale
    Large effect: d = 0.80 → β ≈ 0.20 on theta scale
    """
    print("\n" + "="*80)
    print("POWER TO DETECT MEANINGFUL EFFECTS")
    print("="*80)

    # Cohen's benchmarks converted to theta scale
    # (theta SD ≈ 0.25 from IRT, so d = 0.20 → β = 0.20 * 0.25 = 0.05)
    effect_sizes = {
        'Small (β=0.05)': 0.05,
        'Medium (β=0.12)': 0.12,
        'Large (β=0.20)': 0.20
    }

    z_crit = stats.norm.ppf(1 - alpha/2)

    print(f"\nStandard error: SE = {se:.4f}")
    print(f"Sample size: N = {n_obs}")
    print(f"\n{'Effect Size':<25} {'Beta':<10} {'Power':<10}")
    print("=" * 50)

    powers = {}
    for name, beta in effect_sizes.items():
        ncp = abs(beta) / se
        power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        powers[name] = power
        print(f"{name:<25} {beta:<10.3f} {power:<10.3f}")

    print("\n📊 INTERPRETATION:")
    if powers['Small (β=0.05)'] >= 0.80:
        print("   ✅ Study has 80%+ power to detect SMALL effects")
        print("   → NULL finding is informative (not due to low power)")
    elif powers['Medium (β=0.12)'] >= 0.80:
        print("   ✅ Study has 80%+ power to detect MEDIUM effects")
        print("   → NULL could reflect true lack of medium/large effects")
        print("   ⚠️  But small effects (<0.12) may be undetectable")
    elif powers['Large (β=0.20)'] >= 0.80:
        print("   ⚠️  Study only powered for LARGE effects")
        print("   → NULL finding ambiguous (small/medium effects undetectable)")
    else:
        print("   🔴 UNDERPOWERED for all effect sizes")
        print("   → NULL finding uninformative (study too small)")

    return powers

def compute_tost_equivalence(observed_effect, se, df, equivalence_bound=0.05, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence.

    H0: |effect| ≥ equivalence_bound
    H1: |effect| < equivalence_bound (equivalence)

    Reject H0 if BOTH one-sided tests reject at alpha level.
    """
    print("\n" + "="*80)
    print("EQUIVALENCE TESTING (TOST)")
    print("="*80)

    print(f"\nEquivalence bound: β < {equivalence_bound:.3f} (small effect threshold)")
    print(f"Observed effect: β = {observed_effect:.4f}")
    print(f"Standard error: SE = {se:.4f}")
    print(f"Degrees of freedom: df = {df}")

    # Test 1: observed > -equivalence_bound
    t1 = (observed_effect - (-equivalence_bound)) / se
    p1 = stats.t.sf(t1, df)  # Upper tail

    # Test 2: observed < equivalence_bound
    t2 = (equivalence_bound - observed_effect) / se
    p2 = stats.t.sf(t2, df)  # Upper tail

    # TOST p-value = max(p1, p2)
    tost_p = max(p1, p2)

    print(f"\nTest 1: β > -{equivalence_bound:.3f}")
    print(f"  t = {t1:.2f}, p = {p1:.4f}")
    print(f"\nTest 2: β < {equivalence_bound:.3f}")
    print(f"  t = {t2:.2f}, p = {p2:.4f}")
    print(f"\nTOST p-value: {tost_p:.4f}")

    if tost_p < alpha:
        print(f"\n✅ EQUIVALENCE ESTABLISHED (p = {tost_p:.4f} < {alpha})")
        print(f"   → Effect is significantly smaller than {equivalence_bound:.3f}")
        print(f"   → This is a 'TRUE NULL' (not just absence of evidence)")
        print(f"   → Source and destination genuinely show equivalent decline rates")
    else:
        print(f"\n❌ EQUIVALENCE NOT ESTABLISHED (p = {tost_p:.4f} ≥ {alpha})")
        print(f"   → Cannot conclude effect is smaller than {equivalence_bound:.3f}")
        print(f"   → NULL finding remains ambiguous (could be true null or underpowered)")

    # 90% CI for equivalence (corresponds to alpha=0.05 TOST)
    ci_90_lower = observed_effect - stats.t.ppf(1 - alpha, df) * se
    ci_90_upper = observed_effect + stats.t.ppf(1 - alpha, df) * se

    print(f"\n90% CI for effect: [{ci_90_lower:.4f}, {ci_90_upper:.4f}]")
    if ci_90_lower > -equivalence_bound and ci_90_upper < equivalence_bound:
        print(f"   ✅ 90% CI fully contained within equivalence bounds")
        print(f"      [{-equivalence_bound:.3f}, {equivalence_bound:.3f}]")
    else:
        print(f"   ❌ 90% CI extends beyond equivalence bounds")

    return tost_p, (ci_90_lower, ci_90_upper)

def estimate_n_for_power(target_power=0.80, effect_size=0.05, alpha=0.05):
    """
    Estimate sample size needed for target power to detect effect_size.
    """
    print("\n" + "="*80)
    print("SAMPLE SIZE FOR TARGET POWER")
    print("="*80)

    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(target_power)

    # N = [(z_alpha + z_beta) / effect_size]^2 * sigma^2
    # For interaction term, sigma^2 ≈ SE^2 * N (approximate)
    # Simplified: N ≈ [(z_alpha + z_beta) / (effect_size / SE_current)]^2

    # Current SE ≈ 0.013, N = 800
    # SE ∝ 1/sqrt(N) → SE_new = SE_current * sqrt(N_current / N_new)

    # For effect_size with SE = 0.013 at N=800:
    # Required N for detection: N = (SE / effect_size * (z_alpha + z_beta))^2
    # But SE depends on N, so iterative

    # Simplified approximation (conservative):
    se_current = 0.013
    n_current = 800

    # Target: effect_size / SE_new = z_alpha + z_beta
    # SE_new = effect_size / (z_alpha + z_beta)
    # SE_new = se_current * sqrt(n_current / N_new)
    # → N_new = n_current * (se_current / SE_new)^2
    # → N_new = n_current * (se_current * (z_alpha + z_beta) / effect_size)^2

    n_required = n_current * (se_current * (z_alpha + z_beta) / effect_size)**2

    print(f"\nTarget power: {target_power:.2f}")
    print(f"Effect size to detect: β = {effect_size:.3f}")
    print(f"Alpha: {alpha}")
    print(f"Current sample size: N = {n_current}")
    print(f"\n📊 ESTIMATED REQUIRED SAMPLE SIZE:")
    print(f"   N ≈ {n_required:.0f} observations")
    print(f"   (with {n_current / 8:.0f} participants × 8 obs/participant currently)")
    print(f"   → Would need ~{n_required / 8:.0f} participants for {target_power:.0%} power")

    return n_required

def main():
    """Execute power analysis and equivalence testing."""
    print("="*80)
    print("POWER ANALYSIS & EQUIVALENCE TESTING - RQ 6.8.1")
    print("="*80)
    print("\n🔴 MANDATORY for NULL findings (Taxonomy Section 3)")
    print("   - Post-hoc power analysis")
    print("   - Power to detect meaningful effects")
    print("   - TOST equivalence testing")
    print("   - Sample size estimation")

    # Load LMM results (with random slopes - corrected model)
    coef_df = pd.read_csv('results/ch6/6.8.1/data/step05d_lmm_with_slopes_coefficients.csv')

    # Extract interaction term
    interaction_row = coef_df[coef_df['term'].str.contains('log_TSVR') &
                               coef_df['term'].str.contains('location')]

    if len(interaction_row) == 0:
        print("❌ ERROR: Interaction term not found in coefficients table")
        sys.exit(1)

    interaction_row = interaction_row.iloc[0]

    observed_effect = interaction_row['coefficient']
    se = interaction_row['se']
    p_value = interaction_row['p_value']

    print(f"\n📊 INTERACTION EFFECT:")
    print(f"   β = {observed_effect:.4f}")
    print(f"   SE = {se:.4f}")
    print(f"   p = {p_value:.4f}")
    print(f"   N = 800 observations")

    # 1. Post-hoc power
    post_hoc_power = compute_post_hoc_power(observed_effect, se, n_obs=800)

    # 2. Power for meaningful effects
    powers = compute_power_for_meaningful_effects(se, n_obs=800)

    # 3. TOST equivalence
    # df ≈ N - k (k = 4 fixed effects) ≈ 796, but LMM uses adjusted df
    # Conservative: use df = 99 (N groups - 1)
    tost_p, ci_90 = compute_tost_equivalence(
        observed_effect, se, df=99,
        equivalence_bound=0.05  # Small effect on theta scale
    )

    # 4. Sample size for power
    n_req_small = estimate_n_for_power(target_power=0.80, effect_size=0.05)

    # Save results
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    output_path = 'results/ch6/6.8.1/data/step08_power_and_equivalence.txt'
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POWER ANALYSIS & EQUIVALENCE TESTING - RQ 6.8.1\n")
        f.write("="*80 + "\n\n")

        f.write("INTERACTION EFFECT:\n")
        f.write(f"  β = {observed_effect:.4f}\n")
        f.write(f"  SE = {se:.4f}\n")
        f.write(f"  p = {p_value:.4f}\n")
        f.write(f"  N = 800\n\n")

        f.write("POST-HOC POWER:\n")
        f.write(f"  Power for observed effect: {post_hoc_power:.3f}\n\n")

        f.write("POWER FOR MEANINGFUL EFFECTS:\n")
        for name, power in powers.items():
            f.write(f"  {name}: {power:.3f}\n")
        f.write("\n")

        f.write("EQUIVALENCE TESTING (TOST):\n")
        f.write(f"  Equivalence bound: β < 0.05\n")
        f.write(f"  TOST p-value: {tost_p:.4f}\n")
        f.write(f"  90% CI: [{ci_90[0]:.4f}, {ci_90[1]:.4f}]\n")
        if tost_p < 0.05:
            f.write(f"  Result: EQUIVALENCE ESTABLISHED (TRUE NULL)\n")
        else:
            f.write(f"  Result: EQUIVALENCE NOT ESTABLISHED\n")
        f.write("\n")

        f.write("SAMPLE SIZE FOR 80% POWER:\n")
        f.write(f"  To detect small effect (β=0.05): N ≈ {n_req_small:.0f}\n")
        f.write(f"  Current N: 800\n")
        if 800 >= n_req_small:
            f.write(f"  Assessment: ADEQUATELY POWERED for small effects\n")
        else:
            f.write(f"  Assessment: UNDERPOWERED for small effects\n")

    print(f"✅ Saved: {output_path}")

    # Summary CSV
    summary_df = pd.DataFrame({
        'metric': [
            'Observed_effect', 'SE', 'P_value',
            'Power_observed', 'Power_small', 'Power_medium', 'Power_large',
            'TOST_p', 'CI_90_lower', 'CI_90_upper',
            'N_required_small', 'N_current'
        ],
        'value': [
            observed_effect, se, p_value,
            post_hoc_power, powers['Small (β=0.05)'], powers['Medium (β=0.12)'], powers['Large (β=0.20)'],
            tost_p, ci_90[0], ci_90[1],
            n_req_small, 800
        ]
    })

    csv_path = 'results/ch6/6.8.1/data/step08_power_equivalence_summary.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✅ Saved: {csv_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 KEY FINDINGS:")
    print(f"   Power for small effects: {powers['Small (β=0.05)']:.2%}")
    print(f"   TOST p-value: {tost_p:.4f}")
    if powers['Small (β=0.05)'] >= 0.80:
        print("   ✅ Study adequately powered - NULL is informative")
    if tost_p < 0.05:
        print("   ✅ Equivalence established - TRUE NULL confirmed")

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
