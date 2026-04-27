#!/usr/bin/env python3
"""
RQ 6.4.2: Post-Hoc Power Analysis for NULL Contrasts
Compute detectable effect size at 80% power
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
BASE = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.2")
DATA = BASE / "data"
LOGS = BASE / "logs"

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / "step08_power_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_power_repeated_measures(d, n, alpha=0.05, r=0.5):
    """
    Compute power for paired/repeated measures t-test
    d: Cohen's d effect size
    n: number of pairs (participants)
    alpha: significance level
    r: correlation between repeated measures (assumed 0.5)
    """
    # Effective sample size for correlated samples
    # t = d * sqrt(n) / sqrt(2 * (1 - r))
    ncp = d * np.sqrt(n) / np.sqrt(2 * (1 - r))  # non-centrality parameter
    critical_t = stats.t.ppf(1 - alpha/2, df=n-1)  # two-tailed
    power = 1 - stats.nct.cdf(critical_t, df=n-1, nc=ncp) + stats.nct.cdf(-critical_t, df=n-1, nc=ncp)
    return power

def compute_detectable_d(n, alpha=0.05, power=0.80, r=0.5):
    """
    Compute minimum detectable Cohen's d for given power
    Uses iterative search
    """
    # Binary search for d
    d_low, d_high = 0.001, 2.0
    tolerance = 0.001
    
    while d_high - d_low > tolerance:
        d_mid = (d_low + d_high) / 2
        power_mid = compute_power_repeated_measures(d_mid, n, alpha, r)
        
        if power_mid < power:
            d_low = d_mid
        else:
            d_high = d_mid
    
    return (d_low + d_high) / 2

def main():
    logger.info("=== RQ 6.4.2: Post-Hoc Power Analysis ===")
    
    # Load contrast results
    contrasts = pd.read_csv(DATA / "step02_post_hoc_contrasts.csv")
    logger.info(f"Loaded {len(contrasts)} pairwise contrasts")
    
    # Study parameters
    n_participants = 100
    n_paradigms = 3
    n_tests = 4
    n_obs_total = n_participants * n_paradigms * n_tests  # 1200
    
    # For pairwise contrasts: each participant contributes to 2 paradigms
    n_pairs = n_participants * n_tests  # 400 paired observations per contrast
    
    logger.info(f"\nStudy Design:")
    logger.info(f"  N participants: {n_participants}")
    logger.info(f"  N paradigms: {n_paradigms}")
    logger.info(f"  N tests: {n_tests}")
    logger.info(f"  Total observations: {n_obs_total}")
    logger.info(f"  Paired observations per contrast: {n_pairs}")
    
    # Alpha levels
    alpha_uncorrected = 0.05
    alpha_bonferroni = 0.05 / 3  # 3 contrasts
    
    logger.info(f"\nAlpha levels:")
    logger.info(f"  Uncorrected: {alpha_uncorrected}")
    logger.info(f"  Bonferroni: {alpha_bonferroni:.4f}")
    
    # Observed effect sizes from contrasts
    logger.info(f"\n=== Observed Effect Sizes ===")
    
    results = []
    
    for idx, row in contrasts.iterrows():
        contrast_name = row['contrast']
        observed_d = row['cohens_d']
        
        logger.info(f"\n{contrast_name}:")
        logger.info(f"  Observed Cohen's d: {observed_d:.4f}")
        
        # Post-hoc power for observed effect
        power_uncorr = compute_power_repeated_measures(abs(observed_d), n_pairs, alpha_uncorrected)
        power_bonf = compute_power_repeated_measures(abs(observed_d), n_pairs, alpha_bonferroni)
        
        logger.info(f"  Post-hoc power (α=0.05): {power_uncorr:.4f}")
        logger.info(f"  Post-hoc power (Bonferroni): {power_bonf:.4f}")
        
        results.append({
            'contrast': contrast_name,
            'observed_d': observed_d,
            'power_uncorrected': power_uncorr,
            'power_bonferroni': power_bonf
        })
    
    # Minimum detectable effect sizes
    logger.info(f"\n=== Minimum Detectable Effect Sizes (80% Power) ===")
    
    mde_uncorr = compute_detectable_d(n_pairs, alpha_uncorrected, 0.80)
    mde_bonf = compute_detectable_d(n_pairs, alpha_bonferroni, 0.80)
    
    logger.info(f"\nMinimum detectable d (α=0.05, 80% power): {mde_uncorr:.4f}")
    logger.info(f"Minimum detectable d (Bonferroni, 80% power): {mde_bonf:.4f}")
    
    # Interpretation
    logger.info(f"\n=== Power Interpretation ===")
    
    max_observed_d = contrasts['cohens_d'].abs().max()
    
    if max_observed_d < mde_bonf:
        logger.warning(f"⚠ ALL observed effects (|d| < {max_observed_d:.4f}) below detectable threshold (d ≥ {mde_bonf:.4f})")
        logger.info(f"  Study underpowered for Bonferroni-corrected contrasts")
        logger.info(f"  Cannot distinguish 'true null' from 'underpowered'")
    else:
        logger.info(f"✓ Some effects exceed detectable threshold")
    
    # Sample size needed for observed effects
    logger.info(f"\n=== Required Sample Size ===")
    
    for idx, row in contrasts.iterrows():
        contrast_name = row['contrast']
        observed_d = abs(row['cohens_d'])
        
        # Binary search for required N
        n_low, n_high = 10, 10000
        while n_high - n_low > 1:
            n_mid = (n_low + n_high) // 2
            power_mid = compute_power_repeated_measures(observed_d, n_mid, alpha_bonferroni)
            if power_mid < 0.80:
                n_low = n_mid
            else:
                n_high = n_mid
        
        n_required = n_high
        
        logger.info(f"{contrast_name}:")
        logger.info(f"  Observed d={observed_d:.4f} requires N={n_required} pairs (80% power, Bonferroni)")
        logger.info(f"  Current N={n_pairs} pairs → Need {n_required / n_pairs:.1f}x larger sample")
        
        results[idx]['n_required_pairs'] = n_required
        results[idx]['sample_multiplier'] = n_required / n_pairs
    
    # Save results
    power_df = pd.DataFrame(results)
    power_df['mde_uncorrected'] = mde_uncorr
    power_df['mde_bonferroni'] = mde_bonf
    power_df['n_pairs_current'] = n_pairs
    
    power_df.to_csv(DATA / "step08_power_analysis.csv", index=False)
    logger.info(f"\nSaved: {DATA / 'step08_power_analysis.csv'}")
    
    logger.info("\n=== Power Analysis Complete ===")
    logger.info(f"Conclusion: Study has ~{power_df['power_bonferroni'].mean()*100:.1f}% average power")
    logger.info(f"Detectable effects: d ≥ {mde_bonf:.4f} (Bonferroni-corrected)")
    logger.info(f"Observed effects: d = {contrasts['cohens_d'].abs().min():.4f} to {contrasts['cohens_d'].abs().max():.4f}")

if __name__ == "__main__":
    main()
