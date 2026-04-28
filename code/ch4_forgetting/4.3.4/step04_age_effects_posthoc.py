#!/usr/bin/env python3
"""Compute Age Effects by Paradigm and Post-Hoc Contrasts: Compute paradigm-specific age effects (simple slopes) from the 3-way interaction"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_age_effects_posthoc.log"

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute Age Effects and Post-Hoc Contrasts")
        # Load Fixed Effects from Step 2
        log("Loading fixed effects from Step 2...")
        fe_path = RQ_DIR / "data" / "step02_fixed_effects.csv"

        if not fe_path.exists():
            raise FileNotFoundError(f"Fixed effects file not found: {fe_path}")

        df_fe = pd.read_csv(fe_path, encoding='utf-8')
        log(f"{len(df_fe)} fixed effects from {fe_path}")

        # Create a lookup dictionary for easy access
        fe_dict = dict(zip(df_fe['term'], df_fe['coefficient']))
        se_dict = dict(zip(df_fe['term'], df_fe['SE']))
        # Extract Age Effect Components
        log("Extracting age effect components...")

        # Main Age_c effect (applies to all paradigms)
        age_c_main = fe_dict.get('Age_c', 0)
        age_c_main_se = se_dict.get('Age_c', 0)

        # Age_c × paradigm interactions (relative to IFR reference)
        age_c_icr = fe_dict.get("Age_c:C(paradigm, Treatment('IFR'))[T.ICR]", 0)
        age_c_icr_se = se_dict.get("Age_c:C(paradigm, Treatment('IFR'))[T.ICR]", 0)

        age_c_ire = fe_dict.get("Age_c:C(paradigm, Treatment('IFR'))[T.IRE]", 0)
        age_c_ire_se = se_dict.get("Age_c:C(paradigm, Treatment('IFR'))[T.IRE]", 0)

        log(f"Age_c main effect: {age_c_main:.6f} (SE={age_c_main_se:.6f})")
        log(f"Age_c × ICR: {age_c_icr:.6f} (SE={age_c_icr_se:.6f})")
        log(f"Age_c × IRE: {age_c_ire:.6f} (SE={age_c_ire_se:.6f})")
        # Compute Simple Slopes (Age Effect Within Each Paradigm)
        log("Computing simple slopes (age effect by paradigm)...")

        # For IFR (reference): Age effect = Age_c coefficient
        # For ICR: Age effect = Age_c + Age_c:ICR
        # For IRE: Age effect = Age_c + Age_c:IRE

        # Simple slopes
        age_effect_ifr = age_c_main
        age_effect_icr = age_c_main + age_c_icr
        age_effect_ire = age_c_main + age_c_ire

        # SE approximation: For IFR, use main effect SE
        # For ICR/IRE, use sqrt(var_main + var_interaction + 2*cov)
        # Since we don't have covariance, use conservative approximation: sqrt(sum of variances)
        se_ifr = age_c_main_se
        se_icr = np.sqrt(age_c_main_se**2 + age_c_icr_se**2)  # Conservative
        se_ire = np.sqrt(age_c_main_se**2 + age_c_ire_se**2)  # Conservative

        # Compute z and p-values
        def compute_z_p(coef, se):
            if se > 0:
                z = coef / se
                p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
                return z, p
            return np.nan, np.nan

        z_ifr, p_ifr = compute_z_p(age_effect_ifr, se_ifr)
        z_icr, p_icr = compute_z_p(age_effect_icr, se_icr)
        z_ire, p_ire = compute_z_p(age_effect_ire, se_ire)

        log("Age effects by paradigm:")
        log(f"  IFR: {age_effect_ifr:.6f} (SE={se_ifr:.6f}, z={z_ifr:.3f}, p={p_ifr:.4f})")
        log(f"  ICR: {age_effect_icr:.6f} (SE={se_icr:.6f}, z={z_icr:.3f}, p={p_icr:.4f})")
        log(f"  IRE: {age_effect_ire:.6f} (SE={se_ire:.6f}, z={z_ire:.3f}, p={p_ire:.4f})")
        # Create Age Effects DataFrame
        log("Creating age effects dataframe...")

        age_effects = pd.DataFrame({
            'paradigm': ['IFR', 'ICR', 'IRE'],
            'age_effect': [age_effect_ifr, age_effect_icr, age_effect_ire],
            'SE': [se_ifr, se_icr, se_ire],
            'z': [z_ifr, z_icr, z_ire],
            'p_uncorrected': [p_ifr, p_icr, p_ire]
        })

        # Bonferroni correction for 3 paradigms
        age_effects['p_bonferroni'] = age_effects['p_uncorrected'].apply(
            lambda p: min(p * 3, 1.0)
        )
        age_effects['significant_bonferroni'] = age_effects['p_bonferroni'] < 0.05
        # Compute Pairwise Contrasts
        log("Computing pairwise contrasts...")

        # Contrasts are based on the 2-way Age_c × paradigm interaction terms
        # IFR vs ICR: Age_c:ICR (difference from IFR)
        # IFR vs IRE: Age_c:IRE (difference from IFR)
        # ICR vs IRE: Age_c:IRE - Age_c:ICR

        contrast_ifr_icr = age_c_icr  # Already the difference from IFR
        contrast_ifr_ire = age_c_ire  # Already the difference from IFR
        contrast_icr_ire = age_c_ire - age_c_icr  # Difference between ICR and IRE

        # SE for contrasts
        se_ifr_icr = age_c_icr_se  # Direct from model
        se_ifr_ire = age_c_ire_se  # Direct from model
        se_icr_ire = np.sqrt(age_c_icr_se**2 + age_c_ire_se**2)  # Approximate

        # Z and p-values
        z_ifr_icr, p_ifr_icr = compute_z_p(contrast_ifr_icr, se_ifr_icr)
        z_ifr_ire, p_ifr_ire = compute_z_p(contrast_ifr_ire, se_ifr_ire)
        z_icr_ire, p_icr_ire = compute_z_p(contrast_icr_ire, se_icr_ire)

        log("Pairwise contrasts (Age effect differences):")
        log(f"  IFR vs ICR: {contrast_ifr_icr:.6f} (SE={se_ifr_icr:.6f}, z={z_ifr_icr:.3f}, p={p_ifr_icr:.4f})")
        log(f"  IFR vs IRE: {contrast_ifr_ire:.6f} (SE={se_ifr_ire:.6f}, z={z_ifr_ire:.3f}, p={p_ifr_ire:.4f})")
        log(f"  ICR vs IRE: {contrast_icr_ire:.6f} (SE={se_icr_ire:.6f}, z={z_icr_ire:.3f}, p={p_icr_ire:.4f})")
        # Create Contrasts DataFrame
        log("Creating contrasts dataframe...")

        contrasts = pd.DataFrame({
            'contrast': ['IFR vs ICR', 'IFR vs IRE', 'ICR vs IRE'],
            'difference': [contrast_ifr_icr, contrast_ifr_ire, contrast_icr_ire],
            'SE': [se_ifr_icr, se_ifr_ire, se_icr_ire],
            'z': [z_ifr_icr, z_ifr_ire, z_icr_ire],
            'p_uncorrected': [p_ifr_icr, p_ifr_ire, p_icr_ire]
        })

        # Bonferroni correction for 3 contrasts
        contrasts['p_bonferroni'] = contrasts['p_uncorrected'].apply(
            lambda p: min(p * 3, 1.0)
        )
        contrasts['significant_bonferroni'] = contrasts['p_bonferroni'] < 0.05
        # Save Output Files
        log("Saving output files...")

        age_effects_path = RQ_DIR / "data" / "step04_age_effects.csv"
        age_effects.to_csv(age_effects_path, index=False, encoding='utf-8')
        log(f"{age_effects_path}")

        contrasts_path = RQ_DIR / "data" / "step04_contrasts.csv"
        contrasts.to_csv(contrasts_path, index=False, encoding='utf-8')
        log(f"{contrasts_path}")
        # Validation
        log("Checking output format...")

        # Age effects validation
        if len(age_effects) == 3:
            log("Age effects: 3 paradigms")
        else:
            log(f"Age effects: {len(age_effects)} rows (expected 3)")

        if set(age_effects['paradigm']) == {'IFR', 'ICR', 'IRE'}:
            log("All paradigms present: IFR, ICR, IRE")
        else:
            log(f"Missing paradigms")

        # Contrasts validation
        if len(contrasts) == 3:
            log("Contrasts: 3 pairwise comparisons")
        else:
            log(f"Contrasts: {len(contrasts)} rows (expected 3)")

        # Check dual p-values (Decision D068)
        if 'p_uncorrected' in contrasts.columns and 'p_bonferroni' in contrasts.columns:
            log("Dual p-values present (Decision D068 compliance)")
        else:
            log("Missing p-value columns")

        # Check significance
        n_sig_age = age_effects['significant_bonferroni'].sum()
        n_sig_contrast = contrasts['significant_bonferroni'].sum()
        log(f"Significant age effects (Bonferroni): {n_sig_age} of 3")
        log(f"Significant contrasts (Bonferroni): {n_sig_contrast} of 3")
        # Final Summary
        log("Step 04 complete")
        log("")
        log(f"  Age effects computed for 3 paradigms")
        log(f"  IFR age effect: {age_effect_ifr:.4f} (p={p_ifr:.4f})")
        log(f"  ICR age effect: {age_effect_icr:.4f} (p={p_icr:.4f})")
        log(f"  IRE age effect: {age_effect_ire:.4f} (p={p_ire:.4f})")
        log(f"  Significant contrasts: {n_sig_contrast} of 3")
        log(f"  Output: data/step04_age_effects.csv, data/step04_contrasts.csv")
        log(f"  Ready for Step 5 (prepare plot data by age tertiles)")

        # Interpretation
        log("\n")
        if n_sig_age == 0:
            log("  No significant age effects on memory performance at Bonferroni-corrected level")
        else:
            log(f"  {n_sig_age} paradigm(s) show significant age-related decline")

        if n_sig_contrast == 0:
            log("  No significant differences in age effects between paradigms")
            log("  Age-related memory changes are similar across IFR, ICR, and IRE")
        else:
            log(f"  {n_sig_contrast} pairwise contrast(s) are significant")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        sys.exit(1)
