#!/usr/bin/env python3
"""
Step 02: Compute Attenuation Ratios for RQ 7.2.2
=================================================
Purpose: Calculate attenuation ratios as percentage reduction in age coefficients
when controlling for cognitive tests (including retention predictors).

Scientific Context:
Attenuation ratio = (beta_bivariate - beta_controlled) / beta_bivariate
- >70%: Substantial attenuation (supports VR scaffolding hypothesis)
- 30-70%: Partial attenuation
- <30%: Minimal attenuation

The VR scaffolding hypothesis predicts substantial attenuation because cognitive
tests should capture most age-related variance if VR provides environmental support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_compute_attenuation.log"

(RQ_DIR / "logs").mkdir(exist_ok=True)
(RQ_DIR / "data").mkdir(exist_ok=True)

def log(msg):
    """Log message to both file and console"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def compute_attenuation(beta_bivariate, beta_controlled):
    """
    Compute attenuation ratio and percentage.
    Can exceed 100% in suppression effects (sign reversal).
    """
    if abs(beta_bivariate) < 1e-10:
        return np.nan, np.nan

    attenuation_ratio = (beta_bivariate - beta_controlled) / beta_bivariate
    attenuation_percent = attenuation_ratio * 100

    return attenuation_ratio, attenuation_percent

def classify_attenuation(percent):
    """Classify attenuation magnitude"""
    if pd.isna(percent):
        return "undefined"
    elif percent < 0:
        return "negative_attenuation"
    elif percent < 30:
        return "minimal"
    elif percent < 70:
        return "partial"
    elif percent <= 100:
        return "substantial"
    else:
        return "suppression"

def main():
    """Main attenuation computation function"""
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 70)
    log("STEP 02: COMPUTE ATTENUATION RATIOS")
    log("=" * 70)

    # 1. Load coefficient table from Step 01
    log("\n1. Loading coefficient table from Step 01...")

    coef_file = RQ_DIR / "data" / "step01_coefficient_table.csv"
    if not coef_file.exists():
        log(f"ERROR: Cannot find coefficient table at {coef_file}")
        sys.exit(1)

    coef_df = pd.read_csv(coef_file)
    log(f"  Loaded {len(coef_df)} domains")
    log(f"  Columns: {list(coef_df.columns)}")

    # 2. Compute attenuation for each domain
    log("\n2. Computing attenuation for each domain...")

    attenuation_results = []
    for _, row in coef_df.iterrows():
        domain = row['domain']
        beta_biv = row['beta_bivariate']
        beta_ctrl = row['beta_controlled']

        if pd.isna(beta_biv) or pd.isna(beta_ctrl):
            log(f"\n  {domain.upper()}: SKIPPED (missing coefficients)")
            continue

        ratio, percent = compute_attenuation(beta_biv, beta_ctrl)
        classification = classify_attenuation(percent)

        log(f"\n  {domain.upper()} DOMAIN:")
        log(f"    Bivariate beta:  {beta_biv:.4f} (SE={row['se_bivariate']:.4f}, p={row['p_bivariate']:.4f})")
        log(f"    Controlled beta: {beta_ctrl:.4f} (SE={row['se_controlled']:.4f}, p={row['p_controlled']:.4f})")
        log(f"    R2 bivariate:    {row['r2_bivariate']:.4f}")
        log(f"    R2 controlled:   {row['r2_controlled']:.4f}")
        log(f"    Attenuation:     {percent:.1f}%")
        log(f"    Classification:  {classification}")
        log(f"    N:               {int(row['n'])}")

        if percent > 100:
            log("    *** SUPPRESSION EFFECT DETECTED ***")
            log("    Age coefficient reversed sign after controlling for cognitive tests")

        attenuation_results.append({
            'domain': domain,
            'n': int(row['n']),
            'beta_bivariate': beta_biv,
            'se_bivariate': row['se_bivariate'],
            'p_bivariate': row['p_bivariate'],
            'r2_bivariate': row['r2_bivariate'],
            'beta_controlled': beta_ctrl,
            'se_controlled': row['se_controlled'],
            'p_controlled': row['p_controlled'],
            'r2_controlled': row['r2_controlled'],
            'attenuation_ratio': ratio,
            'attenuation_percent': percent,
            'classification': classification,
        })

    attenuation_df = pd.DataFrame(attenuation_results)

    # 3. Save outputs
    log("\n3. Saving outputs...")

    output_file = RQ_DIR / "data" / "step02_attenuation_ratios.csv"
    attenuation_df.to_csv(output_file, index=False)
    log(f"  Saved attenuation ratios to: {output_file}")

    # Save effect classification report
    classification_file = RQ_DIR / "data" / "step02_effect_classification.txt"
    with open(classification_file, 'w') as f:
        f.write("ATTENUATION EFFECT CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("Classification Thresholds:\n")
        f.write("  <0%: Negative attenuation (unexpected)\n")
        f.write("  0-30%: Minimal attenuation\n")
        f.write("  30-70%: Partial attenuation\n")
        f.write("  70-100%: Substantial attenuation\n")
        f.write("  >100%: Suppression effect (sign reversal)\n\n")

        f.write("Results by Domain:\n")
        for _, row in attenuation_df.iterrows():
            f.write(f"\n{row['domain'].upper()} DOMAIN (n={int(row['n'])}):\n")
            f.write(f"  Bivariate: beta={row['beta_bivariate']:.4f}, p={row['p_bivariate']:.4f}\n")
            f.write(f"  Controlled: beta={row['beta_controlled']:.4f}, p={row['p_controlled']:.4f}\n")
            f.write(f"  Attenuation: {row['attenuation_percent']:.1f}%\n")
            f.write(f"  Classification: {row['classification']}\n")

            if row['attenuation_percent'] > 100:
                f.write("  *** SUPPRESSION EFFECT ***\n")

        # Overall interpretation
        overall = attenuation_df[attenuation_df['domain'] == 'overall']
        if len(overall) > 0:
            pct = overall.iloc[0]['attenuation_percent']
            f.write("\n" + "=" * 60 + "\n")
            f.write("INTERPRETATION:\n")

            if pct > 100:
                f.write("The suppression effect (>100% attenuation) strongly supports\n")
                f.write("the VR scaffolding hypothesis. After accounting for cognitive\n")
                f.write("abilities (including retention measures), age becomes a POSITIVE\n")
                f.write("predictor, suggesting older adults leverage VR environmental\n")
                f.write("support more effectively.\n")
            elif pct > 70:
                f.write("The substantial attenuation (>70%) supports the VR scaffolding\n")
                f.write("hypothesis. Cognitive tests capture most age-related variance.\n")
            elif pct > 30:
                f.write("The partial attenuation (30-70%) provides moderate support for\n")
                f.write("the VR scaffolding hypothesis.\n")
            else:
                f.write("The minimal attenuation (<30%) does not support the VR scaffolding\n")
                f.write("hypothesis.\n")

            f.write("\nNote: Controlled model includes retention predictors\n")
            f.write("(RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T) added in Phase 2 update.\n")

    log(f"  Saved effect classification to: {classification_file}")

    # Key finding summary
    overall_row = attenuation_df[attenuation_df['domain'] == 'overall']
    if len(overall_row) > 0:
        pct = overall_row.iloc[0]['attenuation_percent']
        cls = overall_row.iloc[0]['classification']
        log(f"\n{'=' * 70}")
        log(f"KEY FINDING: Overall attenuation = {pct:.1f}% ({cls})")
        if pct > 100:
            log("SUPPRESSION EFFECT: Age coefficient reversed sign")
        log(f"{'=' * 70}")

    log("\nStep 02 complete: Attenuation ratios computed")

    return attenuation_df

if __name__ == "__main__":
    main()
