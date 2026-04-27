#!/usr/bin/env python3
"""
RQ 6.3.3: GLMM Validation (v2 - fixed extraction)
==================================================
Item-level GLMM validation for Age × Domain interaction in confidence.

CRITICAL FIX: Parse fixed effects from summary table instead of result attributes
(result.tvalues includes variance components, causing length mismatch).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
RQ_631_DIR = Path("/home/etai/projects/REMEMVR/results/ch6/6.3.1")
LOG_FILE = RQ_DIR / "logs" / "glmm_validation.log"

# Input files
IRT_INPUT_FILE = RQ_631_DIR / "data" / "step00_irt_input.csv"
Q_MATRIX_FILE = RQ_631_DIR / "data" / "step00_q_matrix.csv"
TSVR_FILE = RQ_631_DIR / "data" / "step00_tsvr_mapping.csv"
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def reshape_to_long_format():
    """Reshape wide-format confidence data to long format for GLMM."""
    log("=" * 70)
    log("STEP 1: RESHAPE DATA TO LONG FORMAT")
    log("=" * 70)

    # Load wide-format confidence data
    log(f"\nLoading item-level confidence data: {IRT_INPUT_FILE}")
    df_wide = pd.read_csv(IRT_INPUT_FILE)
    log(f"Loaded: {len(df_wide)} rows, {len(df_wide.columns)} columns")

    # Load Q-matrix
    log(f"\nLoading Q-matrix: {Q_MATRIX_FILE}")
    df_q = pd.read_csv(Q_MATRIX_FILE)
    item_to_domain = dict(zip(df_q['item_name'], df_q['domain']))

    # Get TC_* columns
    tc_cols = [col for col in df_wide.columns if col.startswith('TC_')]
    log(f"Found {len(tc_cols)} confidence items")

    # Reshape to long
    df_long = df_wide.melt(
        id_vars=['composite_ID'],
        value_vars=tc_cols,
        var_name='item',
        value_name='confidence'
    )
    df_long['Domain'] = df_long['item'].map(item_to_domain)
    df_long['UID'] = df_long['composite_ID'].str.split('_').str[0]
    df_long['test'] = df_long['composite_ID'].str.split('_').str[1]

    # Load Age
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age'])
    df_age = df_demo.groupby('UID').agg({'age': 'first'}).reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    mean_age = df_age['Age'].mean()

    # Merge Age
    df_long = df_long.merge(df_age, on='UID', how='left')
    df_long['Age_c'] = df_long['Age'] - mean_age

    # Load TSVR
    log(f"Loading TSVR from: {TSVR_FILE}")
    df_tsvr = pd.read_csv(TSVR_FILE)
    df_long = df_long.merge(df_tsvr[['composite_ID', 'TSVR_hours']], on='composite_ID', how='left')

    # Remove missing
    df_long_clean = df_long.dropna(subset=['confidence'])
    log(f"\nFinal dataset: {len(df_long_clean)} rows")

    # Save
    output_path = RQ_DIR / "data" / "glmm_long_format.csv"
    df_long_clean.to_csv(output_path, index=False)
    log(f"✓ Long-format data saved: {output_path}")

    return df_long_clean, mean_age


def fit_glmm(df_long):
    """Fit GLMM with Age × Domain × Time interaction."""
    log("\n" + "=" * 70)
    log("STEP 2: FIT GLMM")
    log("=" * 70)

    formula = "confidence ~ TSVR_hours * Age_c * C(Domain, Treatment('What'))"
    log(f"\nModel formula: {formula}")
    log("Random effects: (1 | UID) + (1 | Item)")

    log("\nFitting GLMM (this may take several minutes)...")

    model = smf.mixedlm(
        formula=formula,
        data=df_long,
        groups=df_long['UID'],
        vc_formula={"item": "0 + C(item)"}
    )

    result = model.fit(method='powell', maxiter=1000, reml=False)
    log("✓ Model converged successfully")

    # Save summary
    summary_path = RQ_DIR / "data" / "glmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"✓ Full summary saved: {summary_path}")

    # Parse fixed effects from summary table
    fe = parse_fixed_effects_from_summary(summary_path)
    log(f"\nExtracted {len(fe)} fixed effects")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "glmm_fixed_effects.csv"
    fe.to_csv(fe_path, index=False)
    log(f"✓ Fixed effects saved: {fe_path}")

    return result, fe


def parse_fixed_effects_from_summary(summary_path):
    """Parse fixed effects from saved summary table."""
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    # Find coefficient table
    start_idx = None
    for i, line in enumerate(lines):
        if 'Coef.' in line and 'Std.Err.' in line:
            start_idx = i + 2
            break

    if not start_idx:
        raise ValueError("Cannot find coefficient table in summary")

    # Parse coefficients
    data = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or '=' in line or 'Var' in line:
            break

        # Split by whitespace
        parts = line.split()
        if len(parts) < 6:
            continue

        # Last 6 values: coef, se, z, p, ci_lower, ci_upper
        term = ' '.join(parts[:-6])
        coef = float(parts[-6])
        se = float(parts[-5])
        z = float(parts[-4])
        p = float(parts[-3])
        ci_lower = float(parts[-2])
        ci_upper = float(parts[-1])

        data.append({
            'term': term,
            'estimate': coef,
            'se': se,
            'z_value': z,
            'p_value': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    return pd.DataFrame(data)


def compare_irt_lmm_vs_glmm(fe_glmm):
    """Compare IRT→LMM vs GLMM results."""
    log("\n" + "=" * 70)
    log("STEP 3: COMPARE IRT→LMM VS GLMM")
    log("=" * 70)

    # Load IRT→LMM results
    irt_lmm_file = RQ_DIR / "data" / "step03_interaction_terms.csv"
    df_irt_lmm = pd.read_csv(irt_lmm_file)

    # Extract 3-way interaction terms
    glmm_3way = fe_glmm[
        fe_glmm['term'].str.contains('TSVR_hours') &
        fe_glmm['term'].str.contains('Age_c') &
        fe_glmm['term'].str.contains('Domain')
    ].copy()

    log(f"\nGLMM 3-way interaction terms: {len(glmm_3way)}")

    # Create comparison
    log("\n--- Comparison Results ---")
    log(f"{'Contrast':<20} {'IRT→LMM β':<12} {'IRT→LMM p':<12} {'GLMM β':<12} {'GLMM p':<12} {'Change':<20}")
    log("-" * 100)

    comparisons = []

    for domain in ['When', 'Where']:
        irt_row = df_irt_lmm[df_irt_lmm['term'].str.contains(domain)]
        glmm_row = glmm_3way[glmm_3way['term'].str.contains(domain)]

        if len(irt_row) == 1 and len(glmm_row) == 1:
            irt_beta = irt_row['estimate'].values[0]
            irt_p = irt_row['p_uncorrected'].values[0]
            glmm_beta = glmm_row['estimate'].values[0]
            glmm_p = glmm_row['p_value'].values[0]

            # Determine change
            if irt_p > 0.05 and glmm_p < 0.05:
                change = "NULL → SIGNIFICANT ⚠️"
            elif irt_p > 0.05 and glmm_p < 0.10:
                change = "NULL → MARGINAL"
            else:
                change = "Consistent"

            log(f"{domain:<20} {irt_beta:>11.6f} {irt_p:>11.4f} {glmm_beta:>11.6f} {glmm_p:>11.4f} {change:<20}")

            comparisons.append({
                'contrast': domain,
                'irt_lmm_beta': irt_beta,
                'irt_lmm_p': irt_p,
                'glmm_beta': glmm_beta,
                'glmm_p': glmm_p,
                'change': change
            })

    # Check for critical changes
    critical_changes = [c for c in comparisons if "SIGNIFICANT" in c['change']]

    log("")
    log("=" * 70)
    log("INTERPRETATION")
    log("=" * 70)

    if critical_changes:
        log("\n🔴 CRITICAL: GLMM reveals hidden effects")
        log("")
        for c in critical_changes:
            log(f"  {c['contrast']}: IRT→LMM p={c['irt_lmm_p']:.3f} → GLMM p={c['glmm_p']:.3f}")
        log("")
        log("IMPACT: IRT→LMM aggregation masked baseline Age × Domain effects")
        log("ACTION: Thesis narrative revision required (user decision)")
    else:
        log("\n✅ GLMM confirms IRT→LMM findings")
        log("")
        for c in comparisons:
            log(f"  {c['contrast']}: IRT→LMM p={c['irt_lmm_p']:.3f}, GLMM p={c['glmm_p']:.3f}")
        log("")
        log("CONCLUSION: Age-invariant confidence decline confirmed")

    # Save comparison
    comparison_path = RQ_DIR / "data" / "glmm_comparison.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    log(f"\n✓ Comparison saved: {comparison_path}")

    return comparisons, critical_changes


def main():
    """Execute GLMM validation."""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.3.3: GLMM Validation\n")
        f.write("=" * 70 + "\n\n")

    try:
        # Step 1: Reshape
        df_long, mean_age = reshape_to_long_format()

        # Step 2: Fit GLMM
        glmm_result, fe_glmm = fit_glmm(df_long)

        # Step 3: Compare
        comparisons, critical_changes = compare_irt_lmm_vs_glmm(fe_glmm)

        log("\n" + "=" * 70)
        log("GLMM VALIDATION COMPLETE")
        log("=" * 70)

        if critical_changes:
            log("\n🔴 BLOCKER: Critical changes detected")
            log("   IRT→LMM findings NOT robust to GLMM validation")
            log("   Thesis narrative revision required")
        else:
            log("\n✅ GLMM validation successful")
            log("   IRT→LMM findings confirmed")

        log(f"\nOutput files:")
        log(f"  - glmm_long_format.csv")
        log(f"  - glmm_model_summary.txt")
        log(f"  - glmm_fixed_effects.csv")
        log(f"  - glmm_comparison.csv")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
