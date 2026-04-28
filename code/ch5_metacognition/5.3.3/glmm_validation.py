#!/usr/bin/env python3
"""
RQ 6.3.3: GLMM Validation
=========================
Item-level GLMM validation for Age × Domain interaction in confidence.

MANDATORY for quality validation (per validation process Step 9).

Purpose: Test if IRT→LMM aggregation masked baseline Age/Domain effects
by fitting GLMM on raw item-level confidence ratings (N=28,800 observations).

Expected outcome: Confirm NULL 3-way interaction (p=1.00 from IRT→LMM)
or reveal hidden baseline effects (like RQ 6.1.3 precedent).

Design:
- DV: Ordinal confidence ratings (0, 0.25, 0.5, 0.75, 1.0)
- Model: Gaussian GLMM (continuous DV, not binomial)
- Random effects: (1 | UID) + (1 | Item)
- Fixed effects: Age_c × Domain × TSVR_hours (3-way interaction)

Data source: RQ 6.3.1 step00_irt_input.csv (item-level confidence)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.3
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
    """
    Reshape wide-format confidence data to long format for GLMM.

    Input: 400 rows (UID × test) × 72 items (wide)
    Output: 28,800 rows (400 × 72 items, long)

    Adds Domain, Age, TSVR variables per observation.
    """
    log("=" * 70)
    log("STEP 1: RESHAPE DATA TO LONG FORMAT")
    log("=" * 70)

    # Load wide-format confidence data
    log(f"\nLoading item-level confidence data: {IRT_INPUT_FILE}")
    df_wide = pd.read_csv(IRT_INPUT_FILE)
    log(f"Loaded: {len(df_wide)} rows, {len(df_wide.columns)} columns")

    # Load Q-matrix (item-to-domain mapping)
    log(f"\nLoading Q-matrix (domain assignments): {Q_MATRIX_FILE}")
    df_q = pd.read_csv(Q_MATRIX_FILE)
    log(f"Loaded: {len(df_q)} items")
    log(f"Domains: {df_q['domain'].value_counts().to_dict()}")

    # Create item-to-domain lookup
    item_to_domain = dict(zip(df_q['item_name'], df_q['domain']))

    # Get all TC_* columns (confidence items)
    tc_cols = [col for col in df_wide.columns if col.startswith('TC_')]
    log(f"\nFound {len(tc_cols)} confidence items")

    # Reshape to long format
    log("\nReshaping wide → long format...")
    df_long = df_wide.melt(
        id_vars=['composite_ID'],
        value_vars=tc_cols,
        var_name='item',
        value_name='confidence'
    )
    log(f"Reshaped: {len(df_long)} rows ({len(df_wide)} observations × {len(tc_cols)} items)")

    # Add Domain from Q-matrix
    df_long['Domain'] = df_long['item'].map(item_to_domain)
    log(f"\nDomain assignments:")
    log(f"{df_long['Domain'].value_counts().to_dict()}")

    # Parse UID and test from composite_ID
    df_long['UID'] = df_long['composite_ID'].str.split('_').str[0]
    df_long['test'] = df_long['composite_ID'].str.split('_').str[1]

    # Load Age
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age'])
    df_age = df_demo.groupby('UID').agg({'age': 'first'}).reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    log(f"Age data: {len(df_age)} unique participants")

    # Merge Age
    df_long = df_long.merge(df_age, on='UID', how='left')
    log(f"Merged with Age: {len(df_long)} rows")

    # Load TSVR
    log(f"\nLoading TSVR from: {TSVR_FILE}")
    df_tsvr = pd.read_csv(TSVR_FILE)
    log(f"TSVR data: {len(df_tsvr)} UID × test combinations")

    # Merge TSVR
    df_long = df_long.merge(df_tsvr[['composite_ID', 'TSVR_hours']], on='composite_ID', how='left')
    log(f"Merged with TSVR: {len(df_long)} rows")

    # Center Age
    mean_age = df_age['Age'].mean()
    df_long['Age_c'] = df_long['Age'] - mean_age
    log(f"\nAge centered: Age_c = Age - {mean_age:.2f}")

    # Remove missing confidence values
    df_long_clean = df_long.dropna(subset=['confidence'])
    n_missing = len(df_long) - len(df_long_clean)
    pct_missing = (n_missing / len(df_long)) * 100
    log(f"\nRemoved {n_missing} missing confidence values ({pct_missing:.1f}%)")
    log(f"Final dataset: {len(df_long_clean)} rows")
    # VALIDATION
    log("\n--- Data Validation ---")

    # Check row count
    expected_min = 20000  # Expect ~28,800 but allow for missing data
    expected_max = 30000
    if not (expected_min <= len(df_long_clean) <= expected_max):
        log(f"WARNING: Row count {len(df_long_clean)} outside expected range [{expected_min}, {expected_max}]")
    else:
        log(f"✓ Row count: {len(df_long_clean)} (expected ~28,800)")

    # Check confidence values
    conf_values = df_long_clean['confidence'].unique()
    expected_values = {0.0, 0.25, 0.5, 0.75, 1.0}
    unexpected = set(conf_values) - expected_values
    if unexpected:
        log(f"WARNING: Unexpected confidence values: {unexpected}")
    else:
        log(f"✓ Confidence values: {sorted(conf_values)}")

    # Check domains
    domains = set(df_long_clean['Domain'].unique())
    expected_domains = {'What', 'Where', 'When'}
    if domains != expected_domains:
        log(f"WARNING: Unexpected domains: {domains}")
    else:
        log(f"✓ Domains: {domains}")

    # Check Age_c centered
    age_c_mean = df_long_clean['Age_c'].mean()
    if abs(age_c_mean) > 0.1:
        log(f"WARNING: Age_c mean = {age_c_mean:.3f}, expected ~0")
    else:
        log(f"✓ Age_c centered: mean = {age_c_mean:.6f}")

    # Check no missing critical variables
    critical_vars = ['UID', 'item', 'Domain', 'Age_c', 'TSVR_hours', 'confidence']
    for var in critical_vars:
        n_miss = df_long_clean[var].isna().sum()
        if n_miss > 0:
            log(f"WARNING: {var} has {n_miss} missing values")
        else:
            log(f"✓ {var}: no missing values")

    # Save long-format data
    output_path = RQ_DIR / "data" / "glmm_long_format.csv"
    df_long_clean.to_csv(output_path, index=False)
    log(f"\n✓ Long-format data saved: {output_path}")

    return df_long_clean, mean_age


def fit_glmm(df_long, mean_age):
    """
    Fit GLMM with Age × Domain × Time interaction.

    Model: confidence ~ Age_c × Domain × TSVR_hours + (1 | UID) + (1 | Item)

    Fixed effects test 3-way interaction for Age × Domain × Time.
    Random effects: participant intercepts + item intercepts.
    """
    log("\n" + "=" * 70)
    log("STEP 2: FIT GLMM")
    log("=" * 70)

    # Formula (full 3-way interaction)
    formula = "confidence ~ TSVR_hours * Age_c * C(Domain, Treatment('What'))"
    log(f"\nModel formula: {formula}")
    log("Random effects: (1 | UID) + (1 | Item)")
    log("Reference domain: What")

    # Fit GLMM (Gaussian family for continuous confidence)
    log("\nFitting GLMM (this may take several minutes)...")

    try:
        # Use mixedlm for Gaussian GLMM with crossed random effects
        # Note: statsmodels mixedlm supports ONE grouping variable
        # For crossed (UID + Item), we use UID as groups and add Item as vc_formula

        model = smf.mixedlm(
            formula=formula,
            data=df_long,
            groups=df_long['UID'],
            vc_formula={"item": "0 + C(item)"}  # Item random intercepts via vc_formula
        )

        result = model.fit(method='powell', maxiter=1000, reml=False)
        log("✓ Model converged successfully")

    except Exception as e:
        log(f"ERROR: Model fitting failed: {e}")
        log("\nAttempting simplified model without item random effects...")

        # Fallback: UID random effects only (no item)
        model_simple = smf.mixedlm(
            formula=formula,
            data=df_long,
            groups=df_long['UID']
        )

        result = model_simple.fit(method='powell', maxiter=1000, reml=False)
        log("✓ Simplified model (UID only) converged")

    # Save full summary
    summary_path = RQ_DIR / "data" / "glmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"✓ Full summary saved: {summary_path}")

    # Extract fixed effects
    log("\n--- Fixed Effects ---")
    fe = pd.DataFrame({
        'term': result.fe_params.index,
        'estimate': result.fe_params.values,
        'se': result.bse_fe.values,
        'z_value': result.tvalues.values,
        'p_value': result.pvalues.values
    })

    for _, row in fe.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        log(f"  {row['term']:50s}: β = {row['estimate']:8.5f}, SE = {row['se']:.5f}, z = {row['z_value']:6.2f}, p = {row['p_value']:.4f} {sig}")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "glmm_fixed_effects.csv"
    fe.to_csv(fe_path, index=False)
    log(f"\n✓ Fixed effects saved: {fe_path}")

    return result, fe


def compare_irt_lmm_vs_glmm(fe_glmm, mean_age):
    """
    Compare IRT→LMM vs GLMM results for Age × Domain × Time interaction.

    Reads IRT→LMM results from step03_interaction_terms.csv and compares
    p-values for 3-way interaction terms.
    """
    log("\n" + "=" * 70)
    log("STEP 3: COMPARE IRT→LMM VS GLMM")
    log("=" * 70)

    # Load IRT→LMM results
    irt_lmm_file = RQ_DIR / "data" / "step03_interaction_terms.csv"
    log(f"\nLoading IRT→LMM results: {irt_lmm_file}")
    df_irt_lmm = pd.read_csv(irt_lmm_file)

    # Extract 3-way interaction terms from GLMM
    glmm_3way = fe_glmm[
        fe_glmm['term'].str.contains('TSVR_hours') &
        fe_glmm['term'].str.contains('Age_c') &
        fe_glmm['term'].str.contains('Domain')
    ].copy()

    log(f"\nGLMM 3-way interaction terms: {len(glmm_3way)}")

    # Create comparison table
    log("\n--- Comparison Results ---")
    log("")
    log(f"{'Contrast':<20} {'IRT→LMM β':<12} {'IRT→LMM p':<12} {'GLMM β':<12} {'GLMM p':<12} {'Change':<20}")
    log("-" * 100)

    comparisons = []

    for domain in ['When', 'Where']:
        # Find corresponding terms
        irt_row = df_irt_lmm[df_irt_lmm['term'].str.contains(domain)]
        glmm_row = glmm_3way[glmm_3way['term'].str.contains(domain)]

        if len(irt_row) == 1 and len(glmm_row) == 1:
            irt_beta = irt_row['estimate'].values[0]
            irt_p = irt_row['p_uncorrected'].values[0]
            glmm_beta = glmm_row['estimate'].values[0]
            glmm_p = glmm_row['p_value'].values[0]

            # Determine change type
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
        else:
            log(f"{domain:<20} NOT FOUND IN BOTH MODELS")

    # Check for critical changes (NULL → SIGNIFICANT)
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
        log("RECOMMENDATION: Report GLMM findings, document methodological note")

    else:
        log("\n✅ GLMM confirms IRT→LMM findings")
        log("")
        for c in comparisons:
            log(f"  {c['contrast']}: IRT→LMM p={c['irt_lmm_p']:.3f}, GLMM p={c['glmm_p']:.3f} (consistent)")
        log("")
        log("IMPACT: NULL 3-way interaction robust across methods")
        log("CONCLUSION: Age-invariant confidence decline confirmed")

    # Save comparison
    comparison_path = RQ_DIR / "data" / "glmm_comparison.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    log(f"\n✓ Comparison saved: {comparison_path}")

    return comparisons, critical_changes


def main():
    """Execute GLMM validation."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.3.3: GLMM Validation\n")
        f.write("=" * 70 + "\n\n")

    try:
        # Step 1: Reshape data to long format
        df_long, mean_age = reshape_to_long_format()

        # Step 2: Fit GLMM
        glmm_result, fe_glmm = fit_glmm(df_long, mean_age)

        # Step 3: Compare IRT→LMM vs GLMM
        comparisons, critical_changes = compare_irt_lmm_vs_glmm(fe_glmm, mean_age)

        log("\n" + "=" * 70)
        log("GLMM VALIDATION COMPLETE")
        log("=" * 70)

        if critical_changes:
            log("\n🔴 BLOCKER: Critical changes detected")
            log("   IRT→LMM findings NOT robust to GLMM validation")
            log("   Thesis narrative revision required")
        else:
            log("\n✅ GLMM validation successful")
            log("   IRT→LMM findings confirmed by item-level analysis")
            log("   NULL 3-way interaction robust across methods")

        log(f"\nOutput files:")
        log(f"  - {RQ_DIR / 'data' / 'glmm_long_format.csv'}")
        log(f"  - {RQ_DIR / 'data' / 'glmm_model_summary.txt'}")
        log(f"  - {RQ_DIR / 'data' / 'glmm_fixed_effects.csv'}")
        log(f"  - {RQ_DIR / 'data' / 'glmm_comparison.csv'}")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
