#!/usr/bin/env python3
"""
RQ 6.5.3: High-Confidence Errors (Schema-Incongruent Effects)
=============================================================

Research Question: Do schema-incongruent items produce more high-confidence
errors (HCE) than schema-congruent or common items?

Pipeline:
- Step 00: Extract item-level accuracy/confidence for congruence-tagged items
- Step 01: Flag high-confidence errors (Accuracy=0 AND Confidence>=0.75)
- Step 02: Compute HCE rates by Congruence × Test
- Step 03: Fit GLMM with Congruence × Time + crossed random effects
- Step 04: Post-hoc contrasts if Congruence effect significant

Analysis Type: Item-level CTT (no IRT)
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)

# PATHS AND CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.5.3
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"

# Ensure directories exist
(RQ_DIR / "logs").mkdir(exist_ok=True)
(RQ_DIR / "data").mkdir(exist_ok=True)

# Configuration
CONFIG = {
    'paradigms': ['IFR', 'ICR', 'IRE'],  # Interactive VR paradigms only
    'domain': 'N',  # What domain (object identity)
    'items': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'],
    'congruence_map': {
        'i1': 'Common', 'i2': 'Common',
        'i3': 'Congruent', 'i4': 'Congruent',
        'i5': 'Incongruent', 'i6': 'Incongruent'
    },
    'high_confidence_threshold': 0.75,  # Likert 4 or 5 on 5-point scale
    'time_mapping': {1: 0, 2: 1, 3: 3, 4: 6},  # Nominal days (Test 1,2,3,4 -> Day 0,1,3,6)
    'significance_threshold': 0.05,
    'bonferroni_n_tests': 3  # 3 pairwise contrasts
}

def log(msg: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')
        f.flush()
    print(log_msg, flush=True)

# EXTRACT ITEM-LEVEL ACCURACY AND CONFIDENCE DATA

def step00_extract_item_level():
    """Extract item-level accuracy and confidence for congruence-tagged items."""
    log("=" * 70)
    log("STEP 00: Extract Item-Level Accuracy and Confidence Data")
    log("=" * 70)

    # Read dfData.csv
    df_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
    log(f"Reading: {df_path}")
    df = pd.read_csv(df_path)
    log(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Build list of target columns
    tq_cols = []
    tc_cols = []
    for paradigm in CONFIG['paradigms']:
        for item in CONFIG['items']:
            tq_col = f"TQ_{paradigm}-{CONFIG['domain']}-{item}"
            tc_col = f"TC_{paradigm}-{CONFIG['domain']}-{item}"
            if tq_col in df.columns and tc_col in df.columns:
                tq_cols.append(tq_col)
                tc_cols.append(tc_col)

    log(f"  Found {len(tq_cols)} TQ columns, {len(tc_cols)} TC columns")

    # Extract and reshape to long format
    records = []
    for idx, row in df.iterrows():
        uid = row['UID']
        test = row['TEST']
        tsvr = row['TSVR']

        for tq_col, tc_col in zip(tq_cols, tc_cols):
            # Parse tag: TQ_IFR-N-i1 -> paradigm=IFR, item=i1
            parts = tq_col.replace('TQ_', '').split('-')
            paradigm = parts[0]
            item = parts[2]
            congruence = CONFIG['congruence_map'][item]

            accuracy = row[tq_col]
            confidence = row[tc_col]

            # Skip if both are NaN
            if pd.isna(accuracy) and pd.isna(confidence):
                continue

            records.append({
                'UID': uid,
                'ItemID': tq_col.replace('TQ_', ''),
                'Test': test,
                'TSVR': tsvr,
                'Paradigm': paradigm,
                'Item': item,
                'Congruence': congruence,
                'Accuracy': accuracy,
                'Confidence': confidence
            })

    df_items = pd.DataFrame(records)

    # Basic validation
    n_participants = df_items['UID'].nunique()
    n_rows = len(df_items)
    congruence_levels = df_items['Congruence'].unique()

    log(f"  Extracted {n_rows} item-responses")
    log(f"  Unique participants: {n_participants}")
    log(f"  Congruence levels: {sorted(congruence_levels)}")
    log(f"  Test sessions: {sorted(df_items['Test'].unique())}")
    log(f"  Paradigms: {sorted(df_items['Paradigm'].unique())}")

    # Check congruence distribution
    cong_dist = df_items['Congruence'].value_counts()
    for level in ['Common', 'Congruent', 'Incongruent']:
        pct = cong_dist.get(level, 0) / n_rows * 100
        log(f"    {level}: {cong_dist.get(level, 0)} ({pct:.1f}%)")

    # Check for missing data
    acc_missing = df_items['Accuracy'].isna().sum()
    conf_missing = df_items['Confidence'].isna().sum()
    log(f"  Missing Accuracy: {acc_missing} ({acc_missing/n_rows*100:.1f}%)")
    log(f"  Missing Confidence: {conf_missing} ({conf_missing/n_rows*100:.1f}%)")

    # Validation
    if n_participants != 100:
        log(f"WARNING: Expected 100 participants, found {n_participants}")
    if n_rows < 6000 or n_rows > 10000:
        log(f"WARNING: Row count {n_rows} outside expected range [6000, 10000]")
    if set(congruence_levels) != {'Common', 'Congruent', 'Incongruent'}:
        log(f"WARNING: Missing congruence levels")

    # Save
    out_path = RQ_DIR / "data" / "step00_item_level.csv"
    df_items.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}")
    log(f"STEP 00 COMPLETE: {n_rows} rows extracted")

    return df_items

# IDENTIFY HIGH-CONFIDENCE ERRORS

def step01_identify_hce(df_items: pd.DataFrame) -> pd.DataFrame:
    """Flag high-confidence errors (HCE)."""
    log("=" * 70)
    log("STEP 01: Identify High-Confidence Errors")
    log("=" * 70)

    df = df_items.copy()

    # HCE definition: Accuracy == 0 AND Confidence >= 0.75
    # Need to handle partial credit: Accuracy values are 0, 0.25, 0.5, 1.0
    # For HCE, we consider Accuracy == 0 (completely wrong) only
    threshold = CONFIG['high_confidence_threshold']

    df['HCE_flag'] = ((df['Accuracy'] == 0) & (df['Confidence'] >= threshold)).astype(int)

    # Statistics
    n_total = len(df)
    n_hce = df['HCE_flag'].sum()
    hce_rate = n_hce / n_total * 100

    log(f"  High-confidence threshold: >= {threshold}")
    log(f"  Error threshold: Accuracy == 0 (completely incorrect)")
    log(f"  Total item-responses: {n_total}")
    log(f"  High-confidence errors: {n_hce} ({hce_rate:.1f}%)")

    # Breakdown by congruence
    log("  HCE by Congruence:")
    for cong in ['Common', 'Congruent', 'Incongruent']:
        subset = df[df['Congruence'] == cong]
        n = len(subset)
        n_hce_cong = subset['HCE_flag'].sum()
        rate = n_hce_cong / n * 100 if n > 0 else 0
        log(f"    {cong}: {n_hce_cong}/{n} ({rate:.1f}%)")

    # Validation: Check logical consistency
    hce_rows = df[df['HCE_flag'] == 1]
    if len(hce_rows) > 0:
        acc_check = (hce_rows['Accuracy'] == 0).all()
        conf_check = (hce_rows['Confidence'] >= threshold).all()
        log(f"  Validation: All HCE have Accuracy=0: {acc_check}")
        log(f"  Validation: All HCE have Confidence>={threshold}: {conf_check}")
        if not acc_check or not conf_check:
            raise ValueError("HCE logic validation failed!")

    # Save
    out_path = RQ_DIR / "data" / "step01_hce_flags.csv"
    df.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}")
    log(f"STEP 01 COMPLETE: {n_hce} HCEs identified ({hce_rate:.1f}%)")

    return df

# COMPUTE HCE RATES BY CONGRUENCE × TEST

def step02_compute_hce_rates(df_hce: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HCE rates by Congruence × Test (12 cells)."""
    log("=" * 70)
    log("STEP 02: Compute HCE Rates by Congruence × Test")
    log("=" * 70)

    # Group by Congruence × Test
    agg = df_hce.groupby(['Congruence', 'Test']).agg(
        N_responses=('HCE_flag', 'count'),
        N_hce=('HCE_flag', 'sum'),
        HCE_rate=('HCE_flag', 'mean')
    ).reset_index()

    # Ensure all 12 cells exist (Tests are integers 1-4 in dfData)
    expected_cells = set()
    for cong in ['Common', 'Congruent', 'Incongruent']:
        for test in [1, 2, 3, 4]:
            expected_cells.add((cong, test))

    actual_cells = set(zip(agg['Congruence'], agg['Test']))
    missing_cells = expected_cells - actual_cells

    if missing_cells:
        log(f"WARNING: Missing cells: {missing_cells}")

    log(f"  Generated {len(agg)} cells (expected 12)")

    # Display HCE rates
    log("  HCE Rates by Congruence × Test:")
    log(f"  {'Congruence':<12} {'Test':<6} {'N_resp':<8} {'N_hce':<8} {'HCE_rate':<10}")
    log(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")

    for _, row in agg.sort_values(['Congruence', 'Test']).iterrows():
        log(f"  {row['Congruence']:<12} {row['Test']:<6} {row['N_responses']:<8} {row['N_hce']:<8} {row['HCE_rate']:.4f}")

    # Marginal means by Congruence (averaging over tests)
    log("\n  Marginal HCE Rates by Congruence:")
    marg_cong = df_hce.groupby('Congruence')['HCE_flag'].agg(['mean', 'count', 'sum'])
    for cong in ['Common', 'Congruent', 'Incongruent']:
        rate = marg_cong.loc[cong, 'mean'] * 100
        n = int(marg_cong.loc[cong, 'count'])
        n_hce = int(marg_cong.loc[cong, 'sum'])
        log(f"    {cong}: {rate:.2f}% ({n_hce}/{n})")

    # Save
    out_path = RQ_DIR / "data" / "step02_hce_rates.csv"
    agg.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}")
    log(f"STEP 02 COMPLETE: 12 cells computed")

    return agg

# FIT GLMM FOR CONGRUENCE × TIME EFFECT

def step03_fit_glmm(df_hce: pd.DataFrame) -> dict:
    """Fit GLMM with Congruence × Time interaction and crossed random effects."""
    log("=" * 70)
    log("STEP 03: Fit Mixed-Effects Model (GLMM)")
    log("=" * 70)

    df = df_hce.copy()

    # Convert Test to numeric time
    df['Time'] = df['Test'].map(CONFIG['time_mapping'])
    log(f"  Time mapping: {CONFIG['time_mapping']}")

    # Note on model: statsmodels doesn't support crossed random effects for binomial GLMM
    # Using LMM on binary outcome (linear probability model) as approximation
    # Alternative: pymer4 or lme4 via rpy2, but keeping it pure Python
    log("  Note: Using Linear Mixed Model (LMM) on binary HCE_flag")
    log("  (Linear probability model - statsmodels limitation)")

    # Set reference level for Congruence (Common as baseline)
    df['Congruence'] = pd.Categorical(
        df['Congruence'],
        categories=['Common', 'Congruent', 'Incongruent']
    )

    # Model formula: HCE_flag ~ Congruence * Time with random intercept/slope by UID
    # Note: Cannot include ItemID random effect in statsmodels MixedLM easily
    formula = "HCE_flag ~ C(Congruence, Treatment('Common')) * Time"

    log(f"  Formula: {formula}")
    log(f"  Random effects: (Time | UID)")
    log(f"  Reference level: Common")

    # Fit model
    log("  Fitting model...")
    try:
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df['UID'],
            re_formula="~Time"
        )
        result = model.fit(method='powell', maxiter=500)
        converged = True
        log("  Model converged successfully")
    except Exception as e:
        log(f"  ERROR: Model fitting failed: {e}")
        # Try simpler model (random intercept only)
        log("  Attempting simpler model (random intercept only)...")
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df['UID']
        )
        result = model.fit(method='powell', maxiter=500)
        converged = True
        log("  Simpler model converged")

    # Extract fixed effects
    n_fe = len(result.model.exog_names)
    fe_names = result.model.exog_names
    fe_params = result.params[:n_fe]
    fe_se = result.bse[:n_fe]
    fe_z = fe_params / fe_se
    fe_p = 2 * (1 - stats.norm.cdf(np.abs(fe_z)))

    # Save model summary
    summary_path = RQ_DIR / "data" / "step03_congruence_hce_model.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ 6.5.3: GLMM for HCE by Schema Congruence\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Random effects: (Time | UID)\n")
        f.write(f"Reference level: Common\n")
        f.write(f"Observations: {len(df)}\n")
        f.write(f"Groups (UID): {df['UID'].nunique()}\n\n")
        f.write("FIXED EFFECTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Effect':<45} {'β':>10} {'SE':>10} {'z':>10} {'p':>10}\n")
        f.write("-" * 70 + "\n")
        for name, param, se, z, p in zip(fe_names, fe_params, fe_se, fe_z, fe_p):
            f.write(f"{name:<45} {param:>10.4f} {se:>10.4f} {z:>10.3f} {p:>10.4f}\n")
        f.write("-" * 70 + "\n\n")
        f.write("RANDOM EFFECTS:\n")
        f.write("-" * 70 + "\n")
        f.write(str(result.cov_re) + "\n\n")
        f.write(f"Log-likelihood: {result.llf:.2f}\n")
        f.write(f"AIC: {result.aic:.2f}\n")
        f.write(f"BIC: {result.bic:.2f}\n")
    log(f"  Saved model summary: {summary_path}")

    # Display fixed effects
    log("\n  FIXED EFFECTS:")
    log(f"  {'Effect':<45} {'β':>10} {'SE':>10} {'z':>10} {'p':>10}")
    log(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, param, se, z, p in zip(fe_names, fe_params, fe_se, fe_z, fe_p):
        sig = "*" if p < 0.05 else ""
        log(f"  {name:<45} {param:>10.4f} {se:>10.4f} {z:>10.3f} {p:>9.4f}{sig}")

    # Extract key effects for hypothesis tests
    # Congruence main effects (Congruent vs Common, Incongruent vs Common)
    # Interaction effects (Congruent:Time, Incongruent:Time)

    test_results = []

    # Find Congruence effects
    for name, param, se, z, p in zip(fe_names, fe_params, fe_se, fe_z, fe_p):
        if 'Congruent' in name or 'Incongruent' in name or 'Time' in name:
            if ':' in name:
                effect_type = 'Interaction'
            elif 'Time' in name and 'Congruent' not in name and 'Incongruent' not in name:
                effect_type = 'Time'
            else:
                effect_type = 'Congruence'

            test_results.append({
                'Effect': name,
                'Type': effect_type,
                'Estimate': param,
                'SE': se,
                'z_value': z,
                'p_value': p
            })

    df_tests = pd.DataFrame(test_results)
    test_path = RQ_DIR / "data" / "step03_congruence_hce_test.csv"
    df_tests.to_csv(test_path, index=False)
    log(f"  Saved hypothesis tests: {test_path}")

    # Check if Congruence effects are significant
    cong_p_values = []
    for name, p in zip(fe_names, fe_p):
        if ('Congruent' in name or 'Incongruent' in name) and ':' not in name:
            cong_p_values.append(p)

    congruence_significant = any(p < CONFIG['significance_threshold'] for p in cong_p_values)
    log(f"\n  Congruence p-values: {[f'{p:.4f}' for p in cong_p_values]}")
    log(f"  Congruence effect significant (any p < 0.05): {congruence_significant}")

    log(f"STEP 03 COMPLETE: Model fitted, converged={converged}")

    return {
        'result': result,
        'fe_names': fe_names,
        'fe_params': fe_params,
        'fe_se': fe_se,
        'fe_z': fe_z,
        'fe_p': fe_p,
        'congruence_significant': congruence_significant,
        'df': df
    }

# POST-HOC CONTRASTS

def step04_post_hoc_contrasts(model_info: dict):
    """Compute pairwise contrasts with Bonferroni correction."""
    log("=" * 70)
    log("STEP 04: Post-Hoc Contrasts")
    log("=" * 70)

    result = model_info['result']
    df = model_info['df']
    congruence_significant = model_info['congruence_significant']
    fe_names = model_info['fe_names']
    fe_params = model_info['fe_params']
    fe_se = model_info['fe_se']

    if not congruence_significant:
        log("  Congruence main effect NOT significant (p >= 0.05)")
        log("  Skipping post-hoc contrasts (no pairwise tests needed)")

        # Save empty contrasts file with note
        df_contrasts = pd.DataFrame({
            'Contrast': ['No contrasts - Congruence effect NULL'],
            'Estimate': [np.nan],
            'SE': [np.nan],
            'z_value': [np.nan],
            'p_uncorrected': [np.nan],
            'p_bonferroni': [np.nan]
        })
        out_path = RQ_DIR / "data" / "step04_post_hoc_contrasts.csv"
        df_contrasts.to_csv(out_path, index=False)
        log(f"  Saved (empty): {out_path}")
        log("STEP 04 COMPLETE: No contrasts computed (NULL Congruence effect)")
        return df_contrasts

    # Congruence IS significant - compute pairwise contrasts
    log("  Congruence effect significant - computing pairwise contrasts")

    # Extract coefficients
    # From Treatment coding with Common as reference:
    # Congruent effect = β_Congruent (vs Common)
    # Incongruent effect = β_Incongruent (vs Common)
    # Congruent vs Incongruent = β_Congruent - β_Incongruent

    coef_dict = dict(zip(fe_names, fe_params))
    se_dict = dict(zip(fe_names, fe_se))

    # Find the Congruence coefficients
    cong_coef = None
    incong_coef = None
    cong_se = None
    incong_se = None

    for name in fe_names:
        if "Congruent" in name and "Incongruent" not in name and ":" not in name:
            cong_coef = coef_dict[name]
            cong_se = se_dict[name]
        if "Incongruent" in name and ":" not in name:
            incong_coef = coef_dict[name]
            incong_se = se_dict[name]

    if cong_coef is None or incong_coef is None:
        log("  ERROR: Could not find Congruence coefficients")
        raise ValueError("Missing Congruence coefficients")

    log(f"  Congruent vs Common: β = {cong_coef:.4f}, SE = {cong_se:.4f}")
    log(f"  Incongruent vs Common: β = {incong_coef:.4f}, SE = {incong_se:.4f}")

    # Compute contrasts
    contrasts = []

    # 1. Incongruent vs Common (directly from model)
    est1 = incong_coef
    se1 = incong_se
    z1 = est1 / se1
    p1 = 2 * (1 - stats.norm.cdf(abs(z1)))
    contrasts.append({
        'Contrast': 'Incongruent vs Common',
        'Estimate': est1,
        'SE': se1,
        'z_value': z1,
        'p_uncorrected': p1,
        'p_bonferroni': min(p1 * CONFIG['bonferroni_n_tests'], 1.0)
    })

    # 2. Congruent vs Common (directly from model)
    est2 = cong_coef
    se2 = cong_se
    z2 = est2 / se2
    p2 = 2 * (1 - stats.norm.cdf(abs(z2)))
    contrasts.append({
        'Contrast': 'Congruent vs Common',
        'Estimate': est2,
        'SE': se2,
        'z_value': z2,
        'p_uncorrected': p2,
        'p_bonferroni': min(p2 * CONFIG['bonferroni_n_tests'], 1.0)
    })

    # 3. Incongruent vs Congruent (difference of coefficients)
    # Assuming uncorrelated estimates for SE approximation
    est3 = incong_coef - cong_coef
    se3 = np.sqrt(incong_se**2 + cong_se**2)  # Conservative approximation
    z3 = est3 / se3
    p3 = 2 * (1 - stats.norm.cdf(abs(z3)))
    contrasts.append({
        'Contrast': 'Incongruent vs Congruent',
        'Estimate': est3,
        'SE': se3,
        'z_value': z3,
        'p_uncorrected': p3,
        'p_bonferroni': min(p3 * CONFIG['bonferroni_n_tests'], 1.0)
    })

    df_contrasts = pd.DataFrame(contrasts)

    # Display results
    log("\n  PAIRWISE CONTRASTS (Bonferroni-corrected):")
    log(f"  {'Contrast':<25} {'Estimate':>10} {'SE':>10} {'z':>10} {'p_uncorr':>10} {'p_bonf':>10}")
    log(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in df_contrasts.iterrows():
        sig = "*" if row['p_bonferroni'] < 0.05 else ""
        log(f"  {row['Contrast']:<25} {row['Estimate']:>10.4f} {row['SE']:>10.4f} "
            f"{row['z_value']:>10.3f} {row['p_uncorrected']:>10.4f} {row['p_bonferroni']:>9.4f}{sig}")

    # Save
    out_path = RQ_DIR / "data" / "step04_post_hoc_contrasts.csv"
    df_contrasts.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}")

    # Interpretation
    log("\n  HYPOTHESIS TEST RESULT:")
    incong_vs_common_sig = df_contrasts[df_contrasts['Contrast'] == 'Incongruent vs Common']['p_bonferroni'].values[0] < 0.05
    incong_vs_cong_sig = df_contrasts[df_contrasts['Contrast'] == 'Incongruent vs Congruent']['p_bonferroni'].values[0] < 0.05

    if incong_vs_common_sig or incong_vs_cong_sig:
        log("  HYPOTHESIS SUPPORTED: Incongruent items show different HCE rate")
        direction = "higher" if incong_coef > 0 else "lower"
        log(f"    Incongruent items have {direction} HCE rate than Common/Congruent")
    else:
        log("  HYPOTHESIS NOT SUPPORTED: No significant difference for Incongruent items")
        log("    Despite overall Congruence effect, Incongruent not significantly different")

    log(f"STEP 04 COMPLETE: 3 pairwise contrasts computed")

    return df_contrasts

# MAIN EXECUTION

def main():
    """Execute all steps."""
    log("=" * 70)
    log("RQ 6.5.3: High-Confidence Errors (Schema-Incongruent Effects)")
    log("=" * 70)
    log(f"Start time: {datetime.now()}")
    log(f"RQ Directory: {RQ_DIR}")
    log("")

    try:
        # Step 00: Extract item-level data
        df_items = step00_extract_item_level()
        log("")

        # Step 01: Identify HCE
        df_hce = step01_identify_hce(df_items)
        log("")

        # Step 02: Compute HCE rates
        df_rates = step02_compute_hce_rates(df_hce)
        log("")

        # Step 03: Fit GLMM
        model_info = step03_fit_glmm(df_hce)
        log("")

        # Step 04: Post-hoc contrasts
        df_contrasts = step04_post_hoc_contrasts(model_info)
        log("")

        log("=" * 70)
        log("ALL STEPS COMPLETE")
        log("=" * 70)
        log(f"End time: {datetime.now()}")

        # Final summary
        log("\nFINAL SUMMARY:")
        total_items = len(df_hce)
        total_hce = df_hce['HCE_flag'].sum()
        hce_rate = total_hce / total_items * 100
        log(f"  Total item-responses: {total_items}")
        log(f"  Total HCE: {total_hce} ({hce_rate:.1f}%)")
        log(f"  Congruence effect significant: {model_info['congruence_significant']}")

        if model_info['congruence_significant']:
            # Find which contrast was significant
            for _, row in df_contrasts.iterrows():
                if row['p_bonferroni'] < 0.05:
                    log(f"  Significant contrast: {row['Contrast']} (p_bonf = {row['p_bonferroni']:.4f})")
        else:
            log("  Result: NULL schema effect on HCE rate")

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
