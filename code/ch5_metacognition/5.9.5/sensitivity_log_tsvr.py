#!/usr/bin/env python3
"""
Sensitivity Analysis: Verify three-way interaction with log_TSVR time scale

PURPOSE: Address validation concern about time variable inconsistency between
Ch5 5.5.1 (TSVR_hours) and Ch6 6.8.1 (log_TSVR).

This script reruns the Step 3 three-way interaction model using log_TSVR
instead of TSVR_hours to verify the dissociation finding is robust to
time scale choice.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "sensitivity_log_tsvr.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

try:
    log("[START] Sensitivity Analysis: log_TSVR Time Scale")

    # Load merged data from Step 3
    df = pd.read_csv(RQ_DIR / "data" / "step03_merged_data_long.csv")
    log(f"[LOAD] {len(df)} rows from step03_merged_data_long.csv")

    # Create log_TSVR variable (match Ch6 6.8.1 approach)
    # Add small constant to avoid log(0)
    df['log_TSVR'] = np.log(df['TSVR_hours'] + 0.1)
    log(f"[TRANSFORM] Created log_TSVR from TSVR_hours")
    log(f"  TSVR_hours range: {df['TSVR_hours'].min():.2f} - {df['TSVR_hours'].max():.2f}")
    log(f"  log_TSVR range: {df['log_TSVR'].min():.2f} - {df['log_TSVR'].max():.2f}")

    # Fit three-way interaction model with log_TSVR
    log("\n[FIT] Three-way interaction: theta ~ measure * location * log_TSVR")

    model = smf.mixedlm(
        formula='theta ~ measure * location * log_TSVR',
        data=df,
        groups=df['UID'],
        re_formula='~1'
    ).fit(reml=False)

    log(f"[CONVERGED] Model converged: {model.converged}")

    # Extract fixed effects
    fe_table = model.summary().tables[1]
    if isinstance(fe_table, pd.DataFrame):
        df_fixed = fe_table.reset_index()
    else:
        df_fixed = pd.DataFrame(fe_table.data[1:], columns=fe_table.data[0])

    df_fixed.columns = ['term', 'beta', 'se', 'z', 'p_uncorrected', 'ci_lower', 'ci_upper']

    # Convert to numeric
    for col in ['beta', 'se', 'z', 'p_uncorrected']:
        df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')

    # Apply Bonferroni correction (2 tests: main 3-way + sensitivity)
    df_fixed['p_bonferroni'] = (df_fixed['p_uncorrected'] * 2).clip(upper=1.0)

    # Extract three-way interaction
    three_way = df_fixed[df_fixed['term'].str.contains('measure.*location.*log_TSVR', regex=True)]

    if len(three_way) > 0:
        row = three_way.iloc[0]
        log("\n[THREE-WAY INTERACTION RESULTS]")
        log(f"  Term: {row['term']}")
        log(f"  Beta: {row['beta']:.6f}")
        log(f"  SE: {row['se']:.6f}")
        log(f"  z-statistic: {row['z']:.2f}")
        log(f"  p-value (uncorrected): {row['p_uncorrected']:.6f}")
        log(f"  p-value (Bonferroni): {row['p_bonferroni']:.6f}")
        log(f"  95% CI: [{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}]")

        # Compare with original TSVR_hours result
        log("\n[COMPARISON WITH ORIGINAL]")
        log("  Original (TSVR_hours): p_bonf = 0.0000 (highly significant)")
        log(f"  Sensitivity (log_TSVR): p_bonf = {row['p_bonferroni']:.6f}")

        if row['p_bonferroni'] < 0.05:
            log("\n[CONCLUSION] ✓ ROBUST: Three-way interaction remains significant with log_TSVR")
            log("  The dissociation finding is NOT sensitive to time scale choice.")
        else:
            log("\n[CONCLUSION] ✗ SENSITIVE: Three-way interaction becomes non-significant with log_TSVR")
            log("  The dissociation finding IS sensitive to time scale choice.")
    else:
        log("[ERROR] Could not find three-way interaction term in model results")

    # Save full results
    output_path = RQ_DIR / "data" / "sensitivity_log_tsvr_results.csv"
    df_fixed.to_csv(output_path, index=False, encoding='utf-8')
    log(f"\n[SAVE] {output_path.name} ({len(df_fixed)} terms)")

    # Save comparison summary
    summary_path = RQ_DIR / "results" / "sensitivity_log_tsvr_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SENSITIVITY ANALYSIS: log_TSVR TIME SCALE\n")
        f.write("="*80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Verify three-way interaction (measure × location × time) is robust to\n")
        f.write("time scale choice (TSVR_hours vs log_TSVR).\n\n")
        f.write("MOTIVATION:\n")
        f.write("- Ch5 5.5.1 (accuracy): Uses TSVR_hours (linear time)\n")
        f.write("- Ch6 6.8.1 (confidence): Uses log_TSVR (logarithmic time)\n")
        f.write("- RQ 6.9.5 Step 3 original: Uses TSVR_hours\n\n")
        f.write("RESULTS:\n")
        f.write("-"*80 + "\n")
        if len(three_way) > 0:
            row = three_way.iloc[0]
            f.write(f"Original (TSVR_hours):  p_bonf < 0.001 (highly significant)\n")
            f.write(f"Sensitivity (log_TSVR): p_bonf = {row['p_bonferroni']:.6f}\n\n")
            if row['p_bonferroni'] < 0.05:
                f.write("CONCLUSION: ✓ ROBUST\n")
                f.write("The dissociation finding is NOT sensitive to time scale choice.\n")
                f.write("Both linear and logarithmic time scales detect the three-way interaction.\n\n")
                f.write("IMPLICATION FOR THESIS:\n")
                f.write("The metacognitive-accuracy dissociation is a robust finding that holds\n")
                f.write("across different time parameterizations. The choice of TSVR_hours in the\n")
                f.write("primary analysis does not bias the conclusion.\n")
            else:
                f.write("CONCLUSION: ✗ SENSITIVE\n")
                f.write("The dissociation finding IS sensitive to time scale choice.\n")
                f.write("Linear time detects interaction, logarithmic time does not.\n\n")
                f.write("IMPLICATION FOR THESIS:\n")
                f.write("The dissociation may depend on functional form assumptions. Consider:\n")
                f.write("1. Reporting both analyses in main text\n")
                f.write("2. Discussing theoretical reasons for time scale choice\n")
                f.write("3. Testing additional functional forms (polynomial, power law)\n")
        else:
            f.write("ERROR: Could not extract three-way interaction results.\n")
        f.write("\n" + "="*80 + "\n")

    log(f"[SAVE] {summary_path.name}")
    log("[SUCCESS] Sensitivity analysis complete")

except Exception as e:
    log(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    sys.exit(1)
