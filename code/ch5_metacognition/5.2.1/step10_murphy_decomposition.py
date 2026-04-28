#!/usr/bin/env python3
"""
RQ 6.2.1 - Step 10: Murphy (1973) Brier Score Decomposition
============================================================

Decomposes the Brier score into three additive components:

    Brier = Reliability - Resolution + Uncertainty

Where:
  - Reliability: weighted mean of (mean_conf_k - mean_acc_k)^2 across bins k
    → Lower is better. Measures how close predicted probabilities match
      observed frequencies within each confidence bin.
  - Resolution: weighted mean of (mean_acc_k - overall_acc)^2 across bins k
    → Higher is better. Measures how well confidence discriminates between
      correct and incorrect responses.
  - Uncertainty: overall_acc * (1 - overall_acc)
    → Base rate variance. Not under participant control.

Confidence bins are the 5 discrete Likert levels (0, 0.25, 0.5, 0.75, 1.0),
matching the ECE computation in step04.

Reference: Murphy, A. H. (1973). A new vector partition of the probability
score. Journal of Applied Meteorology, 12(4), 595-600.

Date: 2026-04-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.1
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root
LOG_FILE = RQ_DIR / "logs" / "step10_murphy_decomposition.log"
DFDATA_FILE = PROJECT_ROOT / "data/cache/dfData.csv"
BRIER_FILE = RQ_DIR / "data" / "step03_brier_scores.csv"

TSVR_FILE = RQ_DIR / "data" / "step00c_tsvr_mapping.csv"

OUTPUT_FILE = RQ_DIR / "data" / "step10_murphy_decomposition.csv"
OUTPUT_DETAIL_FILE = RQ_DIR / "data" / "step10_murphy_bin_detail.csv"

EXPECTED_TESTS = 4
EXPECTED_PARTICIPANTS = 100


def log(msg):
    """Log message to file and console."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def get_item_level_pairs(df, test_val, interactive_items, tq_items, tc_items):
    """Extract all confidence-accuracy pairs for a given test session."""
    df_test = df[df['TEST'] == test_val]
    pairs = []

    for _, row in df_test.iterrows():
        for item in interactive_items:
            tq_col = tq_items[item]
            tc_col = tc_items[item]
            accuracy = row[tq_col]
            confidence = row[tc_col]

            if pd.notna(accuracy) and pd.notna(confidence):
                pairs.append({
                    'UID': row['UID'],
                    'accuracy': 1.0 if float(accuracy) == 1.0 else 0.0,
                    'confidence': float(confidence),
                })

    return pd.DataFrame(pairs)


def murphy_decomposition(df_pairs):
    """
    Compute Murphy (1973) Brier decomposition for a set of confidence-accuracy pairs.

    Parameters
    ----------
    df_pairs : DataFrame with columns 'confidence' and 'accuracy'

    Returns
    -------
    dict with keys: reliability, resolution, uncertainty, brier_reconstructed,
                    brier_direct, n_items, overall_accuracy, bin_details
    """
    N = len(df_pairs)
    overall_acc = df_pairs['accuracy'].mean()

    # Uncertainty = base rate variance (binary outcomes after recoding)
    uncertainty = overall_acc * (1 - overall_acc)

    # Brier score computed directly
    brier_direct = ((df_pairs['confidence'] - df_pairs['accuracy']) ** 2).mean()

    # Bin by discrete confidence levels
    # The 5-point scale maps to: 0, 0.25, 0.5, 0.75, 1.0
    # Round to nearest to handle any floating point issues
    df_pairs = df_pairs.copy()
    df_pairs['conf_bin'] = df_pairs['confidence'].round(2)

    reliability = 0.0
    resolution = 0.0
    bin_details = []

    for conf_val, bin_data in df_pairs.groupby('conf_bin'):
        n_k = len(bin_data)
        w_k = n_k / N  # weight = proportion of items in this bin

        mean_acc_k = bin_data['accuracy'].mean()
        mean_conf_k = bin_data['confidence'].mean()

        rel_k = (mean_conf_k - mean_acc_k) ** 2
        res_k = (mean_acc_k - overall_acc) ** 2

        reliability += w_k * rel_k
        resolution += w_k * res_k

        bin_details.append({
            'conf_bin': conf_val,
            'n_items': n_k,
            'weight': w_k,
            'mean_confidence': mean_conf_k,
            'mean_accuracy': mean_acc_k,
            'reliability_k': rel_k,
            'resolution_k': res_k,
        })

    brier_reconstructed = reliability - resolution + uncertainty

    return {
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'brier_reconstructed': brier_reconstructed,
        'brier_direct': brier_direct,
        'n_items': N,
        'overall_accuracy': overall_acc,
        'bin_details': bin_details,
    }


def main():
    # Clear log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 70)
    log("STEP 10: Murphy (1973) Brier Score Decomposition")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # Load item-level data (same source as step03 and step04)
    # -------------------------------------------------------------------------
    if not DFDATA_FILE.exists():
        raise FileNotFoundError(f"dfData.csv not found: {DFDATA_FILE}")

    df = pd.read_csv(DFDATA_FILE)
    log(f"Loaded dfData: {len(df)} rows")

    # Match TQ/TC item pairs
    tq_cols = [c for c in df.columns if c.startswith('TQ_')]
    tc_cols = [c for c in df.columns if c.startswith('TC_')]
    tq_items = {c.replace('TQ_', ''): c for c in tq_cols}
    tc_items = {c.replace('TC_', ''): c for c in tc_cols}
    matched_items = set(tq_items.keys()) & set(tc_items.keys())
    log(f"Matched TQ/TC item pairs: {len(matched_items)}")

    # Filter to interactive paradigms (same filter as step03)
    interactive_items = [item for item in matched_items
                         if any(tag in item for tag in ['-N-', '-L-', '-U-', '-D-', '-O-'])]
    log(f"Interactive paradigm items: {len(interactive_items)}")

    if len(interactive_items) == 0:
        log("WARNING: No interactive items found, using all matched items")
        interactive_items = list(matched_items)

    # Normalize confidence if needed (check scale)
    sample_tc = df[[tc_items[item] for item in interactive_items[:5]]].values.flatten()
    sample_tc = sample_tc[~np.isnan(sample_tc)]
    if sample_tc.max() > 1:
        log(f"Confidence scale detected: 1-5 (max={sample_tc.max():.1f}), will normalize to 0-1")
        needs_normalization = True
    else:
        log(f"Confidence scale detected: 0-1 (max={sample_tc.max():.2f})")
        needs_normalization = False

    # -------------------------------------------------------------------------
    # Compute decomposition per test session
    # -------------------------------------------------------------------------
    test_values = sorted(df['TEST'].unique())
    log(f"\nTest sessions: {test_values}")

    session_results = []
    all_bin_details = []

    for test_val in test_values:
        test_label = f'T{int(test_val)}'
        log(f"\n--- {test_label} ---")

        df_pairs = get_item_level_pairs(df, test_val, interactive_items, tq_items, tc_items)
        log(f"  Item-level pairs: {len(df_pairs)}")

        if needs_normalization:
            df_pairs['confidence'] = (df_pairs['confidence'] - 1) / 4

        result = murphy_decomposition(df_pairs)

        log(f"  Overall accuracy:    {result['overall_accuracy']:.4f}")
        log(f"  Uncertainty:         {result['uncertainty']:.6f}")
        log(f"  Reliability:         {result['reliability']:.6f}")
        log(f"  Resolution:          {result['resolution']:.6f}")
        log(f"  Brier (direct):      {result['brier_direct']:.6f}")
        log(f"  Brier (reconstructed): {result['brier_reconstructed']:.6f}")

        # Validate: reconstructed should match direct
        discrepancy = abs(result['brier_reconstructed'] - result['brier_direct'])
        log(f"  Reconstruction error: {discrepancy:.2e}")
        if discrepancy > 1e-6:
            log(f"  WARNING: Reconstruction discrepancy > 1e-6!")
        else:
            log(f"  VALIDATION - PASS: Brier = Reliability - Resolution + Uncertainty")

        session_results.append({
            'test': test_label,
            'reliability': result['reliability'],
            'resolution': result['resolution'],
            'uncertainty': result['uncertainty'],
            'brier_direct': result['brier_direct'],
            'brier_reconstructed': result['brier_reconstructed'],
            'reconstruction_error': discrepancy,
            'overall_accuracy': result['overall_accuracy'],
            'n_items': result['n_items'],
        })

        for bd in result['bin_details']:
            bd['test'] = test_label
            all_bin_details.append(bd)

    # -------------------------------------------------------------------------
    # Cross-validate against step03 Brier scores
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("CROSS-VALIDATION: Step 10 vs Step 03 Brier Scores")
    log("=" * 70)

    if BRIER_FILE.exists():
        df_brier_step03 = pd.read_csv(BRIER_FILE)
        step03_means = df_brier_step03.groupby('TEST')['brier_score'].mean()

        for row in session_results:
            test_label = row['test']
            if test_label in step03_means.index:
                step03_val = step03_means[test_label]
                step10_val = row['brier_direct']
                diff = abs(step03_val - step10_val)
                log(f"  {test_label}: step03={step03_val:.6f}, step10={step10_val:.6f}, diff={diff:.2e}")
                if diff > 0.001:
                    log(f"  WARNING: Discrepancy > 0.001 for {test_label}")
            else:
                log(f"  {test_label}: not found in step03 output")
    else:
        log("  WARNING: step03_brier_scores.csv not found, skipping cross-validation")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(session_results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    log(f"\nSaved: {OUTPUT_FILE}")

    df_bin = pd.DataFrame(all_bin_details)
    df_bin.to_csv(OUTPUT_DETAIL_FILE, index=False)
    log(f"Saved: {OUTPUT_DETAIL_FILE}")

    # -------------------------------------------------------------------------
    # Brier and ECE LMMs (previously computed ad-hoc, now traceable)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("BRIER & ECE LINEAR MIXED MODELS")
    log("=" * 70)

    # Load TSVR mapping for continuous time variable
    df_tsvr = pd.read_csv(TSVR_FILE)

    # --- Brier LMM ---
    df_brier = pd.read_csv(BRIER_FILE)
    df_brier = df_brier.merge(df_tsvr[['composite_ID', 'TSVR_hours']], on='composite_ID')

    log("\n--- Brier Score LMM ---")
    log(f"  Model: brier_score ~ TSVR_hours + (TSVR_hours | UID)")
    log(f"  N = {len(df_brier)} observations, {df_brier['UID'].nunique()} participants")

    brier_lmm = smf.mixedlm(
        "brier_score ~ TSVR_hours", df_brier, groups=df_brier["UID"],
        re_formula="~TSVR_hours"
    ).fit(reml=False)

    log(f"\n  Fixed effects:")
    for name in brier_lmm.fe_params.index:
        coef = brier_lmm.fe_params[name]
        se = brier_lmm.bse_fe[name]
        z = brier_lmm.tvalues[name]
        p = brier_lmm.pvalues[name]
        ci = brier_lmm.conf_int().loc[name]
        log(f"    {name}: β = {coef:.6f}, SE = {se:.6f}, z = {z:.2f}, "
            f"p = {p:.4f}, CI95 [{ci[0]:.5f}, {ci[1]:.5f}]")

    # Per-day coefficient for thesis reporting
    coef_day = brier_lmm.fe_params['TSVR_hours'] * 24
    se_day = brier_lmm.bse_fe['TSVR_hours'] * 24
    log(f"\n  Per-day effect: β = {coef_day:.6f}, SE = {se_day:.6f}")

    # LRT vs null
    brier_null = smf.mixedlm(
        "brier_score ~ 1", df_brier, groups=df_brier["UID"],
        re_formula="~TSVR_hours"
    ).fit(reml=False)
    lrt_stat = -2 * (brier_null.llf - brier_lmm.llf)
    lrt_p = stats.chi2.sf(lrt_stat, df=1)
    log(f"  LRT: χ² = {lrt_stat:.4f}, p = {lrt_p:.6f}")

    # --- ECE LMM (per-participant) ---
    log("\n--- ECE LMM (per-participant) ---")

    # Compute per-participant ECE with binary accuracy
    ece_records = []
    bins_ece = [-0.001, 0.125, 0.375, 0.625, 0.875, 1.001]
    bin_labels_ece = ['bin_0', 'bin_025', 'bin_05', 'bin_075', 'bin_1']

    df_raw = pd.read_csv(DFDATA_FILE)
    tq_cols_raw = [c for c in df_raw.columns if c.startswith('TQ_')]
    tc_cols_raw = [c for c in df_raw.columns if c.startswith('TC_')]
    tq_items_raw = {c.replace('TQ_', ''): c for c in tq_cols_raw}
    tc_items_raw = {c.replace('TC_', ''): c for c in tc_cols_raw}
    matched_raw = set(tq_items_raw.keys()) & set(tc_items_raw.keys())
    interactive_raw = [i for i in matched_raw
                       if any(t in i for t in ['-N-', '-L-', '-U-', '-D-', '-O-'])]

    for _, row in df_raw.iterrows():
        pairs = []
        for item in interactive_raw:
            acc_raw = row[tq_items_raw[item]]
            conf = row[tc_items_raw[item]]
            if pd.notna(acc_raw) and pd.notna(conf):
                acc = 1.0 if float(acc_raw) == 1.0 else 0.0
                pairs.append((float(conf), acc))
        if not pairs:
            continue

        df_p = pd.DataFrame(pairs, columns=['confidence', 'accuracy'])
        df_p['conf_bin'] = pd.cut(df_p['confidence'], bins=bins_ece, labels=bin_labels_ece)

        total, weighted_err = 0, 0.0
        for bl in bin_labels_ece:
            bd = df_p[df_p['conf_bin'] == bl]
            if len(bd) > 0:
                weighted_err += abs(bd['confidence'].mean() - bd['accuracy'].mean()) * len(bd)
                total += len(bd)

        if total > 0:
            uid = row['UID']
            test_val_raw = row['TEST']
            comp_id = f"{uid}_T{int(test_val_raw)}"
            ece_records.append({
                'UID': uid,
                'TEST': f"T{int(test_val_raw)}",
                'composite_ID': comp_id,
                'ECE': weighted_err / total,
            })

    df_ece = pd.DataFrame(ece_records)
    df_ece = df_ece.merge(df_tsvr[['composite_ID', 'TSVR_hours']], on='composite_ID')

    log(f"  Model: ECE ~ TSVR_hours + (TSVR_hours | UID)")
    log(f"  N = {len(df_ece)} observations, {df_ece['UID'].nunique()} participants")

    ece_lmm = smf.mixedlm(
        "ECE ~ TSVR_hours", df_ece, groups=df_ece["UID"],
        re_formula="~TSVR_hours"
    ).fit(reml=False)

    log(f"\n  Fixed effects:")
    for name in ece_lmm.fe_params.index:
        coef = ece_lmm.fe_params[name]
        se = ece_lmm.bse_fe[name]
        z = ece_lmm.tvalues[name]
        p = ece_lmm.pvalues[name]
        ci = ece_lmm.conf_int().loc[name]
        log(f"    {name}: β = {coef:.6f}, SE = {se:.6f}, z = {z:.2f}, "
            f"p = {p:.4f}, CI95 [{ci[0]:.5f}, {ci[1]:.5f}]")

    coef_day_ece = ece_lmm.fe_params['TSVR_hours'] * 24
    se_day_ece = ece_lmm.bse_fe['TSVR_hours'] * 24
    log(f"\n  Per-day effect: β = {coef_day_ece:.6f}, SE = {se_day_ece:.6f}")

    ece_null = smf.mixedlm(
        "ECE ~ 1", df_ece, groups=df_ece["UID"],
        re_formula="~TSVR_hours"
    ).fit(reml=False)
    lrt_stat_ece = -2 * (ece_null.llf - ece_lmm.llf)
    lrt_p_ece = stats.chi2.sf(lrt_stat_ece, df=1)
    log(f"  LRT: χ² = {lrt_stat_ece:.4f}, p = {lrt_p_ece:.6f}")

    # Descriptive means for thesis
    log("\n--- Descriptive means ---")
    for tl in ['T1', 'T2', 'T3', 'T4']:
        b_mean = df_brier[df_brier['TEST'] == tl]['brier_score'].mean()
        e_mean = df_ece[df_ece['TEST'] == tl]['ECE'].mean()
        log(f"  {tl}: Brier = {b_mean:.3f}, ECE = {e_mean:.3f}")

    # -------------------------------------------------------------------------
    # Summary table for thesis
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY TABLE (for thesis reporting)")
    log("=" * 70)
    log(f"\n{'Test':<8} {'Reliability':<14} {'Resolution':<14} {'Uncertainty':<14} {'Brier':<10}")
    log("-" * 60)
    for _, row in df_results.iterrows():
        log(f"{row['test']:<8} {row['reliability']:<14.6f} {row['resolution']:<14.6f} "
            f"{row['uncertainty']:<14.6f} {row['brier_direct']:<10.6f}")

    t1 = df_results[df_results['test'] == 'T1'].iloc[0]
    t4 = df_results[df_results['test'] == 'T4'].iloc[0]
    log(f"\nDay 0 → Day 6 changes:")
    log(f"  Reliability: {t1['reliability']:.6f} → {t4['reliability']:.6f} "
        f"(Δ = {t4['reliability'] - t1['reliability']:+.6f})")
    log(f"  Resolution:  {t1['resolution']:.6f} → {t4['resolution']:.6f} "
        f"(Δ = {t4['resolution'] - t1['resolution']:+.6f})")
    log(f"  Uncertainty: {t1['uncertainty']:.6f} → {t4['uncertainty']:.6f} "
        f"(Δ = {t4['uncertainty'] - t1['uncertainty']:+.6f})")

    log("\nDone.")


if __name__ == '__main__':
    main()
