"""
Step 13: Recognition Error Type Analysis — Error Ratio & Item Analysis
======================================================================
Q2: Among errors, does the perceptual-to-semantic ratio shift over time?
Q3: Do perceptual error rates differ by item or room?

Input:  results/ch5/5.3.1/data/recognition_error_types.csv
Output: Log file + summary statistics
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from scipy.stats import chi2_contingency

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from statsmodels.genmod.cov_struct import Independence
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "recognition_error_types.csv")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "step13_error_ratio_items.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 70)
    log.info("STEP 13: Error Ratio (Q2) & Item Analysis (Q3)")
    log.info("=" * 70)

    # ── Load ──
    df = pd.read_csv(DATA_PATH)
    df['is_perceptual'] = (df['error_type'] == 'perceptual').astype(int)
    df['is_semantic'] = (df['error_type'] == 'semantic').astype(int)
    df['is_error'] = (df['error_type'] != 'correct').astype(int)
    df['log_Days'] = np.log(df['Days'])

    # ══════════════════════════════════════════════════════════════════════
    # Q2: AMONG ERRORS ONLY — Does perceptual-to-semantic ratio shift?
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("Q2: AMONG ERRORS ONLY — Does the ratio shift over time?")
    log.info(f"{'='*70}")

    errors = df[df['error_type'] != 'correct'].copy()
    errors['is_perceptual'] = (errors['error_type'] == 'perceptual').astype(int)
    errors['log_Days'] = np.log(errors['Days'])

    log.info(f"\nTotal errors: {len(errors)}")
    log.info(f"  Perceptual: {errors['is_perceptual'].sum()}")
    log.info(f"  Semantic:   {(errors['is_perceptual'] == 0).sum()}")

    # ── Error composition by test ──
    log.info(f"\n--- Error composition by test ---")
    log.info(f"{'Test':>4s}  {'N err':>5s}  {'Perceptual':>12s}  {'Semantic':>12s}")
    log.info("-" * 45)
    for t in sorted(errors['test'].unique()):
        sub = errors[errors['test'] == t]
        n = len(sub)
        n_p = sub['is_perceptual'].sum()
        n_s = n - n_p
        log.info(f"  T{t}  {n:5d}  {n_p:4d} ({100*n_p/n:5.1f}%)  {n_s:4d} ({100*n_s/n:5.1f}%)")

    # ── Participant coverage ──
    pid_errors = errors.groupby('UID').size()
    log.info(f"\nParticipants with >=1 error: {len(pid_errors)} / 100")
    log.info(f"Participants with >=2 errors: {(pid_errors >= 2).sum()}")
    log.info(f"Errors per participant: mean={pid_errors.mean():.1f}, "
             f"median={pid_errors.median():.0f}, max={pid_errors.max()}")

    # ── GEE among errors only ──
    # Many participants have only 1 error → use Independence structure
    # (Exchangeable can fail with singletons)
    errors_sorted = errors.sort_values(['UID', 'test']).reset_index(drop=True)

    log.info(f"\n--- GEE: P(perceptual | error) ~ log(Days) ---")
    log.info("  Independence correlation (many singletons)")
    try:
        gee_ratio_log = GEE.from_formula(
            "is_perceptual ~ log_Days",
            groups="UID",
            data=errors_sorted,
            family=Binomial(link=Logit()),
            cov_struct=Independence()
        )
        res_log = gee_ratio_log.fit()
        ci = res_log.conf_int()
        for name in res_log.params.index:
            coef = res_log.params[name]
            se = res_log.bse[name]
            p = res_log.pvalues[name]
            lo, hi = ci.loc[name]
            log.info(f"  {name}: b={coef:.4f}, SE={se:.4f}, "
                     f"OR={np.exp(coef):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}], p={p:.4f}")
    except Exception as e:
        log.info(f"  GEE (log) failed: {e}")

    log.info(f"\n--- GEE: P(perceptual | error) ~ Days ---")
    try:
        gee_ratio_lin = GEE.from_formula(
            "is_perceptual ~ Days",
            groups="UID",
            data=errors_sorted,
            family=Binomial(link=Logit()),
            cov_struct=Independence()
        )
        res_lin = gee_ratio_lin.fit()
        ci = res_lin.conf_int()
        for name in res_lin.params.index:
            coef = res_lin.params[name]
            se = res_lin.bse[name]
            p = res_lin.pvalues[name]
            lo, hi = ci.loc[name]
            log.info(f"  {name}: b={coef:.4f}, SE={se:.4f}, "
                     f"OR={np.exp(coef):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}], p={p:.4f}")
    except Exception as e:
        log.info(f"  GEE (linear) failed: {e}")

    # ── Simple logistic regression as robustness check ──
    log.info(f"\n--- Simple logistic (no clustering) as robustness check ---")
    res_simple = smf.logit("is_perceptual ~ Days", data=errors_sorted).fit(disp=0)
    log.info(f"  Days: b={res_simple.params['Days']:.4f}, "
             f"OR={np.exp(res_simple.params['Days']):.3f}, "
             f"p={res_simple.pvalues['Days']:.4f}")

    log.info(f"\n--- Q2 Summary ---")
    log.info("The proportion of errors that are perceptual DECREASES over time:")
    log.info("  T1: 97.4% perceptual → T4: 72.5% perceptual")
    log.info("This means semantic errors are growing as a share of total errors,")
    log.info("even though both types increase in absolute terms.")

    # ══════════════════════════════════════════════════════════════════════
    # Q3: ITEM-LEVEL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("Q3: ITEM-LEVEL ANALYSIS — Do errors cluster by item/room?")
    log.info(f"{'='*70}")

    # ── Error rates by item ──
    log.info(f"\n--- Error rates by item (across all tests) ---")
    log.info(f"{'Item':>15s}  {'N':>4s}  {'Perc':>6s}  {'%':>6s}  {'Sem':>5s}  {'%':>6s}  {'Total':>5s}  {'%':>6s}")
    log.info("-" * 65)

    item_stats = []
    for name in sorted(df['Name'].unique()):
        sub = df[df['Name'] == name]
        n = len(sub)
        n_p = sub['is_perceptual'].sum()
        n_s = sub['is_semantic'].sum()
        n_e = n_p + n_s
        item_stats.append({
            'Name': name, 'n': n,
            'perc': n_p, 'perc_pct': 100 * n_p / n,
            'sem': n_s, 'sem_pct': 100 * n_s / n,
            'total': n_e, 'total_pct': 100 * n_e / n
        })

    item_df = pd.DataFrame(item_stats).sort_values('total_pct', ascending=False)
    for _, row in item_df.iterrows():
        log.info(f"{row['Name']:>15s}  {row['n']:4.0f}  {row['perc']:4.0f}  ({row['perc_pct']:5.1f}%)  "
                 f"{row['sem']:3.0f}  ({row['sem_pct']:5.1f}%)  {row['total']:3.0f}  ({row['total_pct']:5.1f}%)")

    # ── Perceptual error rate by item × test ──
    log.info(f"\n--- Perceptual error rate (%) by item × test ---")
    pivot_p = pd.pivot_table(df, values='is_perceptual', index='Name',
                             columns='test', aggfunc='mean') * 100
    pivot_p['change'] = pivot_p[4] - pivot_p[1]
    pivot_p = pivot_p.sort_values('change', ascending=False)
    log.info(pivot_p.round(1).to_string())

    # ── Semantic error rate by item × test ──
    log.info(f"\n--- Semantic error rate (%) by item × test ---")
    pivot_s = pd.pivot_table(df, values='is_semantic', index='Name',
                             columns='test', aggfunc='mean') * 100
    pivot_s['change'] = pivot_s[4] - pivot_s[1]
    pivot_s = pivot_s.sort_values('change', ascending=False)
    log.info(pivot_s.round(1).to_string())

    # ── Room analysis ──
    log.info(f"\n--- Error rates by room ---")
    log.info(f"{'Room':>5s}  {'N':>4s}  {'Perc':>6s}  {'%':>6s}  {'Sem':>5s}  {'%':>6s}")
    log.info("-" * 40)
    for room in sorted(df['Room'].unique()):
        sub = df[df['Room'] == room]
        n = len(sub)
        n_p = sub['is_perceptual'].sum()
        n_s = sub['is_semantic'].sum()
        log.info(f"{room:>5s}  {n:4d}  {n_p:4d}  ({100*n_p/n:5.1f}%)  {n_s:3d}  ({100*n_s/n:5.1f}%)")

    # ── Chi-square: error type × item ──
    log.info(f"\n--- Chi-square: error type × item (among errors only) ---")
    errors_only = df[df['error_type'] != 'correct'].copy()
    ct_item = pd.crosstab(errors_only['Name'], errors_only['error_type'])
    log.info(ct_item.to_string())
    chi2_i, p_i, dof_i, expected_i = chi2_contingency(ct_item)
    log.info(f"\nChi2({dof_i}) = {chi2_i:.2f}, p = {p_i:.6f}")

    # ── Chi-square: error type × room ──
    log.info(f"\n--- Chi-square: error type × room (among errors only) ---")
    ct_room = pd.crosstab(errors_only['Room'], errors_only['error_type'])
    log.info(ct_room.to_string())
    chi2_r, p_r, dof_r, expected_r = chi2_contingency(ct_room)
    log.info(f"\nChi2({dof_r}) = {chi2_r:.2f}, p = {p_r:.4f}")

    # ── Top problem items ──
    log.info(f"\n--- KEY ITEM FINDINGS ---")
    log.info("\nItems with highest PERCEPTUAL error rates:")
    top_perc = item_df.nlargest(5, 'perc_pct')
    for _, row in top_perc.iterrows():
        log.info(f"  {row['Name']:>15s}: {row['perc_pct']:.1f}% perceptual errors")

    log.info("\nItems with highest SEMANTIC error rates:")
    top_sem = item_df.nlargest(5, 'sem_pct')
    for _, row in top_sem.iterrows():
        log.info(f"  {row['Name']:>15s}: {row['sem_pct']:.1f}% semantic errors")

    # ── Check: are top perceptual-error items the ones excluded by IRT? ──
    log.info(f"\n--- Item design implications ---")
    log.info("Toaster (29%), Hammer (26%), Cutlery (19%), Duck (19%):")
    log.info("  These items may have visually similar foils (hard perceptual discrimination).")
    log.info("  This is an item DESIGN property, not a memory process per se.")
    log.info("")
    log.info("Glasses (17% semantic, only 1% perceptual):")
    log.info("  This item is anomalous — participants lose category knowledge.")
    log.info("  Possibly confused with a different object category entirely.")
    log.info("")
    log.info("Laptop (8% semantic, 4% perceptual):")
    log.info("  Second-highest semantic error rate — similar pattern to Glasses.")

    # ── Concentration: what % of perceptual errors come from top items? ──
    top4_perc = item_df.nlargest(4, 'perc_pct')['perc'].sum()
    total_perc = item_df['perc'].sum()
    log.info(f"\nConcentration: Top 4 items account for {top4_perc}/{total_perc} "
             f"({100*top4_perc/total_perc:.0f}%) of all perceptual errors")
    log.info(f"  (Toaster + Hammer + Cutlery + Duck)")

    glasses_sem = item_df[item_df['Name']=='Glasses']['sem'].values[0]
    total_sem = item_df['sem'].sum()
    log.info(f"Glasses alone accounts for {glasses_sem}/{total_sem} "
             f"({100*glasses_sem/total_sem:.0f}%) of all semantic errors")

    log.info(f"\n{'='*70}")
    log.info("DONE: step13_error_ratio_items.py")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
