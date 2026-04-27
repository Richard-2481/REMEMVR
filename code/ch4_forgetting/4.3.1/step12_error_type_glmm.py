"""
Step 12: Recognition Error Type Analysis — Binomial GEE Models
===============================================================
Q1: Do perceptual and semantic error rates follow different trajectories?

Models run:
  A. P(perceptual error) ~ log(Days) + (1|UID)  [GEE, exchangeable]
  B. P(semantic error)   ~ log(Days) + (1|UID)  [GEE, exchangeable]
  C. P(any error)        ~ log(Days) + (1|UID)  [GEE, exchangeable]

ALSO runs with linear Days (not log) to check whether scale matters:
  D. P(perceptual error) ~ Days + (1|UID)
  E. P(semantic error)   ~ Days + (1|UID)
  F. P(any error)        ~ Days + (1|UID)

ALSO runs a JOINT model to directly test the interaction:
  G. P(error) ~ log(Days) * error_type + (1|UID)   [on errors + correct, stacked]
  H. P(error) ~ Days * error_type + (1|UID)

This allows us to say whether the two error types have DIFFERENT slopes,
not just whether each slope is individually significant.

Input:  results/ch5/5.3.1/data/recognition_error_types.csv
Output: Log file + summary statistics
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from statsmodels.genmod.cov_struct import Exchangeable
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "recognition_error_types.csv")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "step12_error_type_glmm.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def run_gee(formula, data, label):
    """Run a binomial GEE and print results."""
    log.info(f"\n--- {label} ---")
    log.info(f"Formula: {formula}")
    log.info(f"N observations: {len(data)}")
    log.info(f"N clusters (UID): {data['UID'].nunique()}")

    # Outcome variable
    outcome_var = formula.split("~")[0].strip()
    n_events = data[outcome_var].sum()
    n_total = len(data)
    log.info(f"Events: {n_events}/{n_total} ({100*n_events/n_total:.2f}%)")

    try:
        model = GEE.from_formula(
            formula,
            groups="UID",
            data=data,
            family=Binomial(link=Logit()),
            cov_struct=Exchangeable()
        )
        result = model.fit()

        log.info(f"\nCoefficients:")
        log.info(f"{'Parameter':>20s}  {'Coef':>8s}  {'SE':>8s}  {'z':>8s}  {'p':>8s}  "
                 f"{'OR':>8s}  {'OR_lo':>8s}  {'OR_hi':>8s}")
        log.info("-" * 90)

        ci = result.conf_int()
        for name in result.params.index:
            coef = result.params[name]
            se = result.bse[name]
            z = result.tvalues[name]
            p = result.pvalues[name]
            lo, hi = ci.loc[name]
            or_val = np.exp(coef)
            or_lo = np.exp(lo)
            or_hi = np.exp(hi)
            log.info(f"{name:>20s}  {coef:8.4f}  {se:8.4f}  {z:8.3f}  {p:8.4f}  "
                     f"{or_val:8.3f}  {or_lo:8.3f}  {or_hi:8.3f}")

        return result
    except Exception as e:
        log.info(f"  FAILED: {e}")
        return None


def main():
    log.info("=" * 70)
    log.info("STEP 12: Recognition Error Type — Binomial GEE Models")
    log.info("=" * 70)

    # ── Load and prepare ──
    df = pd.read_csv(DATA_PATH)
    df['is_perceptual'] = (df['error_type'] == 'perceptual').astype(int)
    df['is_semantic'] = (df['error_type'] == 'semantic').astype(int)
    df['is_error'] = (df['error_type'] != 'correct').astype(int)
    df['log_Days'] = np.log(df['Days'])

    # Sort by UID for GEE
    df = df.sort_values(['UID', 'test', 'Item']).reset_index(drop=True)

    log.info(f"\nData: {len(df)} rows, {df['UID'].nunique()} participants")
    log.info(f"Days range: {df['Days'].min():.3f} to {df['Days'].max():.3f}")
    log.info(f"log(Days) range: {df['log_Days'].min():.3f} to {df['log_Days'].max():.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: Separate models with log(Days)
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("SECTION 1: SEPARATE MODELS — log(Days) as time predictor")
    log.info(f"{'='*70}")
    log.info("\nNOTE: log(Days) captures PROPORTIONAL change.")
    log.info("  Semantic: 0.2% → 3.7% = 18.5× increase (huge on log scale)")
    log.info("  Perceptual: 6.2% → 9.7% = 1.6× increase (modest on log scale)")
    log.info("  Both increase ~3.5 percentage points in ABSOLUTE terms.")

    res_perc_log = run_gee("is_perceptual ~ log_Days", df, "A. P(perceptual) ~ log(Days)")
    res_sem_log = run_gee("is_semantic ~ log_Days", df, "B. P(semantic) ~ log(Days)")
    res_any_log = run_gee("is_error ~ log_Days", df, "C. P(any error) ~ log(Days)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: Separate models with linear Days
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("SECTION 2: SEPARATE MODELS — linear Days as time predictor")
    log.info(f"{'='*70}")
    log.info("\nNOTE: Linear Days captures ABSOLUTE change per day.")
    log.info("  This should better reflect what the plot shows.")

    res_perc_lin = run_gee("is_perceptual ~ Days", df, "D. P(perceptual) ~ Days")
    res_sem_lin = run_gee("is_semantic ~ Days", df, "E. P(semantic) ~ Days")
    res_any_lin = run_gee("is_error ~ Days", df, "F. P(any error) ~ Days")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: Joint stacked model — direct interaction test
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("SECTION 3: JOINT STACKED MODEL — Direct interaction test")
    log.info(f"{'='*70}")
    log.info("\nTo directly test whether the two error types have DIFFERENT slopes,")
    log.info("we create a stacked/long dataset with one row per observation per error type,")
    log.info("and test the error_type × time interaction.")

    # Create stacked dataset: each original row appears twice,
    # once for "did a perceptual error occur?" and once for "did a semantic error occur?"
    df_perc = df[['UID', 'test', 'Days', 'log_Days', 'is_perceptual']].copy()
    df_perc.rename(columns={'is_perceptual': 'occurred'}, inplace=True)
    df_perc['etype'] = 'perceptual'

    df_sem = df[['UID', 'test', 'Days', 'log_Days', 'is_semantic']].copy()
    df_sem.rename(columns={'is_semantic': 'occurred'}, inplace=True)
    df_sem['etype'] = 'semantic'

    df_stacked = pd.concat([df_perc, df_sem], ignore_index=True)
    df_stacked = df_stacked.sort_values(['UID', 'etype', 'test']).reset_index(drop=True)

    log.info(f"\nStacked dataset: {len(df_stacked)} rows (2 × {len(df)})")
    log.info(f"  Perceptual events: {df_stacked[df_stacked['etype']=='perceptual']['occurred'].sum()}")
    log.info(f"  Semantic events:   {df_stacked[df_stacked['etype']=='semantic']['occurred'].sum()}")

    # Treatment coding: semantic as reference → interaction tells us if perceptual slope differs
    run_gee(
        "occurred ~ log_Days * C(etype, Treatment(reference='semantic'))",
        df_stacked,
        "G. Stacked: occurred ~ log(Days) × error_type"
    )

    run_gee(
        "occurred ~ Days * C(etype, Treatment(reference='semantic'))",
        df_stacked,
        "H. Stacked: occurred ~ Days × error_type"
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: Summary comparison
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("SECTION 4: SUMMARY COMPARISON")
    log.info(f"{'='*70}")

    log.info("\n--- log(Days) models (proportional change) ---")
    if res_perc_log and res_sem_log:
        b_p = res_perc_log.params['log_Days']
        p_p = res_perc_log.pvalues['log_Days']
        b_s = res_sem_log.params['log_Days']
        p_s = res_sem_log.pvalues['log_Days']
        log.info(f"  Perceptual: b = {b_p:.4f}, p = {p_p:.4f}")
        log.info(f"  Semantic:   b = {b_s:.4f}, p = {p_s:.4f}")
        log.info(f"  Semantic slope is {b_s/b_p:.1f}× larger (on log scale)")

    log.info("\n--- Linear Days models (absolute change per day) ---")
    if res_perc_lin and res_sem_lin:
        b_p = res_perc_lin.params['Days']
        p_p = res_perc_lin.pvalues['Days']
        b_s = res_sem_lin.params['Days']
        p_s = res_sem_lin.pvalues['Days']
        log.info(f"  Perceptual: b = {b_p:.4f}, p = {p_p:.4f}")
        log.info(f"  Semantic:   b = {b_s:.4f}, p = {p_s:.4f}")
        if b_s != 0:
            log.info(f"  Perceptual slope is {b_p/b_s:.1f}× larger (on linear scale)")

    log.info(f"\n--- Interpretation ---")
    log.info("The choice of time scale matters enormously here:")
    log.info("  - log(Days): semantic wins because 0.2%→3.7% is a larger RATIO")
    log.info("  - linear Days: perceptual should win because 6.2%→9.7% is a larger ABSOLUTE change")
    log.info("Both are 'correct' — they answer different questions:")
    log.info("  - log: 'Which error type is ACCELERATING fastest?'")
    log.info("  - linear: 'Which error type ADDS MORE errors per day?'")
    log.info("For Jason's question ('what drives the steeper recognition decline'),")
    log.info("the linear scale is more directly relevant — it tells you which")
    log.info("error type contributes more additional errors over time.")

    log.info(f"\n{'='*70}")
    log.info("DONE: step12_error_type_glmm.py")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
