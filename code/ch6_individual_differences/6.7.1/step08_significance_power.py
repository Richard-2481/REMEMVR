#!/usr/bin/env python3
"""Step 08: Multiple comparison corrections and post-hoc power analysis"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.stats.power import FTestAnovaPower
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_cohens_f2, compute_post_hoc_power

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_significance_power.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 08: Significance and Power Analysis")

        # Load regression results
        results_path = RQ_DIR / 'data' / 'step03_reverse_models.csv'
        df_results = pd.read_csv(results_path)

        # Load effect sizes
        effect_path = RQ_DIR / 'data' / 'step05_effect_sizes.csv'
        df_effects = pd.read_csv(effect_path)

        n = 100  # Sample size
        alpha_family = 0.05
        n_tests = len(df_results)

        power_results = []

        for _, row in df_results.iterrows():
            model_name = row['model']
            p_uncorr = row['p_uncorrected']
            p_bonf = row['p_bonferroni']

            # Chapter-level correction (28 RQs in Ch7)
            p_chapter = min(p_uncorr * 28, 1.0)

            # FDR correction (Benjamini-Hochberg) for 2 tests
            p_values = df_results['p_uncorrected'].values
            p_values_sorted = np.sort(p_values)
            ranks = np.argsort(np.argsort(p_values)) + 1
            p_fdr = min(p_uncorr * n_tests / ranks[list(df_results['model']).index(model_name)], 1.0)

            log(f"[{model_name}]")
            log(f"  p_uncorrected: {p_uncorr:.4f}")
            log(f"  p_bonferroni: {p_bonf:.4f}")
            log(f"  p_chapter: {p_chapter:.4f}")
            log(f"  p_fdr: {p_fdr:.4f}")

            # Post-hoc power
            f2 = df_effects[df_effects['model']==model_name]['cohens_f2'].values[0]
            power = compute_post_hoc_power(n=n, k_predictors=1, r2=row['R2'], alpha=alpha_family)

            log(f"  Achieved power (alpha=0.05): {power:.4f}")

            # Minimum detectable f² at 80% power
            power_analysis = FTestAnovaPower()
            try:
                min_f2 = power_analysis.solve_power(
                    effect_size=None,
                    nobs=n,
                    alpha=alpha_family,
                    power=0.80
                )
                log(f"  Min detectable f² (80% power): {min_f2:.4f}")
            except:
                min_f2 = np.nan
                log(f"  Min detectable f² calculation failed")

            power_adequate = power >= 0.80

            power_results.append({
                'model': model_name,
                'p_uncorrected': p_uncorr,
                'p_bonferroni': p_bonf,
                'p_chapter': p_chapter,
                'p_fdr': p_fdr,
                'achieved_power': power,
                'min_detectable_f2': min_f2,
                'power_adequate': power_adequate
            })

        df_power = pd.DataFrame(power_results)
        output_path = RQ_DIR / 'data' / 'step08_significance_power.csv'
        df_power.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
