#!/usr/bin/env python3
"""step04_compute_contrasts: Quantify age effect on forgetting rate for each domain (What, Where)"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import extract_marginal_age_slopes_by_domain

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY
LOG_FILE = RQ_DIR / "logs" / "step04_compute_contrasts.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute Domain-Specific Age Effects")
        # Load Input Data
        log("Loading LMM model from Step 2c...")
        lmm_model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"

        from statsmodels.regression.mixed_linear_model import MixedLMResults
        lmm_model = MixedLMResults.load(str(lmm_model_path))
        log(f"LMM model from {lmm_model_path.name}")

        n_groups = len(lmm_model.model.group_labels)
        log(f"Model has {n_groups} groups (participants)")
        log(f"Model converged: {lmm_model.converged}")
        # Extract Domain-Specific Marginal Age Effects
        log("Extracting marginal age effects by domain...")
        log("Evaluation timepoint: TSVR = 72.0 hours (Day 3, midpoint)")
        log("Using delta method for SE propagation")

        # Extract marginal age slopes for each domain at Day 3
        age_effects = extract_marginal_age_slopes_by_domain(
            lmm_result=lmm_model,
            eval_timepoint=72.0,  # Day 3 (midpoint of observation window)
            domain_var="domain",
            age_var="Age_c",
            time_linear="TSVR_hours",
            time_log="log_TSVR"
        )

        log("Age effect extraction complete")
        log(f"Extracted age effects for {len(age_effects)} domains")

        # Log results
        for idx, row in age_effects.iterrows():
            log(f"{row['domain']}: age_slope={row['age_slope']:.6f}, SE={row['se']:.6f}, p={row['p']:.4f}")
        # Save Outputs
        log("Saving age effects by domain...")
        output_path = RQ_DIR / "data" / "step04_age_effects_by_domain.csv"
        age_effects.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(age_effects)} rows, {len(age_effects.columns)} columns)")

        # Create summary text report
        log("Creating summary report...")
        summary_path = RQ_DIR / "results" / "step04_summary.txt"

        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("STEP 04: Domain-Specific Age Effects on Forgetting Rate")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("ANALYSIS METHOD:")
        summary_lines.append("  Marginal age effects extracted from 3-way Age × Domain × Time LMM")
        summary_lines.append("  Evaluation timepoint: TSVR = 72.0 hours (Day 3, midpoint)")
        summary_lines.append("  Standard errors: Delta method for linear combinations")
        summary_lines.append("")
        summary_lines.append("RESULTS:")
        summary_lines.append("")

        for idx, row in age_effects.iterrows():
            summary_lines.append(f"  {row['domain']}:")
            summary_lines.append(f"    Age slope:  {row['age_slope']:.6f} (theta units per 1-year age increase)")
            summary_lines.append(f"    SE:         {row['se']:.6f}")
            summary_lines.append(f"    z:          {row['z']:.3f}")
            summary_lines.append(f"    p-value:    {row['p']:.4f}")
            summary_lines.append(f"    95% CI:     [{row['CI_lower']:.6f}, {row['CI_upper']:.6f}]")

            # Interpretation
            if row['p'] < 0.05:
                direction = "negative" if row['age_slope'] < 0 else "positive"
                summary_lines.append(f"    Interpretation: Significant {direction} age effect (p < 0.05)")
            else:
                summary_lines.append(f"    Interpretation: Non-significant age effect (p >= 0.05)")
            summary_lines.append("")

        # Overall interpretation
        summary_lines.append("NOTE: When domain excluded due to floor effect (RQ 5.2.1)")
        summary_lines.append("")
        summary_lines.append("OVERALL INTERPRETATION:")
        n_sig = (age_effects['p'] < 0.05).sum()
        n_domains = len(age_effects)
        if n_sig == 0:
            summary_lines.append("  No significant age effects detected in any domain.")
            summary_lines.append("  Age does not differentially affect forgetting rate across domains.")
        elif n_sig == n_domains:
            summary_lines.append(f"  Significant age effects detected in ALL {n_domains} domains.")
            summary_lines.append("  Age affects forgetting rate across analyzed episodic memory domains.")
        else:
            sig_domains = age_effects.loc[age_effects['p'] < 0.05, 'domain'].tolist()
            summary_lines.append(f"  Significant age effects detected in {n_sig}/{n_domains} domains: {', '.join(sig_domains)}")
            summary_lines.append("  Age effects are domain-specific.")

        summary_lines.append("")
        summary_lines.append("=" * 80)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        log(f"{summary_path.name}")
        # Validation
        log("Validating outputs...")

        # Check output structure
        expected_cols = ['domain', 'age_slope', 'se', 'z', 'p', 'CI_lower', 'CI_upper']
        assert list(age_effects.columns) == expected_cols, f"Columns mismatch: {age_effects.columns}"
        assert len(age_effects) == 2, f"Expected 2 rows (When excluded), got {len(age_effects)}"

        # Check domains (When excluded)
        expected_domains = {'What', 'Where'}
        actual_domains = set(age_effects['domain'])
        assert actual_domains == expected_domains, f"Domains mismatch (When excluded): {actual_domains}"

        # Check value ranges
        assert all(age_effects['se'] > 0), "All SEs must be positive"
        assert all((age_effects['p'] >= 0) & (age_effects['p'] <= 1)), "P-values must be in [0, 1]"
        assert all(age_effects['CI_lower'] < age_effects['CI_upper']), "CI_lower must be < CI_upper"

        log("All checks passed")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
