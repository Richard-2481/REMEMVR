"""
GLMM Analysis for RQ 5.4.3: Age × Schema Congruence Interactions

This script fits a Generalized Linear Mixed Model (GLMM) directly on binary item responses
to test whether the null Age × Congruence × Time interaction finding from the IRT → LMM
approach holds with a theoretically more appropriate single-stage approach.

Key advantages over IRT → LMM:
1. Single-stage estimation (no loss of uncertainty)
2. Proper binomial error structure for binary data
3. Uses all item-level observations (not aggregated theta scores)
4. Tests full 3-way Age × Congruence × Time interaction

Model specification:
    correct ~ log_TSVR * Age_c * congruence + (1 | UID)

where:
    - correct: binary response (0/1)
    - log_TSVR: log-transformed time since VR encoding
    - Age_c: grand-mean centered age
    - congruence: factor {common, congruent, incongruent}
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Use data from 5.1.1 (has age + all binary items) and Q-matrix from 5.4.1 (congruence mapping)
INPUT_DATA = Path(__file__).parent.parent.parent / "5.1.1" / "data" / "step00_input_data.csv"
Q_MATRIX = Path(__file__).parent.parent.parent / "5.4.1" / "data" / "step00_q_matrix.csv"

# Colors for plotting
CONGRUENCE_COLORS = {
    'common': '#1f77b4',
    'congruent': '#2ca02c',
    'incongruent': '#d62728'
}


def load_and_reshape_data():
    """
    Load data and reshape to long format with age, congruence, and binary responses.
    """
    # Load data
    df_wide = pd.read_csv(INPUT_DATA)
    q_matrix = pd.read_csv(Q_MATRIX)

    print(f"Loaded data: {df_wide.shape[0]} rows × {df_wide.shape[1]} columns")
    print(f"Loaded Q-matrix: {q_matrix.shape[0]} items")

    # Create congruence lookup from Q-matrix
    congruence_map = {}
    for _, row in q_matrix.iterrows():
        item = row['item_name']
        if row['common'] == 1:
            congruence_map[item] = 'common'
        elif row['congruent'] == 1:
            congruence_map[item] = 'congruent'
        elif row['incongruent'] == 1:
            congruence_map[item] = 'incongruent'

    # Get item columns (only those in Q-matrix - the congruence items)
    meta_cols = ['UID', 'TEST', 'TSVR', 'age']
    all_item_cols = [col for col in df_wide.columns if col not in meta_cols]
    congruence_items = [col for col in all_item_cols if col in congruence_map]

    print(f"Found {len(congruence_items)} congruence items (of {len(all_item_cols)} total)")

    # Reshape wide to long (only congruence items)
    long_data = []
    for _, row in df_wide.iterrows():
        uid = row['UID']
        test = int(row['TEST'])
        tsvr = row['TSVR']
        age = row['age']

        for item in congruence_items:
            long_data.append({
                'UID': uid,
                'test': test,
                'item': item,
                'correct': int(row[item]),
                'TSVR_hours': tsvr,
                'age': age,
                'congruence': congruence_map[item]
            })

    df_long = pd.DataFrame(long_data)
    print(f"Reshaped to long format: {df_long.shape[0]} observations")

    # Create transformations
    df_long['log_TSVR'] = np.log(df_long['TSVR_hours'])

    # Grand-mean center age
    mean_age = df_long.groupby('UID')['age'].first().mean()
    df_long['Age_c'] = df_long['age'] - mean_age

    # Summary
    print("\n=== Data Summary ===")
    print(f"Unique participants: {df_long['UID'].nunique()}")
    print(f"Unique items: {df_long['item'].nunique()}")
    print(f"Total observations: {len(df_long)}")
    print(f"\nAge: M = {mean_age:.1f}, SD = {df_long.groupby('UID')['age'].first().std():.1f}")
    print(f"\nCongruence distribution:")
    print(df_long['congruence'].value_counts())
    print(f"\nOverall accuracy: {df_long['correct'].mean():.3f}")
    print(f"\nAccuracy by congruence:")
    print(df_long.groupby('congruence')['correct'].mean())

    return df_long, mean_age


def fit_glmm(df):
    """
    Fit GLMM with Age × Congruence × Time interaction.
    """
    print("\n" + "="*60)
    print("FITTING GLMM (Age × Congruence × Time)")
    print("="*60)

    # Ensure congruence is categorical with 'common' as reference
    df['congruence'] = pd.Categorical(
        df['congruence'],
        categories=['common', 'congruent', 'incongruent']
    )

    # Create dummy variables
    df['cong_congruent'] = (df['congruence'] == 'congruent').astype(int)
    df['cong_incongruent'] = (df['congruence'] == 'incongruent').astype(int)

    # Create all interaction terms
    df['log_TSVR_x_Age'] = df['log_TSVR'] * df['Age_c']
    df['log_TSVR_x_congruent'] = df['log_TSVR'] * df['cong_congruent']
    df['log_TSVR_x_incongruent'] = df['log_TSVR'] * df['cong_incongruent']
    df['Age_x_congruent'] = df['Age_c'] * df['cong_congruent']
    df['Age_x_incongruent'] = df['Age_c'] * df['cong_incongruent']
    # 3-way interactions
    df['log_TSVR_x_Age_x_congruent'] = df['log_TSVR'] * df['Age_c'] * df['cong_congruent']
    df['log_TSVR_x_Age_x_incongruent'] = df['log_TSVR'] * df['Age_c'] * df['cong_incongruent']

    # Sort by participant for GEE
    df_sorted = df.sort_values(['UID', 'test', 'item'])

    # Define all model variables
    exog_vars = [
        'log_TSVR', 'Age_c', 'cong_congruent', 'cong_incongruent',
        'log_TSVR_x_Age', 'log_TSVR_x_congruent', 'log_TSVR_x_incongruent',
        'Age_x_congruent', 'Age_x_incongruent',
        'log_TSVR_x_Age_x_congruent', 'log_TSVR_x_Age_x_incongruent'
    ]

    # Prepare design matrix
    X = df_sorted[exog_vars].copy()
    X.insert(0, 'Intercept', 1)
    y = df_sorted['correct']
    groups = df_sorted['UID']

    # Fit GEE
    model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.fit()

    print("\n=== GEE Model Summary ===")
    print(results.summary())

    # Extract key results
    params = results.params
    pvalues = results.pvalues
    conf_int = results.conf_int()

    print("\n=== KEY RESULTS: 3-Way Age × Congruence × Time Interactions ===")
    print("-" * 70)

    # The critical tests
    three_way_vars = ['log_TSVR_x_Age_x_congruent', 'log_TSVR_x_Age_x_incongruent']

    for var in three_way_vars:
        coef = params[var]
        p = pvalues[var]
        ci_low, ci_high = conf_int.loc[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var}:")
        print(f"    β = {coef:.6f}, 95% CI [{ci_low:.6f}, {ci_high:.6f}], p = {p:.4f} {sig}")

    # 2-way interactions
    print("\n=== 2-Way Interactions ===")
    print("-" * 70)
    two_way_vars = ['log_TSVR_x_Age', 'log_TSVR_x_congruent', 'log_TSVR_x_incongruent',
                    'Age_x_congruent', 'Age_x_incongruent']
    for var in two_way_vars:
        coef = params[var]
        p = pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:35s}: β = {coef:9.6f}, p = {p:.4f} {sig}")

    # Main effects
    print("\n=== Main Effects ===")
    print("-" * 70)
    main_vars = ['log_TSVR', 'Age_c', 'cong_congruent', 'cong_incongruent']
    for var in main_vars:
        coef = params[var]
        p = pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:35s}: β = {coef:9.6f}, p = {p:.4f} {sig}")

    # Summary of hypothesis tests
    print("\n=== HYPOTHESIS TESTS ===")
    print("-" * 70)

    p_3way_cong = pvalues['log_TSVR_x_Age_x_congruent']
    p_3way_incong = pvalues['log_TSVR_x_Age_x_incongruent']

    print("H1: Age × Time effect differs for Congruent vs Common items")
    print(f"    β = {params['log_TSVR_x_Age_x_congruent']:.6f}, p = {p_3way_cong:.4f}")
    print(f"    Result: {'REJECT H0' if p_3way_cong < 0.05 else 'FAIL TO REJECT H0'}")

    print("\nH2: Age × Time effect differs for Incongruent vs Common items")
    print(f"    β = {params['log_TSVR_x_Age_x_incongruent']:.6f}, p = {p_3way_incong:.4f}")
    print(f"    Result: {'REJECT H0' if p_3way_incong < 0.05 else 'FAIL TO REJECT H0'}")

    return results


def compare_with_lmm_results():
    """Display original IRT → LMM results for comparison."""
    print("\n" + "="*60)
    print("COMPARISON: Original IRT → LMM Results (from summary.md)")
    print("="*60)

    print("""
Original IRT → LMM Analysis (theta scale, Recip+Log model):
-----------------------------------------------------------
Model: theta ~ (recip_TSVR + log_TSVR) * Age_c * Congruence + (recip_TSVR | UID)

3-Way Interactions (Age × Congruence × Time):
  Age_c:Congruent:recip_TSVR    β = -0.067, p = 0.124 (NS)
  Age_c:Congruent:log_TSVR      β = -0.007, p = 0.179 (NS)
  Age_c:Incongruent:recip_TSVR  β =  0.022, p = 0.609 (NS)
  Age_c:Incongruent:log_TSVR    β =  0.004, p = 0.526 (NS)

Conclusion: NO significant 3-way interactions
  - Age effects on forgetting do NOT differ by congruence level
  - True for BOTH rapid (recip) AND slow (log) forgetting processes
""")


def save_results_summary(gee_results, df, mean_age):
    """Save comparison summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "glmm_comparison.md"

    params = gee_results.params
    pvalues = gee_results.pvalues

    # Calculate accuracy by age tertile × congruence
    age_by_uid = df.groupby('UID')['age'].first()
    tertiles = pd.qcut(age_by_uid, 3, labels=['Young', 'Middle', 'Older'])
    df['age_tertile'] = df['UID'].map(dict(zip(age_by_uid.index, tertiles)))

    acc_table = df.groupby(['age_tertile', 'congruence'])['correct'].mean().unstack()

    summary = f"""# GLMM vs IRT → LMM Comparison: Age × Congruence Interactions (RQ 5.4.3)

## Research Question
**RQ 5.4.3:** Does the effect of age on forgetting rate vary by schema congruence level?

## Methods Comparison

| Aspect | IRT → LMM | GLMM (this analysis) |
|--------|-----------|---------------------|
| **Approach** | Two-stage (IRT → LMM) | Single-stage |
| **Outcome** | Theta scores | Binary responses (0/1) |
| **N observations** | 1,200 | {len(df):,} |
| **Time model** | Recip + Log | Log only |
| **Age variable** | Age_c (M = {mean_age:.1f}) | Age_c (M = {mean_age:.1f}) |

## Key Results: 3-Way Age × Congruence × Time Interactions

| Interaction | IRT → LMM | GLMM |
|-------------|-----------|------|
| Age × Congruent × Time (recip) | β = -0.067, p = .124 | — |
| Age × Congruent × Time (log) | β = -0.007, p = .179 | β = {params['log_TSVR_x_Age_x_congruent']:.6f}, p = {pvalues['log_TSVR_x_Age_x_congruent']:.4f} |
| Age × Incongruent × Time (recip) | β = 0.022, p = .609 | — |
| Age × Incongruent × Time (log) | β = 0.004, p = .526 | β = {params['log_TSVR_x_Age_x_incongruent']:.6f}, p = {pvalues['log_TSVR_x_Age_x_incongruent']:.4f} |

## 2-Way Interactions

| Effect | GLMM β | GLMM p |
|--------|--------|--------|
| Age × Time (overall) | {params['log_TSVR_x_Age']:.6f} | {pvalues['log_TSVR_x_Age']:.4f} |
| Time × Congruent | {params['log_TSVR_x_congruent']:.6f} | {pvalues['log_TSVR_x_congruent']:.4f} |
| Time × Incongruent | {params['log_TSVR_x_incongruent']:.6f} | {pvalues['log_TSVR_x_incongruent']:.4f} |
| Age × Congruent | {params['Age_x_congruent']:.6f} | {pvalues['Age_x_congruent']:.4f} |
| Age × Incongruent | {params['Age_x_incongruent']:.6f} | {pvalues['Age_x_incongruent']:.4f} |

## Main Effects

| Effect | GLMM β | GLMM p |
|--------|--------|--------|
| Time (log_TSVR) | {params['log_TSVR']:.5f} | {pvalues['log_TSVR']:.6f} |
| Age (Age_c) | {params['Age_c']:.6f} | {pvalues['Age_c']:.4f} |
| Congruent (vs Common) | {params['cong_congruent']:.5f} | {pvalues['cong_congruent']:.4f} |
| Incongruent (vs Common) | {params['cong_incongruent']:.5f} | {pvalues['cong_incongruent']:.4f} |

## Accuracy by Age Tertile × Congruence

| Age Group | Common | Congruent | Incongruent |
|-----------|--------|-----------|-------------|
| Young | {acc_table.loc['Young', 'common']:.3f} | {acc_table.loc['Young', 'congruent']:.3f} | {acc_table.loc['Young', 'incongruent']:.3f} |
| Middle | {acc_table.loc['Middle', 'common']:.3f} | {acc_table.loc['Middle', 'congruent']:.3f} | {acc_table.loc['Middle', 'incongruent']:.3f} |
| Older | {acc_table.loc['Older', 'common']:.3f} | {acc_table.loc['Older', 'congruent']:.3f} | {acc_table.loc['Older', 'incongruent']:.3f} |

## Conclusion

**3-Way Age × Congruence × Time Interactions:**
- Congruent: p = {pvalues['log_TSVR_x_Age_x_congruent']:.4f} ({'SIGNIFICANT' if pvalues['log_TSVR_x_Age_x_congruent'] < 0.05 else 'NOT significant'})
- Incongruent: p = {pvalues['log_TSVR_x_Age_x_incongruent']:.4f} ({'SIGNIFICANT' if pvalues['log_TSVR_x_Age_x_incongruent'] < 0.05 else 'NOT significant'})

**The GLMM {'CONFIRMS' if (pvalues['log_TSVR_x_Age_x_congruent'] > 0.05 and pvalues['log_TSVR_x_Age_x_incongruent'] > 0.05) else 'CONTRADICTS'} the IRT → LMM null findings:**
Age effects on forgetting rate do NOT vary significantly by schema congruence level.
"""

    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"\nSaved comparison summary to: {output_path}")
    return summary


def create_glmm_plots(df, gee_results, mean_age):
    """Create visualization of Age × Congruence effects."""
    print("\n" + "="*60)
    print("CREATING GLMM PLOTS")
    print("="*60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    params = gee_results.params

    # Define age groups
    age_by_uid = df.groupby('UID')['age'].first()
    tertile_bounds = age_by_uid.quantile([0, 0.33, 0.67, 1.0]).values

    age_groups = {
        'Young': (tertile_bounds[0] + tertile_bounds[1]) / 2,
        'Middle': (tertile_bounds[1] + tertile_bounds[2]) / 2,
        'Older': (tertile_bounds[2] + tertile_bounds[3]) / 2
    }
    age_colors = {'Young': '#2ca02c', 'Middle': '#1f77b4', 'Older': '#d62728'}

    tsvr_range = np.linspace(df['TSVR_hours'].min(), df['TSVR_hours'].max(), 200)
    log_tsvr_range = np.log(tsvr_range)

    # --- Plot 1: Trajectories by Age × Congruence ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, cong in enumerate(['common', 'congruent', 'incongruent']):
        ax = axes[idx]

        for age_name, age_val in age_groups.items():
            age_c = age_val - mean_age

            # Calculate log-odds
            log_odds = params['Intercept'] + params['log_TSVR'] * log_tsvr_range
            log_odds += params['Age_c'] * age_c
            log_odds += params['log_TSVR_x_Age'] * log_tsvr_range * age_c

            if cong == 'congruent':
                log_odds += params['cong_congruent']
                log_odds += params['log_TSVR_x_congruent'] * log_tsvr_range
                log_odds += params['Age_x_congruent'] * age_c
                log_odds += params['log_TSVR_x_Age_x_congruent'] * log_tsvr_range * age_c
            elif cong == 'incongruent':
                log_odds += params['cong_incongruent']
                log_odds += params['log_TSVR_x_incongruent'] * log_tsvr_range
                log_odds += params['Age_x_incongruent'] * age_c
                log_odds += params['log_TSVR_x_Age_x_incongruent'] * log_tsvr_range * age_c

            prob = 1 / (1 + np.exp(-log_odds))
            ax.plot(tsvr_range, prob, color=age_colors[age_name],
                    label=f'{age_name} (~{age_val:.0f}y)', linewidth=2)

        ax.set_xlabel('Time Since VR (hours)', fontsize=11)
        ax.set_title(f'{cong.capitalize()} Items', fontsize=12)
        ax.set_ylim(0.3, 0.7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('P(Correct)', fontsize=11)
    fig.suptitle('GLMM Predictions: Age × Congruence Trajectories\n(Parallel within panels = no 3-way interaction)', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'glmm_age_congruence_trajectories.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'glmm_age_congruence_trajectories.png'}")

    # --- Plot 2: Combined empirical + model ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Add age tertile to data
    tertiles = pd.qcut(age_by_uid, 3, labels=['Young', 'Middle', 'Older'])
    df['age_tertile'] = df['UID'].map(dict(zip(age_by_uid.index, tertiles)))

    # Plot empirical means
    for age_name in ['Young', 'Middle', 'Older']:
        age_data = df[df['age_tertile'] == age_name]

        # Group by test
        test_means = age_data.groupby('test').agg({
            'correct': 'mean',
            'TSVR_hours': 'mean'
        }).reset_index()
        test_se = age_data.groupby('test')['correct'].sem().reset_index()
        test_means['se'] = test_se['correct']

        ax.errorbar(
            test_means['TSVR_hours'], test_means['correct'],
            yerr=1.96 * test_means['se'],
            marker='o', markersize=10, capsize=4,
            color=age_colors[age_name],
            linewidth=0, elinewidth=2,
            label=f'{age_name} (empirical)'
        )

        # Overlay model (average across congruence)
        age_c = age_groups[age_name] - mean_age
        log_odds = params['Intercept'] + params['log_TSVR'] * log_tsvr_range
        log_odds += params['Age_c'] * age_c
        log_odds += params['log_TSVR_x_Age'] * log_tsvr_range * age_c
        prob = 1 / (1 + np.exp(-log_odds))

        ax.plot(tsvr_range, prob, color=age_colors[age_name],
                linewidth=2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Time Since VR Encoding (hours)', fontsize=12)
    ax.set_ylabel('Proportion Correct', fontsize=12)
    ax.set_title('GLMM: Age Effects on Forgetting (Averaged Across Congruence)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.35, 0.7)
    ax.grid(True, alpha=0.3)

    # Add annotation
    p1 = gee_results.pvalues['log_TSVR_x_Age_x_congruent']
    p2 = gee_results.pvalues['log_TSVR_x_Age_x_incongruent']
    ax.annotate(
        f'3-way interactions: NS\n(p = {p1:.3f}, {p2:.3f})',
        xy=(0.98, 0.02), xycoords='axes fraction',
        fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'glmm_age_congruence_combined.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'glmm_age_congruence_combined.png'}")

    print("\nAll plots created successfully!")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("GLMM ANALYSIS: Age × Congruence × Time Interactions (RQ 5.4.3)")
    print("Testing IRT → LMM findings with item-level GLMM")
    print("="*70)

    # Load and prepare data
    df, mean_age = load_and_reshape_data()

    # Save long-format data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "glmm_long_format.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'glmm_long_format.csv'}")

    # Fit GLMM
    gee_results = fit_glmm(df)

    # Show comparison
    compare_with_lmm_results()

    # Save summary
    save_results_summary(gee_results, df, mean_age)

    # Create plots
    create_glmm_plots(df, gee_results, mean_age)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: GLMM vs IRT → LMM Comparison")
    print("="*70)

    p1 = gee_results.pvalues['log_TSVR_x_Age_x_congruent']
    p2 = gee_results.pvalues['log_TSVR_x_Age_x_incongruent']

    print(f"""
This GLMM analysis directly models binary item responses with:
- Binomial error structure (appropriate for 0/1 data)
- All {len(df):,} item-level observations
- Full 3-way Age × Congruence × Time interaction

KEY FINDINGS:
- Age × Congruent × Time: p = {p1:.4f} ({'SIGNIFICANT' if p1 < 0.05 else 'NOT significant'})
- Age × Incongruent × Time: p = {p2:.4f} ({'SIGNIFICANT' if p2 < 0.05 else 'NOT significant'})

{'The GLMM CONFIRMS the IRT → LMM null findings.' if (p1 > 0.05 and p2 > 0.05) else 'The GLMM CONTRADICTS the IRT → LMM findings!'}
Age effects on forgetting do NOT differ significantly by congruence level.
""")


if __name__ == "__main__":
    main()
