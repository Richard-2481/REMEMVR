"""
Step 10: Confidence Response Pattern Analysis

Purpose: MANDATORY analysis per taxonomy Section 8.3
         - % participants using full scale (1-5)
         - % using extremes only (1s and 5s)
         - SD of ratings per participant
         - Flag restricted range (limits calibration)

Date: 2025-12-27
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_raw_confidence_data():
    """Load IRT input (raw confidence responses)."""
    print("\n" + "="*80)
    print("LOADING RAW CONFIDENCE DATA")
    print("="*80)

    # Load IRT input (wide format with raw TC_* responses)
    irt_input = pd.read_csv('results/ch6/6.8.1/data/step00_irt_input.csv')

    print(f"\nIRT input shape: {irt_input.shape}")
    print(f"Columns: composite_ID + {irt_input.shape[1]-1} TC_* items")

    # Get TC_* item columns
    tc_cols = [col for col in irt_input.columns if col.startswith('TC_')]
    print(f"TC_* items: {len(tc_cols)}")

    return irt_input, tc_cols

def analyze_per_participant_patterns(irt_input, tc_cols):
    """Analyze response patterns for each participant."""
    print("\n" + "="*80)
    print("PER-PARTICIPANT RESPONSE PATTERNS")
    print("="*80)

    results = []

    for idx, row in irt_input.iterrows():
        composite_id = row['composite_ID']

        # Get confidence ratings for this participant
        ratings = row[tc_cols].dropna().values

        if len(ratings) == 0:
            continue  # Skip if no data

        # Unique values used
        unique_vals = np.unique(ratings)
        n_unique = len(unique_vals)

        # Full scale usage (all 5 values: 0, 0.25, 0.5, 0.75, 1.0)
        expected_vals = {0.0, 0.25, 0.5, 0.75, 1.0}
        uses_full_scale = set(unique_vals) == expected_vals

        # Extremes only (only 0 and 1)
        uses_extremes_only = set(unique_vals).issubset({0.0, 1.0})

        # SD of ratings
        rating_sd = np.std(ratings)

        # Mean rating
        rating_mean = np.mean(ratings)

        # % extreme responses (0 or 1)
        pct_extremes = np.mean((ratings == 0.0) | (ratings == 1.0)) * 100

        results.append({
            'composite_ID': composite_id,
            'n_ratings': len(ratings),
            'n_unique_values': n_unique,
            'uses_full_scale': uses_full_scale,
            'uses_extremes_only': uses_extremes_only,
            'rating_mean': rating_mean,
            'rating_sd': rating_sd,
            'pct_extremes': pct_extremes,
            'unique_values': str(sorted(unique_vals))
        })

    df = pd.DataFrame(results)
    return df

def summarize_response_patterns(df):
    """Summarize patterns across all participants."""
    print("\n" + "="*80)
    print("AGGREGATE RESPONSE PATTERNS")
    print("="*80)

    n_total = len(df)

    # Full scale usage
    n_full_scale = df['uses_full_scale'].sum()
    pct_full_scale = n_full_scale / n_total * 100

    # Extremes only
    n_extremes_only = df['uses_extremes_only'].sum()
    pct_extremes_only = n_extremes_only / n_total * 100

    # SD statistics
    mean_sd = df['rating_sd'].mean()
    median_sd = df['rating_sd'].median()

    # Restricted range (SD < 0.20 indicates low variability)
    n_restricted = (df['rating_sd'] < 0.20).sum()
    pct_restricted = n_restricted / n_total * 100

    print(f"\nTotal participants: {n_total}")
    print(f"\n📊 SCALE USAGE:")
    print(f"   Full scale (all 5 values): {n_full_scale} ({pct_full_scale:.1f}%)")
    print(f"   Extremes only (0 and 1): {n_extremes_only} ({pct_extremes_only:.1f}%)")
    print(f"\n📊 RATING VARIABILITY:")
    print(f"   Mean SD: {mean_sd:.3f}")
    print(f"   Median SD: {median_sd:.3f}")
    print(f"   Restricted range (SD < 0.20): {n_restricted} ({pct_restricted:.1f}%)")

    if pct_extremes_only > 20:
        print(f"\n🔴 WARNING: {pct_extremes_only:.1f}% use extremes only")
        print(f"   → Extreme response style may affect calibration")
    elif pct_full_scale < 30:
        print(f"\n⚠️  Only {pct_full_scale:.1f}% use full scale")
        print(f"   → Most participants underutilize rating scale")
    else:
        print(f"\n✅ Reasonable scale usage ({pct_full_scale:.1f}% use full scale)")

    if mean_sd < 0.25:
        print(f"\n⚠️  Low variability (mean SD = {mean_sd:.3f})")
        print(f"   → Participants show restricted range (limited discrimination)")
    else:
        print(f"\n✅ Adequate variability (mean SD = {mean_sd:.3f})")

    # Distribution of unique values
    print(f"\n📊 UNIQUE VALUES DISTRIBUTION:")
    unique_counts = df['n_unique_values'].value_counts().sort_index()
    for n_vals, count in unique_counts.items():
        pct = count / n_total * 100
        print(f"   {n_vals} unique values: {count} ({pct:.1f}%)")

    return {
        'n_total': n_total,
        'n_full_scale': n_full_scale,
        'pct_full_scale': pct_full_scale,
        'n_extremes_only': n_extremes_only,
        'pct_extremes_only': pct_extremes_only,
        'mean_sd': mean_sd,
        'median_sd': median_sd,
        'n_restricted': n_restricted,
        'pct_restricted': pct_restricted
    }

def visualize_response_patterns(df):
    """Create visualizations of response patterns."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # 1. Histogram of rating SDs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df['rating_sd'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.20, color='r', linestyle='--', linewidth=2, label='Low variability threshold')
    axes[0, 0].set_xlabel('Rating SD', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Rating Variability (SD)', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram of unique values
    unique_counts = df['n_unique_values'].value_counts().sort_index()
    axes[0, 1].bar(unique_counts.index, unique_counts.values, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Unique Values Used', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Scale Usage Diversity', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # 3. Histogram of % extremes
    axes[1, 0].hist(df['pct_extremes'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=80, color='r', linestyle='--', linewidth=2, label='High extreme usage (80%)')
    axes[1, 0].set_xlabel('% Extreme Responses (0 or 1)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Extreme Response Style', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Mean vs SD scatter
    axes[1, 1].scatter(df['rating_mean'], df['rating_sd'], alpha=0.5, s=30)
    axes[1, 1].set_xlabel('Mean Rating', fontsize=11)
    axes[1, 1].set_ylabel('Rating SD', fontsize=11)
    axes[1, 1].set_title('Mean vs Variability', fontsize=12)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = 'results/ch6/6.8.1/plots/confidence_response_patterns.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n✅ Saved: {plot_path}")

def compare_source_vs_destination(irt_input):
    """Compare response patterns for source vs destination items."""
    print("\n" + "="*80)
    print("SOURCE VS DESTINATION RESPONSE PATTERNS")
    print("="*80)

    # Separate source (-U-) and destination (-D-) items
    source_cols = [col for col in irt_input.columns if '-U-' in col and col.startswith('TC_')]
    dest_cols = [col for col in irt_input.columns if '-D-' in col and col.startswith('TC_')]

    print(f"\nSource items: {len(source_cols)}")
    print(f"Destination items: {len(dest_cols)}")

    # Aggregate ratings
    source_ratings = irt_input[source_cols].values.flatten()
    source_ratings = source_ratings[~np.isnan(source_ratings)]

    dest_ratings = irt_input[dest_cols].values.flatten()
    dest_ratings = dest_ratings[~np.isnan(dest_ratings)]

    print(f"\nSource ratings: N = {len(source_ratings)}")
    print(f"  Mean: {source_ratings.mean():.3f}")
    print(f"  SD: {source_ratings.std():.3f}")
    print(f"  % extremes: {np.mean((source_ratings == 0) | (source_ratings == 1))*100:.1f}%")

    print(f"\nDestination ratings: N = {len(dest_ratings)}")
    print(f"  Mean: {dest_ratings.mean():.3f}")
    print(f"  SD: {dest_ratings.std():.3f}")
    print(f"  % extremes: {np.mean((dest_ratings == 0) | (dest_ratings == 1))*100:.1f}%")

    # Statistical comparison
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(source_ratings, dest_ratings, alternative='two-sided')

    print(f"\nMann-Whitney U test (Source vs Destination):")
    print(f"  U = {stat:.0f}")
    print(f"  p = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  🔴 SIGNIFICANT DIFFERENCE (p < 0.05)")
        print(f"     → Source and destination elicit different confidence ratings")
    else:
        print(f"  ✅ NO DIFFERENCE (p ≥ 0.05)")
        print(f"     → Similar response patterns for source and destination")

def main():
    """Execute confidence response pattern analysis."""
    print("="*80)
    print("CONFIDENCE RESPONSE PATTERN ANALYSIS - RQ 6.8.1")
    print("="*80)
    print("\n🔴 MANDATORY per Taxonomy Section 8.3")
    print("   - % participants using full scale")
    print("   - % using extremes only")
    print("   - SD of ratings per participant")
    print("   - Flag restricted range")

    # Load data
    irt_input, tc_cols = load_raw_confidence_data()

    # Analyze per-participant patterns
    df = analyze_per_participant_patterns(irt_input, tc_cols)

    # Summarize
    summary = summarize_response_patterns(df)

    # Visualize
    visualize_response_patterns(df)

    # Source vs destination comparison
    compare_source_vs_destination(irt_input)

    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    # Save per-participant data
    df_path = 'results/ch6/6.8.1/data/step10_response_patterns_per_participant.csv'
    df.to_csv(df_path, index=False, float_format='%.4f')
    print(f"✅ Saved: {df_path}")

    # Save summary
    summary_path = 'results/ch6/6.8.1/data/step10_response_patterns_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONFIDENCE RESPONSE PATTERN SUMMARY - RQ 6.8.1\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total participants: {summary['n_total']}\n\n")

        f.write("SCALE USAGE:\n")
        f.write(f"  Full scale (all 5 values): {summary['n_full_scale']} ({summary['pct_full_scale']:.1f}%)\n")
        f.write(f"  Extremes only (0 and 1): {summary['n_extremes_only']} ({summary['pct_extremes_only']:.1f}%)\n\n")

        f.write("RATING VARIABILITY:\n")
        f.write(f"  Mean SD: {summary['mean_sd']:.3f}\n")
        f.write(f"  Median SD: {summary['median_sd']:.3f}\n")
        f.write(f"  Restricted range (SD < 0.20): {summary['n_restricted']} ({summary['pct_restricted']:.1f}%)\n\n")

        if summary['pct_extremes_only'] > 20:
            f.write(f"WARNING: {summary['pct_extremes_only']:.1f}% use extremes only (extreme response bias)\n")
        if summary['mean_sd'] < 0.25:
            f.write(f"WARNING: Low variability (mean SD = {summary['mean_sd']:.3f})\n")

        f.write("\n✅ VALIDATION REQUIREMENT MET: Response patterns documented\n")

    print(f"✅ Saved: {summary_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 KEY FINDINGS:")
    print(f"   Full scale usage: {summary['pct_full_scale']:.1f}%")
    print(f"   Extremes only: {summary['pct_extremes_only']:.1f}%")
    print(f"   Mean rating SD: {summary['mean_sd']:.3f}")

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
