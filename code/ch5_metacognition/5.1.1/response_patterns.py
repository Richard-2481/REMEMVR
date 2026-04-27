#!/usr/bin/env python3
"""
Response Pattern Analysis for RQ 6.1.1
Analyzes confidence rating response patterns (Section 8.3 requirement)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_response_patterns(irt_input_path, output_dir):
    """
    Analyze confidence rating response patterns.

    Checks:
    - % participants using full scale (all 5 values: 0, 1, 2, 3, 4)
    - % using extremes only (0 and 4)
    - Mean SD of ratings per participant
    - Response style heterogeneity
    """
    print("="*80)
    print("RESPONSE PATTERN ANALYSIS")
    print("="*80)

    # Load IRT input (wide format: composite_ID x items)
    print(f"\nLoading IRT input from {irt_input_path}...")
    data = pd.read_csv(irt_input_path)

    # Extract UID from composite_ID (format: A###_N or P###_N)
    data['UID'] = data['composite_ID'].str.replace(r'_\d+$', '', regex=True)

    # Get confidence item columns (all except composite_ID and UID)
    item_cols = [col for col in data.columns if col not in ['composite_ID', 'UID']]

    print(f"Found {len(item_cols)} confidence items (TC_* columns)")
    print(f"N = {len(data)} observations (composite_IDs)")
    print(f"N = {data['UID'].nunique()} unique participants")

    # Response pattern analysis per composite_ID (observation level)
    patterns = []

    for idx, row in data.iterrows():
        composite_id = row['composite_ID']
        uid = row['UID']

        # Get non-missing confidence ratings for this observation
        ratings = row[item_cols].dropna()

        if len(ratings) == 0:
            continue

        unique_values = set(ratings.unique())

        # Full scale usage (all 5 values present: 0, 1, 2, 3, 4)
        full_scale = unique_values == {0, 1, 2, 3, 4}

        # Extremes only (only 0 and 4, no middle values)
        extremes_only = unique_values.issubset({0, 4})

        # Rating SD
        rating_sd = ratings.std()

        # Rating mean
        rating_mean = ratings.mean()

        # Count of each response category
        n_0 = (ratings == 0).sum()
        n_1 = (ratings == 1).sum()
        n_2 = (ratings == 2).sum()
        n_3 = (ratings == 3).sum()
        n_4 = (ratings == 4).sum()

        patterns.append({
            'composite_ID': composite_id,
            'UID': uid,
            'n_items': len(ratings),
            'full_scale': full_scale,
            'extremes_only': extremes_only,
            'rating_sd': rating_sd,
            'rating_mean': rating_mean,
            'n_categories_used': len(unique_values),
            'n_0': n_0,
            'n_1': n_1,
            'n_2': n_2,
            'n_3': n_3,
            'n_4': n_4
        })

    patterns_df = pd.DataFrame(patterns)

    # Summary statistics
    print("\n" + "="*80)
    print("RESPONSE PATTERN SUMMARY")
    print("="*80)

    n_obs = len(patterns_df)
    pct_full_scale = (patterns_df['full_scale'].sum() / n_obs) * 100
    pct_extremes = (patterns_df['extremes_only'].sum() / n_obs) * 100
    mean_sd = patterns_df['rating_sd'].mean()
    mean_categories = patterns_df['n_categories_used'].mean()

    print(f"\nObservation-Level Patterns (N={n_obs} composite_IDs):")
    print(f"  Full scale usage (all 5 values):  {pct_full_scale:5.1f}%  ({patterns_df['full_scale'].sum()} obs)")
    print(f"  Extremes only (0 and 4):          {pct_extremes:5.1f}%  ({patterns_df['extremes_only'].sum()} obs)")
    print(f"  Mean rating SD:                   {mean_sd:5.2f}")
    print(f"  Mean categories used:             {mean_categories:5.1f} / 5")

    # Response category distribution
    print("\nResponse Category Distribution (across all ratings):")
    total_ratings = patterns_df[['n_0', 'n_1', 'n_2', 'n_3', 'n_4']].sum()
    total_n = total_ratings.sum()

    categories = {
        'n_0': '0',
        'n_1': '1',
        'n_2': '2',
        'n_3': '3',
        'n_4': '4'
    }

    for cat, label in categories.items():
        count = total_ratings[cat]
        pct = (count / total_n) * 100
        print(f"  {label:>4s}:  {count:6.0f} ({pct:5.1f}%)")

    # Participant-level aggregation (average across tests)
    participant_patterns = patterns_df.groupby('UID').agg({
        'full_scale': 'mean',  # % of tests with full scale
        'extremes_only': 'mean',  # % of tests with extremes only
        'rating_sd': 'mean',  # Average SD across tests
        'n_categories_used': 'mean'  # Average categories used
    }).reset_index()

    participant_patterns.rename(columns={
        'full_scale': 'pct_full_scale',
        'extremes_only': 'pct_extremes_only',
        'rating_sd': 'mean_sd',
        'n_categories_used': 'mean_categories'
    }, inplace=True)

    print("\n" + "="*80)
    print("PARTICIPANT-LEVEL SUMMARY (Averaged Across Tests)")
    print("="*80)

    n_participants = len(participant_patterns)

    # Classify participants by dominant response style
    # Consistent full scale users (use full scale in >50% of tests)
    consistent_full = (participant_patterns['pct_full_scale'] > 0.5).sum()

    # Consistent extreme users (use extremes in >50% of tests)
    consistent_extreme = (participant_patterns['pct_extremes_only'] > 0.5).sum()

    # Moderate users (neither full nor extreme dominant)
    moderate = n_participants - consistent_full - consistent_extreme

    print(f"\nParticipant Response Style Classification (N={n_participants}):")
    print(f"  Consistent full scale users:   {consistent_full:3d} ({consistent_full/n_participants*100:5.1f}%)")
    print(f"  Consistent extreme users:      {consistent_extreme:3d} ({consistent_extreme/n_participants*100:5.1f}%)")
    print(f"  Moderate/mixed users:          {moderate:3d} ({moderate/n_participants*100:5.1f}%)")

    print(f"\nParticipant-Level Statistics:")
    print(f"  Mean SD (avg across tests):    {participant_patterns['mean_sd'].mean():5.2f}")
    print(f"  Mean categories used:          {participant_patterns['mean_categories'].mean():5.1f} / 5")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Additional insight: bimodal pattern check
    # High use of extremes (0 + 4) vs middle (1 + 2 + 3)
    extreme_pct = (total_ratings['n_0'] + total_ratings['n_4']) / total_n * 100
    middle_pct = (total_ratings['n_1'] + total_ratings['n_2'] + total_ratings['n_3']) / total_n * 100

    print(f"\nResponse Distribution Pattern:")
    print(f"  Extremes (0 + 4):  {extreme_pct:5.1f}%")
    print(f"  Middle (1-3):      {middle_pct:5.1f}%")

    if extreme_pct > 60:
        print("  → BIMODAL: Participants primarily use low/high confidence (binary-like)")
        print("     This may explain GRM threshold ordering violations")
    elif extreme_pct > 50:
        print("  → SLIGHT BIMODAL TENDENCY: Preference for extreme ratings")
    else:
        print("  → GRADED: Participants use full range of confidence levels")

    if pct_extremes > 50:
        print("\n⚠️  WARNING: >50% of observations use extremes only (0 and 4)")
        print("   This violates GRM assumptions (ordered polytomous responses)")
        print("   Recommendation: Consider binary IRT model (2PL) instead of GRM")
    elif pct_extremes > 30:
        print("\n⚠️  CAUTION: >30% of observations use extremes only")
        print("   Substantial extreme response style may affect GRM calibration")
        print("   Threshold ordering violations may result from restricted range")
    else:
        print("\n✓  Acceptable: <30% extreme response style")
        print("   GRM model appropriate for this response distribution")

    if pct_full_scale < 10:
        print("\n⚠️  NOTE: <10% of observations use full 5-point scale")
        print("   Most participants use restricted range (limits calibration precision)")
    elif pct_full_scale < 30:
        print("\n✓  Moderate full scale usage (10-30%)")
    else:
        print("\n✓  Good full scale usage (>30%)")

    # Save detailed results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_level_path = output_dir / 'response_patterns_observation_level.csv'
    participant_level_path = output_dir / 'response_patterns_participant_level.csv'
    summary_path = output_dir / 'response_patterns_summary.txt'

    patterns_df.to_csv(obs_level_path, index=False)
    participant_patterns.to_csv(participant_level_path, index=False)

    # Write summary report
    with open(summary_path, 'w') as f:
        f.write("RESPONSE PATTERN ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"RQ 6.1.1 Confidence Rating Response Patterns\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        f.write(f"Observation-Level Patterns (N={n_obs}):\n")
        f.write(f"  Full scale usage:    {pct_full_scale:5.1f}%\n")
        f.write(f"  Extremes only:       {pct_extremes:5.1f}%\n")
        f.write(f"  Mean rating SD:      {mean_sd:5.2f}\n")
        f.write(f"  Mean categories:     {mean_categories:5.1f} / 5\n\n")

        f.write(f"Response Category Distribution:\n")
        for cat, label in categories.items():
            count = total_ratings[cat]
            pct = (count / total_n) * 100
            f.write(f"  {label:>4s}:  {count:6.0f} ({pct:5.1f}%)\n")
        f.write(f"\n  Extremes (0 + 4):  {extreme_pct:5.1f}%\n")
        f.write(f"  Middle (1-3):      {middle_pct:5.1f}%\n\n")

        f.write(f"Participant-Level Classification (N={n_participants}):\n")
        f.write(f"  Consistent full scale:  {consistent_full:3d} ({consistent_full/n_participants*100:5.1f}%)\n")
        f.write(f"  Consistent extreme:     {consistent_extreme:3d} ({consistent_extreme/n_participants*100:5.1f}%)\n")
        f.write(f"  Moderate/mixed:         {moderate:3d} ({moderate/n_participants*100:5.1f}%)\n\n")

        f.write("Interpretation:\n")
        if extreme_pct > 60:
            f.write("  → BIMODAL response pattern (binary-like confidence)\n")
            f.write("  → May explain GRM threshold ordering violations\n")
        elif extreme_pct > 50:
            f.write("  → SLIGHT BIMODAL TENDENCY\n")
        else:
            f.write("  → GRADED response pattern\n")

        if pct_extremes > 50:
            f.write("  ⚠️  WARNING: >50% extreme response style - consider binary IRT\n")
        elif pct_extremes > 30:
            f.write("  ⚠️  CAUTION: >30% extreme response style\n")
        else:
            f.write("  ✓  Acceptable extreme response style (<30%)\n")

        if pct_full_scale < 10:
            f.write("  ⚠️  NOTE: <10% full scale usage (restricted range)\n")
        elif pct_full_scale < 30:
            f.write("  ✓  Moderate full scale usage\n")
        else:
            f.write("  ✓  Good full scale usage\n")

    print(f"\nResults saved to:")
    print(f"  {obs_level_path}")
    print(f"  {participant_level_path}")
    print(f"  {summary_path}")

    return patterns_df, participant_patterns

if __name__ == '__main__':
    irt_input_path = Path('data/step00_irt_input.csv')
    output_dir = Path('results')

    patterns_df, participant_patterns = analyze_response_patterns(irt_input_path, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
