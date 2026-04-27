"""
Response Pattern Analysis for RQ 6.2.3
Section 1.4 Requirement from improvement_taxonomy.md

Analyzes confidence rating response patterns:
- % participants using full scale (1-5)
- % extremes only (1s and 5s)
- SD of ratings per participant
"""

import pandas as pd
import numpy as np

# Load item-level data
data = pd.read_csv('data/step00_item_level.csv')

# Confidence values in dataset: 0.2, 0.4, 0.6, 0.8, 1.0 (5 levels)
# Map to 1-5 scale for analysis
data['Confidence_1_5'] = (data['Confidence'] * 5).astype(int)

# Initialize results
results = []

for uid in data['UID'].unique():
    participant_data = data[data['UID'] == uid]
    ratings = participant_data['Confidence_1_5']
    
    # Full scale usage (all 5 values: 1, 2, 3, 4, 5)
    unique_ratings = set(ratings)
    full_scale = len(unique_ratings) == 5
    
    # Extremes only (only 1s and 5s)
    extremes_only = unique_ratings.issubset({1, 5})
    
    # SD of ratings
    rating_sd = ratings.std()
    
    # Count by response level
    counts = ratings.value_counts()
    
    results.append({
        'UID': uid,
        'full_scale': full_scale,
        'extremes_only': extremes_only,
        'rating_sd': rating_sd,
        'n_unique_levels': len(unique_ratings),
        'n_responses': len(ratings),
        'pct_1': counts.get(1, 0) / len(ratings) * 100,
        'pct_2': counts.get(2, 0) / len(ratings) * 100,
        'pct_3': counts.get(3, 0) / len(ratings) * 100,
        'pct_4': counts.get(4, 0) / len(ratings) * 100,
        'pct_5': counts.get(5, 0) / len(ratings) * 100
    })

results_df = pd.DataFrame(results)

# Summary statistics
pct_full_scale = (results_df['full_scale'].sum() / len(results_df)) * 100
pct_extremes_only = (results_df['extremes_only'].sum() / len(results_df)) * 100
mean_sd = results_df['rating_sd'].mean()
median_unique = results_df['n_unique_levels'].median()

# Print summary
print("=" * 60)
print("CONFIDENCE RESPONSE PATTERN ANALYSIS")
print("=" * 60)
print(f"\nParticipants: {len(results_df)}")
print(f"Total responses: {len(data)}")
print(f"\nFull scale usage (all 5 levels): {pct_full_scale:.1f}%")
print(f"Extremes only (1s and 5s): {pct_extremes_only:.1f}%")
print(f"Mean rating SD: {mean_sd:.2f}")
print(f"Median unique levels used: {median_unique:.0f}")
print(f"\nRating SD distribution:")
print(f"  Min: {results_df['rating_sd'].min():.2f}")
print(f"  Q1:  {results_df['rating_sd'].quantile(0.25):.2f}")
print(f"  Med: {results_df['rating_sd'].quantile(0.50):.2f}")
print(f"  Q3:  {results_df['rating_sd'].quantile(0.75):.2f}")
print(f"  Max: {results_df['rating_sd'].max():.2f}")
print(f"\nAverage % by rating level:")
print(f"  1 (lowest):  {results_df['pct_1'].mean():.1f}%")
print(f"  2:           {results_df['pct_2'].mean():.1f}%")
print(f"  3 (midpoint):{results_df['pct_3'].mean():.1f}%")
print(f"  4:           {results_df['pct_4'].mean():.1f}%")
print(f"  5 (highest): {results_df['pct_5'].mean():.1f}%")

# Save detailed results
results_df.to_csv('data/response_patterns.csv', index=False)
print(f"\nDetailed results saved to: data/response_patterns.csv")

# Flag restricted range (SD < 0.8 as potential issue)
restricted_range = (results_df['rating_sd'] < 0.8).sum()
pct_restricted = (restricted_range / len(results_df)) * 100
print(f"\nRestricted range (SD < 0.8): {restricted_range} participants ({pct_restricted:.1f}%)")

# Check for potential bias
if pct_extremes_only > 5:
    print(f"\n⚠️ WARNING: {pct_extremes_only:.1f}% participants use extremes only")
    print("   This may inflate gamma values (binary confidence reduces discriminability)")
elif pct_full_scale < 50:
    print(f"\n⚠️ NOTE: Only {pct_full_scale:.1f}% use full scale")
    print("   Limited scale usage may reduce resolution measurement precision")
else:
    print(f"\n✅ Response patterns acceptable ({pct_full_scale:.1f}% full scale usage)")

print("=" * 60)
