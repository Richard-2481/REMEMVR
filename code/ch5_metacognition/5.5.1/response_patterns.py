"""
Response Pattern Analysis for RQ 6.5.1
Taxonomy Section 8.3 - MANDATORY for confidence RQs

Analyzes confidence rating patterns per validation.md Section 1.4 requirement:
1. % participants using full scale (all 5 Likert categories)
2. % participants using extremes only (1s and 5s)
3. Mean SD of ratings per participant
4. Flag restricted range (limits calibration quality)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
base_path = Path("/home/etai/projects/REMEMVR/results/ch6/6.5.1")
data_path = base_path / "data"
master_data_path = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")

print("="*80)
print("Response Pattern Analysis - RQ 6.5.1")
print("Taxonomy Section 8.3 + validation.md Section 1.4")
print("="*80)

# Load raw confidence data from dfData.csv
# TC_* columns are 5-level Likert: 0, 0.25, 0.5, 0.75, 1.0
dfData = pd.read_csv(master_data_path)
print(f"\nLoaded dfData.csv: {len(dfData)} rows")

# Filter to TC_* confidence columns with congruence tags (i1-i6)
tc_cols = [col for col in dfData.columns if col.startswith('TC_') and any(f'i{i}' in col for i in range(1, 7))]
print(f"Found {len(tc_cols)} TC_* confidence items with congruence tags (i1-i6)")

if len(tc_cols) == 0:
    print("❌ ERROR: No TC_* items found. Check column naming.")
    exit(1)

# Extract confidence ratings for each participant
uids = dfData['UID'].unique()
print(f"Participants: {len(uids)}")

##############################################################################
# ANALYZE RESPONSE PATTERNS PER PARTICIPANT
##############################################################################

print("\n" + "="*80)
print("ANALYZING RESPONSE PATTERNS")
print("="*80)

results = []

for uid in uids:
    uid_data = dfData[dfData['UID'] == uid]

    # Extract all TC_* ratings for this participant (across all test sessions)
    all_ratings = []
    for col in tc_cols:
        ratings = uid_data[col].dropna().values
        all_ratings.extend(ratings)

    if len(all_ratings) == 0:
        continue  # Skip participants with no confidence ratings

    all_ratings = np.array(all_ratings)

    # Pattern 1: Full scale usage (all 5 Likert values: 0, 0.25, 0.5, 0.75, 1.0)
    unique_values = set(all_ratings)
    expected_values = {0.0, 0.25, 0.5, 0.75, 1.0}
    uses_full_scale = expected_values.issubset(unique_values)

    # Pattern 2: Extremes only (only 0 and 1.0)
    uses_extremes_only = unique_values.issubset({0.0, 1.0})

    # Pattern 3: SD of ratings (variability)
    rating_sd = all_ratings.std()

    # Pattern 4: Mean rating
    rating_mean = all_ratings.mean()

    # Count of ratings
    n_ratings = len(all_ratings)

    results.append({
        'UID': uid,
        'n_ratings': n_ratings,
        'uses_full_scale': uses_full_scale,
        'uses_extremes_only': uses_extremes_only,
        'rating_mean': rating_mean,
        'rating_sd': rating_sd,
        'n_unique_values': len(unique_values)
    })

# Convert to DataFrame
patterns_df = pd.DataFrame(results)

##############################################################################
# SUMMARY STATISTICS
##############################################################################

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

n_participants = len(patterns_df)
pct_full_scale = (patterns_df['uses_full_scale'].sum() / n_participants) * 100
pct_extremes_only = (patterns_df['uses_extremes_only'].sum() / n_participants) * 100
mean_sd = patterns_df['rating_sd'].mean()
median_sd = patterns_df['rating_sd'].median()

print(f"\nFull scale usage (all 5 Likert values):")
print(f"  N participants: {patterns_df['uses_full_scale'].sum()} / {n_participants}")
print(f"  Percentage: {pct_full_scale:.1f}%")

print(f"\nExtremes only (only 0 and 1.0):")
print(f"  N participants: {patterns_df['uses_extremes_only'].sum()} / {n_participants}")
print(f"  Percentage: {pct_extremes_only:.1f}%")

print(f"\nRating variability (SD):")
print(f"  Mean SD: {mean_sd:.3f}")
print(f"  Median SD: {median_sd:.3f}")
print(f"  Min SD: {patterns_df['rating_sd'].min():.3f}")
print(f"  Max SD: {patterns_df['rating_sd'].max():.3f}")

# Flag restricted range (SD < 0.10 = very low variability)
low_variability = patterns_df[patterns_df['rating_sd'] < 0.10]
pct_low_variability = (len(low_variability) / n_participants) * 100

print(f"\nRestricted range (SD < 0.10):")
print(f"  N participants: {len(low_variability)} / {n_participants}")
print(f"  Percentage: {pct_low_variability:.1f}%")

##############################################################################
# INTERPRETATION
##############################################################################

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

issues = []

if pct_full_scale < 50:
    print(f"\n⚠️ LOW FULL SCALE USAGE ({pct_full_scale:.1f}%)")
    print("  Many participants do not use all 5 Likert categories.")
    print("  May indicate scale compression or restricted response style.")
    issues.append("Low full scale usage")
else:
    print(f"\n✅ ADEQUATE FULL SCALE USAGE ({pct_full_scale:.1f}%)")
    print("  Majority of participants use all 5 Likert categories.")

if pct_extremes_only > 10:
    print(f"\n⚠️ HIGH EXTREMES-ONLY USAGE ({pct_extremes_only:.1f}%)")
    print("  Some participants only use endpoints (0 and 1.0).")
    print("  May indicate dichotomous thinking or misunderstanding of scale.")
    issues.append("High extremes-only usage")
else:
    print(f"\n✅ LOW EXTREMES-ONLY USAGE ({pct_extremes_only:.1f}%)")
    print("  Few participants restrict to endpoints only.")

if mean_sd < 0.20:
    print(f"\n⚠️ LOW RATING VARIABILITY (mean SD = {mean_sd:.3f})")
    print("  Participants show low variability in confidence ratings.")
    print("  May indicate response bias (e.g., always rating '3' on 1-5 scale).")
    issues.append("Low rating variability")
else:
    print(f"\n✅ ADEQUATE RATING VARIABILITY (mean SD = {mean_sd:.3f})")
    print("  Participants show appropriate variability in ratings.")

if pct_low_variability > 20:
    print(f"\n⚠️ MANY PARTICIPANTS WITH RESTRICTED RANGE ({pct_low_variability:.1f}%)")
    print("  Over 20% of participants have SD < 0.10.")
    print("  May limit confidence-accuracy calibration quality.")
    issues.append("Restricted range in >20% participants")

##############################################################################
# SAVE RESULTS
##############################################################################

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Participant-level patterns
patterns_df.to_csv(data_path / "response_patterns_by_participant.csv", index=False)
print(f"✅ Saved: {data_path / 'response_patterns_by_participant.csv'}")

# Summary statistics
summary_df = pd.DataFrame({
    'metric': [
        'N participants',
        'Full scale usage (%)',
        'Extremes only (%)',
        'Mean rating SD',
        'Median rating SD',
        'Restricted range (SD < 0.10) (%)'
    ],
    'value': [
        n_participants,
        pct_full_scale,
        pct_extremes_only,
        mean_sd,
        median_sd,
        pct_low_variability
    ]
})

summary_df.to_csv(data_path / "response_patterns_summary.csv", index=False)
print(f"✅ Saved: {data_path / 'response_patterns_summary.csv'}")

# Text report
report_path = data_path / "response_patterns_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RESPONSE PATTERN ANALYSIS - RQ 6.5.1\n")
    f.write("Taxonomy Section 8.3 + validation.md Section 1.4\n")
    f.write("="*80 + "\n\n")

    f.write("PURPOSE:\n")
    f.write("Analyze confidence rating patterns to detect response biases that may\n")
    f.write("limit confidence-accuracy calibration quality.\n\n")

    f.write("-"*80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("-"*80 + "\n\n")

    f.write(f"N participants: {n_participants}\n")
    f.write(f"N TC_* items: {len(tc_cols)}\n\n")

    f.write(f"Full scale usage (all 5 Likert values): {pct_full_scale:.1f}%\n")
    f.write(f"Extremes only (0 and 1.0 only): {pct_extremes_only:.1f}%\n")
    f.write(f"Mean rating SD: {mean_sd:.3f}\n")
    f.write(f"Median rating SD: {median_sd:.3f}\n")
    f.write(f"Restricted range (SD < 0.10): {pct_low_variability:.1f}%\n\n")

    f.write("-"*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n\n")

    if len(issues) == 0:
        f.write("✅ NO RESPONSE PATTERN ISSUES DETECTED\n\n")
        f.write("Participants show:\n")
        f.write("  - Adequate full scale usage (≥50%)\n")
        f.write("  - Low extremes-only usage (<10%)\n")
        f.write("  - Adequate rating variability (mean SD ≥0.20)\n")
        f.write("  - Few participants with restricted range (<20% with SD<0.10)\n\n")
        f.write("Confidence ratings have good psychometric properties for calibration analysis.\n")
    else:
        f.write(f"⚠️ {len(issues)} RESPONSE PATTERN ISSUES DETECTED:\n\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"{i}. {issue}\n")
        f.write("\nIMPLICATIONS:\n")
        f.write("Response biases may limit confidence-accuracy calibration quality.\n")
        f.write("Consider:\n")
        f.write("  - Participant training on scale usage\n")
        f.write("  - Alternative confidence scales (e.g., 9-point, continuous slider)\n")
        f.write("  - Post-hoc adjustment for response styles\n")

    f.write("\n" + "-"*80 + "\n")
    f.write("DATA FILES\n")
    f.write("-"*80 + "\n\n")
    f.write(f"  - {data_path / 'response_patterns_by_participant.csv'} (participant-level)\n")
    f.write(f"  - {data_path / 'response_patterns_summary.csv'} (summary statistics)\n")

print(f"✅ Saved: {report_path}")

print("\n" + "="*80)
print("RESPONSE PATTERN ANALYSIS COMPLETE")
print("="*80)
print(f"\n✅ Full scale usage: {pct_full_scale:.1f}%")
print(f"⚠️ Extremes only: {pct_extremes_only:.1f}%")
print(f"✅ Mean rating SD: {mean_sd:.3f}")

if len(issues) > 0:
    print(f"\n⚠️ {len(issues)} issues detected:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✅ No response pattern issues detected")

print("\nFiles created:")
print(f"  - {data_path / 'response_patterns_by_participant.csv'}")
print(f"  - {data_path / 'response_patterns_summary.csv'}")
print(f"  - {report_path}")
print("\nNext: Update validation.md with all new checks")
