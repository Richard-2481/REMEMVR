"""
Compute demographically-adjusted normative percentiles for RAVLT and RPM.

RAVLT: Stricker & Christianson (2021) regression-based T-scores
  - Uses raw → scaled score lookup (Tables A2/A3) then regression formulas
  - Adjusts for age, age², sex, and education

RPM Set I: Murphy et al. (2023) age-stratified percentile cut-offs
  - Uses Table 1b percentile bands by age cohort

Output: Per-participant T-scores/percentiles, then sample-level summary.
"""

import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# Education mapping: Australian categories → Stricker years
# =============================================================================
EDUCATION_MAP = {
    0: 9,   # Year 9 or lower
    1: 10,  # Year 10
    2: 12,  # Year 12
    3: 12,  # Certificate 1 & 2
    4: 13,  # Certificate 3 & 4
    5: 14,  # Diploma / Advanced Diploma
    6: 16,  # Bachelor's Degree
    7: 17,  # Graduate Certificate / Diploma
    8: 18,  # Master's Degree
    9: 20,  # Doctoral Degree
}

# =============================================================================
# Stricker Table A2: Raw score → Scaled score (primary variables)
# Format: SS: (Trials1-5_ranges, 30min_ranges, SumTrials_ranges, RecPC_ranges)
# We only need Trials 1-5 Total and 30-Min Recall for now
# =============================================================================

def raw_to_ss_trials15(raw):
    """Stricker Table A2: Trials 1-5 Total raw → scaled score."""
    lookup = [
        (0, 14, 0), (15, 17, 1), (18, 20, 2), (21, 23, 3), (24, 25, 4),
        (26, 27, 5), (28, 30, 6), (31, 33, 7), (34, 36, 8), (37, 40, 9),
        (41, 43, 10), (44, 47, 11), (48, 50, 12), (51, 54, 13), (55, 57, 14),
        (58, 61, 15), (62, 63, 16), (64, 65, 17), (66, 67, 18), (68, 68, 19),
        (69, 75, 20),
    ]
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    return np.nan


def raw_to_ss_30min(raw):
    """Stricker Table A2: 30-Min Delayed Recall raw → scaled score."""
    lookup = [
        (0, 0, 3), (1, 1, 4), (2, 2, 5), (3, 3, 6), (4, 4, 7),
        (5, 6, 8), (7, 7, 9), (8, 8, 10), (9, 9, 11), (10, 11, 12),
        (12, 12, 13), (13, 13, 14), (14, 14, 15), (15, 15, 17),
    ]
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    return np.nan


# Stricker Table A3: Secondary variables
def raw_to_ss_trial6(raw):
    """Trial 6 (short delay) raw → scaled score."""
    lookup = [
        (0, 0, 2), (1, 2, 4), (3, 3, 5), (4, 4, 6), (5, 5, 7),
        (6, 6, 8), (7, 7, 9), (8, 8, 10), (9, 10, 11), (11, 11, 12),
        (12, 12, 13), (13, 13, 14), (14, 14, 15), (15, 15, 16),
    ]
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    return np.nan


def raw_to_ss_ltpr(raw):
    """Long-term % retention raw → scaled score."""
    lookup = [
        (0, 0, 3), (7, 18, 4), (20, 33, 5), (36, 45, 6), (46, 56, 7),
        (57, 64, 8), (67, 71, 9), (73, 79, 10), (80, 86, 11),
        (87, 92, 12), (93, 93, 13), (100, 100, 15),
    ]
    # Handle gaps: find closest range
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    # Interpolate for values in gaps
    if raw < 0:
        return 3
    if raw > 100:
        return 15
    # Find nearest bracket
    for i, (lo, hi, ss) in enumerate(lookup):
        if i + 1 < len(lookup):
            next_lo = lookup[i + 1][0]
            if hi < raw < next_lo:
                return ss  # Use lower bracket
    return np.nan


def raw_to_ss_listb(raw):
    """List B raw → scaled score."""
    lookup = [
        (0, 0, 2), (1, 1, 4), (2, 2, 6), (3, 3, 7), (4, 4, 9),
        (5, 5, 11), (6, 6, 13), (7, 7, 14), (8, 8, 16), (9, 9, 17),
        (10, 10, 18), (11, 11, 19), (12, 15, 20),
    ]
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    return np.nan


def raw_to_ss_trial1(raw):
    """Trial 1 raw → scaled score."""
    lookup = [
        (0, 0, 0), (1, 1, 2), (2, 2, 3), (3, 3, 6), (4, 4, 8),
        (5, 5, 10), (6, 6, 12), (7, 7, 14), (8, 8, 15), (9, 9, 16),
        (10, 10, 18), (11, 11, 19), (12, 15, 20),
    ]
    for lo, hi, ss in lookup:
        if lo <= raw <= hi:
            return ss
    return np.nan


# =============================================================================
# Stricker T-score formulas
# =============================================================================

def tscore_trials15(ss, age, male, educ):
    """Fully adjusted T-score for Trials 1-5 Total."""
    predicted = (10.2048820335 + age * 0.0696731708 + male * -2.0691847063
                 + educ * 0.2076286782 + age**2 * -0.0014410120)
    residual = (ss - predicted) / 1
    t = round(50 + ((residual + 0.0000000637336) / 0.23569807))
    return t


def tscore_30min(ss, age, male, educ):
    """Fully adjusted T-score for 30-Min Delayed Recall."""
    predicted = (12.4118437425 + age * -0.0016432817 + male * -1.8612455591
                 + educ * 0.1380628944 + age**2 * -0.0007027918)
    residual = (ss - predicted) / 1
    t = round(50 + ((residual + 0.0000001024411) / 0.25299505))
    return t


def tscore_trial6(ss, age, male, educ):
    """Fully adjusted T-score for Trial 6 (short delay)."""
    predicted = (11.7981182251 + age * 0.0154689603 + age**2 * -0.0008517340
                 + male * -1.6396477808 + educ * 0.1436500033)
    residual = (ss - predicted) / 1
    t = round(50 + ((residual + 0.0000000285607) / 0.25404381))
    return t


def tscore_ltpr(ss, age, male, educ):
    """Fully adjusted T-score for Long-Term % Retention."""
    predicted = (13.3565938006 + age * -0.0448177870 + age**2 * -0.0000815093
                 + male * -1.3287251835 + educ * 0.0567713694)
    sd_adj = 2.0060197485 + age**2 * 0.0000707217
    residual = (ss - predicted) / sd_adj
    t = round(50 + ((residual + 0.00000042085768) / 0.12427314))
    return t


def tscore_listb(ss, age, male, educ):
    """Fully adjusted T-score for List B."""
    predicted = (8.9167820377 + age * 0.0780069203 + age**2 * -0.0013677187
                 + male * -1.1375184278 + educ * 0.1914262059)
    sd_adj = 2.3996448266 + age**2 * -0.0000533322
    residual = (ss - predicted) / sd_adj
    t = round(50 + ((residual + 0.00000967380900) / 0.12287159))
    return t


def tscore_trial1(ss, age, male, educ):
    """Fully adjusted T-score for Trial 1."""
    predicted = (10.5554207904 + age * 0.0361599800 + age**2 * -0.0009181852
                 + male * -1.2432854518 + educ * 0.1518778446)
    residual = (ss - predicted) / 1
    t = round(50 + ((residual - 0.0000001305326) / 0.26867547))
    return t


def tscore_to_percentile(t):
    """Convert T-score (mean=50, SD=10) to percentile."""
    return stats.norm.cdf((t - 50) / 10) * 100


# =============================================================================
# Murphy et al. (2023) RPM Set I percentile mapping
# =============================================================================

def rpm_percentile(score, age):
    """Map RPM Set I score to approximate percentile using Murphy Table 1b."""
    # Determine age cohort
    if age < 35:
        cohort = '18-34'
    elif age < 50:
        cohort = '35-49'
    elif age < 65:
        cohort = '50-64'
    elif age < 80:
        cohort = '65-79'
    else:
        cohort = '80-89'

    # Percentile cut-offs from Table 1b (score at each percentile)
    cutoffs = {
        '18-34': [(1, 6), (5, 7), (10, 8), (25, 9), (50, 10), (75, 11), (90, 12)],
        '35-49': [(1, 1), (5, 6), (10, 7), (25, 8), (50, 9), (75, 11), (90, 12)],
        '50-64': [(1, 1), (5, 5), (10, 5), (25, 8), (50, 9), (75, 10), (90, 11), (95, 12)],
        '65-79': [(1, 1), (5, 3), (10, 4), (25, 6), (50, 8), (75, 9), (90, 11), (95, 12)],
        '80-89': [(5, 2), (10, 3), (25, 5), (50, 7), (75, 9), (90, 10), (95, 11)],
    }

    bands = cutoffs[cohort]
    # Find where this score falls
    for i, (pct, cutoff_score) in enumerate(bands):
        if score <= cutoff_score:
            if i == 0:
                return pct / 2  # Below lowest cutoff
            prev_pct = bands[i - 1][0]
            prev_score = bands[i - 1][1]
            if score <= prev_score:
                continue
            # Interpolate between percentile bands
            return prev_pct + (pct - prev_pct) * (score - prev_score) / (cutoff_score - prev_score)
    # Above highest cutoff
    return (bands[-1][0] + 100) / 2


# =============================================================================
# Benedict et al. (1996) BVMT-R age-group norms
# =============================================================================

# Table 1: M, SD by age group for each measure
BVMT_NORMS = {
    # age_group: (total_recall_M, total_recall_SD, delayed_M, delayed_SD, pct_retained_M, pct_retained_SD)
    '18-30': (28.2, 4.1, 10.6, 1.4, 96, 8),
    '31-54': (25.3, 5.1, 9.5, 1.9, 93, 10),
    '55-69': (22.2, 5.5, 8.4, 2.1, 91, 12),
    '70-88': (19.9, 6.3, 7.3, 2.9, 85, 16),
}


def bvmt_age_group(age):
    """Assign BVMT-R age group per Benedict Table 1."""
    if age <= 30:
        return '18-30'
    elif age <= 54:
        return '31-54'
    elif age <= 69:
        return '55-69'
    else:
        return '70-88'


def bvmt_percentile(score, age, measure_idx):
    """Compute percentile for BVMT-R score using Benedict norms.
    measure_idx: 0=total_recall, 1=delayed, 2=pct_retained
    """
    group = bvmt_age_group(age)
    norm_m = BVMT_NORMS[group][measure_idx * 2]
    norm_sd = BVMT_NORMS[group][measure_idx * 2 + 1]
    z = (score - norm_m) / norm_sd
    return stats.norm.cdf(z) * 100


# =============================================================================
# Main computation
# =============================================================================

def main():
    df = pd.read_csv('data/dfData.csv')

    # Get one row per participant (session 1)
    demo = df.groupby('UID').first().reset_index()

    # Map education
    demo['educ_years'] = demo['education'].map(EDUCATION_MAP)
    demo['male'] = demo['sex'].astype(int)  # 0=F, 1=M

    # Compute RAVLT derived scores
    demo['ravlt_trials15'] = (demo['ravlt-trial-1-score'] + demo['ravlt-trial-2-score']
                              + demo['ravlt-trial-3-score'] + demo['ravlt-trial-4-score']
                              + demo['ravlt-trial-5-score'])
    demo['ravlt_delayed'] = demo['ravlt-delayed-recall-score']
    demo['ravlt_trial6'] = demo['ravlt-free-recall-score']  # Trial 6 = post-interference recall
    demo['ravlt_trial1'] = demo['ravlt-trial-1-score']
    demo['ravlt_listb'] = demo['ravlt-distraction-trial-score']

    # Long-term % retention = 100 * (30-min delay / trial 5)
    demo['ravlt_ltpr'] = 100 * demo['ravlt_delayed'] / demo['ravlt-trial-5-score']
    # Exclude participants with trial 5 = 0 (N=4) from LTPR computation
    demo.loc[demo['ravlt-trial-5-score'] == 0, 'ravlt_ltpr'] = np.nan

    # Convert raw → scaled scores
    demo['ss_trials15'] = demo['ravlt_trials15'].apply(raw_to_ss_trials15)
    demo['ss_30min'] = demo['ravlt_delayed'].apply(raw_to_ss_30min)
    demo['ss_trial6'] = demo['ravlt_trial6'].apply(raw_to_ss_trial6)
    demo['ss_ltpr'] = demo['ravlt_ltpr'].apply(
        lambda x: raw_to_ss_ltpr(int(round(x))) if pd.notna(x) else np.nan)
    demo['ss_listb'] = demo['ravlt_listb'].apply(raw_to_ss_listb)
    demo['ss_trial1'] = demo['ravlt_trial1'].apply(raw_to_ss_trial1)

    # Compute T-scores
    demo['t_trials15'] = demo.apply(
        lambda r: tscore_trials15(r['ss_trials15'], r['age'], r['male'], r['educ_years']), axis=1)
    demo['t_30min'] = demo.apply(
        lambda r: tscore_30min(r['ss_30min'], r['age'], r['male'], r['educ_years']), axis=1)
    demo['t_trial6'] = demo.apply(
        lambda r: tscore_trial6(r['ss_trial6'], r['age'], r['male'], r['educ_years']), axis=1)
    demo['t_ltpr'] = demo.apply(
        lambda r: tscore_ltpr(r['ss_ltpr'], r['age'], r['male'], r['educ_years'])
        if pd.notna(r['ss_ltpr']) else np.nan, axis=1)
    demo['t_listb'] = demo.apply(
        lambda r: tscore_listb(r['ss_listb'], r['age'], r['male'], r['educ_years']), axis=1)
    demo['t_trial1'] = demo.apply(
        lambda r: tscore_trial1(r['ss_trial1'], r['age'], r['male'], r['educ_years']), axis=1)

    # Convert T-scores → percentiles
    for var in ['trials15', '30min', 'trial6', 'ltpr', 'listb', 'trial1']:
        demo[f'pct_{var}'] = demo[f't_{var}'].apply(tscore_to_percentile)

    # RPM percentile
    demo['rpm_score'] = demo['rpm-score']
    demo['pct_rpm'] = demo.apply(
        lambda r: rpm_percentile(r['rpm_score'], r['age']), axis=1)

    # BVMT-R percentiles (Benedict et al., 1996 age-group norms)
    demo['bvmt_total'] = demo['bvmt-total-recall']
    demo['bvmt_delayed'] = demo['bvmt-delayed-recall-score']
    demo['bvmt_pctret'] = demo['bvmt-percent-retained']

    demo['pct_bvmt_total'] = demo.apply(
        lambda r: bvmt_percentile(r['bvmt_total'], r['age'], 0), axis=1)
    demo['pct_bvmt_delayed'] = demo.apply(
        lambda r: bvmt_percentile(r['bvmt_delayed'], r['age'], 1), axis=1)
    demo['pct_bvmt_pctret'] = demo.apply(
        lambda r: bvmt_percentile(r['bvmt_pctret'], r['age'], 2), axis=1)

    # =================================================================
    # Summary output
    # =================================================================
    print("=" * 70)
    print("NORMATIVE PERCENTILE COMPUTATIONS")
    print("=" * 70)

    measures = [
        ('RAVLT Total Learning (Trials 1-5)', 'ravlt_trials15', 't_trials15', 'pct_trials15'),
        ('RAVLT Delayed Recall', 'ravlt_delayed', 't_30min', 'pct_30min'),
        ('RAVLT Long-Term % Retention', 'ravlt_ltpr', 't_ltpr', 'pct_ltpr'),
        ('BVMT-R Total Recall (Trials 1-3)', 'bvmt_total', None, 'pct_bvmt_total'),
        ('BVMT-R Delayed Recall', 'bvmt_delayed', None, 'pct_bvmt_delayed'),
        ('BVMT-R Percent Retained', 'bvmt_pctret', None, 'pct_bvmt_pctret'),
        ('RPM Set I', 'rpm_score', None, 'pct_rpm'),
    ]

    print(f"\n{'Measure':<40} {'Raw M(SD)':<16} {'Mean T':<10} {'Mean %ile':<10} {'Median %ile'}")
    print("-" * 90)

    for label, raw_col, t_col, pct_col in measures:
        raw_m = demo[raw_col].mean()
        raw_sd = demo[raw_col].std()
        pct_mean = demo[pct_col].mean()
        pct_median = demo[pct_col].median()

        if t_col:
            t_mean = demo[t_col].mean()
            print(f"{label:<40} {raw_m:>5.1f} ({raw_sd:>4.1f})   T={t_mean:>5.1f}    {pct_mean:>5.1f}      {pct_median:>5.1f}")
        else:
            print(f"{label:<40} {raw_m:>5.1f} ({raw_sd:>4.1f})   --        {pct_mean:>5.1f}      {pct_median:>5.1f}")

    # Verify against existing table values
    print("\n" + "=" * 70)
    print("VERIFICATION against Table 2.2")
    print("=" * 70)
    print(f"RAVLT Total Learning: reported 46th %ile, computed {demo['pct_trials15'].mean():.1f}th (mean)")
    print(f"  Raw M = {demo['ravlt_trials15'].mean():.2f} (table: 50.60)")
    print(f"  Raw SD = {demo['ravlt_trials15'].std():.2f} (table: 8.42)")

    # Per-participant data for inspection
    print("\n" + "=" * 70)
    print("PER-PARTICIPANT T-SCORES (first 10)")
    print("=" * 70)
    cols = ['UID', 'age', 'male', 'educ_years', 't_trials15', 'pct_trials15', 't_30min', 'pct_30min', 'pct_rpm']
    print(demo[cols].head(10).to_string(index=False))

    # T-score distribution check
    print("\n" + "=" * 70)
    print("T-SCORE DISTRIBUTIONS (should be approx M=50, SD=10 if sample matches norms)")
    print("=" * 70)
    for var, label in [('t_trials15', 'Trials 1-5'), ('t_30min', '30-Min Recall'),
                        ('t_trial6', 'Trial 6'), ('t_ltpr', 'LT % Retention'),
                        ('t_listb', 'List B'), ('t_trial1', 'Trial 1')]:
        t = demo[var]
        print(f"  {label:<20} M={t.mean():>5.1f}  SD={t.std():>5.1f}  Range=[{t.min():.0f}, {t.max():.0f}]")


if __name__ == '__main__':
    main()
