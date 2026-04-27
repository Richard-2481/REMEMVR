#!/usr/bin/env python3
"""
RQ 6.4.2: Confidence Response Patterns Analysis
Section 1.4 requirement: Document confidence rating scale usage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.2")
DATA = BASE / "data"
PLOTS = BASE / "plots"
LOGS = BASE / "logs"

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / "step10_confidence_response_patterns.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== RQ 6.4.2: Confidence Response Patterns ===")
    logger.info("Section 1.4 requirement: Analyze confidence rating patterns")
    
    # Load raw confidence ratings from source RQ
    logger.info("\nLoading confidence ratings from Ch6 6.4.1...")
    
    conf_source = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.1/data/step00_irt_input.csv")
    
    if not conf_source.exists():
        logger.error(f"EXPECTATIONS ERROR: Source data missing: {conf_source}")
        return
    
    df = pd.read_csv(conf_source)
    logger.info(f"Loaded {len(df)} item-level ratings")
    
    # Check columns
    logger.info(f"Columns: {list(df.columns)}")
    
    # Identify confidence column (Response column in IRT input)
    if 'Response' in df.columns:
        conf_col = 'Response'
    elif 'Confidence' in df.columns:
        conf_col = 'Confidence'
    else:
        logger.error(f"Cannot find confidence column. Available: {list(df.columns)}")
        return
    
    # Convert to numeric
    df[conf_col] = pd.to_numeric(df[conf_col], errors='coerce')
    df = df.dropna(subset=[conf_col])
    
    logger.info(f"Confidence ratings: N={len(df)}")
    logger.info(f"Range: {df[conf_col].min():.0f} to {df[conf_col].max():.0f}")
    logger.info(f"Mean: {df[conf_col].mean():.2f}, SD: {df[conf_col].std():.2f}")
    
    # Overall distribution
    logger.info("\n=== Overall Rating Distribution ===")
    value_counts = df[conf_col].value_counts().sort_index()
    logger.info(value_counts)
    
    for val in sorted(df[conf_col].unique()):
        pct = (df[conf_col] == val).sum() / len(df) * 100
        logger.info(f"  {val}: {pct:.1f}%")
    
    # Participant-level patterns
    logger.info("\n=== Participant-Level Patterns ===")
    
    participant_stats = []
    
    for uid in df['UID'].unique():
        uid_ratings = df[df['UID'] == uid][conf_col]
        
        n_ratings = len(uid_ratings)
        unique_vals = uid_ratings.nunique()
        mean_rating = uid_ratings.mean()
        sd_rating = uid_ratings.std()
        
        # Full scale usage: all 5 values (1, 2, 3, 4, 5)
        full_scale = (unique_vals == 5)
        
        # Extremes only: only 1s and 5s
        extremes_only = all(r in [1, 5] for r in uid_ratings if not np.isnan(r))
        
        # Percent 1s and 5s
        pct_1 = (uid_ratings == 1).sum() / n_ratings * 100
        pct_5 = (uid_ratings == 5).sum() / n_ratings * 100
        pct_extremes = pct_1 + pct_5
        
        participant_stats.append({
            'UID': uid,
            'n_ratings': n_ratings,
            'unique_values': unique_vals,
            'mean_rating': mean_rating,
            'sd_rating': sd_rating,
            'full_scale_use': full_scale,
            'extremes_only': extremes_only,
            'pct_1s': pct_1,
            'pct_5s': pct_5,
            'pct_extremes': pct_extremes
        })
    
    stats_df = pd.DataFrame(participant_stats)
    
    # Summary statistics
    logger.info(f"\nParticipant-level summary (N={len(stats_df)} participants):")
    logger.info(f"  Full scale usage (all 5 values): {stats_df['full_scale_use'].sum()} ({stats_df['full_scale_use'].mean()*100:.1f}%)")
    logger.info(f"  Extremes only (1s and 5s): {stats_df['extremes_only'].sum()} ({stats_df['extremes_only'].mean()*100:.1f}%)")
    logger.info(f"  Mean rating SD: {stats_df['sd_rating'].mean():.2f} (range: {stats_df['sd_rating'].min():.2f} to {stats_df['sd_rating'].max():.2f})")
    logger.info(f"  Mean % extremes: {stats_df['pct_extremes'].mean():.1f}%")
    
    # Restricted range check (SD < 0.8)
    restricted_range = (stats_df['sd_rating'] < 0.8).sum()
    logger.info(f"  Restricted range (SD < 0.8): {restricted_range} ({restricted_range/len(stats_df)*100:.1f}%)")
    
    # Save results
    stats_df.to_csv(DATA / "step10_response_patterns_by_participant.csv", index=False)
    logger.info(f"\nSaved: {DATA / 'step10_response_patterns_by_participant.csv'}")
    
    # Create summary
    summary = {
        'metric': [
            'Total ratings',
            'N participants',
            'Rating mean',
            'Rating SD',
            '% Full scale use',
            '% Extremes only',
            'Mean participant SD',
            '% Restricted range (SD<0.8)',
            'Mean % extremes per participant'
        ],
        'value': [
            len(df),
            len(stats_df),
            df[conf_col].mean(),
            df[conf_col].std(),
            stats_df['full_scale_use'].mean() * 100,
            stats_df['extremes_only'].mean() * 100,
            stats_df['sd_rating'].mean(),
            (stats_df['sd_rating'] < 0.8).mean() * 100,
            stats_df['pct_extremes'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DATA / "step10_response_patterns_summary.csv", index=False)
    logger.info(f"Saved: {DATA / 'step10_response_patterns_summary.csv'}")
    
    # Visualize
    logger.info("\n=== Creating Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Overall distribution
    ax = axes[0, 0]
    value_counts = df[conf_col].value_counts().sort_index()
    ax.bar(value_counts.index, value_counts.values, edgecolor='black')
    ax.set_title("Overall Confidence Rating Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Confidence Rating (1=Low, 5=High)")
    ax.set_ylabel("Frequency")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Participant SD distribution
    ax = axes[0, 1]
    ax.hist(stats_df['sd_rating'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.8, color='r', linestyle='--', linewidth=2, label='Restricted range threshold')
    ax.set_title("Participant Rating Variability (SD)", fontsize=12, fontweight='bold')
    ax.set_xlabel("SD of Ratings per Participant")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Full scale vs Extremes
    ax = axes[1, 0]
    categories = ['Full Scale\n(All 5 values)', 'Extremes Only\n(1s and 5s)', 'Other']
    counts = [
        stats_df['full_scale_use'].sum(),
        stats_df['extremes_only'].sum(),
        len(stats_df) - stats_df['full_scale_use'].sum() - stats_df['extremes_only'].sum()
    ]
    ax.bar(categories, counts, edgecolor='black', color=['green', 'red', 'gray'], alpha=0.7)
    ax.set_title("Response Style Categories", fontsize=12, fontweight='bold')
    ax.set_ylabel("N Participants")
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, (cat, count) in enumerate(zip(categories, counts)):
        pct = count / len(stats_df) * 100
        ax.text(i, count + 2, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')
    
    # 4. % Extremes per participant
    ax = axes[1, 1]
    ax.hist(stats_df['pct_extremes'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=50, color='r', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_title("% Extreme Ratings (1s + 5s) per Participant", fontsize=12, fontweight='bold')
    ax.set_xlabel("% Extreme Ratings")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS / "confidence_response_patterns.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()
    
    logger.info("\n=== Confidence Response Patterns Complete ===")
    logger.info(f"Key finding: {stats_df['full_scale_use'].mean()*100:.1f}% use full scale, {stats_df['extremes_only'].mean()*100:.1f}% use extremes only")
    logger.info(f"Mean participant SD: {stats_df['sd_rating'].mean():.2f} ({'adequate' if stats_df['sd_rating'].mean() >= 0.8 else 'restricted range'})")

if __name__ == "__main__":
    main()
