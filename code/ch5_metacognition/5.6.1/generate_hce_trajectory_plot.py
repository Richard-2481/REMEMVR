#!/usr/bin/env python3
"""
Generate HCE Trajectory Plot for RQ 6.6.1

Purpose: Visualize high-confidence error rate decline over 6-day retention interval
Input: data/step04_hce_trajectory_data.csv (4 timepoints with 95% CIs)
Output: plots/hce_trajectory.png (publication-quality 300 DPI)

Date: 2025-12-27
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set paths
DATA_FILE = '/home/etai/projects/REMEMVR/results/ch6/6.6.1/data/step04_hce_trajectory_data.csv'
OUTPUT_DIR = '/home/etai/projects/REMEMVR/results/ch6/6.6.1/plots'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'hce_trajectory.png')

# Create plots directory if missing
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trajectory data
print(f"Loading trajectory data from {DATA_FILE}...")
data = pd.read_csv(DATA_FILE)

print(f"Data loaded: {len(data)} timepoints")
print(data)

# Extract columns
time_hours = data['time'].values  # Hours since encoding
hce_rate = data['HCE_rate_mean'].values * 100  # Convert to percentage
ci_lower = data['CI_lower'].values * 100
ci_upper = data['CI_upper'].values * 100
test_labels = [f"T{i}" for i in data['test'].values]

# Convert hours to days for plotting
time_days = time_hours / 24

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot trajectory line
ax.plot(time_days, hce_rate, 'o-', color='#d62728', linewidth=2.5,
        markersize=10, label='HCE Rate', zorder=3)

# Plot confidence interval as shaded region
ax.fill_between(time_days, ci_lower, ci_upper,
                alpha=0.25, color='#d62728', label='95% CI')

# Annotate timepoints with test labels
for i, (t, h, label) in enumerate(zip(time_days, hce_rate, test_labels)):
    ax.annotate(label, xy=(t, h), xytext=(0, 10),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold')

# Annotate with statistical finding
ax.text(0.98, 0.97,
        r'$\beta$ = -0.003, p < .001' + '\n' +
        'Days coefficient (REML LMM)\n' +
        '35% decline (4.87% → 3.17%)',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Set axis labels
ax.set_xlabel('Time Since Encoding (Days)', fontsize=14, fontweight='bold')
ax.set_ylabel('High-Confidence Error Rate (%)', fontsize=14, fontweight='bold')

# Set title
ax.set_title('Metacognitive Recalibration: HCE Rate Decreases Over Time\n' +
             'RQ 6.6.1 - High-Confidence Errors (Confidence ≥ 0.75, Accuracy = 0)',
             fontsize=15, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(-0.5, 7)
ax.set_ylim(0, 7)

# Grid for readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save at 300 DPI (publication quality)
print(f"Saving plot to {OUTPUT_FILE}...")
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved successfully at 300 DPI")

# Also save as PDF for thesis inclusion
pdf_file = OUTPUT_FILE.replace('.png', '.pdf')
plt.savefig(pdf_file, bbox_inches='tight')
print(f"✓ PDF version saved: {pdf_file}")

plt.close()

print("\n" + "="*60)
print("HCE TRAJECTORY PLOT GENERATED")
print("="*60)
print(f"Output: {OUTPUT_FILE}")
print(f"Format: PNG (300 DPI) + PDF")
print(f"Pattern: Monotonic decline from Day 0 to Day 6")
print(f"Finding: Metacognitive recalibration (confidence adjusts to memory quality)")
print("="*60)
