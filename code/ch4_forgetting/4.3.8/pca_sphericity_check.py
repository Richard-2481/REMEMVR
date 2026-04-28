"""
PCA Sphericity Check for RQ 5.3.8 Clustering
Purpose: Quantify variance explained by principal components to validate K-means sphericity assumption
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load standardized features (6 z-scores from Step 1)
features = pd.read_csv('/home/etai/projects/REMEMVR/results/ch5/5.3.8/data/step01_standardized_features.csv')

# Extract 6 feature columns (exclude UID)
feature_cols = [
    'Total_Intercept_Free_z', 'Total_Slope_Free_z',
    'Total_Intercept_Cued_z', 'Total_Slope_Cued_z',
    'Total_Intercept_Recognition_z', 'Total_Slope_Recognition_z'
]

X = features[feature_cols].values

print("="*60)
print("PCA SPHERICITY CHECK: RQ 5.3.8")
print("="*60)
print(f"Input: {X.shape[0]} participants x {X.shape[1]} features")
print()

# Fit PCA
pca = PCA()
pca.fit(X)

# Extract variance explained
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS:")
print("-"*60)
for i in range(len(variance_explained)):
    print(f"PC{i+1}: {variance_explained[i]:.3f} ({variance_explained[i]*100:.1f}%)")
    print(f"     Cumulative: {cumulative_variance[i]:.3f} ({cumulative_variance[i]*100:.1f}%)")
print()

# Check sphericity criterion from plan.md
pc1_variance = variance_explained[0]
print("SPHERICITY ASSESSMENT:")
print("-"*60)
print(f"PC1 variance explained: {pc1_variance:.3f} ({pc1_variance*100:.1f}%)")
print(f"Threshold (from plan.md): 0.70 (70%)")
print()

if pc1_variance > 0.70:
    print("⚠️  SPHERICITY VIOLATED: PC1 explains >70% variance")
    print("    → Data has dominant axis, K-means may be suboptimal")
    print("    → GMM with unconstrained covariance recommended")
else:
    print("✅ SPHERICITY MET: PC1 explains <70% variance")
    print("    → No dominant axis, K-means appropriate")
print()

# Additional insights
pc2_variance = variance_explained[1]
pc3_variance = variance_explained[2]

print("DIMENSIONALITY ASSESSMENT:")
print("-"*60)
print(f"PC1+PC2: {cumulative_variance[1]:.3f} ({cumulative_variance[1]*100:.1f}%)")
print(f"PC1+PC2+PC3: {cumulative_variance[2]:.3f} ({cumulative_variance[2]*100:.1f}%)")
print()

# Interpretation
if cumulative_variance[0] > 0.70:
    print("→ Clustering driven by single dominant dimension")
elif cumulative_variance[1] > 0.70:
    print("→ Clustering driven by 2 dominant dimensions")
elif cumulative_variance[2] > 0.70:
    print("→ Clustering driven by 3 dominant dimensions")
else:
    print("→ Clustering requires 4+ dimensions (complex structure)")
print()

# Save results
output_df = pd.DataFrame({
    'component': [f'PC{i+1}' for i in range(len(variance_explained))],
    'variance_explained': variance_explained,
    'cumulative_variance': cumulative_variance
})

output_path = '/home/etai/projects/REMEMVR/results/ch5/5.3.8/data/pca_sphericity_results.csv'
output_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print()

# Create scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(variance_explained)+1), variance_explained, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.70, color='r', linestyle='--', label='Sphericity threshold (70%)')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('PCA Scree Plot: K-means Sphericity Check')
plt.xticks(range(1, len(variance_explained)+1))
plt.ylim([0, max(variance_explained)*1.1])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = '/home/etai/projects/REMEMVR/results/ch5/5.3.8/plots/pca_scree_plot.png'
plt.savefig(plot_path, dpi=300)
print(f"Scree plot saved to: {plot_path}")
print()

print("="*60)
print("PCA CHECK COMPLETE")
print("="*60)
