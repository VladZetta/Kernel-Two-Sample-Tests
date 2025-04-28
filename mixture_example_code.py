"""
Generate example plots for Gaussian Mixture Models with different component means.
This script produces a horizontal-format figure showing a 2-component Gaussian mixture
with distributions P and Q, where Q has one component with shifted mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)  # For reproducibility

# Define parameters
n_samples = 300
n_components = 2

# Base means for distribution P
means_p = np.array([
    [-2, 0],  # First component
    [2, 0]    # Second component
])

# Weights (equal for both distributions)
weights = np.array([0.5, 0.5])

# Covariance matrices (identity for both components)
cov = np.eye(2)

# Distribution P - sample from each component according to weights
indices_p = np.random.choice(n_components, size=n_samples, p=weights)
samples_p = np.vstack([
    np.random.multivariate_normal(means_p[i], cov) for i in indices_p
])

# Distribution Q - same as P except for the second component's mean
means_q = means_p.copy()
means_q[1] = [4, 0]  # Shifted mean for the second component

indices_q = np.random.choice(n_components, size=n_samples, p=weights)
samples_q = np.vstack([
    np.random.multivariate_normal(means_q[i], cov) for i in indices_q
])

# Create a more horizontal figure
plt.figure(figsize=(8, 4))

# Plot the samples
plt.scatter(samples_p[:, 0], samples_p[:, 1], alpha=0.6, label='Distribution P', s=15)
plt.scatter(samples_q[:, 0], samples_q[:, 1], alpha=0.6, label='Distribution Q (shifted component)', s=15, c='red')

# Add labels and legend
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Example: Component Means Difference\nGaussian Mixture (2 components)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust to make the plot more horizontal while keeping enough vertical space
plt.xlim(-6, 8)
plt.ylim(-3, 3)

# Save the figure
plt.savefig('mixture_example_means.png', dpi=300, bbox_inches='tight')
plt.show() 