import numpy as np
import matplotlib.pyplot as plt
# Assume plotting style is set up (e.g., using seaborn)

np.random.seed(42)
n_samples = 200
n_components = 2 # Keeping it 2 components as requested previously

# Define means for 2 components in distribution P
means_p = np.array([[-2, 0], [2, 0]])

# Define means for 2 components in distribution Q (second component is shifted)
means_q = np.array([[-2, 0], [4, 0]])  # Shifted second component

# Define covariance (same for both distributions)
cov = np.identity(2)

# Define weights
weights = np.array([0.5, 0.5])

# Sample Distribution P
p_indices = np.random.choice(n_components, size=n_samples, p=weights)
dist_p = np.vstack([np.random.multivariate_normal(means_p[i], cov) for i in p_indices])

# Sample Distribution Q (with shifted second component)
q_indices = np.random.choice(n_components, size=n_samples, p=weights)
dist_q = np.vstack([np.random.multivariate_normal(means_q[i], cov) for i in q_indices])

# Plotting
plt.figure(figsize=(8, 4)) # Horizontal format

plt.scatter(dist_p[:, 0], dist_p[:, 1], label='Distribution P', alpha=0.7, s=15)
plt.scatter(dist_q[:, 0], dist_q[:, 1], label='Distribution Q (shifted component)', alpha=0.7, s=15, c='red')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Example: Component Means Difference\nGaussian Mixture (2 components)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Set axis limits to make the plot more horizontal
plt.xlim(-6, 8)
plt.ylim(-3, 3)

# Save the new figure
plt.savefig('figs/mixture_example_means.png', dpi=300, bbox_inches='tight')
plt.show()