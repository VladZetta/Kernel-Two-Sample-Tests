"""
Generate a plot showing MMD test power vs. difference magnitude for two types of differences:
1. Component means difference
2. Weights difference

This recreates Figure (h) without the variance difference line.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# Set the style for a scientific publication
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.6

# Data for the plot (based on the image)
# X-axis: Difference Magnitude
difference_magnitudes = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# Y-axis: Test Power
# For component means difference (blue line)
component_means_power = np.array([0.1, 0.4, 0.45, 0.5, 0.55, 0.55, 0.55, 0.6, 0.55])
# Confidence intervals for component means
component_means_ci = np.array([0.15, 0.2, 0.2, 0.15, 0.3, 0.35, 0.3, 0.2, 0.25])

# For weights difference (orange line)
# Almost perfect power after small difference
weights_power = np.array([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# Very narrow confidence intervals for weights
weights_ci = np.array([0.15, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

# Create a horizontal figure
plt.figure(figsize=(8, 5))

# Plot the lines with confidence intervals
plt.plot(difference_magnitudes, component_means_power, 'o-', label='Components difference', color='#1f77b4')
plt.fill_between(difference_magnitudes, 
                 component_means_power - component_means_ci, 
                 component_means_power + component_means_ci,
                 alpha=0.2, color='#1f77b4')

plt.plot(difference_magnitudes, weights_power, 'o-', label='Weights difference', color='#ff7f0e')
plt.fill_between(difference_magnitudes, 
                 weights_power - weights_ci, 
                 np.minimum(weights_power + weights_ci, 1.0),  # Cap at 1.0
                 alpha=0.2, color='#ff7f0e')

# Set axis limits and labels
plt.xlim(0, 2.05)
plt.ylim(0, 1.05)
plt.xlabel('Difference Magnitude')
plt.ylabel('Test Power')
plt.title('MMD Test Power by Difference Type\n(mean of 3 runs with 95% CI)')

# Add legend
plt.legend(loc='lower right')

# Save the figure
plt.savefig('power_vs_difference_type.png', dpi=300, bbox_inches='tight')
plt.show() 