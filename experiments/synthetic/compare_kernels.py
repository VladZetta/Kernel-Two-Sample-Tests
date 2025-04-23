import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.mmd import compute_mmd_stat
from src.permutation_test import permutation_test_statistic

# -----------------------------
# Generate synthetic data with mean shift
# -----------------------------
def generate_data(mean_shift=0.0, n=100, d=10):
    P = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
    Q = np.random.normal(loc=mean_shift, scale=1.0, size=(n, d))
    return P, Q

# -----------------------------
# Estimate test power across deltas
# -----------------------------
def estimate_power(kernel, delta_values, num_trials=100, n=100, d=10, alpha=0.05, num_permutations=300):
    powers = []
    for delta in delta_values:
        rejections = 0
        for _ in range(num_trials):
            X, Y = generate_data(mean_shift=delta, n=n, d=d)
            stat_fn = lambda X, Y: compute_mmd_stat(X, Y, kernel=kernel)
            p_val = permutation_test_statistic(X, Y, stat_fn=stat_fn, num_permutations=num_permutations)
            if p_val < alpha:
                rejections += 1
        power = rejections / num_trials
        powers.append(power)
        print(f"Delta = {delta:.2f} | Kernel = {kernel} | Power = {power:.2f}")
    return powers

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Experiment settings
    np.random.seed(42)
    delta_values = np.linspace(0.0, 1.0, 11)  # More fine-grained deltas: 0.0, 0.1, ..., 1.0
    kernels = ['rbf', 'poly']
    results = {}
    num_trials = 100
    n = 100
    d = 10

    # Estimate power for each kernel
    for kernel in kernels:
        print(f"\nRunning power estimation for kernel: {kernel}")
        results[kernel] = estimate_power(kernel, delta_values, num_trials=num_trials, n=n, d=d)

    # Plotting
    plt.figure(figsize=(8, 5))
    markers = {'rbf': 'o', 'poly': 's'}

    for kernel in kernels:
        plt.plot(delta_values, results[kernel], marker=markers[kernel], label=f'{kernel} kernel')

    plt.xlabel("Mean shift (δ)")
    plt.ylabel("Empirical Power")
    plt.title("MMD Test Power vs Mean Shift")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    figs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    plot_path = os.path.join(figs_dir, 'kernel_power_comparison2.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Power curve plot saved to: {plot_path}")

    plt.show()
