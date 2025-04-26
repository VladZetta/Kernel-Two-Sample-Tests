import sys
import os

# Setup path to import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.mmd import compute_mmd_stat
from src.permutation_test import permutation_test_statistic

# -----------------------------
# Helper: Generate Gaussian data with mean shift
# -----------------------------
def generate_data(mean_shift=0.0, n=100, d=10):
    P = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
    Q = np.random.normal(loc=mean_shift, scale=1.0, size=(n, d))
    return P, Q

# -----------------------------
# Helper: Estimate empirical power
# -----------------------------
def estimate_power(kernel, delta_values, num_trials=30, n=100, d=10, alpha=0.05, num_permutations=300):
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
        print(f"Kernel: {kernel} | Delta: {delta:.2f} | Power: {power:.2f}")
    return powers

# -----------------------------
# Main function
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Experiment settings
    delta_values = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    kernels = ['rbf', 'laplacian', 'matern', 'poly2', 'poly3']
    num_trials = 30
    n = 100
    d = 10
    alpha = 0.05

    results = {}

    for kernel in kernels:
        print(f"\nEstimating power for kernel: {kernel}")
        results[kernel] = estimate_power(kernel, delta_values,
                                         num_trials=num_trials,
                                         n=n, d=d,
                                         alpha=alpha,
                                         num_permutations=300)

    # Plotting
    plt.figure(figsize=(8, 5))
    markers = {'rbf': 'o', 'laplacian': 's', 'matern': '^', 'poly2': 'D', 'poly3': 'P'}

    for kernel in kernels:
        plt.plot(delta_values, results[kernel], marker=markers.get(kernel, 'o'), label=kernel)

    plt.xlabel("Mean Shift ($\\delta$)")
    plt.ylabel("Empirical Power")
    plt.title(f"MMD Test Power vs Mean Shift (n={n}, d={d})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save plot
    figs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    plot_path = os.path.join(figs_dir, 'power_curve_all_kernels.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Power curve saved at {plot_path}")

    plt.show()
