import os
import sys
# add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.mmd import mmd_test
from generate_data import generate_gaussian_shift


def run_power_vs_delta(n, d, deltas, num_trials, alpha, num_permutations):
    """
    Compute empirical power of the MMD test over a list of mean shifts (deltas) for fixed n, d.
    """
    powers = []
    print(f"Computing power vs delta (n={n}, d={d})")
    for delta in tqdm(deltas, desc=f"Testing deltas (n={n})"):
        rejects = 0
        for trial in tqdm(range(num_trials), desc=f"Trials for delta={delta:.2f}", leave=False):
            X, Y = generate_gaussian_shift(n, d, delta)
            _, p_val = mmd_test(X, Y, return_p_value=True, num_permutations=num_permutations)
            if p_val < alpha:
                rejects += 1
        power = rejects / num_trials
        powers.append(power)
        print(f"  Delta={delta:.2f}: Power={power:.4f} ({rejects}/{num_trials} rejections)")
    return powers


def run_power_vs_n(delta, d, sample_sizes, num_trials, alpha, num_permutations):
    """
    Compute empirical power of the MMD test over a list of sample sizes for fixed delta, d.
    """
    powers = []
    print(f"Computing power vs sample size (delta={delta}, d={d})")
    for n in tqdm(sample_sizes, desc=f"Testing sample sizes (delta={delta})"):
        rejects = 0
        for trial in tqdm(range(num_trials), desc=f"Trials for n={n}", leave=False):
            X, Y = generate_gaussian_shift(n, d, delta)
            _, p_val = mmd_test(X, Y, return_p_value=True, num_permutations=num_permutations)
            if p_val < alpha:
                rejects += 1
        power = rejects / num_trials
        powers.append(power)
        print(f"  n={n}: Power={power:.4f} ({rejects}/{num_trials} rejections)")
    return powers


def main():
    sns.set(style='whitegrid')
    os.makedirs(os.path.join(project_root, 'figs'), exist_ok=True)

    # Experiment settings
    np.random.seed(0)
    alpha = 0.05
    num_permutations = 200
    num_trials = 100
    
    print("Starting synthetic experiments with settings:")
    print(f"  Alpha: {alpha}")
    print(f"  Permutations per test: {num_permutations}")
    print(f"  Trials per condition: {num_trials}")

    # 1) Power vs Delta for d=1, various n
    print("\n=== Experiment 1: Power vs Delta (d=1) ===")
    deltas = np.linspace(0, 1.0, 11)
    sample_sizes = [50, 100, 200]
    results_delta = {}
    for n in sample_sizes:
        results_delta[n] = run_power_vs_delta(n, d=1, deltas=deltas,
                                               num_trials=num_trials,
                                               alpha=alpha,
                                               num_permutations=num_permutations)
    plt.figure(figsize=(8, 5))
    for n, powers in results_delta.items():
        plt.plot(deltas, powers, marker='o', label=f'n={n}')
    plt.xlabel('Mean shift \u03B4')
    plt.ylabel('Empirical Power')
    plt.title('MMD Test Power vs Mean Shift (d=1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'power_vs_delta_d1.png'), dpi=600)
    print(f"Figure saved: {os.path.join('figs', 'power_vs_delta_d1.png')}")
    plt.close()

    # 2) Power vs Sample Size for d=1, various delta
    print("\n=== Experiment 2: Power vs Sample Size (d=1) ===")
    sample_sizes2 = [20, 50, 100, 200]
    deltas2 = [0.2, 0.5, 1.0]
    results_n = {}
    for delta in deltas2:
        results_n[delta] = run_power_vs_n(delta, d=1, sample_sizes=sample_sizes2,
                                         num_trials=num_trials,
                                         alpha=alpha,
                                         num_permutations=num_permutations)
    plt.figure(figsize=(8, 5))
    for delta, powers in results_n.items():
        plt.plot(sample_sizes2, powers, marker='o', label=f'\u03B4={delta}')
    plt.xlabel('Sample Size n')
    plt.ylabel('Empirical Power')
    plt.title('MMD Test Power vs Sample Size (d=1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'power_vs_n_d1.png'), dpi=600)
    print(f"Figure saved: {os.path.join('figs', 'power_vs_n_d1.png')}")
    plt.close()

    # 3) Power vs Dimension for fixed n and delta
    print("\n=== Experiment 3: Power vs Dimension ===")
    dims = [1, 2, 3, 5, 8, 10, 15, 20]
    n3 = 100
    delta3 = 0.5
    powers_dim = []
    print(f"Computing power vs dimension (n={n3}, delta={delta3})")
    for d in tqdm(dims, desc="Testing dimensions"):
        pwr = run_power_vs_delta(n3, d=d, deltas=[delta3],
                                  num_trials=num_trials,
                                  alpha=alpha,
                                  num_permutations=num_permutations)
        powers_dim.append(pwr[0])
        print(f"  d={d}: Power={pwr[0]:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(dims, powers_dim, marker='o')
    plt.xlabel('Dimension d')
    plt.ylabel('Empirical Power')
    plt.title(f'MMD Test Power vs Dimension (n={n3}, \u03B4={delta3})')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'power_vs_dimension.png'), dpi=600)
    print(f"Figure saved: {os.path.join('figs', 'power_vs_dimension.png')}")
    plt.close()

    print('\nSynthetic experiments complete. All figures saved to figs/')


if __name__ == '__main__':
    main()
