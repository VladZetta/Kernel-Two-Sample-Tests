import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.mmd import mmd_test, median_heuristic
from experiments.synthetic.generate_data import generate_gaussian_shift

def run_bandwidth_experiment(n, d, delta, bandwidth_multiples, num_trials, alpha, num_permutations):
    """
    Test MMD power across different bandwidth values (as multiples of median heuristic).
    """
    powers = []
    print(f"Computing power vs bandwidth (n={n}, d={d}, delta={delta})")
    
    for bw_multiple in tqdm(bandwidth_multiples, desc="Testing bandwidth multiples"):
        rejects = 0
        for trial in tqdm(range(num_trials), desc=f"Trials for bw={bw_multiple:.2f}×", leave=False):
            # Generate data
            X, Y = generate_gaussian_shift(n, d, delta)
            
            # Compute baseline median bandwidth
            Z = np.vstack([X, Y])
            median_bw = median_heuristic(X, Y)
            
            # Use the multiple of median bandwidth
            bandwidth = bw_multiple * median_bw
            
            # Run MMD test with custom bandwidth
            _, p_val = mmd_test(X, Y, kernel='rbf', bandwidth=bandwidth, 
                              return_p_value=True, num_permutations=num_permutations)
            
            if p_val < alpha:
                rejects += 1
                
        power = rejects / num_trials
        powers.append(power)
        print(f"  Bandwidth={bw_multiple:.2f}× median: Power={power:.4f} ({rejects}/{num_trials} rejections)")
    
    return powers

def main():
    sns.set(style='whitegrid')
    os.makedirs(os.path.join(project_root, 'figs'), exist_ok=True)

    # Experiment settings
    np.random.seed(42)
    alpha = 0.05
    num_permutations = 200
    num_trials = 50  # Reduced from 100 to speed up experimentation
    
    # Fixed parameters for bandwidth experiment
    n = 100
    d = 2
    delta = 0.3  # Choose a moderate effect size where bandwidth will matter
    
    # Range of bandwidth multiples to test
    bandwidth_multiples = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    
    print(f"Starting bandwidth experiment with settings:")
    print(f"  n={n}, d={d}, delta={delta}")
    print(f"  Alpha: {alpha}")
    print(f"  Permutations per test: {num_permutations}")
    print(f"  Trials per bandwidth: {num_trials}")
    
    # Run the experiment
    powers = run_bandwidth_experiment(
        n=n, d=d, delta=delta, 
        bandwidth_multiples=bandwidth_multiples,
        num_trials=num_trials, 
        alpha=alpha, 
        num_permutations=num_permutations
    )
    
    # Plot and save results
    plt.figure(figsize=(9, 6))
    plt.plot(bandwidth_multiples, powers, marker='o', linestyle='-', linewidth=2)
    plt.xscale('log')  # Log scale for better visualization of bandwidth range
    plt.xlabel('Bandwidth Multiple (× median heuristic)')
    plt.ylabel('Empirical Power')
    plt.title(f'MMD Test Power vs Bandwidth\n(n={n}, d={d}, δ={delta})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                label='Median heuristic (1.0×)')
    plt.ylim([0, 1.05])
    plt.xlim([bandwidth_multiples.min()*0.8, bandwidth_multiples.max()*1.2])
    
    # Add annotations for key points
    max_power_idx = np.argmax(powers)
    plt.scatter(bandwidth_multiples[max_power_idx], powers[max_power_idx], 
                color='green', s=100, zorder=10)
    plt.annotate(f'Max power: {powers[max_power_idx]:.3f} at {bandwidth_multiples[max_power_idx]}×', 
                 xy=(bandwidth_multiples[max_power_idx], powers[max_power_idx]),
                 xytext=(bandwidth_multiples[max_power_idx]*1.1, powers[max_power_idx]-0.1),
                 arrowprops=dict(arrowstyle="->", color='black'))
    
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(project_root, 'figs', 'power_vs_bandwidth.png')
    plt.savefig(output_path, dpi=600)
    print(f"Figure saved: {output_path}")
    
    # Also save numerical results
    results = np.column_stack((bandwidth_multiples, powers))
    results_path = os.path.join(project_root, 'figs', 'bandwidth_results.csv')
    np.savetxt(results_path, results, delimiter=',', 
               header='bandwidth_multiple,power', comments='')
    print(f"Results saved: {results_path}")
    
if __name__ == '__main__':
    main()