import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.mmd import mmd_test, compute_mmd_stat

def run_sanity_check(n_per_sample, d, num_trials, alpha, num_permutations):
    """
    Run a sanity check experiment by randomly splitting a dataset into two samples
    and verifying that the test correctly handles H0 cases.
    """
    p_values = []
    mmd_stats = []
    rejections = 0
    
    print(f"Running sanity check experiment (n={n_per_sample}, d={d})")
    for trial in tqdm(range(num_trials), desc="Trials"):
        # Generate a single dataset
        data = np.random.normal(0, 1, size=(2*n_per_sample, d))
        
        # Randomly shuffle and split into two samples
        np.random.shuffle(data)
        X = data[:n_per_sample]
        Y = data[n_per_sample:]
        
        # Ensure we use the standard permutation test approach
        mmd_stat, p_value = mmd_test(
            X, Y, 
            kernel='rbf',
            bandwidth='median',
            return_p_value=True, 
            num_permutations=num_permutations
        )
        
        mmd_stats.append(mmd_stat)
        p_values.append(p_value)
        
        if p_value < alpha:
            rejections += 1
    
    type_I_error = rejections / num_trials
    print(f"Type I error rate: {type_I_error:.4f} (expected {alpha:.4f})")
    print(f"Number of rejections: {rejections}/{num_trials}")
    
    return mmd_stats, p_values, type_I_error

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs(os.path.join(project_root, 'figs'), exist_ok=True)

    # Experiment settings
    np.random.seed(42)
    alpha = 0.05
    num_permutations = 200  # Increased from 200 for more stable null distribution
    num_trials = 2000
    n_per_sample = 100
    d = 10
    
    print(f"Starting sanity check experiment with settings:")
    print(f"  Alpha: {alpha}")
    print(f"  Permutations per test: {num_permutations}")
    print(f"  Trials: {num_trials}")
    print(f"  Samples per group: {n_per_sample}")
    print(f"  Dimensions: {d}")
    
    # Run the experiment
    mmd_stats, p_values, type_I_error = run_sanity_check(
        n_per_sample=n_per_sample, 
        d=d, 
        num_trials=num_trials,
        alpha=alpha,
        num_permutations=num_permutations
    )
    
    # Create a nicer 2x1 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Histogram of MMD statistics (ensure non-negative display)
    # First check if we have negative values and handle appropriately
    mmd_stats = np.array(mmd_stats)
    if np.any(mmd_stats < 0):
        print(f"Warning: {np.sum(mmd_stats < 0)} negative MMD^2 values detected.")
        print(f"Min value: {np.min(mmd_stats)}, Max value: {np.max(mmd_stats)}")
        # For plotting purposes, shift to ensure non-negativity
        mmd_stats_plot = mmd_stats - np.min(mmd_stats) if np.min(mmd_stats) < 0 else mmd_stats
        shifted = True
    else:
        mmd_stats_plot = mmd_stats
        shifted = False
    
    sns.histplot(mmd_stats_plot, kde=True, ax=ax1, color='skyblue', bins=15)
    ax1.set_xlabel('MMD^2 Statistic' + (' (shifted)' if shifted else ''))
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of MMD^2 Under H_0')
    
    # 2. Histogram of p-values (should be uniform under H_0)
    hist_p = sns.histplot(p_values, kde=False, ax=ax2, color='salmon', bins=10, 
                         stat='count', element='bars')
    
    # Add reference line for uniform distribution
    bin_width = 0.1  # 10 bins from 0 to 1
    expected_count = num_trials * bin_width
    ax2.axhline(y=expected_count, color='black', linestyle='--', 
                label=f'Expected uniform count: {expected_count:.1f}')
    
    ax2.set_xlabel('p-value')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of p-values Under H_0')
    ax2.set_xlim(0, 1)
    ax2.legend()
    
    # Calculate KS test for uniformity of p-values
    from scipy import stats
    ks_stat, ks_pval = stats.kstest(p_values, 'uniform')
    
    # Add text with summary statistics
    plt.figtext(0.5, 0.01, 
                f'Type I Error Rate: {type_I_error:.3f} (expected: {alpha:.3f})\n'
                f'KS test for p-value uniformity: p={ks_pval:.4f}\n'
                f'Sample Size: n1=n2={n_per_sample}, Dimensions: d={d}',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.suptitle('MMD Test Sanity Check: Random Splits of the Same Distribution', fontsize=14)
    
    # Save figure
    output_path = os.path.join(project_root, 'figs', 'sanity_check.png')
    plt.savefig(output_path, dpi=600)
    print(f"Figure saved: {output_path}")
    
    # Also save the raw data for reference
    results = np.column_stack((mmd_stats, p_values))
    np.savetxt(os.path.join(project_root, 'figs', 'sanity_check_results.csv'),
               results, delimiter=',', header='mmd_stat,p_value', comments='')
    
if __name__ == '__main__':
    main()