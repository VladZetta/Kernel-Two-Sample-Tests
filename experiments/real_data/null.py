import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import Parallel, delayed

from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd_optimized import mmd_test, median_heuristic

# Output directories
FIG_DIR = "figures/mnist_null"
RESULTS_DIR = "results"

# Create directories if they don't exist
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the kernel experiments for comparison
# Using only the kernels available in your implementation
KERNEL_VARIANTS = [
    {'name': 'RBF (median)', 'kernel': 'rbf', 'bandwidth': 'median', 'extra_params': {}},
    {'name': 'RBF (fixed)', 'kernel': 'rbf', 'bandwidth': 1.0, 'extra_params': {}},
    {'name': 'Linear', 'kernel': 'linear', 'bandwidth': None, 'extra_params': {}},
    {'name': 'Polynomial', 'kernel': 'poly', 'bandwidth': None, 'extra_params': {'degree': 3, 'coef0': 1}}
]

def subsample_data(data, labels, digit, max_samples=1000):
    """Subsample data for a specific digit."""
    # Filter the data for the specific digit
    digit_indices = np.where(labels == digit)[0]
    # If there are more samples than max_samples, randomly choose a subset
    if len(digit_indices) > max_samples:
        indices = np.random.choice(digit_indices, max_samples, replace=False)
    else:
        indices = digit_indices
    return data[indices]

def compute_null_distribution(data, n_permutations=1000, batch_size=100):
    """
    Compute the null distribution by randomly splitting the same data.
    Returns the null distribution and the observed statistic.
    """
    n = len(data)
    half_n = n // 2
    
    # Compute observed statistic (this is just a dummy split since we're working with same data)
    np.random.shuffle(data)
    group1 = data[:half_n]
    group2 = data[half_n:2*half_n]  # Ensure equal sizes
    
    observed_stat = mmd_test(
        group1, group2,
        kernel='rbf',
        bandwidth='median',
        return_p_value=False
    )
    
    # Compute null distribution in batches
    null_distribution = []
    num_batches = n_permutations // batch_size + (1 if n_permutations % batch_size else 0)
    
    for batch in range(num_batches):
        batch_results = Parallel(n_jobs=-1)(
            delayed(compute_single_null)(data, half_n) 
            for _ in range(min(batch_size, n_permutations - batch * batch_size))
        )
        null_distribution.extend(batch_results)
        print(f"Completed batch {batch+1}/{num_batches} of null distribution computation")
    
    return np.array(null_distribution), observed_stat, half_n

def compute_single_null(data, half_n):
    """Compute a single entry in the null distribution."""
    # Shuffle the data
    indices = np.random.permutation(len(data))
    # Split into two groups
    group1 = data[indices[:half_n]]
    group2 = data[indices[half_n:2*half_n]]
    
    # Compute MMD statistic
    stat = mmd_test(
        group1, group2,
        kernel='rbf',
        bandwidth='median',
        return_p_value=False
    )
    return stat

def compute_power(data1, data2, kernel_config, n_trials=100):
    """
    Compute the statistical power of a kernel test.
    Returns power percentage and average permutation time.
    """
    n = min(len(data1), len(data2))
    
    # Ensure equal sample sizes
    if len(data1) > n:
        idx1 = np.random.choice(len(data1), n, replace=False)
        data1 = data1[idx1]
    if len(data2) > n:
        idx2 = np.random.choice(len(data2), n, replace=False)
        data2 = data2[idx2]
    
    # Extract kernel parameters
    kernel_name = kernel_config['kernel']
    bandwidth = kernel_config['bandwidth']
    extra_params = kernel_config['extra_params']
    
    # Compute power by running multiple trials
    rejections = 0
    total_time = 0
    
    for trial in range(n_trials):
        start_time = time.time()
        try:
            _, p_value = mmd_test(
                data1, data2,
                kernel=kernel_name,
                bandwidth=bandwidth,
                return_p_value=True,
                num_permutations=100,  # Use 100 permutations for p-value
                **extra_params
            )
            # Count rejection of null hypothesis at alpha=0.05
            if p_value < 0.05:
                rejections += 1
        except Exception as e:
            print(f"Error with {kernel_name} kernel: {str(e)}")
            p_value = np.nan
        
        end_time = time.time()
        total_time += (end_time - start_time)
    
    # Calculate power and average permutation time
    power = (rejections / n_trials) * 100
    avg_time = total_time / n_trials
    
    return power, avg_time, n

def plot_null_distribution(null_dist, observed_stat, n, output_file):
    """Plot the null distribution with observed statistic marked in red."""
    plt.figure(figsize=(10, 6))
    sns.histplot(null_dist, kde=True, stat="density")
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2)
    
    # Add text annotation for the observed statistic
    plt.text(observed_stat*1.05, plt.gca().get_ylim()[1]*0.9, 
             f'Observed MMD²={observed_stat:.6f}', 
             color='red', fontsize=12)
    
    plt.title(f'Null Distribution for MNIST Sanity Split (n=m={n})')
    plt.xlabel('MMD² Statistic')
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Null distribution plot saved to {output_file}")
    
    # Also save the raw null distribution data for later use
    np.savez(
        os.path.join(RESULTS_DIR, "null_distribution_data.npz"),
        null_distribution=null_dist,
        observed_statistic=observed_stat,
        sample_size=n
    )
    print(f"Raw null distribution data saved to {os.path.join(RESULTS_DIR, 'null_distribution_data.npz')}")

def compare_kernel_variants(data0, data1, output_csv):
    """
    Compare different kernel variants on digits 0 vs 1 and save results.
    Returns a DataFrame with comparison metrics.
    """
    results = []
    
    for config in KERNEL_VARIANTS:
        print(f"\nEvaluating {config['name']} kernel...")
        
        try:
            power, perm_time, n = compute_power(data0, data1, config)
            
            # For RBF, compute and store the median bandwidth
            if config['kernel'] == 'rbf' and config['bandwidth'] == 'median':
                bandwidth_value = median_heuristic(data0[:100], data1[:100])  # Sample for faster computation
                bandwidth_str = f"{bandwidth_value:.2f}"
            elif config['kernel'] == 'rbf' and isinstance(config['bandwidth'], (int, float)):
                bandwidth_str = f"{config['bandwidth']:.2f}"
            else:
                bandwidth_str = "N/A"
            
            results.append({
                'Kernel': config['name'],
                'n=m': n,
                'Power (%)': power,
                'Perm. Time (s)': perm_time,
                'Bandwidth': bandwidth_str
            })
            
            print(f"{config['name']}: Power={power:.2f}%, Time={perm_time:.2f}s, n=m={n}")
            
        except Exception as e:
            print(f"Error evaluating {config['name']}: {str(e)}")
            results.append({
                'Kernel': config['name'],
                'n=m': 'Error',
                'Power (%)': 'Error',
                'Perm. Time (s)': 'Error',
                'Bandwidth': 'Error'
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Kernel comparison results saved to {output_csv}")
    
    # Visualize the results
    plt.figure(figsize=(12, 5))
    
    # Plot power
    plt.subplot(1, 2, 1)
    kernels = df['Kernel'].tolist()
    powers = df['Power (%)'].tolist()
    powers = [float(p) if not isinstance(p, str) else 0 for p in powers]
    plt.bar(kernels, powers)
    plt.ylabel('Power (%)')
    plt.title('Statistical Power by Kernel')
    plt.xticks(rotation=45)
    
    # Plot time
    plt.subplot(1, 2, 2)
    times = df['Perm. Time (s)'].tolist()
    times = [float(t) if not isinstance(t, str) else 0 for t in times]
    plt.bar(kernels, times)
    plt.ylabel('Time (s)')
    plt.title('Permutation Time by Kernel')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "kernel_comparison.png"))
    plt.close()
    
    return df

def main():
    time_start = time.time()
    print("Starting simplified MMD experiment...")
    
    # Load MNIST data
    data, labels = load_mnist()
    print(f"Loaded MNIST dataset with {len(data)} samples")
    
    # 1. Compute and plot null distribution for sanity check
    print("\n=== Computing null distribution for sanity check ===")
    # Use digit 0 for sanity check
    digit0_data = subsample_data(data, labels, digit=0, max_samples=2000)
    null_dist, observed_stat, n = compute_null_distribution(digit0_data, n_permutations=1000)
    plot_null_distribution(
        null_dist, observed_stat, n,
        output_file=os.path.join(FIG_DIR, "fig_mnist_null.pdf")
    )
    
    # 2. Compare kernel variants on digits 0 vs 1
    print("\n=== Comparing kernel variants on digits 0 vs 1 ===")
    digit0_data = subsample_data(data, labels, digit=0, max_samples=1000)
    digit1_data = subsample_data(data, labels, digit=1, max_samples=1000)
    
    print(f"Digit 0: {len(digit0_data)} samples")
    print(f"Digit 1: {len(digit1_data)} samples")
    
    results_df = compare_kernel_variants(
        digit0_data, digit1_data,
        output_csv=os.path.join(RESULTS_DIR, "kernel_comparison.csv")
    )
    
    # Print total runtime
    end_time = time.time()
    print(f"\nTotal experiment time: {end_time - time_start:.2f} seconds")
    print(f"Results saved to the {RESULTS_DIR} directory")
    print(f"Figures saved to the {FIG_DIR} directory")

if __name__ == '__main__':
    main()