import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

# Import the functions from the original code (assuming they're available)
from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd_optimized import mmd_test, compute_mmd_stat
from joblib import Parallel, delayed
from src.permutation_test import permutation_test_statistic

# Define the kernel experiments (using the same from the original code)
KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',    'bandwidth':'median'},
    {'kernel':'rbf',    'bandwidth':1.0},
    {'kernel':'linear', 'bandwidth':None},
    {'kernel':'poly',   'bandwidth':None, 'degree':2, 'coef0':0},
    {'kernel':'poly',   'bandwidth':None, 'degree':3, 'coef0':1},
    {'kernel':'laplace','bandwidth':'median'},
    {'kernel':'laplace','bandwidth':0.5},
]

def subsample_data(data, labels, digit, num_samples=50):
    """Subsample data for a specific digit with a smaller sample size."""
    digit_indices = np.where(labels == digit)[0]
    indices = np.random.choice(digit_indices, num_samples, replace=False)
    return data[indices]

def add_noise(images, noise_level=0.1):
    """Add Gaussian noise to images."""
    noisy_images = images.copy()
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_images += noise
    # Clip to maintain the same range as original images
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images

def add_blur(images, sigma=1.0):
    """Add Gaussian blur to images."""
    blurred_images = np.zeros_like(images)
    for i in range(len(images)):
        blurred_images[i] = gaussian_filter(images[i], sigma=sigma)
    return blurred_images

def run_single_experiment(group1, group2, cfg, experiment_name=""):
    """Run a single MMD experiment with the given configuration."""
    params = cfg.copy()
    name = params.pop('kernel')
    bw = params.pop('bandwidth')
    
    try:
        # Compute the MMD statistic
        stat = compute_mmd_stat(
            group1, group2,
            kernel=name,
            bandwidth=bw,
            **params
        )
        
        # Time the permutation test
        perm_start_time = time.time()
        
        pval = permutation_test_statistic(
            group1, group2, 
            stat_fn=compute_mmd_stat,
            num_permutations=100,
            n_jobs=1,
            kernel=name,
            bandwidth=bw,
            **params
        )
        
        perm_end_time = time.time()
        perm_time = perm_end_time - perm_start_time
        
        print(f"{experiment_name} - {name:7s} | bw={str(bw):6s} → MMD²={stat:.4e}, p={pval:.4f}, time={perm_time:.2f}s")
        
        return {
            'kernel': name,
            'bandwidth': bw,
            'mmd2': stat,
            'p_value': pval,
            'perm_time': perm_time,
            'experiment': experiment_name
        }
    except Exception as e:
        print(f"Error with {name} kernel, bandwidth={bw}: {str(e)}")
        return {
            'kernel': name,
            'bandwidth': bw,
            'mmd2': np.nan,
            'p_value': np.nan,
            'perm_time': np.nan,
            'experiment': experiment_name
        }

def main():
    print("Starting enhanced MMD experiments...")
    # Ensure output directories exist
    FIG_DIR = "figures/new"
    RESULTS_DIR = "results"
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load MNIST data
    data, labels = load_mnist()
    
    all_results = []
    
    # Experiment 1: Limited data (50 samples per digit)
    print("\n=== Experiment 1: MMD Test with Limited Data (50 samples per digit) ===")
    digit0_small = subsample_data(data, labels, digit=0, num_samples=50)
    digit1_small = subsample_data(data, labels, digit=1, num_samples=50)
    
    print(f"Digit 0: {digit0_small.shape[0]} samples")
    print(f"Digit 1: {digit1_small.shape[0]} samples")
    
    # Run experiments in parallel
    small_sample_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(digit0_small, digit1_small, cfg, "Limited Data (n=50)") 
        for cfg in KERNEL_EXPERIMENTS
    )
    all_results.extend(small_sample_results)
    
    # Experiment 2: Subtle differences (noise)
    print("\n=== Experiment 2: MMD Test with Subtle Differences (Original vs. Noisy) ===")
    digit0_original = subsample_data(data, labels, digit=0, num_samples=100)
    digit0_noisy = add_noise(digit0_original, noise_level=0.1)
    
    noisy_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(digit0_original, digit0_noisy, cfg, "Subtle Diff (Noise)") 
        for cfg in KERNEL_EXPERIMENTS
    )
    all_results.extend(noisy_results)
    
    # Experiment 3: Subtle differences (blur)
    print("\n=== Experiment 3: MMD Test with Subtle Differences (Original vs. Blurred) ===")
    digit0_blurred = add_blur(digit0_original, sigma=1.0)
    
    blur_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(digit0_original, digit0_blurred, cfg, "Subtle Diff (Blur)") 
        for cfg in KERNEL_EXPERIMENTS
    )
    all_results.extend(blur_results)
    
    # Experiment 4: H0 is true (random split of same class)
    print("\n=== Experiment 4: MMD Test when H0 is True (Random Split of Same Class) ===")
    # Get a larger sample first
    digit0_all = subsample_data(data, labels, digit=0, num_samples=200)
    # Then randomly split
    indices = np.random.permutation(len(digit0_all))
    split_idx = len(indices) // 2
    digit0_group1 = digit0_all[indices[:split_idx]]
    digit0_group2 = digit0_all[indices[split_idx:]]
    
    print(f"Group 1: {digit0_group1.shape[0]} samples")
    print(f"Group 2: {digit0_group2.shape[0]} samples")
    
    h0_true_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(digit0_group1, digit0_group2, cfg, "H0 True (Split)") 
        for cfg in KERNEL_EXPERIMENTS
    )
    all_results.extend(h0_true_results)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "enhanced_mmd_results.csv"), index=False)
    print("\nAll results summary:\n", results_df)
    
    # Visualize results - grouped by experiment
    experiments = results_df['experiment'].unique()
    kernels = results_df['kernel'].unique()
    
    # Create a figure for p-values across experiments
    plt.figure(figsize=(15, 10))
    
    for i, exp in enumerate(experiments):
        plt.subplot(2, 2, i+1)
        exp_data = results_df[results_df['experiment'] == exp]
        
        # Sort kernels by p-value for better visualization
        exp_data = exp_data.sort_values('p_value')
        
        bars = plt.bar(exp_data['kernel'], exp_data['p_value'])
        plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
        plt.ylabel("p-value")
        plt.title(f"Significance by Kernel: {exp}")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Set consistent y-axis limits
        
        # Color bars based on significance
        for j, bar in enumerate(bars):
            if exp_data.iloc[j]['p_value'] < 0.05:
                bar.set_color('green')  # Significant
            else:
                bar.set_color('gray')   # Not significant
                
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd_enhanced_pvalues.png"))
    
    # Create a summary visualization comparing all experiments
    plt.figure(figsize=(15, 10))
    
    # 1. Plot average p-values by experiment
    plt.subplot(2, 2, 1)
    avg_pvals = results_df.groupby('experiment')['p_value'].mean().sort_values()
    bars = plt.bar(avg_pvals.index, avg_pvals.values)
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.ylabel("Average p-value")
    plt.title("Average p-value by Experiment")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Color bars based on significance
    for j, bar in enumerate(bars):
        if avg_pvals.iloc[j] < 0.05:
            bar.set_color('green')  # Significant
        else:
            bar.set_color('gray')   # Not significant
    
    # 2. Plot rejection rates (power) by experiment
    plt.subplot(2, 2, 2)
    rejection_rates = results_df.groupby('experiment')['p_value'].apply(
        lambda x: (x < 0.05).mean() * 100
    ).sort_values(ascending=False)
    
    plt.bar(rejection_rates.index, rejection_rates.values)
    plt.ylabel("Rejection Rate (%)")
    plt.title("Power by Experiment (% of Kernels with p < 0.05)")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # 3. Plot MMD² statistics by experiment
    plt.subplot(2, 2, 3)
    avg_mmd = results_df.groupby('experiment')['mmd2'].mean().sort_values(ascending=False)
    plt.bar(avg_mmd.index, avg_mmd.values)
    plt.ylabel("Average MMD² statistic")
    plt.title("Average MMD² by Experiment")
    plt.xticks(rotation=45)
    plt.yscale('log')  # Log scale for better visualization
    
    # 4. Plot best performing kernel for each experiment
    plt.subplot(2, 2, 4)
    
    best_kernels = []
    for exp in experiments:
        exp_data = results_df[results_df['experiment'] == exp]
        # For H0 True experiment, "best" means closest to not rejecting
        if exp == "H0 True (Split)":
            best_kernel = exp_data.loc[exp_data['p_value'].idxmax()]['kernel']
        else:
            # For other experiments, "best" means strongest rejection
            best_kernel = exp_data.loc[exp_data['p_value'].idxmin()]['kernel']
        
        best_kernels.append({
            'experiment': exp,
            'best_kernel': best_kernel
        })
        
    best_df = pd.DataFrame(best_kernels)
    
    # Create a table-like visualization
    plt.axis('off')  # Turn off axis
    plt.table(
        cellText=best_df.values,
        colLabels=best_df.columns,
        loc='center',
        cellLoc='center'
    )
    plt.title("Best Performing Kernel by Experiment")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd_experiment_summary.png"))
    
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIG_DIR}")
    
    # Create a comprehensive summary table
    summary_data = []
    for exp in experiments:
        exp_data = results_df[results_df['experiment'] == exp]
        rejection_rate = (exp_data['p_value'] < 0.05).mean() * 100
        avg_mmd = exp_data['mmd2'].mean()
        
        if exp == "H0 True (Split)":
            # For H0 True, we want high p-values (no false positives)
            best_kernel = exp_data.loc[exp_data['p_value'].idxmax()]['kernel']
            best_p = exp_data['p_value'].max()
        else:
            # For other experiments, we want low p-values (high power)
            best_kernel = exp_data.loc[exp_data['p_value'].idxmin()]['kernel']
            best_p = exp_data['p_value'].min()
            
        summary_data.append({
            'Experiment': exp,
            'Avg MMD²': f"{avg_mmd:.6e}",
            'Rejection Rate (%)': f"{rejection_rate:.1f}",
            'Best Kernel': best_kernel,
            'Best p-value': f"{best_p:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False)
    print("\nExperiment summary:\n", summary_df)

if __name__ == '__main__':
    main()