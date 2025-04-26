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

# Define the kernel experiments (using a subset for clearer visualization)
KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',    'bandwidth':'median', 'display_name': 'RBF (median)'},
    {'kernel':'rbf',    'bandwidth':1.0, 'display_name': 'RBF (1.0)'},
    {'kernel':'linear', 'bandwidth':None, 'display_name': 'Linear'},
    {'kernel':'poly',   'bandwidth':None, 'degree':2, 'coef0':0, 'display_name': 'Poly (d=2)'},
    {'kernel':'laplace','bandwidth':'median', 'display_name': 'Laplace (median)'},
    {'kernel':'matern', 'bandwidth':'median', 'length_scale':1.0, 'nu':1.5, 'display_name': 'Matern (median)'},
]

def subsample_data(data, labels, digit, num_samples=100):
    """Subsample data for a specific digit with a fixed sample size."""
    digit_indices = np.where(labels == digit)[0]
    indices = np.random.choice(digit_indices, num_samples, replace=False)
    return data[indices]

def add_noise(images, noise_level):
    """Add Gaussian noise to images."""
    noisy_images = images.copy()
    noise = np.random.normal(0, noise_level, images.shape)
    noisy_images += noise
    # Clip to maintain the same range as original images
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images

def add_blur(images, sigma):
    """Add Gaussian blur to images."""
    blurred_images = np.zeros_like(images)
    for i in range(len(images)):
        blurred_images[i] = gaussian_filter(images[i], sigma=sigma)
    return blurred_images

def run_noise_sensitivity_experiment(data, labels, kernels, noise_levels):
    """
    Run experiment to test sensitivity to different noise levels.
    
    Parameters:
    -----------
    data, labels : Dataset
    kernels : List of kernel configurations
    noise_levels : List of noise levels to test
    
    Returns:
    --------
    DataFrame with results for each kernel and noise level
    """
    results = []
    
    # Get original data
    digit_data = subsample_data(data, labels, digit=0, num_samples=100)
    
    for noise_level in noise_levels:
        print(f"\nTesting noise level: {noise_level}")
        # Add noise to copies of the original data
        noisy_data = add_noise(digit_data, noise_level)
        
        for kernel_config in kernels:
            print(f"  Testing kernel: {kernel_config['display_name']}")
            params = kernel_config.copy()
            display_name = params.pop('display_name')
            name = params.pop('kernel')
            bw = params.pop('bandwidth')
            
            try:
                # Compute MMD statistic
                stat = compute_mmd_stat(
                    digit_data, noisy_data,
                    kernel=name,
                    bandwidth=bw,
                    **params
                )
                
                # Compute p-value with permutation test
                pval = permutation_test_statistic(
                    digit_data, noisy_data, 
                    stat_fn=compute_mmd_stat,
                    num_permutations=100,
                    n_jobs=1,
                    kernel=name,
                    bandwidth=bw,
                    **params
                )
                
                results.append({
                    'kernel': display_name,
                    'noise_level': noise_level,
                    'mmd2': stat,
                    'p_value': pval
                })
                
                print(f"    MMD²={stat:.4e}, p={pval:.4f}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                results.append({
                    'kernel': display_name,
                    'noise_level': noise_level,
                    'mmd2': np.nan,
                    'p_value': np.nan
                })
    
    return pd.DataFrame(results)

def run_blur_sensitivity_experiment(data, labels, kernels, blur_sigmas):
    """
    Run experiment to test sensitivity to different blur intensities.
    
    Parameters:
    -----------
    data, labels : Dataset
    kernels : List of kernel configurations
    blur_sigmas : List of blur sigma values to test
    
    Returns:
    --------
    DataFrame with results for each kernel and blur level
    """
    results = []
    
    # Get original data
    digit_data = subsample_data(data, labels, digit=0, num_samples=100)
    
    for sigma in blur_sigmas:
        print(f"\nTesting blur sigma: {sigma}")
        # Add blur to copies of the original data
        blurred_data = add_blur(digit_data, sigma)
        
        for kernel_config in kernels:
            print(f"  Testing kernel: {kernel_config['display_name']}")
            params = kernel_config.copy()
            display_name = params.pop('display_name')
            name = params.pop('kernel')
            bw = params.pop('bandwidth')
            
            try:
                # Compute MMD statistic
                stat = compute_mmd_stat(
                    digit_data, blurred_data,
                    kernel=name,
                    bandwidth=bw,
                    **params
                )
                
                # Compute p-value with permutation test
                pval = permutation_test_statistic(
                    digit_data, blurred_data, 
                    stat_fn=compute_mmd_stat,
                    num_permutations=100,
                    n_jobs=1,
                    kernel=name,
                    bandwidth=bw,
                    **params
                )
                
                results.append({
                    'kernel': display_name,
                    'blur_sigma': sigma,
                    'mmd2': stat,
                    'p_value': pval
                })
                
                print(f"    MMD²={stat:.4e}, p={pval:.4f}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                results.append({
                    'kernel': display_name,
                    'blur_sigma': sigma,
                    'mmd2': np.nan,
                    'p_value': np.nan
                })
    
    return pd.DataFrame(results)

def create_noise_sensitivity_plots(results_df, save_dir):
    """Create plots for noise sensitivity results."""
    kernels = results_df['kernel'].unique()
    
    # Create figure for p-values
    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        kernel_df = results_df[results_df['kernel'] == kernel]
        plt.plot(kernel_df['noise_level'], kernel_df['p_value'], 
                 marker='o', label=kernel, linewidth=2)
    
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('p-value', fontsize=12)
    plt.title('MMD Test Sensitivity to Noise Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_sensitivity_pvals_2.png'), dpi=300)
    
    # Create figure for MMD² statistics
    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        kernel_df = results_df[results_df['kernel'] == kernel]
        plt.plot(kernel_df['noise_level'], kernel_df['mmd2'], 
                 marker='o', label=kernel, linewidth=2)
    
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('MMD² Statistic', fontsize=12)
    plt.title('MMD² Statistic vs. Noise Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_sensitivity_mmd2_2.png'), dpi=300)

def create_blur_sensitivity_plots(results_df, save_dir):
    """Create plots for blur sensitivity results."""
    kernels = results_df['kernel'].unique()
    
    # Create figure for p-values
    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        kernel_df = results_df[results_df['kernel'] == kernel]
        plt.plot(kernel_df['blur_sigma'], kernel_df['p_value'], 
                 marker='o', label=kernel, linewidth=2)
    
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.xlabel('Blur Intensity (σ)', fontsize=12)
    plt.ylabel('p-value', fontsize=12)
    plt.title('MMD Test Sensitivity to Blur Intensity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'blur_sensitivity_pvals_2.png'), dpi=300)
    
    # Create figure for MMD² statistics
    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        kernel_df = results_df[results_df['kernel'] == kernel]
        plt.plot(kernel_df['blur_sigma'], kernel_df['mmd2'], 
                 marker='o', label=kernel, linewidth=2)
    
    plt.xlabel('Blur Intensity (σ)', fontsize=12)
    plt.ylabel('MMD² Statistic', fontsize=12)
    plt.title('MMD² Statistic vs. Blur Intensity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'blur_sensitivity_mmd2_2.png'), dpi=300)

def main():
    print("Starting MMD sensitivity experiments...")
    # Ensure output directories exist
    FIG_DIR = "figures/new"
    RESULTS_DIR = "results"
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load MNIST data
    data, labels = load_mnist()
    
    # Define noise levels to test (small increments)
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    # Define blur intensities to test
    blur_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Run noise sensitivity experiment
    print("\n=== Running Noise Sensitivity Experiment ===")
    noise_results = run_noise_sensitivity_experiment(data, labels, KERNEL_EXPERIMENTS, noise_levels)
    noise_results.to_csv(os.path.join(RESULTS_DIR, "noise_sensitivity_results.csv"), index=False)
    create_noise_sensitivity_plots(noise_results, FIG_DIR)
    
    # Run blur sensitivity experiment
    print("\n=== Running Blur Sensitivity Experiment ===")
    blur_results = run_blur_sensitivity_experiment(data, labels, KERNEL_EXPERIMENTS, blur_sigmas)
    blur_results.to_csv(os.path.join(RESULTS_DIR, "blur_sensitivity_results.csv"), index=False)
    create_blur_sensitivity_plots(blur_results, FIG_DIR)
    
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIG_DIR}")

if __name__ == '__main__':
    main()