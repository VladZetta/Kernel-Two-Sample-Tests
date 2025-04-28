import os
import sys
# add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from scipy import stats

from src.mmd import mmd_test
from generate_data import generate_gaussian_mixture


def run_power_vs_components(n, d, num_components_list, num_trials, alpha, num_permutations, num_runs=3):
    """
    Compute empirical power of the MMD test for different numbers of Gaussian mixture components.
    Run multiple times for confidence intervals.
    """
    all_powers = []
    print(f"Computing power vs. number of components (n={n}, d={d}, runs={num_runs})")
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        run_powers = []
        
        for num_components in tqdm(num_components_list, desc=f"Run {run+1}: Testing component counts"):
            rejects = 0
            for trial in tqdm(range(num_trials), desc=f"Trials for components={num_components}", leave=False):
                X, Y = generate_gaussian_mixture(n=n, d=d, num_components=num_components,
                                               mean_shift=1.0, mixture_difference='components')
                _, p_val = mmd_test(X, Y, return_p_value=True, num_permutations=num_permutations)
                if p_val < alpha:
                    rejects += 1
            power = rejects / num_trials
            run_powers.append(power)
            print(f"  Components={num_components}: Power={power:.4f} ({rejects}/{num_trials} rejections)")
        
        all_powers.append(run_powers)
    
    # Convert to numpy array for easier computation
    all_powers = np.array(all_powers)  # shape: [num_runs, len(num_components_list)]
    
    # Calculate mean and confidence intervals
    mean_powers = np.mean(all_powers, axis=0)
    
    # Calculate 95% confidence intervals using t-distribution
    if num_runs > 1:
        sem = stats.sem(all_powers, axis=0)
        ci_95 = sem * stats.t.ppf((1 + 0.95) / 2, num_runs-1)
        lower_ci = np.maximum(0, mean_powers - ci_95)
        upper_ci = np.minimum(1, mean_powers + ci_95)
    else:
        lower_ci = mean_powers
        upper_ci = mean_powers
    
    return mean_powers, lower_ci, upper_ci


def run_power_vs_difference_type(n, d, difference_types, num_trials, alpha, num_permutations, 
                                num_components=2, num_runs=3):
    """
    Compute empirical power of the MMD test for different types of mixture differences.
    Run multiple times for confidence intervals.
    """
    powers = {}
    ci_lower = {}
    ci_upper = {}
    
    print(f"Computing power vs. difference type (n={n}, d={d}, components={num_components}, runs={num_runs})")
    
    mean_shifts = np.linspace(0.1, 2.0, 10)  # Different levels of shift/difference
    
    for diff_type in difference_types:
        all_powers = []
        
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs} for {diff_type}")
            run_powers = []
            
            for shift in tqdm(mean_shifts, desc=f"Run {run+1}: Testing {diff_type} differences"):
                rejects = 0
                for trial in tqdm(range(num_trials), desc=f"Trials for shift={shift:.2f}", leave=False):
                    X, Y = generate_gaussian_mixture(n=n, d=d, num_components=num_components,
                                                   mean_shift=shift, mixture_difference=diff_type)
                    _, p_val = mmd_test(X, Y, return_p_value=True, num_permutations=num_permutations)
                    if p_val < alpha:
                        rejects += 1
                power = rejects / num_trials
                run_powers.append(power)
                print(f"  {diff_type.capitalize()} difference (shift={shift:.2f}): Power={power:.4f}")
            
            all_powers.append(run_powers)
        
        # Convert to numpy array for easier computation
        all_powers = np.array(all_powers)  # shape: [num_runs, len(mean_shifts)]
        
        # Calculate mean and confidence intervals
        mean_power = np.mean(all_powers, axis=0)
        powers[diff_type] = mean_power
        
        # Calculate 95% confidence intervals
        if num_runs > 1:
            sem = stats.sem(all_powers, axis=0)
            ci_95 = sem * stats.t.ppf((1 + 0.95) / 2, num_runs-1)
            ci_lower[diff_type] = np.maximum(0, mean_power - ci_95)
            ci_upper[diff_type] = np.minimum(1, mean_power + ci_95)
        else:
            ci_lower[diff_type] = mean_power
            ci_upper[diff_type] = mean_power
    
    return powers, ci_lower, ci_upper, mean_shifts


def generate_mixture_examples(dimensions=2, num_components=2):
    """
    Generate example datasets for visualizing the mixture differences.
    """
    np.random.seed(42)
    samples = {}
    
    for diff_type in ['components', 'weights']:
        X, Y = generate_gaussian_mixture(n=500, d=dimensions, num_components=num_components,
                                        mean_shift=1.0, mixture_difference=diff_type)
        samples[diff_type] = (X, Y)
    
    return samples


def main():
    sns.set(style='whitegrid')
    os.makedirs(os.path.join(project_root, 'figs'), exist_ok=True)

    # Experiment settings
    np.random.seed(42)
    alpha = 0.05
    num_permutations = 200
    num_trials = 30  # Reduced for computational efficiency
    num_runs = 3  # Number of runs for confidence intervals
    
    print("Starting Gaussian mixture experiments with settings:")
    print(f"  Alpha: {alpha}")
    print(f"  Permutations per test: {num_permutations}")
    print(f"  Trials per condition: {num_trials}")
    print(f"  Runs per experiment: {num_runs}")
    
    # Create a single figure with 3 key plots
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Visualization of different mixture types
    print("\n=== Part 1: Example Mixture Distributions ===")
    samples = generate_mixture_examples(num_components=2)
    
    # Create plot 1: Display example mixtures (component means difference)
    ax1 = fig.add_subplot(131)
    
    # Plot the component means difference example
    X, Y = samples['components']
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7, s=20, label='Distribution P', color='blue')
    ax1.scatter(Y[:, 0], Y[:, 1], alpha=0.5, s=20, label='Distribution Q (shifted component)', color='red')
    
    ax1.set_title('Example: Component Means Difference\nGaussian Mixture (2 components)', fontsize=12)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Save individual plot
    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, s=20, label='Distribution P', color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5, s=20, label='Distribution Q (shifted component)', color='red')
    plt.title('Example: Component Means Difference\nGaussian Mixture (2 components)', fontsize=12)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'mixture_example_means.png'), dpi=300)
    plt.close()
    
    # 2. Power comparison across difference types
    print("\n=== Part 2: Power Comparison Across Difference Types ===")
    # Use only components and weights difference types
    diff_types = ['components', 'weights']
    powers_diff_type, ci_lower, ci_upper, mean_shifts = run_power_vs_difference_type(
        n=100, d=2,
        difference_types=diff_types,
        num_trials=num_trials,
        alpha=alpha,
        num_permutations=num_permutations,
        num_components=2,
        num_runs=num_runs
    )
    
    ax2 = fig.add_subplot(132)
    colors = ['#1f77b4', '#ff7f0e'] # Standard matplotlib colors
    
    for i, diff_type in enumerate(diff_types):
        ax2.plot(mean_shifts, powers_diff_type[diff_type], marker='o', 
               label=f'{diff_type.capitalize()} difference', 
               color=colors[i])
        ax2.fill_between(mean_shifts, 
                         ci_lower[diff_type], 
                         ci_upper[diff_type], 
                         alpha=0.2, color=colors[i])
    
    ax2.set_xlabel('Difference Magnitude')
    ax2.set_ylabel('Test Power')
    ax2.set_title('MMD Test Power by Difference Type\n(mean of 3 runs with 95% CI)', fontsize=12)
    ax2.legend()
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    
    # Save individual plot
    plt.figure(figsize=(8, 5))
    for i, diff_type in enumerate(diff_types):
        plt.plot(mean_shifts, powers_diff_type[diff_type], marker='o', 
               label=f'{diff_type.capitalize()} difference', 
               color=colors[i])
        plt.fill_between(mean_shifts, 
                        ci_lower[diff_type], 
                        ci_upper[diff_type], 
                        alpha=0.2, color=colors[i])
    plt.xlabel('Difference Magnitude')
    plt.ylabel('Test Power')
    plt.title('MMD Test Power by Difference Type\n(mean of 3 runs with 95% CI)', fontsize=12)
    plt.legend()
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'power_vs_difference_type_mixture.png'), dpi=300)
    plt.close()
    
    # 3. Power vs. Number of components
    print("\n=== Part 3: Power vs. Number of Components ===")
    # Expand to more components
    component_counts = [2, 3, 4, 5, 6, 8, 10]
    powers_components, comp_ci_lower, comp_ci_upper = run_power_vs_components(
        n=100, d=2, 
        num_components_list=component_counts,
        num_trials=num_trials, 
        alpha=alpha, 
        num_permutations=num_permutations,
        num_runs=num_runs
    )
    
    ax3 = fig.add_subplot(133)
    ax3.plot(component_counts, powers_components, marker='o', color='purple')
    ax3.fill_between(component_counts, comp_ci_lower, comp_ci_upper, alpha=0.2, color='purple')
    
    ax3.set_xlabel('Number of Mixture Components')
    ax3.set_ylabel('Test Power')
    ax3.set_title('MMD Test Power vs. Component Count\n(mean of 3 runs with 95% CI)', fontsize=12)
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    
    # Save individual plot
    plt.figure(figsize=(6, 5))
    plt.plot(component_counts, powers_components, marker='o', color='purple')
    plt.fill_between(component_counts, comp_ci_lower, comp_ci_upper, alpha=0.2, color='purple')
    plt.xlabel('Number of Mixture Components')
    plt.ylabel('Test Power')
    plt.title('MMD Test Power vs. Component Count\n(mean of 3 runs with 95% CI)', fontsize=12)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'figs', 'power_vs_components_mixture.png'), dpi=300)
    plt.close()
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'figs', 'gaussian_mixture_results.png')
    plt.savefig(output_path, dpi=300)
    print(f"Combined figure saved: {output_path}")
    print(f"Individual figures also saved to figs/")
    
    print('\nGaussian mixture experiments complete.')


if __name__ == '__main__':
    main() 