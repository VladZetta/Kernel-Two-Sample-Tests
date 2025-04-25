import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd_optimized import mmd_test, compute_mmd_stat
from joblib import Parallel, delayed

# Define the kernel experiments
KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',    'bandwidth':'median'},
    {'kernel':'rbf',    'bandwidth':1.0},
    {'kernel':'linear', 'bandwidth':None},
    {'kernel':'poly',   'bandwidth':None, 'degree':2, 'coef0':0},
    {'kernel':'poly',   'bandwidth':None, 'degree':3, 'coef0':1},
    {'kernel':'laplace','bandwidth':'median'},
    {'kernel':'laplace','bandwidth':0.5},
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

def run_single_experiment(group1, group2, cfg):
    """Run a single MMD experiment with the given configuration and time the permutation test."""
    params = cfg.copy()
    name = params.pop('kernel')
    bw = params.pop('bandwidth')
    
    try:
        # First compute just the MMD statistic (without permutation test)
        stat = compute_mmd_stat(
            group1, group2,
            kernel=name,
            bandwidth=bw,
            **params
        )
        
        # Now time the permutation test
        perm_start_time = time.time()
        
        # Instead of using mmd_test with return_p_value=True, we'll call the permutation test directly
        from src.permutation_test import permutation_test_statistic
        
        pval = permutation_test_statistic(
            group1, group2, 
            stat_fn=compute_mmd_stat,
            num_permutations=100,
            n_jobs=1,  # Use 1 job here since we're already parallelizing at experiment level
            kernel=name,
            bandwidth=bw,
            **params
        )
        
        perm_end_time = time.time()
        perm_time = perm_end_time - perm_start_time
        
        print(f"{name:7s} | bw={str(bw):6s} → MMD²={stat:.4e}, p={pval:.4f}, time={perm_time:.2f}s")
        
        return {
            'kernel': name,
            'bandwidth': bw,
            'mmd2': stat,
            'p_value': pval,
            'perm_time': perm_time
        }
    except Exception as e:
        print(f"Error with {name} kernel, bandwidth={bw}: {str(e)}")
        return {
            'kernel': name,
            'bandwidth': bw,
            'mmd2': np.nan,
            'p_value': np.nan,
            'perm_time': np.nan
        }

# To measure power, we need to run multiple trials
def measure_power_and_time(data1, data2, kernel_config, n_trials=10):
    """
    Measure statistical power and average permutation time over multiple trials.
    """
    params = kernel_config.copy()
    name = params.pop('kernel')
    bw = params.pop('bandwidth')
    
    rejections = 0
    total_perm_time = 0.0
    
    for trial in range(n_trials):
        # Subsample for each trial to get variation
        if len(data1) > 500:
            idx1 = np.random.choice(len(data1), 500, replace=False)
            sample1 = data1[idx1]
        else:
            sample1 = data1
            
        if len(data2) > 500:
            idx2 = np.random.choice(len(data2), 500, replace=False)
            sample2 = data2[idx2]
        else:
            sample2 = data2
        
        try:
            # Compute statistic
            stat = compute_mmd_stat(
                sample1, sample2,
                kernel=name,
                bandwidth=bw,
                **params
            )
            
            # Time permutation test
            perm_start = time.time()
            from src.permutation_test import permutation_test_statistic
            
            pval = permutation_test_statistic(
                sample1, sample2, 
                stat_fn=compute_mmd_stat,
                num_permutations=100,
                n_jobs=1,
                kernel=name,
                bandwidth=bw,
                **params
            )
            perm_time = time.time() - perm_start
            
            # Count rejection at alpha=0.05
            if pval < 0.05:
                rejections += 1
                
            total_perm_time += perm_time
            
            print(f"Trial {trial+1}/{n_trials}: {name} | p={pval:.4f}, time={perm_time:.2f}s")
            
        except Exception as e:
            print(f"Error in trial {trial+1}: {str(e)}")
    
    # Calculate power and average time
    power = (rejections / n_trials) * 100
    avg_time = total_perm_time / n_trials
    
    return {
        'kernel': name,
        'bandwidth': bw,
        'power': power,
        'avg_perm_time': avg_time,
        'n_trials': n_trials,
        'n': len(sample1)  # Sample size
    }

FIG_DIR = "figures"
RESULTS_DIR = "results"

def main():
    print("Starting MMD experiment with timing...")
    # Ensure output directories exist
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load MNIST data
    data, labels = load_mnist()
    
    # First run: Basic MMD comparison between digits 0 and 1
    print("\n=== Basic MMD Test between Digits 0 and 1 ===")
    digit0_data = subsample_data(data, labels, digit=0, max_samples=1000)
    digit1_data = subsample_data(data, labels, digit=1, max_samples=1000)
    
    print(f"Digit 0: {digit0_data.shape[0]} samples")
    print(f"Digit 1: {digit1_data.shape[0]} samples")
    
    # Run experiments in parallel
    basic_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(digit0_data, digit1_data, cfg) 
        for cfg in KERNEL_EXPERIMENTS
    )
    
    # Convert to DataFrame and save
    basic_df = pd.DataFrame(basic_results)
    basic_df.to_csv(os.path.join(RESULTS_DIR, "basic_mmd_results.csv"), index=False)
    print("\nBasic results summary:\n", basic_df)
    
    """     # Second run: Measure power and time more accurately with multiple trials
        print("\n=== Measuring Power and Time with Multiple Trials ===")
        power_results = []
        for cfg in KERNEL_EXPERIMENTS:
            print(f"\nTesting {cfg['kernel']} kernel, bandwidth={cfg['bandwidth']}...")
            power_result = measure_power_and_time(digit0_data, digit1_data, cfg, n_trials=20)
            power_results.append(power_result)
            print(f"Power: {power_result['power']:.1f}%, Avg Time: {power_result['avg_perm_time']:.3f}s") """
    
    # Convert to DataFrame and save
    #power_df = pd.DataFrame(power_results)
    #power_df.to_csv(os.path.join(RESULTS_DIR, "power_timing_results.csv"), index=False)
    #print("\nPower and timing results summary:\n", power_df)
    
    # Create summary table for paper
    """     paper_results = []
        for result in power_results:
            kernel_name = result['kernel']
            if kernel_name == 'rbf' and result['bandwidth'] == 'median':
                kernel_display = 'RBF (median)'
            elif kernel_name == 'rbf':
                kernel_display = f'RBF ({result["bandwidth"]})'
            elif kernel_name == 'linear':
                kernel_display = 'Linear'
            elif kernel_name == 'poly':
                kernel_display = f'Polynomial (d={cfg["degree"]})'
            elif kernel_name == 'laplace':
                kernel_display = 'Laplace'
            else:
                kernel_display = kernel_name
                
            paper_results.append({
                'Kernel': kernel_display,
                'n=m': result['n'],
                'Power (%)': f"{result['power']:.1f}",
                'Perm. Time (s)': f"{result['avg_perm_time']:.3f}"
            }) """
        
    """     paper_df = pd.DataFrame(paper_results)
        paper_df.to_csv(os.path.join(RESULTS_DIR, "paper_table_data.csv"), index=False)
        print("\nData for paper table:\n", paper_df) """
        
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot MMD statistics
    plt.subplot(1, 3, 1)
    plt.bar(basic_df['kernel'], basic_df['mmd2'])
    plt.yscale('log')
    plt.ylabel("MMD² statistic (log scale)")
    plt.title("MMD² by Kernel")
    plt.xticks(rotation=45)
    
    # Plot p-values
    plt.subplot(1, 3, 2)
    plt.bar(basic_df['kernel'], basic_df['p_value'])
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.ylabel("p-value")
    plt.title("Significance by Kernel")
    plt.xticks(rotation=45)
    plt.legend()
    
    """     # Plot permutation times
        plt.subplot(1, 3, 3)
        plt.bar(power_df['kernel'], power_df['avg_perm_time'])
        plt.ylabel("Time (seconds)")
        plt.title("Avg. Permutation Time by Kernel")
        plt.xticks(rotation=45) """
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd_full_results.png"))
    plt.close()
    
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIG_DIR}")

if __name__ == '__main__':
    main()