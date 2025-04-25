import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
import torch.nn as nn

# Import necessary functions from your modules
from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd import mmd_test
from .train_gan import ConvGenerator, ConvDiscriminator


# ----------------------
#  Classifier Model
# ----------------------
class Classifier(nn.Module):
    """
    Simple convolutional classifier for MNIST digits.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    


# Define kernel configurations for experiment
KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',     'bandwidth':'median'},
    {'kernel':'rbf',     'bandwidth':1.0},
    {'kernel':'linear',  'bandwidth':None},
    {'kernel':'poly',    'bandwidth':None, 'degree':2, 'coef0':0},
    {'kernel':'poly',    'bandwidth':None, 'degree':3, 'coef0':1},
    {'kernel':'laplace', 'bandwidth':'median'},
    {'kernel':'laplace', 'bandwidth':0.5},
]

# Define directories
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def subsample_data(data, labels, digit, max_samples=1000):
    """
    Filter data for a specific digit and subsample if needed.
    """
    digit_data = data[labels == digit]
    if digit_data.shape[0] > max_samples:
        indices = np.random.choice(digit_data.shape[0], max_samples, replace=False)
        digit_data = digit_data[indices]
    return digit_data

def load_gan_model(model_path, z_dim=128, device='cpu'):
    """
    Load a trained GAN generator model.
    """
    G = ConvGenerator(z_dim=z_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded generator from epoch {epoch}")
    
    G.eval()
    return G

def generate_samples(G, n_samples=1000, z_dim=128, device='cpu'):
    """
    Generate samples from the trained generator.
    """
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        samples = G(z).cpu().numpy()
        # Convert from [-1,1] to [0,1] range
        samples = (samples + 1) / 2
    return samples

def  compare_real_vs_generated(real_data, gen_data, num_permutations=200):
    """
    Compare real and generated data using different kernel configurations.
    """
    results = []
    for cfg in KERNEL_EXPERIMENTS:
        params = cfg.copy()
        name = params.pop('kernel')
        bw = params.pop('bandwidth')
        print(f"\nRunning MMD test with kernel: {name}, bandwidth: {bw}")
        stat, pval = mmd_test(
            real_data, gen_data,
            kernel=name,
            bandwidth=bw,
            return_p_value=True,
            num_permutations=num_permutations,
            **params
        )
        
        print(f"{name:7s} | bw={str(bw):6s} → MMD²={stat:.4e}, p={pval:.4f}")
        
        results.append({
            'kernel': name,
            'bandwidth': bw,
            'mmd2': stat,
            'p_value': pval
        })
    
    return results

def compare_across_digits(real_data, labels, gen_data, digit_preds):
    """
    Compare MMD test results across different digits.
    """
    digit_results = {}
    
    for digit in range(10):
        print(f"\nAnalyzing digit {digit}:")
        
        # Get real samples for this digit
        real_digit_samples = subsample_data(real_data, labels, digit, max_samples=500)
        
        # Get generated samples classified as this digit
        gen_digit_indices = np.where(digit_preds == digit)[0]
        if len(gen_digit_indices) > 500:
            gen_digit_indices = np.random.choice(gen_digit_indices, 500, replace=False)
        
        if len(gen_digit_indices) < 20:
            print(f"Too few generated samples for digit {digit} (found {len(gen_digit_indices)}), skipping...")
            continue
            
        gen_digit_samples = gen_data[gen_digit_indices]
        
        print(f"Real samples: {len(real_digit_samples)}, Generated samples: {len(gen_digit_samples)}")
        
        # Run MMD test with RBF kernel
        stat, pval = mmd_test(
            real_digit_samples, gen_digit_samples,
            kernel='rbf',
            bandwidth='median',
            return_p_value=True,
            num_permutations=100
        )
        
        digit_results[digit] = {
            'mmd2': stat,
            'p_value': pval,
            'real_count': len(real_digit_samples),
            'gen_count': len(gen_digit_samples)
        }
        
        print(f"Digit {digit}: MMD²={stat:.4e}, p={pval:.4f}")
    
    return digit_results

def plot_digit_distribution(digit_results):
    """
    Plot MMD² and p-values by digit.
    """
    digits = sorted(digit_results.keys())
    mmd2_values = [digit_results[d]['mmd2'] for d in digits]
    p_values = [digit_results[d]['p_value'] for d in digits]
    
    # Plot MMD²
    plt.figure(figsize=(10, 5))
    plt.bar(digits, mmd2_values)
    plt.xlabel("Digit")
    plt.ylabel("MMD² statistic")
    plt.title("MMD² by digit")
    plt.xticks(digits)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd2_by_digit.png"))
    plt.close()
    
    # Plot p-values
    plt.figure(figsize=(10, 5))
    plt.bar(digits, p_values)
    plt.xlabel("Digit")
    plt.ylabel("Permutation p-value")
    plt.title("Significance by digit")
    plt.xticks(digits)
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pvalue_by_digit.png"))
    plt.close()

def plot_kernel_results(results_df):
    """
    Plot MMD² and p-values by kernel.
    """
    # Bar plot of MMD²
    plt.figure(figsize=(10, 5))
    plt.bar(results_df['kernel'], results_df['mmd2'])
    plt.ylabel("MMD² statistic")
    plt.title("MMD² by kernel")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd2_by_kernel.png"))
    plt.close()

    # Bar plot of p-value
    plt.figure(figsize=(10, 5))
    plt.bar(results_df['kernel'], results_df['p_value'])
    plt.ylabel("Permutation p-value")
    plt.title("Significance by kernel")
    plt.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pvalue_by_kernel.png"))
    plt.close()

def compare_model_checkpoints(checkpoints, real_data, num_samples=1000, z_dim=128, device='cpu'):
    """
    Compare MMD scores across different model checkpoints.
    """
    checkpoint_results = []
    
    for checkpoint_path in checkpoints:
        epoch = os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]
        if epoch == 'final':
            epoch = 'Final'
        
        print(f"\nEvaluating checkpoint: {epoch}")
        
        # Load generator
        G = load_gan_model(checkpoint_path, z_dim=z_dim, device=device)
        
        # Generate samples
        gen_samples = generate_samples(G, n_samples=num_samples, z_dim=z_dim, device=device)
        
        # Calculate MMD with RBF kernel
        stat, pval = mmd_test(
            real_data, gen_samples,
            kernel='rbf',
            bandwidth='median',
            return_p_value=True,
            num_permutations=100
        )
        
        checkpoint_results.append({
            'epoch': epoch,
            'mmd2': stat,
            'p_value': pval
        })
        
        print(f"Epoch {epoch}: MMD²={stat:.4e}, p={pval:.4f}")
    
    return checkpoint_results

def plot_checkpoint_evolution(checkpoint_results):
    """
    Plot MMD² evolution over training epochs.
    """
    epochs = [result['epoch'] for result in checkpoint_results]
    mmd2_values = [result['mmd2'] for result in checkpoint_results]
    
    plt.figure(figsize=(10, 5))
    plt.plot(mmd2_values, marker='o')
    plt.xlabel("Checkpoint")
    plt.ylabel("MMD² statistic")
    plt.title("MMD² evolution over training")
    plt.xticks(range(len(epochs)), epochs, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd2_evolution.png"))
    plt.close()

def main():
    time_start = time.time()
    print("Starting GAN evaluation experiment...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load MNIST data
    real_data, labels = load_mnist()
    print(f"Loaded real MNIST data: {real_data.shape}")
    
    # Load the trained classifier
    classifier_path = os.path.join(MODELS_DIR, 'mnist_classifier.pt')
    if os.path.exists(classifier_path):
        classifier = Classifier().to(device)
        checkpoint = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        print("Loaded classifier for digit prediction")
    else:
        print("Warning: Classifier not found. Digit-specific analysis will be skipped.")
        classifier = None
    
    # Load the final GAN model
    gan_model_path = os.path.join(MODELS_DIR, 'gan_final.pt')
    if not os.path.exists(gan_model_path):
        print(f"Error: GAN model not found at {gan_model_path}")
        return
    
    z_dim = 128
    G = load_gan_model(gan_model_path, z_dim=z_dim, device=device)
    
    # Generate samples
    n_samples = 1000
    print(f"Generating {n_samples} samples...")
    gen_samples = generate_samples(G, n_samples=n_samples, z_dim=z_dim, device=device)
    
    # First, compare real vs generated data with different kernels
    print("\nComparing real vs. generated data with different kernels:")
    n_real = min(n_samples, len(real_data))
    idx = np.random.choice(len(real_data), n_real, replace=False)
    real_subsample = real_data[idx]
    
    kernel_results = compare_real_vs_generated(real_subsample, gen_samples, num_permutations=100)
    results_df = pd.DataFrame(kernel_results)
    print("\nSummary of kernel results:\n", results_df)
    
    # Plot kernel-specific results
    plot_kernel_results(results_df)
    
    # If classifier is available, perform digit-specific analysis
    if classifier is not None:
        print("\nPerforming digit-specific analysis...")
        gen_tensor = torch.tensor(gen_samples.reshape(-1, 1, 28, 28), device=device, dtype=torch.float32)
        with torch.no_grad():
            digit_preds = classifier(gen_tensor).argmax(dim=1).cpu().numpy()
        
        # Get unique predicted digits and their counts
        unique_preds, counts = np.unique(digit_preds, return_counts=True)
        print("\nGenerated digit distribution:")
        for digit, count in zip(unique_preds, counts):
            print(f"Digit {digit}: {count} samples ({count/len(digit_preds)*100:.1f}%)")
        
        # Compare across digits
        digit_results = compare_across_digits(real_data, labels, gen_samples, digit_preds)
        
        # Plot digit-specific results
        if digit_results:
            plot_digit_distribution(digit_results)
    
    # Compare different checkpoints if available
    checkpoint_paths = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) 
                       if f.startswith('gan_epoch_') and f.endswith('.pt')]
    checkpoint_paths.append(gan_model_path)  # Add the final model
    
    if len(checkpoint_paths) > 1:
        print("\nComparing model checkpoints:")
        checkpoint_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]) 
                             if 'epoch' in x else float('inf'))
        
        checkpoint_results = compare_model_checkpoints(
            checkpoint_paths, real_subsample, 
            num_samples=n_samples, z_dim=z_dim, device=device
        )
        
        plot_checkpoint_evolution(checkpoint_results)
    
    end_time = time.time()  
    print(f"\nTotal experiment time: {end_time - time_start:.2f} seconds")
    print(f"Results and figures saved to '{FIG_DIR}/'")

if __name__ == '__main__':
    main()