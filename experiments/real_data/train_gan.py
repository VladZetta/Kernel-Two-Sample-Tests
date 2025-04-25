import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# Ensure directories exist
base_dir = os.path.dirname(__file__)
results_dir = os.path.join(base_dir, 'results')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ----------------------
#  Models Definition
# ----------------------
class ConvGenerator(nn.Module):
    """DCGAN-style convolutional Generator"""
    def __init__(self, z_dim=128, ngf=64, img_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, img_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        out = z.view(z.size(0), z.size(1), 1, 1)
        img = self.net(out)
        # Crop from 32x32 to 28x28
        img = img[:, :, 2:30, 2:30]
        return img.reshape(z.size(0), -1)

    def generate_samples(self, num_samples, device):
        """Generate samples from the trained generator"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim, device=device)
            samples = self(z)
            # Convert from [-1,1] to [0,1] range
            samples = (samples + 1) / 2
            return samples

class ConvDiscriminator(nn.Module):
    """DCGAN-style convolutional Discriminator"""
    def __init__(self, ndf=64, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad2d(2, 0),
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        img = x.view(x.size(0), 1, 28, 28)
        return self.net(img).view(-1, 1)

# ----------------------
#  Training Functions
# ----------------------
def weights_init(m):
    """Initialize network weights"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def train_gan(
    dataloader,
    G, D,
    g_optimizer, d_optimizer,
    criterion,
    z_dim=128,
    n_epochs=50,
    device='cpu',
    save_interval=10,
    save_dir='models'
):
    """Train the GAN and save weights periodically"""
    print("Starting DCGAN training...")
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    # Track losses for plotting
    g_losses = []
    d_losses = []
    
    # Save initial model
    model_save_path = os.path.join(save_dir, f'gan_initial.pt')
    torch.save({
        'epoch': 0,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, model_save_path)
    print(f"Saved initial model to {model_save_path}")

    # Training loop
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        running_g_loss = 0.0
        running_d_loss = 0.0
        batches = 0
        
        for real_imgs, _ in dataloader:
            batches += 1
            bs = real_imgs.size(0)
            real = real_imgs.to(device).view(bs, -1)
            
            # Train Discriminator
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z)
            
            # Real images - label as 1
            d_real = D(real)
            d_real_loss = criterion(d_real, torch.ones(bs, 1, device=device))
            
            # Fake images - label as 0
            d_fake = D(fake.detach())
            d_fake_loss = criterion(d_fake, torch.zeros(bs, 1, device=device))
            
            # Combined loss
            loss_d = d_real_loss + d_fake_loss
            
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()
            
            # Train Generator
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z)
            d_fake = D(fake)
            loss_g = criterion(d_fake, torch.ones(bs, 1, device=device))
            
            g_optimizer.zero_grad()
            loss_g.backward()
            g_optimizer.step()
            
            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
        
        # Calculate average losses for the epoch
        avg_g_loss = running_g_loss / batches
        avg_d_loss = running_d_loss / batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch}/{n_epochs}   D_loss={avg_d_loss:.4f}   G_loss={avg_g_loss:.4f}   time={time.time()-epoch_start:.1f}s")
        
        # Save model checkpoint at specified intervals
        if epoch % save_interval == 0 or epoch == n_epochs:
            model_save_path = os.path.join(save_dir, f'gan_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, model_save_path)
            print(f"Saved checkpoint at epoch {epoch} to {model_save_path}")
            
            # Generate and save sample images
            save_sample_images(G, epoch, device, save_dir=results_dir)
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gan_training_losses.png'))
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'gan_final.pt')
    torch.save({
        'epoch': n_epochs,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
    }, final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    return G, D, g_losses, d_losses

def save_sample_images(G, epoch, device, num_samples=16, save_dir='results'):
    """Generate and save sample images from the generator"""
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, G.z_dim, device=device)
        samples = G(z)
        # Convert from [-1,1] to [0,1] range
        samples = (samples + 1) / 2
        
    # Reshape to image format
    samples = samples.cpu().view(num_samples, 1, 28, 28)
    
    # Plot images
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i][0], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'DCGAN Samples - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'gan_samples_epoch_{epoch}.png'))
    plt.close()
    
    G.train()

# ----------------------
#  Main Function
# ----------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set hyperparameters
    z_dim = 128
    lr = 2e-4
    batch_size = 128
    n_epochs = 50
    save_interval = 5  # Save weights every 5 epochs
    
    # Prepare GAN training data ([-1,1])
    gan_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    gan_dataset = datasets.MNIST('data/', train=True, transform=gan_transform, download=True)
    gan_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    G = ConvGenerator(z_dim=z_dim).to(device)
    D = ConvDiscriminator().to(device)
    
    # Initialize optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    gan_criterion = nn.BCELoss()
    
    # Train GAN
    G, D, g_losses, d_losses = train_gan(
        gan_loader, G, D, g_optimizer, d_optimizer, gan_criterion,
        z_dim=z_dim, n_epochs=n_epochs, device=device,
        save_interval=save_interval, save_dir=models_dir
    )
    
    print("Training complete!")

if __name__ == '__main__':
    main()