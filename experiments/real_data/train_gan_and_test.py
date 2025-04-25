# experiments/real_data/train_gan_and_test.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt

# MMD
from src.mmd import mmd_test
from experiments.real_data.preprocess_mnist import load_mnist

# Ensure results directory exists
tmp_dir = os.path.dirname(__file__)
results_dir = os.path.join(tmp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# ----------------------
#  Models Definition
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


def train_classifier(loader, model, optimizer, criterion, device, n_epochs=3):
    """
    Train the classifier model on MNIST.
    """
    model.to(device)
    model.train()
    for epoch in range(1, n_epochs+1):
        total_loss = 0.0
        correct = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = correct / len(loader.dataset)
        print(f"Classifier Epoch {epoch}/{n_epochs}  Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    model.eval()

class ConvGenerator(nn.Module):
    """DCGAN-style convolutional Generator"""
    def __init__(self, z_dim=128, ngf=64, img_channels=1):
        super().__init__()
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
def train_gan(
    dataloader,
    G, D,
    g_optimizer, d_optimizer,
    criterion,
    z_dim=128,
    n_epochs=50,
    device='cpu'
):
    print("Starting DCGAN training...")
    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    G.apply(weights_init)
    D.apply(weights_init)

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        for real_imgs, _ in dataloader:
            bs = real_imgs.size(0)
            real = real_imgs.to(device).view(bs, -1)
            # Train Discriminator
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z)
            loss_d = criterion(D(real), torch.ones(bs, 1, device=device)) + \
                     criterion(D(fake.detach()), torch.zeros(bs, 1, device=device))
            d_optimizer.zero_grad(); loss_d.backward(); d_optimizer.step()
            # Train Generator
            loss_g = criterion(D(fake), torch.ones(bs, 1, device=device))
            g_optimizer.zero_grad(); loss_g.backward(); g_optimizer.step()
        print(f"Epoch {epoch}/{n_epochs}   D_loss={loss_d.item():.4f}   G_loss={loss_g.item():.4f}   time={time.time()-epoch_start:.1f}s")

# ----------------------
#  Evaluation Function
# ----------------------
def evaluate_mmd(
    G, classifier, real_data,
    z_dim=128, n_gen=1000,
    device='cpu', show_images=True
):
    print("Starting MMD evaluation...")
    G.eval()
    idx = np.random.choice(len(real_data), n_gen, replace=False)
    real = real_data[idx]
    with torch.no_grad():
        z = torch.randn(n_gen, z_dim, device=device)
        fake_flat = G(z).cpu().numpy()
    fake = (fake_flat + 1) / 2

    if show_images:
        num_show = 16
        samples = fake[:num_show].reshape(-1, 1, 28, 28)
        imgs_tensor = torch.tensor(samples, device=device, dtype=torch.float32)
        preds = classifier(imgs_tensor).argmax(1).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i][0], cmap='gray')
            ax.set_title(str(preds[i].item()))
            ax.axis('off')
        plt.suptitle('DCGAN-generated digits')
        plt.tight_layout()
        save_path = os.path.join(results_dir, 'dcgan_generated.png')
        plt.savefig(save_path)
        print(f"Saved images: {save_path}")
        plt.show()

    stat, pval = mmd_test(real, fake, kernel='rbf', bandwidth='median', preprocess=False, return_p_value=True, num_permutations=200)
    print(f"[MMD] stat={stat:.4f}, pval={pval:.4f}")
    return stat, pval

# ----------------------
#  Main Script
# ----------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load MNIST for real data
    data, _ = load_mnist()

    # Train classifier on raw MNIST ([0,1])
    clf_transform = transforms.ToTensor()
    clf_dataset = datasets.MNIST('data/', train=True, transform=clf_transform, download=True)
    clf_loader = DataLoader(clf_dataset, batch_size=256, shuffle=True)
    classifier = Classifier().to(device)
    clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    clf_criterion = nn.CrossEntropyLoss()
    print("Training classifier...")
    train_classifier(clf_loader, classifier, clf_optimizer, clf_criterion, device, n_epochs=3)

    # Prepare GAN training data ([-1,1])
    gan_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    gan_dataset = datasets.MNIST('data/', train=True, transform=gan_transform, download=True)
    gan_loader = DataLoader(gan_dataset, batch_size=128, shuffle=True)

    # Initialize and train DCGAN
    G = ConvGenerator().to(device)
    D = ConvDiscriminator().to(device)
    g_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gan_criterion = nn.BCELoss()
    print("Training DCGAN...")
    train_gan(gan_loader, G, D, g_optimizer, d_optimizer, gan_criterion, z_dim=128, n_epochs=50, device=device)

    # Evaluate generated samples with classifier and MMD
    evaluate_mmd(G, classifier, data, z_dim=128, n_gen=1000, device=device)

if __name__ == '__main__':
    main()