import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure directories exist
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, 'models')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ----------------------
#  Model Definition
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

# ----------------------
#  Training Function
# ----------------------
def train_classifier(loader, model, optimizer, criterion, device, n_epochs=3, save_path=None):
    """
    Train the classifier model on MNIST.
    """
    model.to(device)
    model.train()
    
    # Track metrics
    train_losses = []
    train_accuracies = []
    
    for epoch in range(1, n_epochs+1):
        total_loss = 0.0
        correct = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / len(loader.dataset)
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Classifier Epoch {epoch}/{n_epochs}  Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    
    # Save the trained model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'accuracy': train_accuracies,
            'n_epochs': n_epochs,
        }, save_path)
        print(f"Saved classifier model to {save_path}")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs+1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs+1), train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'classifier_training.png'))
    
    # Set to evaluation mode
    model.eval()
    return model, train_losses, train_accuracies

# ----------------------
#  Evaluation Function
# ----------------------
def evaluate_classifier(model, test_loader, device):
    """
    Evaluate the classifier on test data.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize some predictions
    model.eval()
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            if i >= 18:
                break
                
            img = img.to(device)
            output = model(img)
            pred = output.argmax(dim=1).item()
            
            ax = axes[i//6, i%6]
            ax.imshow(img[0, 0].cpu(), cmap='gray')
            ax.set_title(f"True: {label.item()}\nPred: {pred}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'classifier_predictions.png'))
    
    return accuracy

# ----------------------
#  Main Function
# ----------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    n_epochs = 5  # Increased from 3 for better accuracy
    
    # Load MNIST dataset
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST('data/', train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST('data/', train=False, transform=test_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded MNIST dataset: {len(train_dataset)} training, {len(test_dataset)} test samples")
    
    # Initialize model, loss function and optimizer
    classifier = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    # Train classifier
    save_path = os.path.join(models_dir, 'mnist_classifier.pt')
    print("Training classifier...")
    classifier, train_losses, train_accuracies = train_classifier(
        train_loader, classifier, optimizer, criterion, 
        device, n_epochs=n_epochs, save_path=save_path
    )
    
    # Evaluate on test set
    print("Evaluating classifier on test set...")
    test_accuracy = evaluate_classifier(classifier, test_loader, device)
    
    print("Classifier training and evaluation complete!")

if __name__ == '__main__':
    main()