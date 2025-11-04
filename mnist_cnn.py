"""
Deep Learning CNN for MNIST Handwritten Digit Classification
Dataset: MNIST Handwritten Digits
Goal: Achieve >95% test accuracy and visualize predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CNN_MNIST(nn.Module):
    """
    Convolutional Neural Network for MNIST Classification
    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - Conv3: 64 -> 128 channels, 3x3 kernel
    - FC1: 128*3*3 -> 256
    - FC2: 256 -> 10 (output classes)
    """
    
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Conv Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Conv Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3
        
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128*3*3)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def load_data(batch_size=64):
    """
    Load and preprocess MNIST dataset
    """
    # Data transformations with augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on test data
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def visualize_predictions(model, device, test_loader, num_samples=5):
    """
    Visualize model predictions on sample images
    """
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices].to(device)
    sample_labels = labels[indices].numpy()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(sample_images)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # Move data to CPU for visualization
    sample_images = sample_images.cpu()
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Display image
        img = sample_images[idx].squeeze()
        ax.imshow(img, cmap='gray')
        
        # Add title with prediction and confidence
        pred_label = predictions[idx]
        true_label = sample_labels[idx]
        confidence = probabilities[idx][pred_label] * 100
        
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(
            f'Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.1f}%',
            color=color,
            fontsize=10
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPrediction visualization saved as 'mnist_predictions.png'")
    plt.show()


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(test_losses, label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(test_accs, label='Test Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history saved as 'training_history.png'")
    plt.show()


def main():
    """
    Main training and evaluation pipeline
    """
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    print("="*60)
    print("MNIST Handwritten Digit Classification with CNN")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_data(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing CNN model...")
    model = CNN_MNIST().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    print("\nStarting training...")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}  | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_mnist_cnn.pth')
            print(f'  ✓ New best accuracy! Model saved.')
        
        print('-' * 60)
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    # Check if goal is achieved
    if best_acc >= 95.0:
        print("✓ Goal achieved: Test accuracy > 95%")
    else:
        print(f"✗ Goal not achieved: Test accuracy {best_acc:.2f}% < 95%")
    
    # Load best model
    model.load_state_dict(torch.load('best_mnist_cnn.pth'))
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Visualize predictions
    print("\nVisualizing predictions on 5 sample images...")
    visualize_predictions(model, device, test_loader, num_samples=5)
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - best_mnist_cnn.pth (trained model)")
    print("  - training_history.png (loss and accuracy plots)")
    print("  - mnist_predictions.png (sample predictions)")


if __name__ == '__main__':
    main()
