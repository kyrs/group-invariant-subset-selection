import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bar
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from dataloader import MNISTDataset 
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
# from models import Model, ClassicalModel
from utils import grouped_nt_xent_loss
import torch.nn.functional as F
# --- 1. Define the ResNet9 Architecture ---

class classifierModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)  # Or replace with AdaptiveAvgPool2d if you want
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        feature = self.dropout(x)   # feature before final classification layer
        logits = self.fc(feature)
        return logits, feature

class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dataset = "rot_mnist"):
        super().__init__()

        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))

        # self.classifier = nn.Sequential(
        #     nn.MaxPool2d(3), # Global average pooling for 7x7 -> 1x1 (for MNIST 28x28 images, after 3 pooling layers, it becomes 3x3 or 4x4, so max pool 4)
        #     nn.Flatten(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, num_classes)
        # )
        self.classifier = classifierModel(num_classes)
    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out # Residual connection
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out # Residual connection
        # print("inside classifier: ",out.size())
        out, feature = self.classifier(out)
        return out, feature

# --- 2. Load and Preprocess MNIST Dataset ---
# def load_mnist_data(batch_size=128):
#     # Define transformations for the MNIST dataset
#     # MNIST images are 28x28, grayscale. ResNet9 typically expects 3 channels,
#     # so we will convert to RGB (though it's just repeating the single channel)
#     # and then resize to 32x32 as ResNet models are often trained on larger images.
#     transform = transforms.Compose([
#         transforms.Resize(32), # Resize to 32x32
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std for MNIST
#     ])

#     # Download and load the training data
#     train_dataset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform
#     )
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#     # Download and load the test data
#     test_dataset = torchvision.datasets.MNIST(
#         root='./data', train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2)

#     return train_loader, test_loader
def get_cifar10_dataloaders(
    batch_size: int = 64,
    val_split_ratio: float = 0.1,
    num_workers: int = 2,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoader instances for CIFAR-10 training, validation, and test sets.

    Args:
        batch_size (int): The batch size for the DataLoaders. Defaults to 64.
        val_split_ratio (float): The ratio of the training data to use for the validation set.
                                 Defaults to 0.1 (10%).
        num_workers (int): How many subprocesses to use for data loading. 0 means that the data
                           will be loaded in the main process. Defaults to 2.
        seed (int): Random seed for reproducibility of the train/validation split. Defaults to 42.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A tuple containing (train_loader, val_loader, test_loader).
    """

    # Define transformations for the CIFAR-10 dataset
    # Normalize with mean and standard deviation for CIFAR-10
    # These values are standard for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load the CIFAR-10 training dataset
    # download=True ensures the dataset is downloaded if not already present
    train_val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Load the CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Calculate the size of the validation set
    total_train_size = len(train_val_dataset)
    val_size = int(total_train_size * val_split_ratio)
    train_size = total_train_size - val_size

    # Split the training dataset into training and validation sets
    # Use a fixed seed for reproducibility of the split
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Create DataLoader instances
    # shuffle=True for training data to ensure randomness in batches
    # shuffle=False for validation and test data as order does not matter
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"CIFAR-10 DataLoaders created:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, test_loader
    
def load_mnist_data(batch_size = 512):
    file_path = "../data/ROTMNIST/mnist_all_rotation_normalized_float_combined.amat" 
    ds = MNISTDataset(file_path)
    # dl = DataLoader(ds, batch_size=512)
    total_samples = len(ds)
    # l = [int(d[1]) for d in ds]
    # l = np.array(l)
    # print(np.unique(l))
    # print(ds[0][1])
    # sys.exit(0)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    batch_size = 512 # Define your desired batch size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle validation/test
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
# --- 3. Training and Evaluation Functions ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.train() # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:, None]
            labels = labels.long()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (i + 1))
        print("Validation===>")
        evaluate_model(model, val_loader, device = device)

        print(f"Epoch {epoch+1} finished. Average Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation during inference
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:, None]
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy

# --- Main execution block ---
if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    # Load data
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    # MNIST images are 1 channel, but ResNet9 expects 3 channels due to common usage.
    # The transform 'transforms.ToTensor()' will create a 1-channel tensor.
    # We can either modify ResNet9's first conv layer to accept 1 channel,
    # or ensure the input is transformed to 3 channels.
    # For simplicity and to fit the common ResNet structure, we'll assume the
    # input transform handles the channel expansion (though for MNIST, it's often
    # just repeating the single channel).
    # Let's adjust the first conv layer to accept 1 channel directly for MNIST.
    model = ResNet9(in_channels=1, num_classes=10).to(device)
    # If you want to force 3 channels, you'd add:
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    # into your transform pipeline *after* ToTensor().
    # However, for MNIST, it's more appropriate to handle 1 channel directly.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=NUM_EPOCHS)

    print("\n--- Starting Evaluation ---")
    evaluate_model(model, test_loader, device)

    print("\n--- Training Complete ---")
    # You can save the model if needed
    # torch.save(model.state_dict(), 'resnet9_mnist.pth')
    # print("Model saved to resnet9_mnist.pth")
