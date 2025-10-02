import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple two-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        # Define the layers: input -> hidden -> output
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()  # Non-linear activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second layer (output)

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)  # Apply first layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Apply second layer
        return x

class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# Define a function to train the network (with batch processing)
def train(model, train_features, train_labels, batch_size=32, epochs=100, learning_rate=0.001):
    # Create a DataLoader for batching the data
    dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()  # Loss function (assuming classification task)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=100)  # Optimizer
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_features)

            # Compute the loss
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # Print the loss for this epoch
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Define a function to test the model
def test(model, test_features, test_labels):
    # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class (max probability)
        
        # Calculate accuracy
        correct = (predicted == test_labels).sum().item()
        accuracy = correct / len(test_labels) * 100
        print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy, predicted


def neural_network(train_data, test_data, input_size, learning_rate, hidden_size = 32, output_size = 2, epoch =20, batch_size = 1024, device = "cuda:0"):
    model = TwoLayerNN(input_size, hidden_size, output_size).to(device)

    train_features, train_labels = train_data
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    train(model, train_features, train_labels, batch_size=batch_size, epochs=epoch, learning_rate= learning_rate)

    test_features, test_labels = test_data
    test_features, test_labels = test_features.to(device), test_labels.to(device)

        # Test the model
    acc, y_hat = test(model, test_features, test_labels)
    return acc, y_hat


def logisitic_regression(train_data, test_data, input_size, learning_rate, output_size = 2, epoch = 20, batch_size = 1024, device = "cuda:0"):
    model = LogisticRegression(input_size, output_size).to(device)

    train_features, train_labels = train_data
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    train(model, train_features, train_labels, batch_size=batch_size, epochs=epoch, learning_rate= learning_rate)

    test_features, test_labels = test_data
    test_features, test_labels = test_features.to(device), test_labels.to(device)

        # Test the model
    acc, y_hat = test(model, test_features, test_labels)
    return acc, y_hat