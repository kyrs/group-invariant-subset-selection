import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from copy import deepcopy
from src.real_world_exp.so2.classifier import ResNet9
from src.real_world_exp.se3.se3_classifier import GNNSimpeClassifier
from src.utils import BasicEdgeGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
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
        fet = self.fc1(x)  # Apply first layer
        x = self.relu(fet)  # Apply ReLU activation
        x = self.fc2(x)  # Apply second layer
        return x,fet  # Return output and features from the first layer

class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

class IndexedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return idx, self.features[idx], self.labels[idx]



def trackerInfo( batch_labels, indices, outputs, fet, model=None, args=None):
    trackerInfoDict={}
    with torch.no_grad():
        # Placeholder for tracker information update logic
        # This function should compute and return the necessary tracker information
        # based on the batch features, labels, and indices.
        pred_labels = torch.argmax(outputs, dim = 1)
        curr_acc = (pred_labels==batch_labels).clone().detach().requires_grad_(False).type(torch.float32)
        # criterion = nn.CrossEntropyLoss()
        probabilities = F.softmax(outputs, dim=1)
        
        # Assert: Each row (sample) should sum to 1 (within tolerance)
        row_sums = probabilities.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Probabilities do not sum to 1"
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        
        true_one_hot = F.one_hot(batch_labels, num_classes=args.classNum).float()


    assert (len(curr_acc) == len(indices)== len(entropy)== len(fet)), "Shape mismatch in tracker information"
    # assert (len(curr_acc) == len(grandList), "Shape mismatch in tracker information"
    for idx, curr_acc, etp, feature in zip(indices, curr_acc, entropy, fet):
        i = int(idx.item())  # Convert index to integer 
        trackerInfoDict[i] = {
            "forget": curr_acc.item(),  # Current accuracy
            "fet" : feature.clone().detach().requires_grad_(False).cpu().numpy(),  # Features from the first layer
            "entropy" : etp.item()  # Entropy of the output probabilities
        }
           
    return trackerInfoDict

def updateTracker(trackerDict, trackerInfoDict, epoch, epoch_subset_cutoff =10, epoch_final_cutoff = 20):
    for i, info in trackerInfoDict.items():
        trackerDict[i]["epoch"] =epoch  # Increment epoch count 
        trackerDict[i]["forget"] += (trackerDict[i]["last_ind"] - info["forget"])>0.01
        trackerDict[i]["last_ind"] = info["forget"]  # Update last index processed

        if (epoch==(epoch_final_cutoff)-1):
            trackerDict[i]["feature"] = info["fet"]
            trackerDict[i]["entropy"] = info["entropy"]
        else:
            pass 
        

    # Ensure all entries in trackerDict have the same epoch value

    return trackerDict

# Define a function to train the network (with batch processing)
def train(model, train_features, train_labels, batch_size=32, epochs=100, learning_rate=0.001, returnTracker=False, args=None):
    # Create a DataLoader for batching the data
    print("Training with batch size:", batch_size)
    print("learning rate:", learning_rate)
    dataset = IndexedDataset(train_features, train_labels)
    
    trackerDict = {i:{"epoch":0, "forget":0, "last_ind": 0, "feature":0, "entropy": 0} for i in range(len(train_features))}
    # trackerDict = {i:{"epoch":0, "forget":0, "grand" : 0, "last_ind": 0} for i in range(len(train_features))}
    if args.data in ["tetris"]:
        edgGraphDataset = BasicEdgeGraphDataset(dataset=dataset, oneHotFetFlag=True)
        train_loader = PyGDataLoader(edgGraphDataset, batch_size=batch_size, shuffle=True)


    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()  # Loss function (assuming classification task)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        epoch_loss = 0
        correct, total = 0, 0
        for batch in train_loader:
            # print(batch)
            if args.data in ["tetris"]:
                indices = batch.idx
                batch_labels= batch.y
                batch_labels = batch_labels.long()  # Ensure labels are of type Long for CrossEntropyLoss
                outputs, fet = model(batch)
                
            else:
                indices, batch_features, batch_labels = batch
                # Zero the gradients
                batch_labels = batch_labels.long()  # Ensure labels are of type Long for CrossEntropyLoss
                batch_features = batch_features.float()  # Ensure features are of type Float

                # Forward pass
                outputs, fet = model(batch_features)
           
            # Compute the loss
            optimizer.zero_grad()
            loss = criterion(outputs, batch_labels)
            # Backward pass and optimization
            loss.backward()
            # input()
            optimizer.step()

            epoch_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
            ## NOTE : Ddo we need a zero grad here?
            # model_clone = deepcopy(model)
            trakerInfoDict = trackerInfo(batch_labels, indices, outputs, fet, model, args=args) 
            trackerDict = updateTracker(trackerDict, trakerInfoDict, epoch, epoch_final_cutoff=epochs)
        avg_loss = epoch_loss / len(train_loader)
        acc = 100 * correct / total
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f} training accuracy: {acc:.2f}%')
        # Update tracker information for the epoch
        
        

    #Extract all "epoch" values
    epoch_values = [entry["epoch"] for entry in trackerDict.values()]
    # Assert all are equal to the first one
    assert all(epoch == epoch_values[0] for epoch in epoch_values), "Inconsistent epoch values in trackerDict."
    if returnTracker:
        return trackerDict, model
    else:
        return None, model # Return nothing if tracker is not needed

# Define a function to test the model
def test(model, test_features, test_labels, args=None, batch_size=32):

    test_dataset = IndexedDataset(test_features, test_labels)
    if args.data in ["tetris"]:
        edgGraphDataset = BasicEdgeGraphDataset(dataset=test_dataset, oneHotFetFlag=True)
        test_loader = PyGDataLoader(edgGraphDataset, batch_size=batch_size, shuffle=False)

    else:
       
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()

    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            if args.data in ["tetris"]:
                indices = batch.idx
                batch_labels= batch.y
                labels = batch_labels.long()  # Ensure labels are of type Long for CrossEntropyLoss
                outputs, _ = model(batch)
                
                
            else:
                indices, features, labels = batch
                outputs, _ = model(features)

            _, predicted = torch.max(outputs, 1)
            

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())

    accuracy = correct / total * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy, all_preds


def classifier(train_data, test_data, input_size, learning_rate, batch_size, hidden_size = 32, output_size = 2, epoch = 20, device = "cuda:0", returnTracker=False, args=None):
    if args.data in ["rot_mnist"]:
        model = ResNet9(in_channels=1, num_classes=10, dataset = args.data).to(device)
    elif args.data in ["tetris"]:
        model = GNNSimpeClassifier().to(device)

    else:
        model = TwoLayerNN(input_size, hidden_size, output_size).to(device)

    train_features, train_labels = train_data
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    trackerDict, model = train(model, train_features, train_labels, batch_size=batch_size, epochs=epoch, learning_rate= learning_rate, returnTracker=returnTracker,  args=args)

    test_features, test_labels = test_data
    test_features, test_labels = test_features.to(device), test_labels.to(device)

        # Test the model
    acc, y_hat = test(model, test_features, test_labels, args=args, batch_size=batch_size)
    return acc, y_hat, trackerDict 


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