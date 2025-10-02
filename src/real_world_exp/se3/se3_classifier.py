import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from torch_cluster import knn_graph
# from src.utils import BasicEdgeGraphDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataloader import TetrisDataset, TetrisDatasetPyG, TetrisDataGenerator
# from model_with_baselines import IndexedDataset
# from utils import BasicEdgeGraphDataset
class BasicEdgeGraphDataset(PyGDataset):
    def __init__(self, dataset, oneHotFetFlag, k=3 ):
        super().__init__()
        self.dataset = dataset
        self.k = k
        self.oneHotFetFlag = oneHotFetFlag
        self.num_elements = len(dataset)

    def len(self):
        return self.num_elements

    def get(self, idx):
        idx, pos, label = self.dataset[idx]
        n = pos.size(0)

        # One-hot encoding for nodes
        if self.oneHotFetFlag:
            x = torch.eye(n)
        else:
            raise NotImplementedError("Only one-hot encoding is supported for now.")
        

        # Build edge index using k-NN
        edge_index = knn_graph(pos, k=self.k, loop=False)

        # Compute edge attributes (Euclidean distances)
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)  # [E, 1]
        edge_attr = edge_attr.to(pos.device)  # Ensure edge_attr is on the same device as pos
        x =x.to(pos.device)

        return Data(pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, idx=idx)
# ----------------------------
# GNN Classifier with NNConv
# ----------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import NNConv, global_mean_pool

# ----------------------------
# TETRIS BASE DATASET
# ----------------------------
class TetrisDataset(torch.utils.data.Dataset):
    def __init__(self, num_elements, rotate=True, noise_level=0.01, translation_level=5.0):
        self.num_elements = num_elements
        self.rotate = rotate
        self.noise_level = noise_level
        self.translation_level = translation_level

        self.pos = torch.tensor([
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
        ], dtype=torch.float)

        self.labels = torch.eye(8)
        self.shape_names = ['L-piece', 'J-piece', 'O-piece', 'I-piece',
                            'T-piece', 'Custom 1', 'Custom 2', 'S/Z-piece']

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        shape_idx = torch.randint(0, len(self.pos), (1,)).item()
        shape = self.pos[shape_idx].clone()
        label = self.labels[shape_idx]

        if self.rotate:
            R = torch.linalg.qr(torch.randn(3, 3)).Q
            shape = shape @ R.T

        t = torch.randn(3) * self.translation_level
        shape = shape + t

        if self.noise_level > 0:
            noise = torch.randn_like(shape) * self.noise_level
            shape = shape + noise

        return shape, label, shape_idx

# class IndexedDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return idx, self.features[idx], self.labels[idx]


# ----------------------------
# PyG Dataset Adapter
# ----------------------------
class TetrisDatasetPyG(PyGDataset):
    def __init__(self, num_elements, dataset=None, k=3):
        super().__init__()
        self.num_elements = num_elements
        self.dataset = dataset if dataset else TetrisDataset(num_elements)
        self.k = k

    def len(self):
        return self.num_elements

    def get(self, idx):
        pos, label, shape_idx = self.dataset[idx]
        n = pos.size(0)

        # One-hot encoding for nodes
        x = torch.eye(n)  # [4, 4]

        # Build edge index using k-NN
        edge_index = knn_graph(pos, k=self.k, loop=False)

        # Compute edge attributes (Euclidean distances)
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)  # [E, 1]

        return Data(pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(shape_idx), idx=torch.tensor(idx))




class GNNQM9Classifier(nn.Module):
    def __init__(self, input_dim=11, edge_dim=4, hidden_dim=64, num_classes=2):
        super().__init__()
        self.nn1 = nn.Sequential(nn.Linear(edge_dim, hidden_dim * input_dim), nn.ReLU())
        self.nn2 = nn.Sequential(nn.Linear(edge_dim, hidden_dim * hidden_dim), nn.ReLU())
        self.nn3 = nn.Sequential(nn.Linear(edge_dim, 32 * hidden_dim), nn.ReLU())

        self.conv1 = NNConv(input_dim, hidden_dim, self.nn1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.nn2, aggr='mean')
        self.conv3 = NNConv(hidden_dim, 32, self.nn3, aggr='mean')

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        return self.classifier(x), x


class GNNSimpeClassifier(nn.Module):
    def __init__(self, input_dim=4, edge_dim=1, hidden_dim=64, num_classes=8):
        super().__init__()
        self.nn1 = nn.Sequential(nn.Linear(edge_dim, hidden_dim * input_dim), nn.ReLU())
        self.nn2 = nn.Sequential(nn.Linear(edge_dim, hidden_dim * hidden_dim), nn.ReLU())
        self.nn3 = nn.Sequential(nn.Linear(edge_dim, 32 * hidden_dim), nn.ReLU())

        self.conv1 = NNConv(input_dim, hidden_dim, self.nn1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.nn2, aggr='mean')
        self.conv3 = NNConv(hidden_dim, 32, self.nn3, aggr='mean')

        self.classifier = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        return self.classifier(x), x


# ----------------------------
# Training & Evaluation
# ----------------------------
def train(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        else:
            pass
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, _ = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.y.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / val_total)

        # scheduler.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}: Train Acc = {train_accs[-1]:.2f}%, Val Acc = {val_accs[-1]:.2f}%")

    return train_losses, val_losses, train_accs, val_accs


def evaluate(model, test_loader, dataset):
    device = next(model.parameters()).device
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _ = model(data)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=dataset.dataset.shape_names))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.dataset.shape_names,
                yticklabels=dataset.dataset.shape_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return acc

def test(model, test_features, test_labels, args=None, batch_size=32):
    
    test_dataset = IndexedDataset(test_features, test_labels)
    edgGraphDataset = BasicEdgeGraphDataset(dataset=test_dataset, oneHotFetFlag=True)
    test_loader = PyGDataLoader(edgGraphDataset, batch_size=batch_size, shuffle=False)



    model.eval()

    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            if True:
                indices = batch.idx
                batch_labels= batch.y
                labels = batch_labels.long()  # Ensure labels are of type Long for CrossEntropyLoss
                outputs, _ = model(batch)
                
                
            else:
                indices, features, labels = batch
                outputs, _ = model(features)

            _, predicted = torch.max(outputs, 1)
            # print(outputs, labels)
            # input()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())

    accuracy = correct / total * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy, all_preds

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # train_dataset = TetrisDatasetPyG(8000)
    # val_dataset = TetrisDatasetPyG(1000)
    # test_dataset = TetrisDatasetPyG(1000)
    tetris_generator = TetrisDataGenerator()
    features, labels = tetris_generator.generate_data(num_elements=5000)

    x_train, x_test, y_orig, y_test = train_test_split(features, labels, test_size =0.2, random_state = 42)

    x_train = torch.tensor(x_train, dtype= torch.float32)
    x_test = torch.tensor(x_test, dtype= torch.float32)
    y_orig = torch.tensor(y_orig, dtype= torch.long)
    y_test = torch.tensor(y_test, dtype= torch.long)
    # train_loader = PyGDataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = PyGDataLoader(val_dataset, batch_size=64)
    # test_loader = PyGDataLoader(test_dataset, batch_size=64)
    train_dataset = IndexedDataset(x_train, y_orig)
    edgGraphDataset = BasicEdgeGraphDataset(dataset=train_dataset, oneHotFetFlag=True)
    train_loader = PyGDataLoader(edgGraphDataset, batch_size=32, shuffle=True)

    test_dataset = IndexedDataset(x_test, y_test)
    test_edgGraphDataset = BasicEdgeGraphDataset(dataset=test_dataset, oneHotFetFlag=True)
    val_loader = PyGDataLoader(test_edgGraphDataset, batch_size=32, shuffle=False)

    model = GNNSimpeClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader)
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # print("Evaluating on test set...")
    # test_acc = evaluate(model, test_loader, test_dataset)
    device = next(model.parameters()).device
    x_test, y_test = x_test.to(device), y_test.to(device)
    test(model, test_features=x_test, test_labels=y_test, batch_size=32)

    # torch.save(model.state_dict(), "tetris_gnn_distance.pth")
    # print("Model saved as tetris_gnn_distance.pth")

    return model, test_acc


if __name__ == "__main__":
    model, test_acc = main()
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
