from torch_geometric.data import Data, Dataset as PyGDataset
from torch_cluster import knn_graph
from torch_geometric.data import Data
import torch.nn.functional as F
import torch

from torch_geometric.utils import remove_self_loops
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

class EmbedGraphDataset(PyGDataset):
    def __init__(self, x_fet, k=3 ):
        super().__init__()
        self.x_fet = x_fet
        self.k=k
        self.num_elements = len(x_fet)

    def len(self):
        return self.num_elements

    def get(self, idx):
        pos = self.x_fet[idx]
        n = pos.size(0)
        nelements = torch.tensor([pos.size(0)]) 
        # Build edge index using k-NN
       # fully-connected graph
        edge_index = torch.cartesian_prod(torch.arange(nelements.item()), torch.arange(nelements.item())).T
        # remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        return Data(pos=pos,x = pos.squeeze(), edge_index=edge_index, idx=idx )