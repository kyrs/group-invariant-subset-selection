
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataloader import TetrisDatasetPyG, TetrisDataset
from gnn_module_se3 import Model as eqvModel
# from vanilla_se3 import BaselineMLPModel

try:
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    from torch_geometric.data import DataLoader

def main():
    torch.manual_seed(48) 
    origDataset = TetrisDataset(num_elements=10000, noise_level=0.01, translation_level=5)
    dataset = TetrisDatasetPyG(num_elements=origDataset.num_elements, dataset=origDataset)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset: {dataset[0]}")
    trainData = DataLoader(dataset, batch_size=256)
    flag = "eqv"  # Change to "baseline" for the baseline model
    if flag == "eqv":
        print("Using equivariant model")
        model = eqvModel(hidden_dim=32, emb_dim=8, num_layers=5).to("cuda") # setting dim as 8
    elif flag == "classical":
        print("Using baseline model")
        pass  
    else:
        raise  NonImplementedError(f"Unknown model type: {flag}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    model.train()
    for epoch in range(20): 
        for data in trainData:
            # print(data)
            data = data.to("cuda")
            optimizer.zero_grad()
            pos_out, perm, vout, rot, shape_embed = model(data)
            loss = ((data.pos - pos_out) ** 2).sum(dim=-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            # print(f"embed : {shape_embed.shape}")
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        scheduler.step()
    
        save_path = f"so2_{flag}_model_epoch_{epoch}.pth"  # <-- Change the name if you want
    if flag == "eqv":
        model.eval()
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
if __name__ == "__main__":
    main()