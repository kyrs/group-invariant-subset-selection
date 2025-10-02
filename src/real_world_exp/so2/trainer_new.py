
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from dataloader import MNISTDataset 
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from models import Model, ClassicalModel
from utils import grouped_nt_xent_loss
import torch.nn.functional as F

def main():
    file_path = "../data/ROTMNIST/mnist_all_rotation_normalized_float_combined.amat" 
    ds = MNISTDataset(np.loadtxt(file_path))
    # dl = DataLoader(ds, batch_size=512)
    # print()
    total_samples = len(ds)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    batch_size = 512 # Define your desired batch size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle validation/test
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # for data in dl:
    #     x, labels = data
    #     print(x.size(), labels.size(), labels, x[0])
    #     break
    hparams = {"hidden_dim" :128, "emb_dim": 32}
    flag = "classical"  # "classical" or "eqv"
    if flag == "eqv":
        model = Model(hparams=hparams).to("cuda")
    else:
        # Use classical model for comparison
        model = ClassicalModel(hparams=hparams).to("cuda")  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    # model.train()
    # # contrastive_loss_fn = SupConLoss(contrast_mode='all')
    # for epoch in range(10): 
    #     for data in dl:
    #         x,labels = data 
    #         optimizer.zero_grad()
    #         x = x.to("cuda")
    #         labels = labels.to("cuda")
    #         out, rot, emb = model(x)
    #         out = out.squeeze(1)
    #         # print(f"out shape: {out.shape}, emb shape: {emb.shape} x: {x.shape}")
    #         # input()
    #         recon_loss = F.mse_loss(out, x)
    #         if flag == "eqv":
    #             # Use InfoNCE loss for equivariant model
    #             contrastive_loss = 0
    #         else:
    #             # Use groupwise InfoNCE loss for classical model
    #             # This is to ensure that the contrastive loss is computed within groups of the same label
    #             # which is more suitable for classical models
    #             contrastive_loss = grouped_nt_xent_loss(emb,labels)
    #             # print (f"contrastive_loss: {contrastive_loss}")

    #         total_loss = recon_loss +  contrastive_loss
    #         loss = total_loss
    #         loss.backward()
    #         optimizer.step()
    #         # print(f"embed : {shape_embed.shape}")
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    #     print(f"loss: {loss}, recon_loss: {recon_loss}, contrastive_loss: {contrastive_loss}")
    #     scheduler.step()
if __name__ == "__main__":
    main()