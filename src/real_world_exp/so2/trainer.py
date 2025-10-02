
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from dataloader import MNISTDataset 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import Model, ClassicalModel
from utils import grouped_nt_xent_loss, rotation_contrastive_loss, rotate_batch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split 
class IndexedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return idx, self.features[idx], self.labels[idx]
def main(dataset_name):
    file_path = "../data/ROTMNIST/mnist_all_rotation_normalized_float_combined.amat" 
    ds = MNISTDataset(file_path)
    data_fet, data_label = ds.generate_data()
    # data_fet, data_label = ds.get_data()
    x_train, x_test, y_train, y_test = train_test_split(data_fet, data_label, test_size = 0.2, random_state = 42)

    indexedDataset = IndexedDataset(features=torch.tensor(x_train, dtype=torch.float32), labels=torch.tensor(y_train, dtype=torch.long))
    dl = DataLoader(indexedDataset, batch_size=512)

    # x_train, x_test, y_orig, y_test = train_test_split(data_fet, data_label, test_size = args.test_size, random_state = args.seed)
    hparams = {"hidden_dim" :128, "emb_dim": 32}
    flag = "classical"  # "classical" or "eqv"
    if flag == "eqv":
        model = Model(hparams=hparams).to("cuda")
    else:
        # Use classical model for comparison
        model = ClassicalModel(hparams=hparams).to("cuda")  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    model.train()
    epoch = 20
    # contrastive_loss_fn = SupConLoss(contrast_mode='all')
    for epoch in range(epoch): 
        for data in dl:
            __,x,_ = data 
            optimizer.zero_grad()
            x = x.to("cuda")
            x = x.squeeze(1)
            # labels = labels.to("cuda")
            out, rot, emb = model(x)
            out = out.squeeze(1)
            # print(f"out shape: {out.shape}, emb shape: {emb.shape} x: {x.shape}")
            # input()
            recon_loss = F.mse_loss(out, x)
            if flag == "eqv":
                # Use InfoNCE loss for equivariant model
                contrastive_loss = 0
            else:
                # Use groupwise InfoNCE loss for classical model
                # This is to ensure that the contrastive loss is computed within groups of the same label
                # which is more suitable for classical models
                
                x_rot = rotate_batch(x)
                out_rot, _, emb_rot = model(x_rot)
                recon_loss_rot = F.mse_loss(out_rot, x_rot)
                recon_loss = recon_loss + recon_loss_rot
                recon_loss = recon_loss*10  # Average the reconstruction losses
                contrastive_loss = rotation_contrastive_loss(emb, emb_rot)
                # contrastive_loss = grouped_nt_xent_loss(emb,labels)
                # print (f"contrastive_loss: {contrastive_loss}")

            total_loss = recon_loss +  contrastive_loss
            loss = total_loss
            loss.backward()
            optimizer.step()
            # print(f"embed : {shape_embed.shape}")
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        print(f"loss: {loss}, recon_loss: {recon_loss}, contrastive_loss: {contrastive_loss}")
        scheduler.step()
    save_path = f"so2_{flag}_model_epoch_{epoch}.pth"  # <-- Change the name if you want
    if flag == "eqv":
        model.eval()
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(datset_name = dataset_name)