
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import defaultdict
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as PyGDataset, Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from torch_geometric.utils import remove_self_loops
from dataloader import TetrisDatasetPyG, TetrisDataset
import numpy as np

class GCNDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64, max_nodes=4, out_dim=3):
        super().__init__()
        self.max_nodes = max_nodes
        self.out_dim = out_dim
        
        # Decoder layers
        self.convs = nn.ModuleList([
            GCNConv(latent_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, out_dim),
        ])

        self.norms = nn.ModuleList([
            LayerNorm(latent_dim),     # first input to decoder is z -> latent_dim
            LayerNorm(hidden_dim),
            LayerNorm(hidden_dim),
            LayerNorm(hidden_dim),
        ])
        # self.dropout = nn.Dropout(0.1)

    def compute_edge_weights(self, pos, edge_index):
        """
        Compute edge weights based on Euclidean distances between node positions
        """
        row, col = edge_index
        pos_i = pos[row]
        pos_j = pos[col]
        
        distances = torch.norm(pos_i - pos_j, dim=1)
        sigma = 1.0
        edge_weights =distances.pow(2) 
        edge_weights = edge_weights.clamp(min=1e-4)
        
        return edge_weights

    def expand_latent_to_nodes(self, z, batch, target_num_nodes):
        """
        Expand graph-level latent representation to node-level features
        """
        # z is [batch_size, latent_dim]
        # We need to expand it to [total_nodes, latent_dim]
        expanded = z[batch]  # This maps each node to its graph's embedding
        return expanded

    def forward(self, z, edge_index, batch, pos, target_shape=None):
        # Expand latent representation to node level
        total_nodes = batch.size(0)
        x = self.expand_latent_to_nodes(z, batch, total_nodes)
        
        # Compute edge weights from positions
        edge_weights = self.compute_edge_weights(pos, edge_index)
        
        # Apply decoder layers
        for i, conv in enumerate(self.convs):
            if i < len(self.norms):
                x = self.norms[i](x)
            x = conv(x, edge_index, edge_weights)
            if i < len(self.convs) - 1: # NOTE : VERIY
                x = F.relu(x)
                # x = self.dropout(x)
        
        # x should now be [total_nodes, out_dim] representing reconstructed positions
        return x


# -------------------------------
# GCN ENCODER WITH EDGE WEIGHTS
# -------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, max_nodes=4, hidden_dim=64, latent_dim=128):
        super().__init__()
        self.max_nodes = max_nodes
        # Input dimension is max_nodes for one-hot encoding
        self.convs = nn.ModuleList([
            GCNConv(max_nodes, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, latent_dim),
        ])
        self.norms = nn.ModuleList([
            LayerNorm(self.max_nodes),       # This should match input feature size = 4
            LayerNorm(hidden_dim),           # After conv1: 64
            LayerNorm(hidden_dim),           # After conv2: 64
            LayerNorm(hidden_dim)            # After conv3: 64
        ])
    def compute_edge_weights(self, pos, edge_index):
        """
        Compute edge weights based on Euclidean distances between node positions
        """
        row, col = edge_index
        pos_i = pos[row]  # Source node positions
        pos_j = pos[col]  # Target node positions
        
        # Compute Euclidean distances
        distances = torch.norm(pos_i - pos_j, dim=1)
        
        edge_weights =distances.pow(2) 
        edge_weights = edge_weights.clamp(min=1e-4)
        
        return edge_weights

    def prepare_node_features(self, x, num_nodes_per_graph, batch):
        """
        Ensure consistent one-hot encoding across different graph sizes
        """
        # x is already one-hot encoded from dataset, but we need to pad to max_nodes
        batch_size = batch.max().item() + 1
        max_nodes_in_batch = x.size(1)  # Current one-hot dimension
        
        if max_nodes_in_batch < self.max_nodes:
            # Pad with zeros to reach max_nodes dimension
            padding = torch.zeros(x.size(0), self.max_nodes - max_nodes_in_batch, 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif max_nodes_in_batch > self.max_nodes:
            # Truncate if somehow larger (shouldn't happen with our dataset)
            x = x[:, :self.max_nodes]
            
        return x

    def forward(self, data):
        # Prepare consistent node features
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        x = self.prepare_node_features(x, None, batch)
        
        # Compute edge weights from positions
        edge_weights = self.compute_edge_weights(pos, edge_index)
        
        # Apply GCN layers with edge weights
        for i, conv in enumerate(self.convs):
            # Pass edge weights to GCN layer
            # x = LayerNorm(x, batch)
            if i < len(self.norms):                # skip norm on last layer
                x = self.norms[i](x)
            else:
                pass
            x = conv(x, edge_index, edge_weights)
                
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                # x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        return x


# -------------------------------
# NT-XENT LOSS (InfoNCE)
# -------------------------------

def nt_xent_loss(z1: torch.Tensor,
                 z2: torch.Tensor,
                 temperature: float = 0.5,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric NT-Xent (SimCLR) for a batch of positive pairs (z1, z2).

    Key upgrades vs. the bare version:
    • ε-normalisation to avoid div-by-zero
    • fp32 maths even under AMP
    • self-similarities masked to −∞
    • row-max subtraction to prevent exp() overflow
    """
    B = z1.size(0)

    # 1. L2-normalise (fp32 for safety)
    z1 = F.normalize(z1, dim=1, eps=eps).float()
    z2 = F.normalize(z2, dim=1, eps=eps).float()

    # 2. Full similarity matrix, τ-scaled
    z   = torch.cat([z1, z2], dim=0)            # [2B, D]
    sim = z @ z.T                               # cosine similarities
    sim.div_(temperature)

    # 3. Mask self-similarities
    diag = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(diag, -float('inf'))

    # 4. Row-max stabilisation (helps when τ is small / batch large)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    # 5. Cross-entropy targets
    targets = torch.arange(B, device=sim.device)
    loss_a  = F.cross_entropy(sim[:B], targets + B)
    loss_b  = F.cross_entropy(sim[B:], targets)

    return 0.5 * (loss_a + loss_b)
# def nt_xent_loss(z1, z2, temperature=0.5):
#     """
#     Normalized Temperature-scaled Cross Entropy Loss
#     z1, z2: [batch_size, latent_dim] - representations of positive pairs
#     """
#     batch_size = z1.size(0)
    
#     # Normalize embeddings
#     z1 = F.normalize(z1, dim=1)
#     z2 = F.normalize(z2, dim=1)
    
#     # Concatenate z1 and z2
#     z = torch.cat([z1, z2], dim=0)  # [2*batch_size, latent_dim]
    
#     # Compute similarity matrix
#     sim_matrix = torch.mm(z, z.t()) / temperature  # [2*batch_size, 2*batch_size]
    
#     # Create labels: positive pairs are (i, i+batch_size) for i in [0, batch_size)
#     labels = torch.arange(batch_size).to(z.device)
    
#     # For each sample in z1, the positive is the corresponding sample in z2
#     loss_a = F.cross_entropy(sim_matrix[:batch_size], labels + batch_size)
#     # For each sample in z2, the positive is the corresponding sample in z1  
#     loss_b = F.cross_entropy(sim_matrix[batch_size:], labels)
    
#     return (loss_a + loss_b) / 2


# -------------------------------
# CONTRASTIVE LEARNING TRAINER
# -------------------------------
class ContrastiveLearner:
    def __init__(self, encoder, decoder, device, lr=1e-3, temperature=0.5, reconstruction_weight=0.1):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.temperature = temperature
        self.reconstruction_weight = reconstruction_weight
        
        # Optimizer for both encoder and decoder
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def create_positive_pairs(self, batch):
        """
        Create positive pairs from the same class in the batch
        Returns two batches where corresponding indices are positive pairs
        """
        # Group by class label
        class_groups = defaultdict(list)
        for i, y in enumerate(batch.y):
            class_groups[y.item()].append(i)
        
        # Create positive pairs
        pair_indices_1 = []
        pair_indices_2 = []
        
        for class_indices in class_groups.values():
            if len(class_indices) >= 2:
                # If we have multiple samples from same class, pair them
                pairs = [(class_indices[i], class_indices[j]) 
                        for i in range(len(class_indices)) 
                        for j in range(i+1, len(class_indices))]
                for idx1, idx2 in pairs:
                    pair_indices_1.append(idx1)
                    pair_indices_2.append(idx2)

        if not pair_indices_1:
            return None, None
            
        # Create new batches with positive pairs
        data_list = batch.to_data_list()
        batch1_data = [data_list[i] for i in pair_indices_1]
        batch2_data = [data_list[i] for i in pair_indices_2]
        
        batch1 = Batch.from_data_list(batch1_data).to(self.device)
        batch2 = Batch.from_data_list(batch2_data).to(self.device)
        
        return batch1, batch2

    def compute_reconstruction_loss(self, batch):
        """
        Compute reconstruction loss for the entire batch
        """
        # Encode
        z = self.encoder(batch)
        
        # Decode
        pos_reconstructed = self.decoder(z, batch.edge_index, batch.batch, batch.pos)
        
        # Compute MSE loss between original and reconstructed positions
        reconstruction_loss = F.mse_loss(pos_reconstructed, batch.pos)
        
        return reconstruction_loss, z

    def train_step(self, batch):
        self.encoder.train()
        self.decoder.train()
        batch = batch.to(self.device)
        
        total_loss = 0.0
        contrastive_loss_val = 0.0
        reconstruction_loss_val = 0.0
        
        # 1. Compute reconstruction loss for the entire batch
        recon_loss, _ = self.compute_reconstruction_loss(batch)
        reconstruction_loss_val = recon_loss.item()
        total_loss += self.reconstruction_weight * recon_loss
        
        # 2. Create positive pairs and compute contrastive loss
        batch1, batch2 = self.create_positive_pairs(batch)
        
        if batch1 is not None and batch2 is not None:
            # Get embeddings for both batches (now passing pos as well)
            # z1 = self.encoder(batch1.x, batch1.edge_index, batch1.batch, batch1.pos)
            # z2 = self.encoder(batch2.x, batch2.edge_index, batch2.batch, batch2.pos)
            z1 = self.encoder(batch1)
            z2 = self.encoder(batch2)
            # Compute contrastive loss
            contrastive_loss = nt_xent_loss(z1, z2, self.temperature)
            contrastive_loss_val = contrastive_loss.item()
            total_loss += contrastive_loss
        
        # 3. Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss_val,
            'reconstruction_loss': reconstruction_loss_val
        }

    def train_epoch(self, dataloader):
        total_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0
        }
        num_batches = 0
        
        for batch in dataloader:
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key]
            num_batches += 1
        
        self.scheduler.step()
        
        # Return average losses
        avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
        return avg_losses


# -------------------------------
# TRAINING FUNCTION WITH SAVING
# -------------------------------
def train():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # # Create dataset and dataloader
    # dataset = TetrisDatasetPyG(num_elements=5000)
    # loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    torch.manual_seed(48)
    origDataset = TetrisDataset(num_elements=10000, rotate=True, noise_level=0.001, translation_level=10.0)
    dataset = TetrisDatasetPyG(num_elements=origDataset.num_elements, dataset=origDataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    # Initialize model and trainer
    encoder = GCNEncoder(max_nodes=4, hidden_dim=32, latent_dim=16)
    decoder = GCNDecoder(latent_dim=16, hidden_dim=32, max_nodes=4, out_dim=3)
    trainer = ContrastiveLearner(
        encoder, decoder, device, 
        lr=1e-3, temperature=0.5, reconstruction_weight=0.1
    )
    
    print("Starting contrastive learning training with reconstruction...")
    
    for epoch in range(1, 20):
        avg_losses = trainer.train_epoch(loader)
        
        # if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | "
                f"Total Loss: {avg_losses['total_loss']:.4f} | "
                f"Contrastive: {avg_losses['contrastive_loss']:.4f} | "
                f"Reconstruction: {avg_losses['reconstruction_loss']:.4f} | "
                f"LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
    
    print("Training completed!")
    
    # Save the trained encoder
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, "tetris_encoder.pth")
    decoder_path = os.path.join(save_dir, "tetris_decoder.pth")
    
    # Save model state dict and configuration
    # torch.save({
    #     'model_state_dict': trainer.encoder.state_dict(),
    #     'model_config': {
    #         'max_nodes': 4,
    #         'hidden_dim': 64,
    #         'latent_dim': 128
    #     }
    # }, encoder_path)
    
    # torch.save({
    #     'model_state_dict': trainer.decoder.state_dict(),
    #     'model_config': {
    #         'latent_dim': 128,
    #         'hidden_dim': 64,
    #         'max_nodes': 4,
    #         'out_dim': 3
    #     }
    # }, decoder_path)
    torch.save(trainer.encoder.state_dict(), encoder_path)
    torch.save(trainer.decoder.state_dict(), decoder_path)
    
    print(f"Encoder saved to: {encoder_path}")
    print(f"Decoder saved to: {decoder_path}")
    
    return trainer.encoder, trainer.decoder


# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate_embeddings(encoder, decoder, dataset, device, num_samples=1000):
    """
    Evaluate the quality of learned embeddings and reconstruction
    """
    encoder.eval()
    decoder.eval()
    
    # Sample data for evaluation
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    total_reconstruction_error = 0.0
    num_samples_processed = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            
            # Get embeddings
            embeddings = encoder(batch.x, batch.edge_index, batch.batch, batch.pos)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch.y.cpu())
            
            # Compute reconstruction error
            pos_reconstructed = decoder(embeddings, batch.edge_index, batch.batch, batch.pos)
            reconstruction_error = F.mse_loss(pos_reconstructed, batch.pos)
            total_reconstruction_error += reconstruction_error.item()
            num_samples_processed += 1
            
            if len(all_embeddings) * 64 >= num_samples:
                break
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    avg_reconstruction_error = total_reconstruction_error / num_samples_processed
    
    print(f"Average Reconstruction Error: {avg_reconstruction_error:.6f}")
    
    # Compute within-class vs between-class distances
    from sklearn.metrics import silhouette_score
    
    try:
        silhouette = silhouette_score(embeddings.numpy(), labels.numpy())
        print(f"Silhouette Score: {silhouette:.4f}")
    except:
        print("Could not compute silhouette score")
    
    return embeddings, labels, avg_reconstruction_error


# -------------------------------  
# VISUALIZATION FUNCTION
# -------------------------------
def visualize_reconstruction(encoder, decoder, dataset, device, num_examples=3):
    """
    Visualize original vs reconstructed Tetris pieces
    """
    encoder.eval()
    decoder.eval()
    
    # Get a few samples
    eval_loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)
    batch = next(iter(eval_loader)).to(device)
    
    with torch.no_grad():
        # Encode and decode
        z = encoder(batch.x, batch.edge_index, batch.batch, batch.pos)
        pos_reconstructed = decoder(z, batch.edge_index, batch.batch, batch.pos)
    
    # Convert to numpy for visualization
    original_pos = batch.pos.cpu().numpy()
    reconstructed_pos = pos_reconstructed.cpu().numpy()
    batch_assignment = batch.batch.cpu().numpy()
    
    print("\nReconstruction Comparison:")
    print("=" * 50)
    
    for i in range(num_examples):
        mask = (batch_assignment == i)
        orig_shape = original_pos[mask]
        recon_shape = reconstructed_pos[mask]
        
        print(f"\nExample {i+1} (Class {batch.y[i].item()}):")
        print("Original positions:")
        print(orig_shape)
        print("Reconstructed positions:")
        print(recon_shape)
        print(f"MSE Error: {np.mean((orig_shape - recon_shape)**2):.6f}")
        print("-" * 30)


if __name__ == "__main__":
    # import numpy as np
    
    # Train the model
    trained_encoder, trained_decoder = train()
    
    # Evaluate embeddings and reconstruction
    print("\nEvaluating learned embeddings and reconstruction...")
    # dataset = TetrisDatasetPyG(num_elements=100000)
    
    origDataset = TetrisDataset(num_elements=10000, rotate=True, noise_level=0.01, translation_level=5.0)
    dataset = TetrisDatasetPyG(num_elements=origDataset.num_elements, dataset=origDataset)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    embeddings, labels, recon_error = evaluate_embeddings(
        trained_encoder, trained_decoder, dataset, device
    )
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of unique classes: {len(torch.unique(labels))}")
    
    # Visualize some reconstructions
    print("\nVisualizing reconstructions...")
    # visualize_reconstruction(trained_encoder, trained_decoder, dataset, device, num_examples=3)