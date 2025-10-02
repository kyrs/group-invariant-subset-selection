import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.real_world_exp.so2.models import Model as eqv_so2_model
from src.real_world_exp.so2.models import ClassicalModel as contrast_so2_model 
from src.real_world_exp.se3.gnn_module_se3 import Model as eqv_se3_model
from torch.utils.data import DataLoader
import yaml  # or use json if preferred
from src.utils import EmbedGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.real_world_exp.se3.vanilla_se3 import GCNEncoder as contrast_se3_model

def fetch_embedding_model(args, method):
    if args.data == "rot_mnist":
        hparams = {"hidden_dim" :128, "emb_dim": 32}
        if method == "eqv":
            model = eqv_so2_model(hparams=hparams)
            model.eval()
        elif method == "classical":
            model = contrast_so2_model(hparams=hparams)
        

        else:
            raise ValueError(f"Model for method {args.method} is not defined.")
    elif args.data == "tetris":
        if method == "eqv":
            model = eqv_se3_model(hidden_dim=32, emb_dim=8, num_layers=5)
            model.eval()
        elif method == "classical":
            model = contrast_se3_model(max_nodes=4, hidden_dim=32, latent_dim=16)
    else:
        raise ValueError(f"Model for dataset {args.data} is not defined.")
    return model

def get_embeddings_from_config(model_path, model, dataset, args,method,  batch_size=128):
    """
    Loads model weights from config and returns embeddings for entire dataset (no shuffle).

    Args:
        model (torch.nn.Module): An instance of the model.
        config_path (str): Path to YAML config file with at least 'model_path'.
        dataset (torch.utils.data.Dataset): Dataset to extract embeddings from.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Embeddings of shape (N, D).
    """
    


    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Create DataLoader
    if args.data == 'rot_mnist':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif args.data == 'tetris':
        dataset = EmbedGraphDataset(dataset)
        dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            if args.data == 'rot_mnist':
                
                inputs = inputs.to(args.device)
                inputs = inputs.squeeze(1)  # Add channel dimension if needed
                __, _, embeddings = model(inputs)
            elif args.data == 'tetris':
                inputs = inputs.to(args.device)
                if method == "eqv":
                    _, _, _, _, embeddings = model(inputs)
                else:
                    embeddings = model(inputs)
                

            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def identity_rep(arr):
    if len(arr.shape)==3:
        sampl,row,column = arr.shape
        arr = arr.reshape(sampl, row*column) ## NOTE : VERIY THIS
    else:
        pass
    return arr

def ortho_grp_rep(arr):
    if type(arr) != np.ndarray:
        arr = torch.from_numpy(arr)
    return np.expand_dims(np.linalg.norm(arr, axis=1), axis = 1)

def permute_grp_rep(arr):
    return np.sort(arr, axis = 1)

def compute_pairwise_distances(arr, fn=None,  p = 2):
    if type(arr) != np.ndarray:
        arr = arr.numpy()

    reps = torch.from_numpy(np.float32(fn(arr)))
    
    distance_matrix = torch.cdist(reps, reps, p = p)
    return torch.FloatTensor(distance_matrix)
