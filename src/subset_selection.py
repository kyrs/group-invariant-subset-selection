import torch
import numpy as np
def get_cutstat_subset( pairwise_dists, labels, coverage=0.5, K=20, device='cpu',priors = None):
    pairwise_dists = pairwise_dists.type(torch.float32).to(device)
    labels = labels.type(torch.int64).to(device)
    with torch.no_grad():
        keep_indices = get_cutstat_inds(
                                        labels,
                                        pairwise_dists=pairwise_dists,
                                        coverage=coverage,
                                        device=device,
                                        K = K, priors = priors)

        # wrench dataset has create_subset() method that takes indices

    return keep_indices


def tracker_dict_based_selection(tracker_dict, method='forget',coverage=0.5):
    print(f"Using {method} method for subset selection with coverage {coverage}")
    # add sanity check on tracker_dict like  checking all value are not same across all keys
    if method in ['forget', 'grand', 'entropy']:
        sample_values = [v.get(method, None) for v in tracker_dict.values()]
        if all(val == sample_values[0] for val in sample_values):
            raise Exception(f"Warning: All values for '{method}' are the same.")
    else:
        pass 

    if method == 'forget':
        # Sort by forget value
        sorted_indices = sorted(tracker_dict.keys(), key=lambda x: tracker_dict[x]['forget'])
        selected_indices = sorted_indices[:int(coverage * len(sorted_indices))]
    elif method == 'grand':
        # Sort by grand value
        sorted_indices = sorted(tracker_dict.keys(), key=lambda x: tracker_dict[x]['grand'])
        selected_indices = sorted_indices[:int(coverage * len(sorted_indices))]
    elif method == 'entropy':
        # Sort by entropy value
        sorted_indices = sorted(tracker_dict.keys(), key=lambda x: tracker_dict[x]['entropy'])
        selected_indices = sorted_indices[:int(coverage * len(sorted_indices))]
    elif method == 'herding':
        cons_fet = []
        for i in range(len(tracker_dict)):
            cons_fet.append(tracker_dict[i]['feature'])
        cons_fet = np.vstack(cons_fet)
        
        assert cons_fet.ndim == 2, "cons_fet should be a 2D array"
        # Optional: check for NaNs or Infs
        assert not np.isnan(cons_fet).any(), "NaNs found in cons_fet"
        assert not np.isinf(cons_fet).any(), "Infs found in cons_fet"

        selected_indices = herding_coreset_selection(X = cons_fet, coreset_size  =int(coverage * len(tracker_dict)))
         

    else:
        raise ValueError("Invalid method. Choose from 'forget', 'grand' ")
    return selected_indices

def get_cutstat_inds(labels, pairwise_dists, coverage=0.5, K=20, device='cpu', priors = None):
        N = labels.shape[0]
        dists_sorted = torch.argsort(pairwise_dists)
        neighbors = dists_sorted[:,:K]
        dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
        weights = 1/(1 + dists_nn)
        neighbors = neighbors.to(device)
        dists_nn = dists_nn.to(device)
        weights = weights.to(device)
        cut_vals = (labels[:,None] != labels[None,:]).long()
        cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
        Jp = (weights * cut_neighbors).sum(dim=1)
        weak_counts = torch.bincount(labels)
        weak_pct = weak_counts / weak_counts.sum()
        # prior_probs = 0.5 # Below is the original code
        if priors is not None:
            prior_probs = priors
        else:
            prior_probs = weak_pct[labels]
        mu_vals = (1-prior_probs) * weights.sum(dim=1)
        sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
        sigma_vals = torch.sqrt(sigma_vals)
        normalized = (Jp - mu_vals) / sigma_vals
        normalized = normalized.cpu()
        inds_sorted = torch.argsort(normalized)
        N_select = int(coverage * N)
        conf_inds = inds_sorted[:N_select]
        conf_inds = list(set(conf_inds.tolist()))
        return conf_inds
    
def herding_coreset_selection(X, coreset_size):
    """
    Select indices of data samples using herding-based coreset selection.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        Input data.
    - coreset_size: int
        Number of data points to select.

    Returns:
    - selected_indices: list of int
        Indices of selected data points.
    """
    n_samples, _ = X.shape
    X_mean = np.mean(X, axis=0)
    
    selected_indices = []
    current_sum = np.zeros_like(X_mean)
    
    for _ in range(coreset_size):
        # Compute the inner product of each sample with the residual vector
        residual = X_mean - current_sum / (len(selected_indices) + 1)
        similarities = X @ residual
        
        # Mask already selected indices
        similarities[selected_indices] = -np.inf
        
        # Select the index with the highest similarity
        best_index = np.argmax(similarities)
        selected_indices.append(best_index)
        
        # Update current sum
        current_sum += X[best_index]
        
    return selected_indices