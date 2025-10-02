
from tqdm import tqdm
import numpy as np 
import random
from scipy.stats import zscore
from collections import defaultdict
import torch
from copy import deepcopy

def flip_label_noise(y, p, classNum=2):
    new_label = []
    corrupted_samples = 0
    for idx in tqdm(range(len(y))):
        if classNum == 2:
            # For binary classification, flip the label with probability p
            out = np.random.choice([0, 1], p=[1-p, p])
            if out == 0:
                new_label.append(y[idx])
            else:
                new_label.append(1-y[idx])
                corrupted_samples+=1
        else:
            # For multi-class classification, randomly choose a different class with probability p
            if np.random.rand() < p:
                unique_labels = np.unique(y)
                other_labels = [label for label in unique_labels if label != y[idx]]
                new_label.append(random.choice(other_labels))
                corrupted_samples += 1
            else:
                new_label.append(y[idx])
    print(f"total samples : {len(y)} corrupted samples : {corrupted_samples} p : {p}")
    return new_label


def introduce_systematic_noise(y, noise_rate=0.1, bias_class=0):
    """
    Introduces systematic noise by mislabeling a specific class with another class.
    """
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_rate * n_samples)
    y = np.array(y)
    noisy_indices = np.where(y == bias_class)[0]
    noisy_indices = np.random.choice(noisy_indices, n_noisy, replace=False)
    
    # Mislabel those indices to another class
    unique_labels = np.unique(y)
    for idx in noisy_indices:
        other_labels = [label for label in unique_labels if label != bias_class]
        new_label = random.choice(other_labels)
        y_noisy[idx] = new_label
        
    return y_noisy


# Function to introduce class-dependent noise
def introduce_class_dependent_noise(y, noise_rates=None):
    """
    Introduces class-dependent noise where different classes have different noise rates.
    """
    y_noisy = deepcopy(y)
    # print(type(y))
    if type(y) == np.ndarray:
        # y_noisy = y.copy()
        unique_labels = np.unique(y)
        
        if noise_rates is None:
            # Default noise rate for all classes
            noise_rates = {label: 0.4 for label in unique_labels}
        
        for label in unique_labels:
            indices = np.where(y == label)[0]
            n_samples = len(indices)
            noise_rate = noise_rates[label]
            n_noisy = int(noise_rate * n_samples)
            
            noisy_indices = np.random.choice(indices, n_noisy, replace=False)
            for idx in noisy_indices:
                other_labels = [l for l in unique_labels if l != label]
                new_label = random.choice(other_labels)
                y_noisy[idx] = new_label
    elif type(y) == torch.Tensor:
        
        unique_labels = torch.unique(y)
    
        if noise_rates is None:
            # Default noise rate for all classes
            noise_rates = {label.item(): 0.4 for label in unique_labels}
        
        for label in unique_labels:
            # Find indices where the label appears in y
            indices = (y == label).nonzero(as_tuple=True)[0]
            n_samples = len(indices)
            
            # Get the noise rate for the current label
            noise_rate = noise_rates[label.item()]
            n_noisy = int(noise_rate * n_samples)
            
            # Randomly select indices to be corrupted
            noisy_indices = torch.randperm(n_samples)[:n_noisy]
            
            # For each noisy index, randomly assign a new label
            for idx in noisy_indices:
                # Choose a new label different from the current label
                other_labels = unique_labels[unique_labels != label]
                new_label = other_labels[torch.randint(0, len(other_labels), (1,))].item()
                y_noisy[indices[idx]] = new_label
    else:
        Exception(f"{type(y)} is not supported")
    return y_noisy

def outlier_corruption(X, y):
    z_threshold = 3

    # Step 1: Identify indices of data belonging to each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    # Step 2: Identify outliers based on Z-scores
    z_scores = zscore(X, axis=0)  # Compute Z-scores for each feature (axis=0 for column-wise computation)
    outliers = set()  # To store indices of outliers

    for idx in range(len(X)):
        # If any feature of the data point has a Z-score greater than the threshold, it's an outlier
        if np.any(np.abs(z_scores[idx]) > z_threshold):
            outliers.add(idx)

    # Step 3: Flip the labels of outliers
    # We randomly flip the label of outliers to any other class except their current class
    unique_labels = np.unique(y)
    new_labels = y.copy()

    for idx in outliers:
        # Exclude the current class label to avoid flipping to the same class
        possible_labels = [label for label in unique_labels if label != new_labels[idx]]
        new_label = np.random.choice(possible_labels)
        new_labels[idx] = new_label
    
    return new_labels