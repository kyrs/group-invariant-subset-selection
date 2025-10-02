import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe, faster saves
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import umap  # pip install umap-learn
from tqdm.auto import tqdm

def plot_and_save_umap(
    X,
    y=None,
    method="umap",
    highlight_idx=None,
    out_dir="./plots",
    filename=None,

    # ---- SPEED OPTIMIZED for ~10k ----
    pca_components=15,        # Balance: not too low (5), not too high (50)
    n_neighbors=10,           
    min_dist=0.1,
    metric="euclidean",
    n_epochs=50,            # Let UMAP auto-decide based on data size
    random_state=42,
    unique=True,

    # Advanced NN-descent knobs
    angular_rp_forest=False,

    # ---- Reuse options ----
    embedding_cache=None,
    embedding_path=None,      # ".npy"
    reuse_embedding=True,

    # ---- Plot options ----
    dpi=150,                  # Lower DPI for faster saving
    point_size=15,

    # ---- Progress / logging ----
    umap_verbose=False,       # Disable UMAP's verbose output for speed
):
    """
    Fast UMAP with tqdm progress bars for ~10k points.
    Stages: scale -> PCA -> UMAP -> save.
    """
    # ---- Input handling ----
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 samples for UMAP.")

    print(f"Processing {n} samples with {X.shape[1]} features")

    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"{method}_umap_fast"
    out_path = os.path.join(out_dir, f"{filename}.png")

    # ---- Load cached embedding if available ----
    Z = None
    if embedding_cache is not None:
        Z = np.asarray(embedding_cache, dtype=np.float32)
    elif reuse_embedding and embedding_path and os.path.exists(embedding_path):
        print(f"Loading cached embedding from {embedding_path}")
        Z = np.load(embedding_path).astype(np.float32)

    # ---- Compute embedding if not provided/loaded ----
    if Z is None:
        # 0) Validate input
        if not np.isfinite(X).all():
            n_bad = np.sum(~np.isfinite(X))
            print(f"Warning: Replacing {n_bad} inf/nan values in input")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 1) Standardize
        print("Standardizing...")
        scaler = StandardScaler(copy=True)
        X_std = scaler.fit_transform(X)
        
        # Handle zero-variance features
        zero_var_mask = scaler.scale_ < 1e-10
        if np.any(zero_var_mask):
            print(f"Warning: {np.sum(zero_var_mask)} constant features")
            X_std[:, zero_var_mask] = 0.0
        
        # Clean
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        X_std = np.clip(X_std, -1e6, 1e6)

        # 2) PCA (optional)
        if pca_components and X_std.shape[1] > pca_components:
            print(f"PCA: {X_std.shape[1]} → {pca_components} dims...")
            if not np.isfinite(X_std).all():
                X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                pca = PCA(n_components=pca_components, random_state=random_state)
                X_red = pca.fit_transform(X_std)
                X_red = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
                X_red = np.clip(X_red, -1e6, 1e6)
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"PCA failed: {e}. Using standardized data.")
                X_red = X_std
        else:
            X_red = X_std

        # Final clean
        if not np.isfinite(X_red).all():
            X_red = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)

        X_red = X_red.astype(np.float32)

        # 3) UMAP fit
        print(f"Running UMAP (n_neighbors={min(n_neighbors, n-1)})...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, n - 1),
            min_dist=min_dist,
            metric=metric,
            n_epochs=n_epochs,  # None = auto
            random_state=random_state,
            unique=unique,
            densmap=False,
            low_memory=True,    # IMPORTANT: reduces memory usage
            angular_rp_forest=angular_rp_forest,
            verbose=umap_verbose,
        )
        Z = reducer.fit_transform(X_red).astype(np.float32)
        
        print(f"UMAP complete. Unique points: {len(np.unique(Z, axis=0))}/{n}")

        # 4) Save embedding (optional)
        if embedding_path is not None:
            np.save(embedding_path, Z)
            print(f"Saved embedding to {embedding_path}")

    # ---- Plot ----
    print("Rendering plot...")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if y is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=0.6, 
                  rasterized=True, edgecolors='k', linewidth=0.3)
    else:
        y = np.asarray(y)
        if y.dtype.kind not in "iuf":
            y_plot = LabelEncoder().fit_transform(y)
        else:
            y_plot = y
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_plot, s=point_size,
                        alpha=0.7, cmap="tab10", rasterized=True,
                        edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="Class")

    if highlight_idx is not None:
        if isinstance(highlight_idx, int):
            highlight_idx = [highlight_idx]
        for idx in highlight_idx:
            ax.scatter(Z[idx, 0], Z[idx, 1], s=150,
                       facecolors="none", edgecolors="red", linewidth=2.5)

    ax.set_title(f"UMAP ({method}) - {n} samples", fontsize=12, fontweight='bold')
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot to {out_path}")

    return Z, out_path