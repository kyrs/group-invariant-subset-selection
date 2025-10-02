# Invariant Subset Selection

This project provides an implementation of "Invariant Subset Selection." The primary goal is to understand and leverage the invariant geometric structure within a dataset to select a high-quality subset. The core assumption is that while the dataset's labels may be noisy, the inherent invariance can guide the selection of samples with clean labels. This resulting "clean" subset can then be used to significantly improve the performance of various downstream machine learning tasks.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Experiments](#running-experiments)

## Prerequisites

Before you begin, ensure you have Python 3.8+ installed. All other required packages are listed in `requirements.txt`.

You can install all dependencies using `pip`:
```bash
pip install -r requirements.txt
````

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/invariant-subset-selection.git](https://github.com/your-username/invariant-subset-selection.git)
    cd invariant-subset-selection
    ```

2.  **Download the Rotated MNIST Dataset:**
    The experiment on the Rotated MNIST dataset requires a specific data file.

      - **Download the file:** You must procure the `mnist_all_rotation_normalized_float_combined.amat` file. (Taken from: https://github.com/Bayer-Group/giae/tree/main/giae/so2)
        ```bash
            wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
            unzip mnist_rotation_new.zip
        ```
      - **Place the file** in the following directory (you may need to create the folders):
        ```
        src/real_world_exp/data/ROTMNIST/
        ```

    The final path for the dataset file must be exactly: `src/real_world_exp/data/ROTMNIST/mnist_all_rotation_normalized_float_combined.amat`.

3.  **Train the GNN models for representations:**

    **Please add the appropriate commands for this!!!!!!!!!!!!!!!**
## Running Experiments

This project includes experiments for four datasets demonstrating different types of invariance.

### Synthetic Datasets

  - **Orthogonally Invariant Group:**

    ```bash
    # Placeholder: Command to run the orthogonally invariant experiment
    python main.py --data ortho --method ours --noise_prob 0.45 --subset_size_percent 0.4
    ```

  - **Permutation Invariant Group:**

    ```bash
    # Placeholder: Command to run the permutation invariant experiment
    python main.py --data perm --method ours --noise_prob 0.45 --subset_size_percent 0.4
    ```

### Real-World Datasets

  - **Rotated MNIST (SO(2) Invariance):**

    ```bash
    # Placeholder: Command to run the Rotated MNIST experiment
    python main.py --data rot_mnist --method ours --noise_prob 0.45 --subset_size_percent 0.4
    ```

  - **Tetris (SE(3) Invariance):**

    ```bash
    # Placeholder: Command to run the Tetris experiment
    python main.py --data tetris --method ours --noise_prob 0.45 --subset_size_percent 0.4
    ```

