# Group Invariant Subset Selection

This project provides an implementation of "Group Invariant Subset Selection." The primary objective is to understand and exploit the invariant geometric structure within a dataset to select a high-quality subset. The key intuition is that, although the dataset’s labels may be noisy, the underlying invariances can guide the selection of samples with clean labels. The resulting "clean" subset can then be used to substantially improve the performance of downstream machine learning tasks.


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
    Download the repository and change the dir
    ```bash
    cd group-invariant-subset-selection
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

3.  **Train the  models associated with invariant representation function :**
    - Train the SO(2) and SE(3) invariant encoders and replace the path in config.py
    - Code for training the encoders are provided in src/real_world_exp/se3/train_invariant_enc.py and src/real_world_exp/so2/trainer.py



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

