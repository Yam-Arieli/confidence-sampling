import numpy as np
import torch
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix

def prepare_train_tensors(adata_train, device):
    # Prepare input and output
    X = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X
    y = adata_train.obs["y"].astype("category").cat.codes.values
    num_classes = len(adata_train.obs["y"].unique())

    # One-hot encode targets
    y_onehot = np.eye(num_classes)[y]

    # Convert to torch tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32, device=device)
    return X_tensor, y_tensor, num_classes

def fix_test_genes(adata_train, adata_test, label_column):
    adata_test = adata_test.copy()

    test_label_key = adata_test.obs[label_column].copy()

    # 1. Get gene names
    train_genes = adata_train.var_names
    test_genes = adata_test.var_names

    # 2. Find genes in train but not in test
    missing_genes = train_genes.difference(test_genes)

    # 3. Create zero matrix for missing genes
    zero_data = csr_matrix((adata_test.n_obs, len(missing_genes)))

    # 4. Create AnnData for missing genes
    adata_missing = sc.AnnData(X=zero_data)
    adata_missing.var_names = missing_genes
    adata_missing.obs_names = adata_test.obs_names

    # 5. Concatenate along genes axis (columns)
    adata_test = sc.concat([adata_test, adata_missing], axis=1, join='outer')

    # 6. Reorder genes to match adata_train
    adata_test = adata_test[:, train_genes].copy()

    #
    adata_test.obs[label_column] = test_label_key

    return adata_test

def add_target_y_to_test(adata_train, adata_test, label_column):
    adata_test = adata_test.copy()
    # add 'true_labels' (aka `y`) to test
    # adata_test.obs[label_column] = test_label_key
    temp = pd.merge(left=adata_test.obs[[label_column]],
            right=adata_train.obs[['y', label_column]].drop_duplicates(),
            on=label_column, how='left')
    adata_test.obs['y'] = temp['y'].values
    adata_test = adata_test[~adata_test.obs['y'].isna().values, :]
    return adata_test

def mislabel_trainset(adata_train, label_column, noise_prob: float):
    # Extract the labels
    labels = adata_train.obs[label_column].copy()

    # Get class distribution (as probabilities)
    class_probs = labels.value_counts(normalize=True)

    # Generate a mask for which cells will be noised
    mask = np.random.rand(len(labels)) < noise_prob

    # For each "noised" index, sample a new label from the distribution
    noisy_labels = labels.copy()
    noisy_labels.loc[mask] = np.random.choice(
        a=class_probs.index,  # possible labels
        size=mask.sum(),    # number of samples
        p=class_probs.values  # probability weights
    )

    # Assign back
    adata_train.obs[label_column] = noisy_labels
    return adata_train

def add_noise_to_genes(adata_train, genes_std_noise: float)
    # Ensure X is dense (not sparse)
    X = adata_train.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # Compute std per gene
    gene_stds = X.std(axis=0)

    # Generate noise with per-gene scaling
    noise = np.random.normal(
        loc=0,
        scale=genes_std_noise * gene_stds,  # shape (n_genes,)
        size=X.shape
    )

    # Add noise
    adata_train.X = X + noise
    return adata_train