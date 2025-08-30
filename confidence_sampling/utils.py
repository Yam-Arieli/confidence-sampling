import numpy as np
import torch

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