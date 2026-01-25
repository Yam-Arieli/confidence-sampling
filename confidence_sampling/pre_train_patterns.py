import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def track_training_dynamics(model, X, y, device, optimizer_fn=torch.optim.SGD, 
                            criterion=torch.nn.NLLLoss(), batch_size=256, epochs=100, lr=1e-4):
    """
    Phase 1: The Monitor
    Trains the GOLDSelectNet and tracks the mean Z vector for each class per epoch.
    
    Returns:
        model: Trained model
        z_tensor: Shape (n_epochs, n_patterns, n_classes) - The trajectory of patterns.
    """
    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)
    
    # We need the pattern dimension from the model to initialize the tensor
    # Assuming model.classifier.weight shape is [n_classes, n_patterns]
    n_patterns = model.classifier.weight.shape[1] 
    
    # The Tensor to track pattern dynamics: (Epochs, Patterns, Classes)
    z_tensor = np.zeros((epochs, n_patterns, n_classes))
    
    y_onehot = F.one_hot(y, num_classes=n_classes).float()
    dataset = TensorDataset(X, y_onehot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 1. Train Step
        model.train()
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_label = y_batch.argmax(dim=1).to(device)
            
            optimizer.zero_grad()
            output, _ = model(X_batch) # Ignore Z during training update
            loss = criterion(output, y_label)
            loss.mean().backward()
            optimizer.step()
            
        # 2. Monitor Step (End of Epoch)
        # We need the mean Z for each class.
        model.eval()
        with torch.no_grad():
            # Get Z for all data (in batches to save RAM)
            all_z = []
            eval_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
            for (X_b,) in eval_loader:
                _, z_b = model(X_b.to(device))
                all_z.append(z_b.cpu())
            all_z = torch.cat(all_z, dim=0) # Shape: (n_samples, n_patterns)
            
            # Group by Class and Calculate Mean
            for c in range(n_classes):
                # Mask for current class
                class_mask = (y.cpu() == c)
                if class_mask.sum() > 0:
                    mean_z = all_z[class_mask].mean(dim=0).numpy()
                    z_tensor[epoch, :, c] = mean_z
                
        print(f"Epoch {epoch+1}/{epochs} tracked.")

    return model, z_tensor

def calculate_pattern_importance(z_tensor):
    """
    Phase 1.5: Calculate "Earliness" Weights (AUC)
    
    Args:
        z_tensor: (n_epochs, n_patterns, n_classes)
        
    Returns:
        W_c: (n_patterns, n_classes) - The importance weight of each pattern for each class.
    """
    n_epochs, n_patterns, n_classes = z_tensor.shape
    W_c = np.zeros((n_patterns, n_classes))
    
    # Global Max for Normalization (The "Global Standard" we discussed)
    # This ensures "weak" patterns (that never rise high) get low scores.
    global_max = z_tensor.max() + 1e-8
    
    # Normalize tensor globally
    z_norm = z_tensor / global_max
    
    for c in range(n_classes):
        for p in range(n_patterns):
            # Trajectory of pattern p for class c
            trajectory = z_norm[:, p, c]
            
            # Calculate Area Under Curve (Sum of values)
            # Higher Sum = Started earlier and stayed high
            auc = np.sum(trajectory)
            
            # Normalize by number of epochs so score is roughly 0.0 to 1.0
            W_c[p, c] = auc / n_epochs
            
    return W_c

def calculate_gold_scores(model, X, y, W_c, device, batch_size=256):
    """
    Phase 2: Snapshot & Score Calculation
    
    Args:
        model: Trained GOLDSelectNet
        X: Data (n_samples, n_features)
        y: Labels (n_samples,)
        W_c: Importance Weights (n_patterns, n_classes)
        device: 'cuda' or 'cpu'
        
    Returns:
        scores: np.array of shape (n_samples,) - The calculated GOLD score for each cell.
    """
    model = model.to(device)
    
    # 1. Robust Data Handling (Preserve Dtype)
    if not torch.is_tensor(X):
        X_tensor = torch.tensor(X).to(device)
    else:
        X_tensor = X.to(device)
        
    # y is needed as numpy for indexing class columns in W_c
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = np.array(y)

    # 2. Extract Z Matrix (Snapshot)
    model.eval()
    all_z = []
    
    # Use DataLoader to prevent OOM on large datasets
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (X_batch,) in loader:
            _, z_batch = model(X_batch) # Get the latent Z
            all_z.append(z_batch.cpu())
            
    z_matrix = torch.cat(all_z, dim=0).numpy() # Shape: (n_samples, n_patterns)
        
    # 3. Calculate Scores
    n_samples = z_matrix.shape[0]
    scores = np.zeros(n_samples)
    
    # Score = Dot Product of (Cell's Z) and (Class's Pattern Weights)
    for i in range(n_samples):
        cell_z = z_matrix[i]                 # Vector (n_patterns,)
        cell_class = y_np[i]                 # Scalar Class ID
        class_weights = W_c[:, cell_class]   # Vector (n_patterns,)
        
        scores[i] = np.dot(cell_z, class_weights)
        
    return scores

def subsample_gold_cells(scores, y, percentile=0.2):
    """
    Phase 3: Stratified Selection based on Scores
    
    Args:
        scores: np.array of scores from calculate_gold_scores
        y: Labels (n_samples,)
        percentile: Top fraction to keep (e.g. 0.2)
        
    Returns:
        selected_indices: Indices of the chosen cells in the original dataset
    """
    # Ensure y is numpy
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    else:
        y = np.array(y)
        
    selected_indices = []
    classes = np.unique(y)
    
    for c in classes:
        # Get indices of all cells in this class
        c_indices = np.where(y == c)[0]
        c_scores = scores[c_indices]
        
        # Determine number of cells to keep (k)
        k = int(len(c_indices) * percentile)
        if k < 1: k = 1
        
        # Select Top K
        # argsort is ascending, so we take the last k indices ([-k:])
        top_k_local_indices = np.argsort(c_scores)[-k:]
        top_k_global_indices = c_indices[top_k_local_indices]
        
        selected_indices.extend(top_k_global_indices)
        
    return np.array(selected_indices)

# --- Usage Example ---
# 1. Train and Monitor
# model = GOLDSelectNet(layer_sizes=[1000, 256, 10], pattern_dim=64)
# trained_model, z_history = track_training_dynamics(model, X_train, y_train, device='cuda')

# 2. Calculate Weights (Earliness)
# importance_weights = calculate_pattern_importance(z_history)

# 3. Select Cells
# keep_idx, gold_scores = select_gold_cells(trained_model, X_train, y_train, importance_weights, device='cuda')

# X_subset = X_train[keep_idx]
# y_subset = y_train[keep_idx]