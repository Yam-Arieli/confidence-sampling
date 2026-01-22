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

def select_gold_cells(model, X, y, W_c, device, percentile=0.2):
    """
    Phase 2 & 3: Snapshot and Selection
    
    Args:
        model: Trained GOLDSelectNet
        X, y: Data and Labels
        W_c: The Importance Weights from calculate_pattern_importance
        percentile: Top fraction of cells to keep (e.g., 0.2 for top 20%)
        
    Returns:
        selected_indices: Indices of the chosen cells in X
        scores: The calculated GOLD scores for all cells
    """
    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).cpu().numpy() # Keep y on CPU for indexing
    
    # 1. Get Snapshot (Z matrix for all cells)
    model.eval()
    with torch.no_grad():
        _, z_matrix = model(X)
        z_matrix = z_matrix.cpu().numpy() # Shape: (n_samples, n_patterns)
        
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)
    
    # 2. Calculate Scores
    # Score = Dot Product of (Cell's Z) and (Class's Pattern Weights)
    for i in range(n_samples):
        cell_z = z_matrix[i]       # Vector (n_patterns,)
        cell_class = y[i]          # Scalar Class ID
        class_weights = W_c[:, cell_class] # Vector (n_patterns,) - Importance of patterns for this class
        
        # The GOLD Metric
        scores[i] = np.dot(cell_z, class_weights)
        
    # 3. Select Top Percentile per Class (Stratified Selection)
    selected_indices = []
    classes = np.unique(y)
    
    for c in classes:
        # Get indices of all cells in this class
        c_indices = np.where(y == c)[0]
        c_scores = scores[c_indices]
        
        # Determine threshold
        k = int(len(c_indices) * percentile)
        if k < 1: k = 1
        
        # Get indices of top k scores within this class
        # argsort is ascending, so take from end [-k:]
        top_k_local_indices = np.argsort(c_scores)[-k:]
        top_k_global_indices = c_indices[top_k_local_indices]
        
        selected_indices.extend(top_k_global_indices)
        
    return np.array(selected_indices), scores

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