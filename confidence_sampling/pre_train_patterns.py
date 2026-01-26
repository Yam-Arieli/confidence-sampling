import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class Sparsemax(nn.Module):
    """
    Sparsemax activation function.
    Pytorch implementation of: https://arxiv.org/abs/1602.02068
    """
    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        # Translate input by max for numerical stability
        input = input - input.max(dim=self.dim, keepdim=True)[0].expand_as(input)
        
        # Sort input in descending order
        zs = input.sort(dim=self.dim, descending=True)[0]
        range_values = torch.arange(start=1, end=zs.size(self.dim) + 1, device=input.device).float()
        range_values = range_values.view(1, -1) if self.dim == 1 else range_values

        # Determine threshold indices
        bound = 1 + range_values * zs
        cumsum_zs = torch.cumsum(zs, dim=self.dim)
        is_gt = bound > cumsum_zs
        k = is_gt.sum(dim=self.dim, keepdim=True).float()

        # Compute threshold (tau)
        taus = (cumsum_zs.gather(self.dim, (k - 1).long()) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax output
        return torch.max(torch.zeros_like(input), input - taus)


class GOLDSelectNet(nn.Module):
    def __init__(self, layer_sizes, pattern_dim=64, dropout_rate=0.2, temperature=2.0):
        super(GOLDSelectNet, self).__init__()
        
        # ... (Encoder setup same as before) ...
        # (Copy the encoder loop from previous code)
        input_dim = layer_sizes[0]
        hidden_dims = layer_sizes[1:-1]
        num_classes = layer_sizes[-1]
        
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # --- NEW: Regulation Parameters ---
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout_rate)

        # Pattern Layer
        self.pattern_projection = nn.Linear(current_dim, pattern_dim)
        self.pattern_bn = nn.BatchNorm1d(pattern_dim)
        self.sparsemax = Sparsemax(dim=1)
        
        # Classifier
        self.classifier = nn.Linear(pattern_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        
        z_logits = self.pattern_projection(features)
        z_logits = self.pattern_bn(z_logits)
        
        # --- FIX 1: Dropout (Force Redundancy) ---
        # Randomly kill patterns so the net learns backups
        z_logits = self.dropout(z_logits)
        
        # --- FIX 2: Temperature (Force Softness) ---
        # Shrink the gaps between logits so more patterns survive the cut
        z_logits = z_logits / self.temperature
        
        z = self.sparsemax(z_logits)
        
        logits = self.classifier(z)
        return F.log_softmax(logits, dim=1), z

def track_training_dynamics(model, X, y, device, optimizer_fn=torch.optim.SGD, 
                            criterion=torch.nn.NLLLoss(reduction='none'), # Default to 'none' to match your logic
                            batch_size=256, epochs=100, lr=1e-4):
    
    model = model.to(device)
    
    # 1. Cast Data to Match Model Logic (Float32 is standard for Nets)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    
    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)
    
    # Determine pattern dim from the classifier weights
    n_patterns = model.classifier.weight.shape[1] 
    
    # The Tensor to track pattern dynamics: (Epochs, Patterns, Classes)
    z_tensor = np.zeros((epochs, n_patterns, n_classes))
    
    y_onehot = F.one_hot(y, num_classes=n_classes).float()
    dataset = TensorDataset(X, y_onehot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # --- 1. Train Step ---
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_label = y_batch.argmax(dim=1).to(device)
            
            optimizer.zero_grad()
            
            # Forward (ignore Z for loss calc)
            output, _ = model(X_batch) 
            
            loss = criterion(output, y_label)
            
            # handle reduction='none' vs 'mean' dynamically
            if loss.dim() > 0:
                loss.mean().backward() # Backprop the mean
                total_loss += loss.sum().item() # Track the sum for reporting
            else:
                loss.backward()
                total_loss += loss.item() * X_batch.size(0)

            optimizer.step()
            
        avg_loss = total_loss / n_samples
            
        # --- 2. Monitor Step (End of Epoch) ---
        model.eval()
        with torch.no_grad():
            all_z = []
            # Use a non-shuffled loader for consistent evaluation
            eval_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
            for (X_b,) in eval_loader:
                _, z_b = model(X_b.to(device))
                all_z.append(z_b.cpu())
            all_z = torch.cat(all_z, dim=0) 
            
            # Group by Class and Calculate Mean
            for c in range(n_classes):
                class_mask = (y.cpu() == c)
                if class_mask.sum() > 0:
                    mean_z = all_z[class_mask].mean(dim=0).numpy()
                    z_tensor[epoch, :, c] = mean_z
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | tracked.")

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

def get_cell_patterns(model, X, device, batch_size=256):
    """
    Phase 2: The Snapshot
    Runs a forward pass on the full dataset to extract the active patterns for every cell.
    
    Args:
        model: The trained GOLDSelectNet.
        X: Input data (n_samples, n_features).
        device: 'cuda' or 'cpu'.
        batch_size: Batch size to prevent memory overflow.
        
    Returns:
        z_matrix: Numpy array of shape (n_samples, n_patterns).
                  Each row i is the sparse latent vector for cell i.
    """
    model = model.to(device)
    model.eval()
    
    # Ensure data is Float32 to match model weights
    X_tensor = torch.tensor(X)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_z = []
    
    print("Extracting latent patterns...")
    with torch.no_grad():
        for i, (X_batch,) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            
            # We only care about the second output (z)
            _, z_batch = model(X_batch)
            
            all_z.append(z_batch.cpu())
            
    # Concatenate all batches into one large matrix
    z_matrix = torch.cat(all_z, dim=0).numpy()
    
    print(f"Extraction complete. Matrix shape: {z_matrix.shape}")
    return z_matrix