import torch
import torch.nn.functional as F

def train_and_get_confidence(model, X, y, sampler, optimizer_fn=torch.optim.SGD, criterion=torch.nn.CrossEntropyLoss(),
                             epochs=100, lr=1e-4, momentum=0.9, device=None):
    """
    Train the given model on data X, y with per-example confidence tracking.
    
    Args:
        model: PyTorch model (last layer must be log_softmax)
        optimizer_fn: optimizer class (e.g., torch.optim.SGD)
        criterion: loss function (e.g., nn.NLLLoss(reduction='none'))
        sampler: a function or iterable returning batches of indices
        X: input data tensor, shape [n_samples, n_features]
        y: labels tensor, shape [n_samples] (int class indices)
        epochs: number of epochs to train
        lr: learning rate
        momentum: momentum for optimizer (if applicable)
        device: torch device, e.g., 'cuda' or 'cpu'

    Returns:
        confidences: tensor of shape [n_samples], average confidence per sample
    """
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    
    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)
    
    # One-hot encoding
    y_onehot = F.one_hot(y, num_classes=n_classes).float()
    
    # Optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr, momentum=momentum)
    
    # Track confidence sum per example
    confidence_sum = torch.zeros(n_samples, device=device)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx in sampler(n_samples):
            xb = X[batch_idx]
            yb = y[batch_idx]
            yb_onehot = y_onehot[batch_idx]
            
            optimizer.zero_grad()
            output = model(xb)  # log_softmax output
            loss = criterion(output, yb)
            loss.mean().backward()  # aggregate batch loss
            optimizer.step()
            
            # Calculate probabilities
            probs = output.exp()  # log_softmax → probabilities
            
            # Confidence per sample = probability of true label
            batch_conf = (probs * yb_onehot).sum(dim=1)
            
            # Add to running sum
            confidence_sum[batch_idx] += batch_conf
        
        # Optional: print epoch progress
        # print(f"Epoch {epoch+1}/{epochs} done.")
    
    # Average over epochs
    confidences = confidence_sum / epochs
    return confidences