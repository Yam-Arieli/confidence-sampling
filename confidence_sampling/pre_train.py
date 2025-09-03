import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np

class BaseNet(nn.Module):
    """
    This class is copied from Annotatability.models, and can be
    found at 'https://github.com/nitzanlab/Annotatability/blob/main/Annotatability/models.py'.
    """
    def __init__(self, layer_sizes):
        """
        Initializes a feedforward neural network with variable number of fully-connected layers.

        Args:
            layer_sizes (list of int): Sizes of each layer including input and output layers.
        """
        super(BaseNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output data of shape (batch_size, output_size).
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)

def get_dataloader(X, y_onehot, weighted_sampler=False, device=None, batch_size=256):
    """
    Create a DataLoader for X and one-hot encoded y.
    
    Args:
        X: tensor of shape [n_samples, n_features]
        y_onehot: tensor of shape [n_samples, n_classes] (one-hot)
        use_weighted_sampler: if True, use WeightedRandomSampler with inverse class frequency
        device: torch device, e.g., 'cuda' or 'cpu'
        batch_size: int, batch size for DataLoader

    Returns:
        DataLoader object
    """
    X = X.to(device)
    y_onehot = y_onehot.to(device)
    
    dataset = TensorDataset(X, y_onehot)
    
    if weighted_sampler:
        # Compute weights = 1 / class frequency
        class_counts = y_onehot.sum(dim=0)
        class_weights = 1.0 / class_counts
        sample_weights = (y_onehot * class_weights).sum(dim=1)
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_onehot_batch in dataloader:
        X_batch = X_batch.to(device)
        y_onehot_batch = y_onehot_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)  # log_softmax output
        loss = criterion(output, y_onehot_batch)
        loss.mean().backward()  # aggregate batch loss
        optimizer.step()
        
        total_loss += loss.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def evaluate_model(model, X, y_onehot, device, batch_size=256):
    indices = torch.arange(len(X)).to(device)
    dataset = TensorDataset(X, y_onehot, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = torch.zeros(len(X), dtype=X.dtype, device=device)
    model.eval()
    with torch.no_grad():
        for X_batch, y_onehot_batch, idx_batch in dataloader:
            X_batch = X_batch.to(device)
            y_onehot_batch = y_onehot_batch.to(device)
            output = model(X_batch)
            probs = output.exp()
            right_probs = (probs * y_onehot_batch).sum(dim=1)
            all_probs[idx_batch] = right_probs
    
    return all_probs

def pretrain_and_get_confidence(model, X, y, device=None, optimizer_fn=torch.optim.SGD, criterion=torch.nn.CrossEntropyLoss(),
                                weighted_sampler=True, batch_size=256, epochs=100, lr=1e-4, momentum=0.9):
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
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    
    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)
    
    # One-hot encoding
    y_onehot = F.one_hot(y, num_classes=n_classes).float()

    # DataLoader
    dataloader = get_dataloader(X, y_onehot, weighted_sampler=weighted_sampler,
                                device=device, batch_size=batch_size)
    
    # Optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr, momentum=momentum)
    
    probs_sum = torch.zeros((n_samples,), device=device)
    train_losses = []
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        probs = evaluate_model(model, X, y_onehot, device, batch_size=batch_size)

        train_losses.append(avg_loss)
        probs_sum = probs_sum + probs

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Average over epochs
    confidence = probs_sum / epochs
    return confidence, train_losses

##################################################################################

def compute_forgetting_events(correctness_matrix):
    """
    Compute forgetting events from a boolean correctness matrix.

    Parameters
    ----------
    correctness_matrix : torch.Tensor of shape (epochs, n_samples)
        correctness_matrix[e, i] = True if sample i was correctly classified at epoch e

    Returns
    -------
    forgetting_counts : np.ndarray of shape (n_samples,)
        Number of forgetting events for each sample.
    """
    epochs, n_samples = correctness_matrix.shape
    forgetting_counts = np.zeros(n_samples, dtype=int)

    prev_correct = correctness_matrix[0].cpu().numpy()
    for e in range(1, epochs):
        curr_correct = correctness_matrix[e].cpu().numpy()
        # count correct → incorrect transitions
        forgetting_counts += ((prev_correct == 1) & (curr_correct == 0))
        prev_correct = curr_correct

    return forgetting_counts

def pretrain_and_get_signals(model, X, y, device=None, optimizer_fn=torch.optim.SGD,
                             criterion=torch.nn.CrossEntropyLoss(), weighted_sampler=True,
                             batch_size=256, epochs=100, lr=1e-4, momentum=0.9):
    """
    Train the given model on data X, y and extract signals for sampling methods:
    - average confidence per sample (for AFLite / least_confidence)
    - full probabilities at final epoch (for margin / entropy)
    - forgetting counts (for forgetting-based sampling)

    Returns
    -------
    confidence : np.ndarray of shape (n_samples,)
        Average confidence per sample across epochs.
    final_probs : np.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities at the last epoch.
    forgetting_counts : np.ndarray of shape (n_samples,)
        Number of forgetting events per sample.
    train_losses : list of floats
        Average loss per epoch.
    """
    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)

    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)

    y_onehot = F.one_hot(y, num_classes=n_classes).float()

    dataloader = get_dataloader(X, y_onehot, weighted_sampler=weighted_sampler,
                                device=device, batch_size=batch_size)

    optimizer = optimizer_fn(model.parameters(), lr=lr, momentum=momentum)

    probs_sum = torch.zeros((n_samples,), device=device)
    correctness_matrix = torch.zeros((epochs, n_samples), dtype=torch.bool, device=device)
    train_losses = []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        probs = evaluate_model(model, X, y_onehot, device, batch_size=batch_size)  # shape (n_samples, n_classes)

        # confidence of predicted class
        conf_epoch, preds = probs.max(dim=1)
        correct = preds.eq(y)
        correctness_matrix[epoch] = correct

        # accumulate confidence
        probs_sum += conf_epoch

        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Average over epochs
    confidence = (probs_sum / epochs).cpu().numpy()
    final_probs = probs.detach().cpu().numpy()
    forgetting_counts = compute_forgetting_events(correctness_matrix)

    return confidence, final_probs, forgetting_counts, train_losses