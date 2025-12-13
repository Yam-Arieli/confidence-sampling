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


class LoRANet(nn.Module):
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
        super(LoRANet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            bottle_neck = int(np.sqrt((layer_sizes[i] + layer_sizes[i+1])/2))
            matA = nn.Linear(layer_sizes[i], bottle_neck)
            matB = nn.Linear(bottle_neck, layer_sizes[i+1])
            layers.append(nn.Sequential(matA, matB))
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

def pretrain_and_get_confidence(model, X, y, device=None, optimizer_fn=torch.optim.SGD, criterion=torch.nn.NLLLoss(),
                                weighted_sampler=True, batch_size=256, epochs=100, lr=1e-4, momentum=0.9):
    """
    Train the given model on data X, y with per-example confidence tracking.
    
    Args:
        model: PyTorch model (last layer must be log_softmax)
        optimizer_fn: optimizer class (e.g., torch.optim.SGD)
        criterion: loss function (e.g., nn.CrossEntropyLoss(reduction='none'))
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
    confidence = (probs_sum / epochs).cpu().numpy()
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

def evaluate_model_probs_n_classes_probs(model, X, y_onehot, device, batch_size=256):
    indices = torch.arange(len(X)).to(device)
    dataset = TensorDataset(X, y_onehot, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_classes_probs = torch.zeros((len(X), y_onehot.shape[1]), dtype=X.dtype, device=device)
    all_probs = torch.zeros(len(X), dtype=X.dtype, device=device)
    model.eval()
    with torch.no_grad():
        for X_batch, y_onehot_batch, idx_batch in dataloader:
            X_batch = X_batch.to(device)
            y_onehot_batch = y_onehot_batch.to(device)
            output = model(X_batch)

            probs = output.exp()
            right_probs = (probs * y_onehot_batch).sum(dim=1)

            all_classes_probs[idx_batch] = probs
            all_probs[idx_batch] = right_probs
    
    return all_probs, all_classes_probs

def pretrain_and_get_signals(
    model,
    X,
    y,
    device=None,
    optimizer_fn=torch.optim.SGD,
    criterion=torch.nn.NLLLoss(),
    weighted_sampler=True,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    momentum=0.9,
):
    """
    Train a model on X, y while collecting signals for various sampling strategies.
    
    Signals collected:
    1. Confidence: average of probabilities assigned to the correct label across all epochs.
    2. Final predicted probabilities for all classes: used for uncertainty-based sampling (margin/entropy).
    3. Forgetting counts: number of times a sample flips from correct → incorrect across epochs.
    
    Args:
        model: PyTorch model. Last layer should output log-softmax.
        X: input tensor of shape (n_samples, n_features)
        y: label tensor of shape (n_samples,) (integer class indices)
        device: 'cpu' or 'cuda'
        optimizer_fn: optimizer class (e.g., torch.optim.SGD)
        criterion: loss function (reduction='none' recommended if per-sample losses needed)
        weighted_sampler: whether to use a weighted DataLoader sampler
        batch_size: batch size for training and evaluation
        epochs: number of epochs
        lr: learning rate
        momentum: momentum (if applicable)
    
    Returns:
        confidence: np.ndarray, shape (n_samples,)
            Average probability assigned to the correct label across all epochs.
        final_probs: np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities at the final epoch (used for uncertainty sampling).
        afliteting_counts: np.ndarray, shape (n_samples,)
            Number of afliteting events per sample.
        train_losses: list of floats
            Average training loss per epoch.
    """
    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)

    # One-hot encoding of labels for computing probabilities
    y_onehot = F.one_hot(y, num_classes=n_classes).float()

    # Prepare DataLoader
    dataloader = get_dataloader(
        X, y_onehot, weighted_sampler=weighted_sampler, device=device, batch_size=batch_size
    )

    # Initialize optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr, momentum=momentum)

    # Initialize storage for signals
    # sum_probs will accumulate the probability of the correct label across epochs
    sum_probs = torch.zeros(n_samples, dtype=X.dtype, device=device)
    # correctness_matrix tracks whether each sample was predicted correctly at each epoch
    correctness_matrix = torch.zeros((epochs, n_samples), dtype=torch.bool, device=device)
    train_losses = []

    for epoch in range(epochs):
        # Train one epoch
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

        # Evaluate model to get per-sample probabilities
        # all_probs: probability of the true label for each sample
        # all_classes_probs: full probability distribution for each sample
        all_probs, all_classes_probs = evaluate_model_probs_n_classes_probs(model, X, y_onehot, device, batch_size=batch_size)

        # Track correctness for forgetting events
        # predicted class is argmax over class probabilities
        _, preds = all_classes_probs.max(dim=1)
        correct = preds.eq(y)
        correctness_matrix[epoch] = correct

        # Accumulate true-label probabilities for confidence
        sum_probs += all_probs

        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Compute confidence as the average probability of the correct label over all epochs
    confidence = (sum_probs / epochs).cpu().numpy()

    # Final predicted probabilities for uncertainty-based sampling
    final_probs = all_classes_probs.detach().cpu().numpy()

    # Compute forgetting counts: number of correct → incorrect transitions per sample
    forgetting_counts = compute_forgetting_events(correctness_matrix)

    return confidence, final_probs, forgetting_counts, train_losses

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_onehot_batch in dataloader:
        X_batch = X_batch.to(device)
        y = y_onehot_batch.argmax(dim=1).to(device)

        optimizer.zero_grad()
        output = model(X_batch)        # log_softmax output
        loss = criterion(output, y)    # per-sample loss
        loss.mean().backward()
        optimizer.step()

        total_loss += loss.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss



def pretrain_and_probs_matrix(
    model,
    X,
    y,
    device=None,
    optimizer_fn=torch.optim.SGD,
    criterion=torch.nn.NLLLoss(reduction='none'),
    weighted_sampler=True,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    momentum=0.9,
    weight_power=1.0,   ### NEW: exponent for weighting 1 - confidence
):
    """
    Same docstring as before — omitted for brevity
    """

    model = model.to(device)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    n_samples = X.shape[0]
    n_classes = int(y.max().item() + 1)

    y_onehot = F.one_hot(y, num_classes=n_classes).float()

    # Initial equal weights
    sample_weights = torch.ones(n_samples, device=device)

    # Build initial dataloader
    dataset = TensorDataset(X, y_onehot)

    def make_loader(weights):
        sampler = WeightedRandomSampler(
            weights.cpu().numpy(),
            num_samples=len(weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    if weighted_sampler:
        dataloader = make_loader(sample_weights)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr, momentum=momentum)

    # Storage
    sum_probs = torch.zeros(n_samples, device=device)
    probs_matrix = torch.zeros((epochs, n_samples), device=device)
    probs_matrix_history = []
    train_losses = []

    for epoch in range(epochs):
        # --- Train one epoch ---
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

        # --- Evaluate per-sample probabilities ---
        all_probs, all_classes_probs = evaluate_model_probs_n_classes_probs(
            model, X, y_onehot, device, batch_size=batch_size
        )
        probs_matrix_history.append(all_classes_probs)

        probs_matrix[epoch] = all_probs
        sum_probs += all_probs

        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # --- NEW: update sample weights so low-confidence gets MORE weight ---
        if weighted_sampler:
            with torch.no_grad():
                # Confidence estimate so far (averaged)
                current_confidence = sum_probs / (epoch + 1)

                # Weight = (1 - confidence)^power
                new_weights = (1.0 - current_confidence).clamp(min=0.0) ** weight_power

                # Scale
                new_weights = (new_weights - new_weights.mean()) / (new_weights.std())
                new_weights -= new_weights.min()

                # Avoid zero weights
                new_weights = new_weights + 1e-8

            # plt.hist(new_weights.cpu().numpy(), bins=30)
            # plt.show()
            # Rebuild dataloader for the next epoch
            dataloader = make_loader(new_weights)

    # After all epochs
    confidence = (sum_probs / epochs).cpu().numpy()
    final_probs = all_classes_probs.detach().cpu().numpy()
    probs_matrix_history = torch.stack(probs_matrix_history, dim=0).cpu().numpy()

    return confidence, final_probs, train_losses, probs_matrix, probs_matrix_history