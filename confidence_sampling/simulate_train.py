import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from confidence_sampling.utils import prepare_train_tensors
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_one_epoch(model, optimizer, criterion, X_tensor, y_tensor, batch_size=16):
    model.train()
    epoch_losses = []
    # Shuffle indices for random batching
    permutation = torch.randperm(X_tensor.size(0))
    num_samples = X_tensor.size(0)
    num_batches = int(np.ceil(num_samples / batch_size))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        indices = permutation[start_idx:end_idx]
        batch_x, batch_y = X_tensor[indices], y_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    return epoch_losses

class ComplexNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, droupout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.ln2 = nn.LayerNorm(int(hidden_dim/4))

        self.fc3 = nn.Linear(int(hidden_dim/4), 16)
        self.ln3 = nn.LayerNorm(16)

        self.fc4 = nn.Linear(16, int(hidden_dim/8))
        self.ln4 = nn.LayerNorm(int(hidden_dim/8))

        self.dropout = nn.Dropout(droupout_p)
        self.fc_out = nn.Linear(int(hidden_dim/8), output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout(x)

        return self.fc_out(x)

class ComplexNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, droupout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.ln1 = nn.LayerNorm(32)

        self.fc2 = nn.Linear(32, int(hidden_dim/4))
        self.ln2 = nn.LayerNorm(int(hidden_dim/4))

        self.fc3 = nn.Linear(int(hidden_dim/4), 16)
        self.ln3 = nn.LayerNorm(16)

        self.fc4 = nn.Linear(16, int(hidden_dim/8))
        self.ln4 = nn.LayerNorm(int(hidden_dim/8))

        self.dropout = nn.Dropout(droupout_p)
        self.fc_out = nn.Linear(int(hidden_dim/8), output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout(x)

        return self.fc_out(x)


def eval_result(test_probs, y_test_labels, do_print=False):
    # Get predicted class for each test sample
    test_pred = test_probs.argmax(dim=1).cpu().numpy()

    # Overall accuracy
    acc = accuracy_score(y_test_labels, test_pred)

    # Recall per class
    recall = recall_score(y_test_labels, test_pred, average=None)

    # F1 per class
    f1_per_class = f1_score(y_test_labels, test_pred, average=None)
    # Macro F1 (treats all classes equally)
    f1_macro = f1_score(y_test_labels, test_pred, average='macro')
    # Weighted F1 (accounts for class imbalance)
    f1_weighted = f1_score(y_test_labels, test_pred, average='weighted')

    if do_print:
        print(f"Accuracy: {acc:.4f}")
        for i, (r, f1) in enumerate(zip(recall, f1_per_class)):
            print(f"Class {i}: Recall={r:.4f}, F1={f1:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")

    return acc, recall, f1_per_class, f1_macro, f1_weighted

def do_train(X_tensor, y_tensor, num_classes,
             X_test_tensor, y_test_labels, device,
             hidden_dim=1024, epochs=100, batch_size=16, lr=1e-4, droupout_p=0.1, model=None, do_print=False):
    input_dim = X_tensor.shape[1]
    
    if not model:
        model = ComplexNet2(input_dim, hidden_dim, num_classes, droupout_p=droupout_p).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
    criterion = nn.CrossEntropyLoss()

    losses = []
    test_metrics = []
    
    for epoch in range(epochs):
        epoch_losses = train_one_epoch(model, optimizer, criterion, X_tensor, y_tensor, batch_size)

        # Predict probabilities for adata_test
        with torch.no_grad():
            test_probs = torch.softmax(model(X_test_tensor), dim=1)
        
        acc_test, recall_test, f1_per_class, f1_macro, f1_weighted = eval_result(test_probs, y_test_labels, do_print=False)
        test_metrics.append([acc_test, recall_test, f1_per_class, f1_macro, f1_weighted])
        
        if do_print and ((epoch+1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")
        
        losses.append(epoch_losses)

    # Get probabilities
    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=1)
    
    return model, probs, np.array(losses), test_metrics

def simulate_train(adata_train, adata_test, device, epochs=80, batch_size=16, lr=1e-4, hidden_dim=1024, droupout_p=0.1):
    adata_temp = adata_train.copy()
    X_tensor, y_tensor, num_classes = prepare_train_tensors(adata_temp, device)
    X_test_tensor, y_test_tensor, num_classes_test = prepare_train_tensors(adata_test, device)
    y_test_labels = adata_test.obs["y"].astype("category").cat.codes.values
    model, probs, losses, test_metrics = do_train(X_tensor, y_tensor, num_classes,
                                                  X_test_tensor, y_test_labels,
                                                  device, hidden_dim=hidden_dim, epochs=epochs,
                                                  batch_size=batch_size, lr=lr, droupout_p=droupout_p, do_print=True)

    return model, probs, losses, test_metrics