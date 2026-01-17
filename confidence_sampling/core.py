import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score
from datetime import datetime
import wandb
import numpy as np

# ==============================================================================
# WANDB & LOGGING
# ==============================================================================
def init_new_group_run_generator(config: dict={}):
    group_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        run = wandb.init(
            entity="yam-arieli-hebrew-university-of-jerusalem",
            project="GOLDSelect",
            group=group_id,
            config=config
        )
        yield run

# Shared generator instance
group_run_generator = init_new_group_run_generator()

def eval_result(test_probs, y_test_labels, do_print=False):
    test_pred = test_probs.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test_labels, test_pred)
    recall = recall_score(y_test_labels, test_pred, average=None)
    f1_per_class = f1_score(y_test_labels, test_pred, average=None)
    f1_macro = f1_score(y_test_labels, test_pred, average='macro')
    f1_weighted = f1_score(y_test_labels, test_pred, average='weighted')

    if do_print:
        print(f"Accuracy: {acc:.4f}")
        for i, (r, f1) in enumerate(zip(recall, f1_per_class)):
            print(f"Class {i}: Recall={r:.4f}, F1={f1:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")

    return acc, recall, f1_per_class, f1_macro, f1_weighted

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
class ComplexNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, min_hidden_dim, output_dim, layers_num=3, dropout_p=0.1):
        super().__init__()
        layers_fc = []
        layers_ln = []
        
        min_hidden_dim = max(min_hidden_dim, output_dim)
        hidden_dim = max(min_hidden_dim, hidden_dim)

        for layer in range(layers_num):
            fc = nn.Linear(input_dim, hidden_dim)
            ln = nn.LayerNorm(hidden_dim)

            layers_fc.append(fc)
            layers_ln.append(ln)

            input_dim = hidden_dim
            hidden_dim = max(min_hidden_dim, int(hidden_dim / 2))

        self.layers_fc = nn.ModuleList(layers_fc)
        self.layers_ln = nn.ModuleList(layers_ln)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for fc, ln in zip(self.layers_fc, self.layers_ln):
            x = F.relu(ln(fc(x)))
            x = self.dropout(x)
        return self.fc_out(x)