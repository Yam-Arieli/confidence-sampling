import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from confidence_sampling.utils import prepare_train_tensors
from sklearn.metrics import accuracy_score, recall_score, f1_score
from datetime import datetime
import wandb

# Make sure you wandb.login(key='KEY') before using this code.
def init_new_group_run_generator(config: dict={}):
    group_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Start a new wandb run to track this script.
    while True:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="yam-arieli-hebrew-university-of-jerusalem",
            # Set the wandb project where this run will be logged.
            project="GOLDSelect",
            # The group aggregate all the iterations of the same setting for bootstrap aggregation
            group=group_id,
            # Track hyperparameters and run metadata.
            config=config
        )

        yield run

# Create the generator instance
group_run_generator = init_new_group_run_generator()

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

class ComplexNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers_num=3, droupout_p=0.1):
        super().__init__()
        layers_fc = []
        layers_ln = []
        for layer in range(layers_num):
            fc = nn.Linear(input_dim, hidden_dim)
            ln = nn.LayerNorm(hidden_dim)

            layers_fc.append(fc)
            layers_ln.append(ln)

            input_dim = hidden_dim
            hidden_dim = max(output_dim, int(hidden_dim / 2))

        self.layers_fc = nn.ModuleList(layers_fc)
        self.layers_ln = nn.ModuleList(layers_ln)
        self.dropout = nn.Dropout(droupout_p)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for fc, ln in zip(self.layers_fc, self.layers_ln):
            x = F.relu(ln(fc(x)))
            F.dropout(x, p=self.dropout.p, training=self.training)
            x = self.dropout(x)

        return self.fc_out(x)

def simulate_train(adata_train, adata_test, device, epochs=80,
                   batch_size=16, lr=1e-4, hidden_dim=1024, droupout_p=0.1,
                   eval_last_only=False, scheduler=None, ran_name='lrdy_04',
                   dropout_p=0.1):
    adata_temp = adata_train.copy()
    X_tensor, y_tensor, num_classes = prepare_train_tensors(adata_temp, device)
    X_test_tensor, y_test_tensor, num_classes_test = prepare_train_tensors(adata_test, device)
    y_test_labels = adata_test.obs["y"].astype("category").cat.codes.values
    model, probs, losses, test_metrics = do_train(X_tensor, y_tensor, num_classes,
                                                  X_test_tensor, y_test_labels,
                                                  device, hidden_dim=hidden_dim, epochs=epochs,
                                                  batch_size=batch_size, lr=lr, droupout_p=droupout_p,
                                                  do_print=True, eval_last_only=eval_last_only,
                                                  scheduler=scheduler, ran_name=ran_name, dropout_p=droupout_p)

    return model, probs, losses, test_metrics

def do_train(X_tensor, y_tensor, num_classes,
             X_test_tensor, y_test_labels, device,
             hidden_dim=1024, epochs=100, batch_size=16,
             lr=1e-4, droupout_p=0.1, model=None, do_print=False,
             eval_last_only=False, scheduler=None, ran_name='lrdy_04',
             dropout_p=0.1):
             
    input_dim = X_tensor.shape[1]
    
    if not model:
        model = ComplexNet(input_dim, hidden_dim, num_classes, layers_num=6, droupout_p=droupout_p).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)

    # Initialize scheduler ONLY if the user passed a scheduler constructor
    if scheduler is not None:
        scheduler = scheduler(optimizer)   # <-- allow passing lambda opt: ...

    criterion = nn.CrossEntropyLoss()

    losses = []
    test_metrics = []


    run = group_run_generator.__next__()
    run.config['method'] = ran_name
    run.name = ran_name
    
    cumulative_time = 0.0
    for epoch in range(epochs):
        start_time = datetime.now()
        epoch_losses = train_one_epoch(model, optimizer, criterion, X_tensor, y_tensor, batch_size)

        # STEP SCHEDULER HERE
        if scheduler is not None:
            scheduler.step()               # <-- added

        # Predict probabilities for adata_test
        with torch.no_grad():
            test_probs = torch.softmax(model(X_test_tensor), dim=1)

        end_time = datetime.now()
        cumulative_time += (end_time - start_time).total_seconds()

        acc_test, recall_test, f1_per_class, f1_macro, f1_weighted = eval_result(
            test_probs, y_test_labels, do_print=False
        )
        test_metrics.append([acc_test, recall_test, f1_per_class, f1_macro, f1_weighted])
        run.log({"acc": acc_test, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
                 'epoch': epoch, 'cumulative_time': int(cumulative_time*10)/10})
        
        if do_print and ((epoch+1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")
        
        losses.append(epoch_losses)

    run.finish()
    # Get probabilities
    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=1)
    
    return model, probs, np.array(losses), test_metrics
##################################################################################
############################### Actual Train #####################################
##################################################################################

def run_sampling_experiment(
    sampler_func,
    sample_args,
    label,
    adata_train,
    adata_test,
    device,
    actual_train_params,
    iterations,
    scheduler=None,
    ran_name='lrdy_04',
    dropout_p=0.1
):
    """Run repeated sampling + training experiments and collect test metrics."""
    all_metrics = []
    
    for iteration in range(iterations):
        print(f"iteration: {iteration}")
        
        sample_indices = sampler_func(**sample_args) if sampler_func else None
        adata_sample = adata_train[sample_indices, :].copy() if sample_indices is not None else adata_train
        print(adata_sample.shape)

        # model, probs, losses, test_metrics = simulate_train.simulate_train(
        model, probs, losses, test_metrics = simulate_train(
            adata_sample,
            adata_test,
            device,
            epochs=actual_train_params["epochs"],
            lr=actual_train_params["lr"],
            batch_size=actual_train_params["batch_size"],
            hidden_dim=actual_train_params["hidden_dim"],
            scheduler=scheduler,
            ran_name=ran_name,
            dropout_p=dropout_p
        )

        accuracies = [epoch_results[0] for epoch_results in test_metrics]
        all_metrics.append(test_metrics)

        print(accuracies[-1], end="\n-------------------------\n\n")
