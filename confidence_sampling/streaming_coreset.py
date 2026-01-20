import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from confidence_sampling import sampler
from confidence_sampling.core import ComplexNet, eval_result, group_run_generator
from confidence_sampling.utils import prepare_train_tensors

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_stratified_batches(y, batch_size, min_per_class=3):
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    
    max_batches_by_class = (counts // min_per_class).min()
    max_batches_by_size = n_total // batch_size
    n_batches = int(min(max_batches_by_class, max_batches_by_size))
    if n_batches < 1: n_batches = 1
    
    class_indices = {c: np.where(y == c)[0] for c in classes}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
        
    batches = [[] for _ in range(n_batches)]
    
    # Core
    for i in range(n_batches):
        for c in classes:
            selected = class_indices[c][-min_per_class:]
            class_indices[c] = class_indices[c][:-min_per_class]
            batches[i].extend(selected)
            
    # Leftovers
    leftovers = []
    for c in classes:
        leftovers.extend(class_indices[c])
    np.random.shuffle(leftovers)
    
    current_batch = 0
    for idx in leftovers:
        if len(batches[current_batch]) < batch_size:
            batches[current_batch].append(idx)
        else:
            current_batch += 1
            if current_batch >= n_batches: break
                
    return [np.array(b) for b in batches]

def cull_arena(arena_indices, new_indices, history_sum, history_count, target_size, power=0.4):
    if len(arena_indices) <= target_size:
        return arena_indices

    current_conf = history_sum[arena_indices] / (history_count[arena_indices] + 1e-8)
    current_conf = current_conf.cpu().numpy()
    
    is_rookie = np.isin(arena_indices, new_indices)
    rookie_indices = arena_indices[is_rookie]
    
    veteran_mask = ~is_rookie
    veteran_indices = arena_indices[veteran_mask]
    veteran_conf = current_conf[veteran_mask]
    print(f'veteran_conf avg: ', veteran_conf.mean())
    
    n_rookies = len(rookie_indices)
    n_slots_for_vets = max(0, target_size - n_rookies)

    if n_slots_for_vets > 0 and len(veteran_indices) > 0:
        survivor_local_idxs = sampler.sample_by_conf(veteran_conf, n_slots_for_vets, power=power)
        survivor_vets = veteran_indices[survivor_local_idxs]
    else:
        survivor_vets = np.array([], dtype=int)
        
    new_arena = np.concatenate([rookie_indices, survivor_vets])
    return np.unique(new_arena.astype(int))


# ==============================================================================
# HELPER 1: Data Preparation
# ==============================================================================
def prepare_data_for_evolution(adata_train, adata_test, device):
    """
    Handles data extraction, type conversion (Categorical -> Int), and Tensor creation.
    """
    # 1. Extract Train Data
    X_np = adata_train.X if isinstance(adata_train.X, np.ndarray) else adata_train.X.toarray()
    
    # Handle Categorical 'y' safely
    if hasattr(adata_train.obs['y'], 'cat'):
        y_np = adata_train.obs['y'].cat.codes.values
    else:
        y_np = adata_train.obs['y'].values
        
    if not isinstance(y_np, np.ndarray):
        y_np = np.array(y_np)

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()
    
    # 2. Extract Test Data
    # (Using utils helper if available, or manual extraction)
    X_test_tensor, _, _ = prepare_train_tensors(adata_test, device)
    y_test_labels = adata_test.obs["y"].astype("category").cat.codes.values
    
    # 3. Meta-data
    n_genes = X.shape[1]
    n_classes = int(y.max().item() + 1)
    
    return X, y, y_np, X_test_tensor, y_test_labels, n_genes, n_classes


# ==============================================================================
# HELPER 2: Model Initialization
# ==============================================================================
def init_training_components(n_genes, n_classes, params, device):
    """
    Initializes the Model, Optimizer, and Loss function.
    """
    model = ComplexNet(
        n_genes, 
        params['hidden_dim'], 
        params.get('min_hidden_dim', 128), 
        n_classes, 
        layers_num=params['layers_num'], 
        dropout_p=params.get('dropout_p', 0.1)
    ).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


# ==============================================================================
# HELPER 3: The "Arena" Training Step (FIXED)
# ==============================================================================
def train_arena_step(model, optimizer, criterion, X, y, arena_indices, sub_epochs, device, history_sum, history_count, n_classes, batch_size):
    """
    Trains the model on the current Arena using the specified batch_size.
    """
    # Create DataLoader for Arena
    X_arena = X[arena_indices].to(device)
    y_arena = y[arena_indices].to(device)
    y_arena_oh = F.one_hot(y_arena, num_classes=n_classes).float()
    
    # FIX: Use the user-defined batch_size (32) instead of hardcoded 256
    ds = TensorDataset(X_arena, y_arena_oh, torch.tensor(arena_indices).to(device))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(sub_epochs):
        for bx, by, global_idx in dl:
            optimizer.zero_grad()
            out = model(bx)
            
            loss = criterion(out, by.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            # Confidence tracking
            probs = F.softmax(out, dim=1).detach()
            true_probs = (probs * by).sum(dim=1)

            history_sum.index_add_(0, global_idx, true_probs)
            history_count.index_add_(0, global_idx, torch.ones_like(true_probs))

# ==============================================================================
# HELPER 4: Evaluation & Logging
# ==============================================================================
def evaluate_and_log(run, model, X_test_tensor, y_test_labels, start_time, cycle, batch_i, total_batches, arena_size, ratio_N_to_M):
    """
    Runs evaluation on test set and logs metrics to WandB with scaled x-axis.
    """
    with torch.no_grad():
        test_probs = torch.softmax(model(X_test_tensor), dim=1)
        
    cur_time = (datetime.now() - start_time).total_seconds()
    acc, _, _, f1_macro, f1_weighted = eval_result(test_probs, y_test_labels, do_print=False)
    
    # Calculate scaled epoch for fair comparison plot
    frac_epoch = cycle + (batch_i / total_batches)
    scaled_epoch = frac_epoch * ratio_N_to_M
    
    run.log({
        "acc": acc, 
        "f1_macro": f1_macro, 
        "f1_weighted": f1_weighted,
        "epoch": scaled_epoch,        # Scaled for comparison
        "raw_epoch": frac_epoch,      # Actual passes
        "cycle": cycle,
        "arena_size": arena_size,
        "cumulative_time": int(cur_time*10)/10
    })
    return acc

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def run_evolutionary_experiment(
    adata_train,
    adata_test,
    device,
    actual_train_params,
    iterations=1,
    cycles=5,
    sub_epochs=3,
    ran_name='evolutionary_stream',
    reset_history=False
):
    X, y, y_np, X_test_tensor, y_test_labels, n_genes, n_classes = prepare_data_for_evolution(
        adata_train, adata_test, device
    )

    # 1. SETUP PARAMS
    # This is the "Fresh Batch" size (how many new cells we add)
    # We set this to coreset_size to minimize the number of "Stops" the stream makes.
    coreset_size = int(len(y) * actual_train_params['sample_frac'])
    fresh_batch_size = int(coreset_size / 2)
    
    # This is the "Training" batch size (e.g., 32) used for Gradient Descent
    train_batch_size = actual_train_params['batch_size'] 

    # Ratio for plotting
    ratio_N_to_M = len(y) / (coreset_size if coreset_size > 0 else 1)

    for iteration in range(iterations):
        print(f"\n--- Evolutionary Iteration {iteration+1}/{iterations} ---")
        
        run = group_run_generator.__next__()
        run.config.update(actual_train_params)
        run.config['method'] = ran_name
        run.config['reset_history'] = reset_history
        run.name = f"{ran_name}_iter{iteration}"
        
        model, optimizer, criterion = init_training_components(
            n_genes, n_classes, actual_train_params, device
        )
        
        arena_indices = np.array([], dtype=int)
        history_sum = torch.zeros(len(y), device=device)
        history_count = torch.zeros(len(y), device=device)
        start_time = datetime.now()
        
        for cycle in range(cycles):
            # Create batches for INJECTION (size = coreset_size)
            batches = create_stratified_batches(
                y_np, fresh_batch_size, min_per_class=actual_train_params.get('min_per_class', 3)
            )
            
            for batch_i, new_indices in enumerate(batches):# --- OPTIONAL RESET ---
                if reset_history:
                    history_sum.zero_()
                    history_count.zero_()
                arena_indices = np.concatenate([arena_indices, new_indices])
                arena_indices = np.unique(arena_indices.astype(int))
                
                # TRAIN: Pass the 'train_batch_size'
                train_arena_step(
                    model, optimizer, criterion, X, y, arena_indices, 
                    sub_epochs, device, history_sum, history_count, n_classes,
                    batch_size=train_batch_size # <--- PASSED HERE
                )
                
                # CULL
                arena_indices = cull_arena(
                    arena_indices, new_indices, 
                    history_sum, history_count, 
                    target_size=coreset_size, 
                    power=actual_train_params.get('lrdy_power', 0.4)
                )
                
                # EVALUATE
                acc = evaluate_and_log(
                    run, model, X_test_tensor, y_test_labels, start_time, 
                    cycle, batch_i, len(batches), len(arena_indices), ratio_N_to_M
                )
            
            print(f"Cycle {cycle+1}/{cycles} finished. Test Acc: {acc:.4f}")
            
        run.finish()