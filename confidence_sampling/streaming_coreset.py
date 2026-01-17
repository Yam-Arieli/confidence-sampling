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
# MAIN EVOLUTIONARY LOOP
# ==============================================================================

def run_evolutionary_experiment(
    adata_train,
    adata_test,
    device,
    actual_train_params,
    iterations=1,
    cycles=5,      # How many times to loop through the Full dataset
    sub_epochs=3,  # How many epochs to train on the Arena per step
    ran_name='evolutionary_stream'
):
    """
    Wrapper that mimics 'run_sampling_experiment' but for the streaming method.
    It runs 'iterations' independent experiments.
    """
    
    # 1. PREPARE DATA ONCE
    # We keep X, y on CPU mainly, but convert to Tensor for easier slicing
    X_np = adata_train.X if isinstance(adata_train.X, np.ndarray) else adata_train.X.toarray()
    y_np = adata_train.obs['y'].values
    
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).long()
    
    # Prepare Test Data (on GPU for fast evaluation)
    X_test_tensor, _, _ = prepare_train_tensors(adata_test, device)
    y_test_labels = adata_test.obs["y"].astype("category").cat.codes.values
    
    n_genes = X.shape[1]
    n_classes = int(y.max().item() + 1)
    
    # Params
    batch_size = actual_train_params['batch_size'] # Used for the "Fresh Batch" size
    coreset_size = int(len(y) * actual_train_params['sample_frac'])
    
    # 2. ITERATION LOOP
    for iteration in range(iterations):
        print(f"\n--- Evolutionary Iteration {iteration+1}/{iterations} ---")
        
        # Initialize W&B Run
        run = group_run_generator.__next__()
        run.config.update(actual_train_params)
        run.config['method'] = ran_name
        run.name = f"{ran_name}_iter{iteration}"
        
        # Initialize Model & Optimizer
        model = ComplexNet(
            n_genes, 
            actual_train_params['hidden_dim'], 
            actual_train_params.get('min_hidden_dim', 128), 
            n_classes, 
            layers_num=actual_train_params['layers_num'], 
            dropout_p=actual_train_params.get('dropout_p', 0.1)
        ).to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=actual_train_params['lr'], momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler (Optional - wrapping SGD)
        # Note: Schedulers in streaming are tricky. We apply step() once per Cycle or once per Batch.
        # Here we apply per Cycle to emulate "Epochs".
        scheduler = None
        # If you passed a scheduler factory (lambda) in main script:
        # if 'scheduler_fn' in actual_train_params: ... (needs refactoring to pass fn)
        
        # State
        arena_indices = np.array([], dtype=int)
        history_sum = torch.zeros(len(y), device=device)
        history_count = torch.zeros(len(y), device=device)
        
        cumulative_time = 0.0
        global_step = 0
        
        # Batches (Reshuffle every cycle or compute once? 
        # For true robustness, compute new stratified batches every cycle.)
        
        start_time = datetime.now()
        
        # 3. CYCLES LOOP
        for cycle in range(cycles):
            # Create fresh batches for this cycle
            batches = create_stratified_batches(y_np, batch_size, min_per_class=actual_train_params.get('min_per_class', 3))
            
            for batch_i, new_indices in enumerate(batches):
                
                # A. Merge
                arena_indices = np.concatenate([arena_indices, new_indices])
                arena_indices = np.unique(arena_indices.astype(int))
                
                # B. Train Arena (Sub-Epochs)
                # Create DataLoader for Arena
                X_arena = X[arena_indices].to(device)
                y_arena = y[arena_indices].to(device)
                y_arena_oh = F.one_hot(y_arena, num_classes=n_classes).float()
                
                # Pass global indices to track history
                ds = TensorDataset(X_arena, y_arena_oh, torch.tensor(arena_indices).to(device))
                dl = DataLoader(ds, batch_size=256, shuffle=True)
                
                model.train()
                for _ in range(sub_epochs):
                    for bx, by, global_idx in dl:
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by.argmax(dim=1))
                        loss.backward()
                        optimizer.step()
                        
                        # Track Confidence
                        probs = out.exp().detach()
                        true_probs = (probs * by).sum(dim=1)
                        history_sum.index_add_(0, global_idx, true_probs)
                        history_count.index_add_(0, global_idx, torch.ones_like(true_probs))

                # C. Cull
                arena_indices = cull_arena(
                    arena_indices, new_indices, 
                    history_sum, history_count, 
                    target_size=coreset_size, 
                    power=actual_train_params.get('lrdy_power', 0.4)
                )
                
                # D. Evaluate & Log (User Request: "Eval after each epoch")
                # In streaming, we treat (Batch Step) as a mini-epoch or check-point.
                
                global_step += 1
                
                # We log at every step or every X steps to keep W&B clean?
                # Let's log every step to be granular.
                with torch.no_grad():
                    test_probs = torch.softmax(model(X_test_tensor), dim=1)
                    
                cur_time = (datetime.now() - start_time).total_seconds()
                
                acc, _, _, f1_macro, f1_weighted = eval_result(test_probs, y_test_labels, do_print=False)
                
                # We log 'epoch' as floating point (Cycle + Fraction)
                frac_epoch = cycle + (batch_i / len(batches))
                
                run.log({
                    "acc": acc, 
                    "f1_macro": f1_macro, 
                    "f1_weighted": f1_weighted,
                    "epoch": frac_epoch,
                    "cycle": cycle,
                    "arena_size": len(arena_indices),
                    "cumulative_time": int(cur_time*10)/10
                })
            
            # End of Cycle
            print(f"Cycle {cycle+1}/{cycles} finished. Test Acc: {acc:.4f}")
            
        run.finish()