import numpy as np

##############################
# Core Logistic Weighting
##############################

def numerator_logistic_curve(c, k, c0):
    """
    Computes the numerator of the logistic curve.
    """
    return 1 / (1 + np.exp(-k * (c - c0)))

def compute_k(M, N, pi, k0=1.0, epsilon=1e-6):
    """
    Compute steepness parameter k based on sample size and dataset size.
    """
    ratio = np.sqrt(M / (N + epsilon))       
    k = k0 * ratio / (pi + epsilon)
    k = max(k, 1.0)                
    return k

def weighted_sample_from_confidence(confidence, sample_size, k=None, c0=0.5, k0=1.0, pi=0.2, epsilon=1e-6, **kwargs):
    """
    Sample indices from confidence values using a logistic weighting scheme.
    """
    trainset_size = len(confidence)
    if not k:
        k = compute_k(sample_size, trainset_size, pi=pi, k0=k0, epsilon=epsilon)
    weights = numerator_logistic_curve(confidence, k, c0)
    weights = np.clip(weights, epsilon, None)  

    # normalize weights into probabilities
    probs = weights / (weights.sum())
    probs = (weights - weights.min()) / (weights.max() - weights.min())
    probs = np.power(probs, 0.5)
    
    quantiles = np.linspace(0, 1, sample_size + 1)
    sample_indices = []
    all_indices = np.arange(confidence.shape[0])
    
    for i in range(sample_size):
        q_min = np.quantile(probs, quantiles[i])
        q_max = np.quantile(probs, quantiles[i+1])

        # Get indices within this quantile
        idxs = all_indices[(probs >= q_min) & (probs <= q_max)].copy()

        if len(idxs) > 0:
            chosen = np.random.choice(idxs, size=1, replace=False)
            sample_indices.append(int(chosen))

    return sample_indices

def sample_by_conf(confidence, sample_size, power=0.7, **kwargs):
    """
    Simple quantile-based confidence sampling.
    """
    sample_idxs = []
    quantiles = np.linspace(0, 1, sample_size + 1)
    quantiles = np.power(quantiles, power)
    all_indices = np.arange(confidence.shape[0])

    for i in range(sample_size):
        q_min = np.quantile(confidence, quantiles[i])
        q_max = np.quantile(confidence, quantiles[i+1])

        idxs = all_indices[(confidence >= q_min) & (confidence <= q_max)].copy()

        if len(idxs) > 0:
            chosen = np.random.choice(idxs, size=1, replace=False)
            sample_idxs.extend(list(chosen))

    return sample_idxs

##############################
# Active Learning Samplers
##############################

def sample_by_forgetting(forgetting_counts, M, **kwargs):
    """
    Sample based on forgetting events (Toneva et al., 2018).
    """
    if forgetting_counts.sum() == 0:
        probs = np.ones_like(forgetting_counts).astype(float)
    else:
        probs = forgetting_counts.astype(float)
    probs /= probs.sum()
    
    replace = M > (forgetting_counts > 0).sum()
    return np.random.choice(len(forgetting_counts), size=M, replace=replace, p=probs)

def sample_by_aflite(confidence, M, q=0.2, **kwargs):
    """
    AFLite sampling (LeBras et al., 2020).
    """
    n = len(confidence)
    indices = np.arange(n)

    while len(indices) > M:
        keep_n = max(M, int(len(indices) * (1 - q)))
        order = np.argsort(confidence[indices])
        indices = indices[order[:keep_n]]

    if len(indices) > M:
        indices = np.random.choice(indices, size=M, replace=False)

    return indices

def sample_by_uncertainty(confidence, M, strategy="least_confidence", **kwargs):
    """
    Active learning uncertainty sampling (Joshi et al., 2009).
    """
    if strategy == "least_confidence":
        scores = 1 - confidence 
    elif strategy == "margin":
        if confidence.ndim == 1:
            raise ValueError("Margin requires full probability distributions.")
        part = np.partition(-confidence, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        scores = 1 - (top1 - top2)
    elif strategy == "entropy":
        if confidence.ndim == 1:
            raise ValueError("Entropy requires full probability distributions.")
        eps = 1e-12
        scores = -(confidence * np.log(confidence + eps)).sum(axis=1)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    order = np.argsort(-scores)
    return order[:M]

def sample_by_greedy_k(X, M, **kwargs):
    """
    Greedy k-center sampling (Sener & Savarese, 2018).
    """
    n = X.shape[0]
    indices = []

    # Start with a random point
    first = np.random.randint(0, n)
    indices.append(first)

    # Precompute squared distances
    dist = np.sum((X - X[first]) ** 2, axis=1)

    for _ in range(1, M):
        new_idx = np.argmax(dist)
        indices.append(new_idx)
        new_dist = np.sum((X - X[new_idx]) ** 2, axis=1)
        dist = np.minimum(dist, new_dist)

    return np.array(indices)


##############################
# GOLD Select (Pattern Coverage)
##############################

def sample_by_pattern_coverage(class_z, m, class_id=None, W_c=None, threshold=0.01, **kwargs):
    """
    Selects cells to ensure coverage of active patterns, then fills budget by best score.
    
    Args:
        class_z: (n_class_samples, n_patterns) - Z matrix for this specific class.
        m: (int) - The budget (number of cells to select).
        class_id: (int) - The current class index.
        W_c: (n_patterns, n_classes) - The global Importance Weights matrix.
        threshold: (float) - Minimum weight to consider a pattern "Active".
        
    Returns:
        indices: List of selected indices relative to class_z.
    """
    if W_c is None or class_id is None:
        raise ValueError("W_c (weights) and class_id are required for pattern coverage sampling.")
    
    # Foolproof check: Ensure we got a Matrix, not a 1D score array
    if class_z.ndim != 2:
        raise ValueError(f"Pattern coverage sampling requires a 2D Z-matrix. Got shape {class_z.shape}.")

    n_samples, n_patterns = class_z.shape
    selected_indices = set()
    
    # 1. Identify "Active" Patterns for this Class
    weights = W_c[:, class_id]
    
    active_patterns_idx = np.where(weights > threshold)[0]
    
    # Sort active patterns by importance (Strongest first)
    if len(active_patterns_idx) > 0:
        sorted_patterns_by_importance = active_patterns_idx[np.argsort(weights[active_patterns_idx])[::-1]]
    else:
        sorted_patterns_by_importance = []
    
    # --- PHASE A: COVERAGE ---
    # Pick the best cell for each active pattern
    for p_idx in sorted_patterns_by_importance:
        if len(selected_indices) >= m:
            break
            
        # Find the cell with the MAX activation for this specific pattern
        best_cell_idx = np.argmax(class_z[:, p_idx])
        selected_indices.add(best_cell_idx)
        
    # --- PHASE B: FILLING ---
    # If we still have budget, fill with the cells having the highest Total GOLD Score
    if len(selected_indices) < m:
        remaining_budget = m - len(selected_indices)
        
        # Calculate Total GOLD Score for all cells
        # Score = Dot(Cell_Z, Class_Weights)
        total_scores = np.dot(class_z, weights)
        
        # Sort all cells by score (Desc)
        sorted_cells_by_score = np.argsort(total_scores)[::-1]
        
        for idx in sorted_cells_by_score:
            if remaining_budget <= 0:
                break
            
            if idx not in selected_indices:
                selected_indices.add(idx)
                remaining_budget -= 1
                
    # Return sorted list for deterministic behavior
    return sorted(list(selected_indices))


##############################
# Stratified Utilities
##############################

def compute_class_sample_sizes(y, M, min_per_class=5):
    """
    Compute sample sizes per class, proportional to their frequency in `y`.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if M < n_classes * min_per_class:
        raise ValueError(
            f"Cannot allocate {M} samples with at least {min_per_class} per class "
            f"(need at least {n_classes * min_per_class})."
        )

    # Step 1: assign the minimum per class
    sizes = np.full(n_classes, min_per_class, dtype=int)
    remaining = M - sizes.sum()

    # Step 2: distribute remaining proportionally
    freqs = counts / counts.sum()
    extra_sizes = np.floor(freqs * remaining).astype(int)

    sizes += extra_sizes
    remaining = M - sizes.sum()

    # Step 3: distribute any leftover
    fractions = (freqs * remaining + 1e-9)
    top_classes = np.argsort(-fractions)[:remaining]
    sizes[top_classes] += 1

    return sizes


def stratified_random_sample(y, M, min_per_class=3):
    """
    Stratified random sample of size M.
    """
    sizes = compute_class_sample_sizes(y, M, min_per_class=min_per_class)
    indices = []
    for c, m in enumerate(sizes):
        class_indices = np.where(y == c)[0]
        if m > 0:
            sampled = np.random.choice(class_indices, size=m, replace=(m > len(class_indices)))
            indices.append(sampled)
    return np.concatenate(indices)

def stratified_apply(y, data, M, min_per_class=3, sample_func=sample_by_conf, allow_oversample=False, **kwargs):
    """
    Generic stratified sampling wrapper.
    Updated to pass `class_id` to sample_func.
    """
    sizes = compute_class_sample_sizes(y, M, min_per_class=min_per_class)
    out = []

    for c, m in enumerate(sizes):
        if m <= 0: 
            continue
        class_idx = np.where(y == c)[0]
        if class_idx.size == 0:
            continue

        m_eff = m if allow_oversample else min(m, class_idx.size)
        class_data = None if data is None else data[class_idx]

        # --- UPDATED: Passing class_id=c ---
        local_sel = sample_func(class_data, m_eff, class_id=c, **kwargs)
        # -----------------------------------
        
        local_sel = np.asarray(local_sel, dtype=int)
        out.append(class_idx[local_sel])

        if allow_oversample and m_eff < m:
            extra = np.random.choice(class_idx, size=(m - m_eff), replace=True)
            out.append(extra)

    return np.concatenate(out) if out else np.array([], dtype=int)


# Backward-compatible alias: now works with any "values" (scores/probs/etc.)
def stratified_weighted_sample(y, values, M, min_per_class=3, sample_func=None, **kwargs_sample_func):
    """
    Stratified sampling using arbitrary per-example values, via `sample_func`.
    `values` can be confidence, forgetting counts, probabilities, or Z-Matrix.
    """
    if sample_func is None:
        raise ValueError("Provide a sample_func (e.g., sample_by_conf, sample_by_pattern_coverage, ...)")
    return stratified_apply(y, values, M, min_per_class=min_per_class,
                            sample_func=sample_func, **kwargs_sample_func)


# ---- Wrappers ----

def stratified_forgetting_sample(y, forgetting_counts, M, min_per_class=3):
    return stratified_apply(y, forgetting_counts, M, min_per_class, sample_by_forgetting)

def stratified_aflite_sample(y, confidence, M, min_per_class=3, q=0.2):
    return stratified_apply(y, confidence, M, min_per_class, sample_by_aflite, q=q)

def stratified_uncertainty_sample(y, probs_or_conf, M, min_per_class=3, strategy="least_confidence"):
    return stratified_apply(y, probs_or_conf, M, min_per_class, sample_by_uncertainty, strategy=strategy)

def stratified_greedyk_sample(y, X, M, min_per_class=3):
    return stratified_apply(y, X, M, min_per_class, sample_by_greedy_k)

def stratified_confident_sample(y, X, M, min_per_class=3):
    return stratified_apply(y, X, M, min_per_class, sample_by_conf)