import numpy as np

def numerator_logistic_curve(c, k, c0):
    """
    Computes the numerator of the logistic curve.

    Parameters:
    c (float or np.ndarray): The input value(s).
    k (float): The steepness of the curve.
    c0 (float): The x-value of the sigmoid's midpoint.

    Returns:
    float or np.ndarray: The computed numerator of the logistic curve.
    """
    return 1 / (1 + np.exp(-k * (c - c0)))

import numpy as np

def compute_k(M, N, pi, k0=1.0, epsilon=1e-6):
    """
    Compute a steepness parameter k for the weighting function f(c; k),
    based on sample size, dataset size, and prior fraction of generalizable cells.

    Parameters
    ----------
    M : int
        Subsample size.
    N : int
        Full training set size.
    pi : float
        Prior fraction of generalizable cells in [0,1].
    k0 : float, optional
        Base steepness for small sample sizes (default 5.0).
    epsilon : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    k : float
        Steepness parameter for f(c; k).
    """
    ratio = np.sqrt(M / (N + epsilon))       # sample fraction
    k = k0 * ratio / (pi + epsilon)
    k = max(k, 1.0)                # enforce a minimal steepness
    return k

import numpy as np

def weighted_sample_from_confidence(confidence, sample_size, k=None, c0=0.5, k0=1.0, pi=0.2, epsilon=1e-6):
    """
    Sample indices from confidence values using a logistic weighting scheme.

    Parameters
    ----------
    confidence : np.ndarray
        Array of confidence scores in [0,1].
    sample_size : int
        Number of samples to draw.
    c0 : float
        Midpoint of logistic curve.
    k0 : float, optional
        Base steepness parameter.
    pi : float, optional
        Prior fraction of generalizable cells.
    epsilon : float, optional
        Stability constant to avoid division by zero.

    Returns
    -------
    sample_indices : np.ndarray
        Indices of the sampled elements.
    """
    trainset_size = len(confidence)
    if not k:
        k = compute_k(sample_size, trainset_size, pi=pi, k0=k0, epsilon=epsilon)
    weights = numerator_logistic_curve(confidence, k, c0)
    weights = np.clip(weights, epsilon, None)  # avoid exact 0

    # normalize weights into probabilities
    probs = weights / (weights.sum())
    probs = (weights - weights.min()) / (weights.max() - weights.min())
    probs = np.power(probs, 0.5)
    # probs_cumsum = np.cumulative_sum(probs)
    # Quantile boundaries
    quantiles = np.linspace(0, 1, sample_size + 1)

    # Loop over quantile intervals
    sample_indices = []
    all_indices = np.arange(confidence.shape[0])
    for i in range(sample_size):
        q_min = np.quantile(probs, quantiles[i])
        q_max = np.quantile(probs, quantiles[i+1])
        print(f'{q_min=}, {q_max=}')

        # Get the indices of cells within this quantile range
        idxs = all_indices[(probs >= q_min) & (probs <= q_max)].copy()

        if len(idxs) > 0:
            # Sample one (or more) cell(s) from this bin
            chosen = np.random.choice(idxs, size=1, replace=False)
            sample_indices.append(int(chosen))

    return sample_indices

def sample_by_conf(confidence, sample_size, power=0.7):
    sample_idxs = []
    # Quantile boundaries
    quantiles = np.linspace(0, 1, sample_size + 1)
    quantiles = np.power(quantiles, power)
    all_indices = np.arange(confidence.shape[0])

    # Loop over quantile intervals
    for i in range(sample_size):
        q_min = np.quantile(confidence, quantiles[i])
        q_max = np.quantile(confidence, quantiles[i+1])

        # Get the indices of cells within this quantile range
        idxs = all_indices[(confidence >= q_min) & (confidence <= q_max)].copy()

        if len(idxs) > 0:
            # Sample one (or more) cell(s) from this bin
            chosen = np.random.choice(idxs, size=1, replace=False)
            sample_idxs.extend(list(chosen))

    return sample_idxs

##############################
# Active Learning Samplers
##############################

def sample_by_forgetting(forgetting_counts, M):
    """
    Sample based on forgetting events (Toneva et al., 2018).

    Parameters
    ----------
    forgetting_counts : np.ndarray
        Number of forgetting events per example (higher = more often forgotten).
    M : int
        Number of samples to draw.

    Returns
    -------
    indices : np.ndarray
        Indices of selected samples.
    """
    # if all zeros, fallback to uniform
    if forgetting_counts.sum() == 0:
        probs = np.ones_like(forgetting_counts).astype(float)
    else:
        probs = forgetting_counts.astype(float)
    probs /= probs.sum()
    
    replace = M > (forgetting_counts > 0).sum()  # True if not enough non-zero entries
    return np.random.choice(len(forgetting_counts), size=M, replace=replace, p=probs)

def sample_by_aflite(confidence, M, q=0.2):
    """
    AFLite sampling (LeBras et al., 2020).
    Iteratively removes examples with high confidence until only a fraction remains.

    Parameters
    ----------
    confidence : np.ndarray
        Confidence scores (e.g., model probability on predicted class).
    M : int
        Number of samples to draw.
    q : float
        Fraction to prune at each iteration (default=0.2).

    Returns
    -------
    indices : np.ndarray
        Indices of selected samples.
    """
    n = len(confidence)
    indices = np.arange(n)

    # Iteratively prune most confident
    while len(indices) > M:
        keep_n = max(M, int(len(indices) * (1 - q)))
        order = np.argsort(confidence[indices])  # ascending (low conf = kept)
        indices = indices[order[:keep_n]]

    # If more than M left, randomly downsample
    if len(indices) > M:
        indices = np.random.choice(indices, size=M, replace=False)

    return indices


def sample_by_uncertainty(confidence, M, strategy="least_confidence"):
    """
    Active learning uncertainty sampling (Joshi et al., 2009).

    Parameters
    ----------
    confidence : np.ndarray
        Confidence values (prob of predicted class).
    M : int
        Number of samples to draw.
    strategy : {"least_confidence", "margin", "entropy"}
        Which uncertainty metric to use.

    Returns
    -------
    indices : np.ndarray
        Indices of selected samples.
    """
    if strategy == "least_confidence":
        scores = 1 - confidence  # lower confidence = more uncertain
    elif strategy == "margin":
        # If given class probs: confidence = 2D array (n, n_classes)
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

    order = np.argsort(-scores)  # descending: most uncertain first
    return order[:M]


def sample_by_greedy_k(X, M):
    """
    Greedy k-center sampling (Sener & Savarese, 2018).
    Selects a diverse set of points that maximize coverage.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, d).
    M : int
        Number of samples to select.

    Returns
    -------
    indices : np.ndarray
        Indices of selected samples.
    """
    n = X.shape[0]
    indices = []

    # Start with a random point
    first = np.random.randint(0, n)
    indices.append(first)

    # Precompute squared distances
    dist = np.sum((X - X[first]) ** 2, axis=1)

    for _ in range(1, M):
        # pick farthest point from current set
        new_idx = np.argmax(dist)
        indices.append(new_idx)

        # update distances: min dist to any selected center
        new_dist = np.sum((X - X[new_idx]) ** 2, axis=1)
        dist = np.minimum(dist, new_dist)

    return np.array(indices)

def compute_class_sample_sizes(y, M, min_per_class=5):
    """
    Compute sample sizes per class, proportional to their frequency in `y`,
    while enforcing a minimum per-class sample size.

    Parameters
    ----------
    y : np.ndarray
        Array of class IDs (integers from 0..n_classes-1).
    M : int
        Total number of samples to draw.
    min_per_class : int, optional (default=1)
        Minimum number of samples to allocate per class.

    Returns
    -------
    sizes : np.ndarray
        Array of length n_classes with number of samples to draw from each class.
        Sums to exactly M.
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

    # Step 3: distribute any leftover (due to rounding) to the largest fractional parts
    fractions = (freqs * remaining + 1e-9)  # avoid ties with small noise
    top_classes = np.argsort(-fractions)[:remaining]
    sizes[top_classes] += 1

    return sizes


def stratified_random_sample(y, M, min_per_class=3):
    """
    Stratified random sample of size M.

    Parameters
    ----------
    y : np.ndarray
        Array of class ids.
    M : int
        Total number of samples to draw.

    Returns
    -------
    indices : np.ndarray
        Indices of sampled observations.
    """
    sizes = compute_class_sample_sizes(y, M, min_per_class=min_per_class)
    indices = []
    for c, m in enumerate(sizes):
        class_indices = np.where(y == c)[0]
        if m > 0:
            sampled = np.random.choice(class_indices, size=m, replace=(m > len(class_indices)))
            indices.append(sampled)
    return np.concatenate(indices)


# def stratified_weighted_sample(y, confidence, M, min_per_class=3, sample_func=sample_by_conf, **kwargs_sample_func):
#     """
#     Stratified weighted sampling from each class using confidence scores.

#     Parameters
#     ----------
#     y : np.ndarray
#         Array of class ids.
#     confidence : np.ndarray
#         Confidence values aligned with y.
#     M : int
#         Total number of samples.
#     c0, k0, pi, epsilon : params for weighted_sample_from_confidence

#     Returns
#     -------
#     indices : np.ndarray
#         Indices of sampled observations.
#     """
#     sizes = compute_class_sample_sizes(y, M, min_per_class=min_per_class)
#     indices = []
#     for c, m in enumerate(sizes):
#         if m == 0:
#             continue
#         class_indices = np.where(y == c)[0]
#         class_conf = confidence[class_indices]
#         sample_idxs = sample_func(class_conf, m, **kwargs_sample_func)
#         indices.append(class_indices[sample_idxs])
#     return np.concatenate(indices)

def stratified_apply(y, data, M, min_per_class=3, sample_func=sample_by_conf, allow_oversample=False, **kwargs):
    """
    Generic stratified sampling wrapper.

    Parameters
    ----------
    y : np.ndarray
        Class ids (0..n_classes-1).
    data : np.ndarray or None
        Per-example array sliced per-class and passed to `sample_func`.
        Examples:
          - 1D scores (forgetting counts, confidence)
          - 2D class probabilities (for margin/entropy)
          - 2D features (for k-center)
          - None if `sample_func` doesn't need per-example data
    M : int
        Total number to draw (sum over classes).
    min_per_class : int
        Minimum per class.
    sample_func : callable
        Must accept (class_data, m, **kwargs) and return indices *within* class_data.
    allow_oversample : bool
        If True and m > class size, tops up by random sampling with replacement.

    Returns
    -------
    np.ndarray of global indices.
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

        local_sel = sample_func(class_data, m_eff, **kwargs)
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
    `values` can be confidence, forgetting counts, probabilities, etc.
    """
    if sample_func is None:
        raise ValueError("Provide a sample_func (e.g., sample_by_conf, sample_by_aflite, ...)")
    return stratified_apply(y, values, M, min_per_class=min_per_class,
                            sample_func=sample_func, **kwargs_sample_func)


# ---- Your specific stratified helpers become thin wrappers ----

def stratified_forgetting_sample(y, forgetting_counts, M, min_per_class=3):
    return stratified_apply(y, forgetting_counts, M, min_per_class, sample_by_forgetting)

def stratified_aflite_sample(y, confidence, M, min_per_class=3, q=0.2):
    return stratified_apply(y, confidence, M, min_per_class, sample_by_aflite, q=q)

def stratified_uncertainty_sample(y, probs_or_conf, M, min_per_class=3, strategy="least_confidence"):
    # Accepts either 1D confidences (least_confidence) or 2D class-probabilities (margin/entropy)
    return stratified_apply(y, probs_or_conf, M, min_per_class, sample_by_uncertainty, strategy=strategy)

def stratified_greedyk_sample(y, X, M, min_per_class=3):
    return stratified_apply(y, X, M, min_per_class, sample_by_greedy_k)

def stratified_confident_sample(y, X, M, min_per_class=3):
    return stratified_apply(y, X, M, min_per_class, sample_by_conf)