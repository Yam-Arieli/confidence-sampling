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


def stratified_random_sample(y, M):
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
    sizes = compute_class_sample_sizes(y, M)
    indices = []
    for c, m in enumerate(sizes):
        class_indices = np.where(y == c)[0]
        if m > 0:
            sampled = np.random.choice(class_indices, size=m, replace=(m > len(class_indices)))
            indices.append(sampled)
    return np.concatenate(indices)


def stratified_weighted_sample(y, confidence, M, sample_func=sample_by_conf, **kwargs_sample_func):
    """
    Stratified weighted sampling from each class using confidence scores.

    Parameters
    ----------
    y : np.ndarray
        Array of class ids.
    confidence : np.ndarray
        Confidence values aligned with y.
    M : int
        Total number of samples.
    c0, k0, pi, epsilon : params for weighted_sample_from_confidence

    Returns
    -------
    indices : np.ndarray
        Indices of sampled observations.
    """
    sizes = compute_class_sample_sizes(y, M)
    indices = []
    for c, m in enumerate(sizes):
        if m == 0:
            continue
        class_indices = np.where(y == c)[0]
        class_conf = confidence[class_indices]
        sample_idxs = sample_func(class_conf, m, **kwargs_sample_func)
        indices.append(class_indices[sample_idxs])
    return np.concatenate(indices)