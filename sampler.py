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

def weighted_sample_from_confidence(confidence, sample_size, c0, k0=1.0, pi=0.2, epsilon=1e-6):
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
    k = compute_k(sample_size, trainset_size, pi=pi, k0=k0, epsilon=epsilon)
    weights = numerator_logistic_curve(confidence, k, c0)
    weights = np.clip(weights, epsilon, None)  # avoid exact 0

    # normalize weights into probabilities
    probs = weights / (weights.sum())

    # sample indices according to probs
    sample_indices = np.random.choice(
        np.arange(trainset_size),
        size=sample_size,
        replace=sample_size > trainset_size,
        p=probs
    )

    return sample_indices