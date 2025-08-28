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

def weighted_sample_from_confidence(confidence, sample_size, c0, k0=1.0, pi=0.2, epsilon=1e-6):
    k = compute_k(len(confidence), sample_size, pi=pi, k0=k0, epsilon=epsilon)
    weights = numerator_logistic_curve(confidence, k, c0)
    
    return sampler