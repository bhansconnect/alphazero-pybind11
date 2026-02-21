"""Shared policy divergence and agreement metrics.

Per-sample functions (numpy arrays) are used by mcts_analysis.py.
Batched variants (vectorized over axis=0) are used by network_pareto.py.
"""

import numpy as np


# --- Per-sample functions (1-D arrays) ---

def jensen_shannon_divergence(p, q):
    """JSD(p, q). Bounded [0, ln(2)]."""
    m = 0.5 * (p + q)
    jsd = 0.0
    mask_p = p > 0
    jsd += 0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    mask_q = q > 0
    jsd += 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    return float(jsd)


def total_variation(p, q):
    """TV(p, q). Bounded [0, 1]."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def hellinger_distance(p, q):
    """Hellinger distance. Bounded [0, 1]."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)))


def top_k_agreement(p, q, k):
    """Fraction of top-k moves by p that appear in top-k of q."""
    top_p = set(np.argsort(p)[-k:])
    top_q = set(np.argsort(q)[-k:])
    return len(top_p & top_q) / k


def kl_divergence(p, q, epsilon=1e-10):
    """KL(p || q). Skip terms where p=0. Epsilon-smooth q to avoid log(0)."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + epsilon))))


def policy_entropy(p, epsilon=1e-10):
    """Shannon entropy H(p) = -sum(p * log(p)). Skip zeros."""
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


# --- Batched functions (2-D arrays, axis=0 = samples) ---

def batch_top1_agreement(p, q):
    """Mean top-1 agreement: fraction of samples where argmax matches."""
    return float(np.mean(np.argmax(p, axis=1) == np.argmax(q, axis=1)))


def batch_top_k_agreement(p, q, k):
    """Mean top-k overlap fraction across samples."""
    # Get top-k indices for each row
    top_p = np.argpartition(p, -k, axis=1)[:, -k:]
    top_q = np.argpartition(q, -k, axis=1)[:, -k:]
    n = p.shape[0]
    agreements = np.empty(n)
    for i in range(n):
        agreements[i] = len(set(top_p[i]) & set(top_q[i])) / k
    return float(np.mean(agreements))


def batch_kl_divergence(p, q, epsilon=1e-10):
    """Mean KL(p || q) across samples. Epsilon-smooth q."""
    # Mask where p > 0, compute element-wise KL, sum per sample
    safe_q = q + epsilon
    kl = np.where(p > 0, p * np.log(p / safe_q), 0.0)
    return float(np.mean(np.sum(kl, axis=1)))


def batch_policy_entropy(p, epsilon=1e-10):
    """Mean Shannon entropy across samples."""
    ent = np.where(p > 0, -p * np.log(p), 0.0)
    return float(np.mean(np.sum(ent, axis=1)))


def batch_total_variation(p, q):
    """Mean total variation distance across samples."""
    return float(np.mean(0.5 * np.sum(np.abs(p - q), axis=1)))
