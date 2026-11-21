"""Objective-1 Pearson/phase-null mathematics, in anatomical ROI coordinates.

No diagnosis or cohort data enter fc_spectrum/phase_threshold. A common training
cutoff is a separate fitted object and must be refit inside every inner split.
"""
import numpy as np
from scipy.linalg import eigh

N = 116
ARMS = ('full', 'mp', 'temporal', 'budget', 'fixed')

def correlation(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0, ddof=1)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(sd <= 1e-10):
        raise ValueError('Nonfinite or constant required ROI')
    z = (x - x.mean(axis=0)) / sd
    c = z.T @ z / (len(x) - 1)
    c = (c + c.T) / 2
    np.fill_diagonal(c, 1.)
    return c

def fc_spectrum(x):
    c = correlation(x)
    w, u = eigh(c, check_finite=False, driver='evr')
    return c, w[::-1].copy(), u[:, ::-1].copy()

def phase_threshold(x, random_seed, B=100):
    # Match rmt_corrected.py: independent phases per ROI, DC/Nyquist phases zero.
    # Keep the historical convention explicitly; no silent endpoint correction.
    rng = np.random.default_rng(random_seed)
    amp = np.abs(np.fft.rfft(x, axis=0))
    pool = []
    for _ in range(B):
        ph = rng.uniform(0, 2 * np.pi, size=amp.shape)
        ph[0] = 0
        if len(x) % 2 == 0:
            ph[-1] = 0
        xs = np.fft.irfft(amp * np.exp(1j * ph), n=len(x), axis=0)
        pool.append(eigh(correlation(xs), eigvals_only=True, check_finite=False, driver='evr'))
    return float(np.quantile(np.concatenate(pool), .99, method='linear'))

def reconstruct(w, u, mask):
    return (u * (w * mask)[None, :]) @ u.T

def fit_budget(w, corrected_ranks):
    """Nearest attainable pooled TRAINING mode count, including deterministic ties."""
    target = int(np.sum(corrected_ranks))
    v, counts = np.unique(np.asarray(w).ravel(), return_counts=True)
    # Candidate threshold in each gap; strict > defines inclusion.
    cuts = np.r_[np.nextafter(v[0], -np.inf), (v[:-1] + v[1:]) / 2, np.nextafter(v[-1], np.inf)]
    retained = len(w) * w.shape[1] - np.r_[0, np.cumsum(counts)]
    # Tied count error selects fewer modes, never uses diagnostic labels.
    ix = min(range(len(cuts)), key=lambda j: (abs(int(retained[j]) - target), int(retained[j])))
    return {'cutoff': float(cuts[ix]), 'target': target, 'attained': int(retained[ix]),
            'fixed_k': int(np.clip(np.floor(np.mean(corrected_ranks) + .5), 0, w.shape[1]))}

def make_operators(w, u, threshold, T, arm, budget=None):
    if arm == 'full': mask = np.ones_like(w, dtype=bool)
    elif arm == 'mp': mask = w > (1 + np.sqrt(w.shape[-1] / np.asarray(T)))[:, None] ** 2
    elif arm == 'temporal': mask = w > np.asarray(threshold)[:, None]
    elif arm == 'budget': mask = w > budget['cutoff']
    elif arm == 'fixed': mask = np.broadcast_to(np.arange(w.shape[-1]) < budget['fixed_k'], w.shape)
    else: raise ValueError(arm)
    # SAME scalar for every arm for a given participant/context; retains eigenspace.
    r = (u * (w * mask)[:, None, :]) @ u.transpose(0, 2, 1)
    s = r / np.maximum(w[:, 0], 1e-12)[:, None, None]
    return s.astype(np.float32), mask.sum(axis=1), r.astype(np.float32)

def vectorize(s):
    # Including diagonals matches the entire signed operator given to the GNN.
    return s[:, np.triu_indices(s.shape[1])[0], np.triu_indices(s.shape[1])[1]].astype(np.float64)
