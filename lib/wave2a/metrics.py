"""Statistical metrics used across wave 2a experiments (§3.4–3.8, §4.1).

All rule-based, deterministic. seed=42 for any bootstrap.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except ImportError:  # pragma: no cover
    _HAS_SK = False


# ---------------------------------------------------------------------------
# §3.4 stress extractor
# ---------------------------------------------------------------------------

def compute_sigma(U: float, K: float, R: float, V: float, B: float,
                  weights=(1.0, 1.0, 1.0, 1.0, 1.0)) -> float:
    """σ = Σ w_j * component (default weights all 1)."""
    w1, w2, w3, w4, w5 = weights
    return float(w1 * U + w2 * K + w3 * R + w4 * V + w5 * B)


def sigma_series_from_z(z_list: list[dict]) -> np.ndarray:
    """Extract σ_{i,t} from list of z-tuples (extractor output).

    Uses:
      U = uncertainty.unresolved_gold_slots  (unresolved uncertainty proxy)
      K = len(constraints.violated)          (known contradictions)
      R = 0                                   (no retrieval memory conflicts tracked; extend per env)
      V = risk.unverified_writes             (unverified assumption debt)
      B = 0                                   (no tool/env failure debt in StatefulPuzzle; extend)
    """
    out = np.zeros(len(z_list), dtype=float)
    for t, z in enumerate(z_list):
        U = float(z["u"].get("unresolved_gold_slots", 0))
        K = float(len(z["c"].get("violated", [])))
        R = 0.0
        V = float(z["r"].get("unverified_writes", 0))
        B = 0.0
        out[t] = compute_sigma(U, K, R, V, B)
    return out


# ---------------------------------------------------------------------------
# §3.5 avalanche detector
# ---------------------------------------------------------------------------

def detect_avalanches(e_series: np.ndarray, tau_e: float = 0.5,
                      window_w: int = 2) -> dict:
    """Return A, A^w, D, peak_error, release_speed for the whole trajectory
    (aggregated across all avalanche episodes)."""
    e = np.asarray(e_series, dtype=float)
    H = len(e)
    above = e > tau_e
    A = int(above.sum())
    A_w = float(e[above].sum()) if A > 0 else 0.0

    # avalanche episodes: sequence broken when w consecutive steps below
    episodes = []
    in_ep = False
    start = None
    low_run = 0
    for t in range(H):
        if above[t]:
            if not in_ep:
                start = t
                in_ep = True
            low_run = 0
        else:
            if in_ep:
                low_run += 1
                if low_run >= window_w:
                    episodes.append((start, t - low_run))
                    in_ep = False
                    low_run = 0
    if in_ep:
        episodes.append((start, H - 1))

    D_max = max(((b - a + 1) for a, b in episodes), default=0)
    peak = float(e.max()) if H > 0 else 0.0
    # release speed = max step-to-step decrease
    if H >= 2:
        rel = float(np.max(-np.diff(e)))
    else:
        rel = 0.0
    return {
        "A": A, "A_w": A_w, "D_max": D_max,
        "n_episodes": len(episodes),
        "peak_error": peak,
        "release_speed": rel,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# §3.7 spectral analysis (Welch periodogram) + DFA + power-law fit
# ---------------------------------------------------------------------------

def spectral_slope(x: np.ndarray, nperseg: int | None = None) -> dict:
    """Fit S(f) ~ 1/f^alpha via log-log linear regression on Welch PSD."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    if N < 8 or np.allclose(x, 0):
        return {"alpha": None, "r2": None, "n_freq": 0, "n_samples": N}
    from scipy.signal import welch
    if nperseg is None:
        nperseg = min(N, max(16, N // 4))
    f, P = welch(x, fs=1.0, nperseg=nperseg, scaling="density")
    mask = (f > 0) & (P > 0)
    if mask.sum() < 4:
        return {"alpha": None, "r2": None, "n_freq": int(mask.sum()), "n_samples": N}
    lf = np.log(f[mask]); lP = np.log(P[mask])
    slope, intercept, r, _, _ = stats.linregress(lf, lP)
    return {"alpha": float(-slope), "r2": float(r * r),
            "n_freq": int(mask.sum()), "n_samples": N,
            "intercept": float(intercept)}


def dfa_exponent(x: np.ndarray, min_win: int = 4, max_win: int | None = None) -> dict:
    """Detrended Fluctuation Analysis (DFA). Returns Hurst-like exponent H_dfa."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 16 or np.allclose(x, x.mean()):
        return {"H_dfa": None, "n_windows": 0}
    y = np.cumsum(x - x.mean())
    if max_win is None:
        max_win = N // 4
    windows = np.unique(np.round(np.logspace(np.log10(min_win), np.log10(max_win), 12)).astype(int))
    windows = windows[windows >= 4]
    Fs = []
    for w in windows:
        n_seg = N // w
        if n_seg < 2:
            continue
        rms_vals = []
        for i in range(n_seg):
            seg = y[i * w:(i + 1) * w]
            t_seg = np.arange(w)
            p = np.polyfit(t_seg, seg, 1)
            detr = seg - np.polyval(p, t_seg)
            rms_vals.append(np.sqrt(np.mean(detr ** 2)))
        if rms_vals:
            Fs.append((w, float(np.mean(rms_vals))))
    if len(Fs) < 3:
        return {"H_dfa": None, "n_windows": len(Fs)}
    ws = np.array([w for w, _ in Fs])
    fs = np.array([f for _, f in Fs])
    mask = fs > 0
    if mask.sum() < 3:
        return {"H_dfa": None, "n_windows": int(mask.sum())}
    slope, _, r, _, _ = stats.linregress(np.log(ws[mask]), np.log(fs[mask]))
    return {"H_dfa": float(slope), "r2": float(r * r), "n_windows": int(mask.sum())}


def shuffled_baseline(x: np.ndarray, seed: int = 42, n_shuffle: int = 20) -> dict:
    """Compute spectral slope on n_shuffle shuffles; return mean +- std."""
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(n_shuffle):
        y = rng.permutation(x)
        r = spectral_slope(y)
        if r["alpha"] is not None:
            alphas.append(r["alpha"])
    if not alphas:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(alphas)), "std": float(np.std(alphas)), "n": len(alphas)}


def bootstrap_alpha_vs_shuffled(x: np.ndarray, seed: int = 42,
                                n_boot: int = 200) -> dict:
    """Bootstrap p-value: fraction of shuffled draws whose alpha >= observed.

    Under H0 (no long memory) alpha_shuffled distribution centers near 0.
    Small p (~< 0.01) => reject H0 => evidence of 1/f-like memory.
    """
    obs = spectral_slope(x)
    if obs["alpha"] is None:
        return {"observed_alpha": None, "p_value": None}
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_boot):
        y = rng.permutation(x)
        r = spectral_slope(y)
        if r["alpha"] is not None and r["alpha"] >= obs["alpha"]:
            ge += 1
    p = (ge + 1) / (n_boot + 1)
    return {"observed_alpha": obs["alpha"], "n_boot": n_boot,
            "p_value": float(p)}


# ---------------------------------------------------------------------------
# Power-law vs exponential fit (weak chaos §3.7)
# ---------------------------------------------------------------------------

def fit_power_and_exp(t: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = c t^beta AND y = c exp(lambda t); compare via AIC/BIC.

    Uses only t>=1, y>0 samples. Returns None-fields if not enough data.
    """
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    mask = (t > 0) & (y > 0)
    n = int(mask.sum())
    if n < 4:
        return {"n": n, "power": None, "exp": None, "best": None}
    lt = np.log(t[mask]); ly = np.log(y[mask])
    # power-law: log y = log c + beta * log t
    p_slope, p_int, p_r, _, _ = stats.linregress(lt, ly)
    resid_p = ly - (p_int + p_slope * lt)
    ss_p = float(np.sum(resid_p ** 2))
    # exponential: log y = log c + lambda * t
    e_slope, e_int, e_r, _, _ = stats.linregress(t[mask], ly)
    resid_e = ly - (e_int + e_slope * t[mask])
    ss_e = float(np.sum(resid_e ** 2))
    # 2-parameter Gaussian residual AIC (up to constant)
    k = 2
    aic_p = n * np.log(ss_p / n + 1e-30) + 2 * k
    aic_e = n * np.log(ss_e / n + 1e-30) + 2 * k
    bic_p = n * np.log(ss_p / n + 1e-30) + k * np.log(n)
    bic_e = n * np.log(ss_e / n + 1e-30) + k * np.log(n)
    best = "power" if aic_p < aic_e else "exp"
    return {
        "n": n,
        "power": {"beta": float(p_slope), "log_c": float(p_int),
                  "r2": float(p_r * p_r), "aic": float(aic_p), "bic": float(bic_p)},
        "exp":   {"lambda": float(e_slope), "log_c": float(e_int),
                  "r2": float(e_r * e_r), "aic": float(aic_e), "bic": float(bic_e)},
        "best": best,
        "delta_aic_exp_minus_power": float(aic_e - aic_p),
    }


# ---------------------------------------------------------------------------
# §3.8 fractal geometry: box-counting on 2D grid or 1D binary set
# ---------------------------------------------------------------------------

def box_counting_2d(mask: np.ndarray, scales: list[int] | None = None) -> dict:
    """Compute box-counting dimension D_f on a 2D bool mask."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    L = min(H, W)
    if L < 4 or not mask.any():
        return {"D_f": None, "r2": None, "n_scales": 0}
    if scales is None:
        # Powers of 2 up to L/2
        s = 1
        scales = []
        while s <= L // 2:
            scales.append(s)
            s *= 2
        if len(scales) < 3:
            scales = [1, 2, 4]
    Ns, Rs = [], []
    for r in scales:
        H_r = H // r
        W_r = W // r
        if H_r < 1 or W_r < 1:
            continue
        reshaped = mask[:H_r * r, :W_r * r].reshape(H_r, r, W_r, r).any(axis=(1, 3))
        n = int(reshaped.sum())
        if n > 0:
            Ns.append(n)
            Rs.append(r)
    if len(Ns) < 3:
        return {"D_f": None, "r2": None, "n_scales": len(Ns)}
    lr = np.log(np.array(Rs, dtype=float))
    lN = np.log(np.array(Ns, dtype=float))
    slope, _, r, _, _ = stats.linregress(lr, lN)
    return {"D_f": float(-slope), "r2": float(r * r), "n_scales": len(Ns)}


# ---------------------------------------------------------------------------
# AUROC (stress predicts collapse / avalanche size)
# ---------------------------------------------------------------------------

def auroc(y_true, y_score) -> float | None:
    y_true = np.asarray(y_true); y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    if _HAS_SK:
        return float(roc_auc_score(y_true, y_score))
    # Manual Mann-Whitney U based AUROC
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    U, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(U / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Distribution distance (Kolmogorov-Smirnov, Wasserstein-1)
# ---------------------------------------------------------------------------

def dist_distance(a, b) -> dict:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"ks": None, "wasserstein": None}
    ks, _ = stats.ks_2samp(a, b)
    w1 = stats.wasserstein_distance(a, b)
    return {"ks": float(ks), "wasserstein": float(w1)}


# ---------------------------------------------------------------------------
# Local error Jaccard overlap between two trajectories (same t)
# ---------------------------------------------------------------------------

def error_jaccard(errs_a: list[int], errs_b: list[int]) -> float:
    """Jaccard overlap of *timestep-level* error sets."""
    A = set(int(t) for t in errs_a)
    B = set(int(t) for t in errs_b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))
