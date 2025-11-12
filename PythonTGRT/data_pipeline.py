"""
Data pipeline utilities for building a sliding-window EMG-IMU dataset from Unity-exported logs,
including robust loading, normalization, SNR-based lag estimation helpers, and a PyTorch Dataset.

Constants
---------
NUM_ELEC : int
    Number of EMG electrodes (default 12).
SNR_REF : {"p95","max"}
    Reference used to derive per-channel SNR thresholds.
THR_PCT : float
    Fraction of reference used as threshold per channel (0-1).
AGGR_MODE : {"kofn","any"}
    Channel aggregation rule for activity detection.
AGGR_K : int
    k for the k-of-N rule.
SNR_WIN : int
    RMS window length (samples) used for SNR computation.
SR : int
    Default sampling rate (Hz) used by helpers.
BELOW_MS : int
    Minimum below-threshold duration to mark an "end" (ms).
BRIDGE_MS : int
    Bridge gaps shorter than this in activity mask (ms).

Functions
---------
_parse_csv_numbers(s: str) -> List[float]
    Parse a comma/semicolon separated string into floats, skipping non-numerics.
    Parameters: s — input string.

_pad_or_trim(row: List[float], C: int) -> List[float]
    Ensure a vector has exactly C elements by truncation or zero-padding.
    Parameters: row — values; C — target length.

_load_log_strict(path: str, emg_channels: int = 12, imu_channels: int = 6)
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    Load a Unity log in the form
    ts|label|e1,...,eC | i1,...,i6  (or combined signal field).
    Returns: (ts_us [N], labels [N], emg [N, Ce], imu [N, Ci]).
    Parameters: path — file path; emg_channels; imu_channels.

_running_rms(sig: np.ndarray, win: int) -> np.ndarray
    Per-channel running RMS; returns shape like signal (sig).
    Parameters: sig — (N, C) array; win — window length (samples).

_estimate_lag_full_dataset_snr(
    snr_db: np.ndarray, thr_db_per_ch: np.ndarray, y: np.ndarray, sr: int,
    max_lag_ms: int = 1500, seg_s: float = 5.0, hop_s: float = 2.5
) -> Tuple[int, float, dict]
    Estimate label-to-signal lag by cross-correlating an SNR-based activity mask
    with (y == 3). Returns (shift_samples, shift_ms, info).

_bridge_mask_gaps(mask: np.ndarray, bridge_ms: int, sr: int) -> np.ndarray
    Fill False gaps shorter than bridge_ms between True runs in a boolean series.

_snr_activity_mask(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, sr: int) -> np.ndarray
    Create a boolean activity mask via per-sample channel thresholding,
    k-of-N aggregation, and short-gap bridging.

_over_mask(
    snr_db: np.ndarray, thr_db_per_ch: np.ndarray, mode: str = "kofn", k: int = 3
) -> np.ndarray
    Compute per-time boolean "over-threshold" with 'any' or 'kofn' aggregation.

_shift_labels(y: np.ndarray, shift_samples: int) -> np.ndarray
    Shift label sequence by a number of samples.

_norm_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int
    Normalized cross-correlation lag.

_crosscorr_lag(a_bin: np.ndarray, b_bin: np.ndarray, max_lag: int) -> int
    Wrapper over normalized cross-correlation lag for binary/float series.

rest_noise_rms_from_labels(
    y: np.ndarray,
    emg: np.ndarray,
    sr: int = 1000,
    rest_win: int = 256,
    rest_hop: int = 128,
    guard_ms: int = 400,
    method: str = "percentile",
    percentile: float = 10.0,
) -> np.ndarray
    Estimate per-channel rest-noise RMS using only clean rest (y==0) windows,
    excluding ±guard_ms around transitions. Falls back to global low-percentile
    RMS if insufficient clean rest is available. Returns (C,).

_compute_snr_db(emg: np.ndarray, noise_rms: np.ndarray, rms_win: int = SNR_WIN) -> np.ndarray
    Compute per-channel SNR in dB as running_RMS(emg) / noise_rms.

_per_channel_thresholds(snr_db: np.ndarray, pct: float, ref: str = "p95") -> np.ndarray
    Derive per-channel SNR(dB) thresholds as pct {95th-percentile | max}.

Classes
-------
class EMGIMUTextDataset(torch.utils.data.Dataset)
    Build a PyTorch dataset of overlapping windows with majority-vote labels and
    optional augmentations.

    __init__(
        path: str,
        window_size: int = 512,
        patch_len: int = 8,
        overlap: float = 0.5,
        snr_shift: int = 0,
        *,
        mu: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        save_stats_to: str | None = None,
        augment: bool = False
    )
        Steps:
        1) Load ts, y, EMG (NUM_ELEC), IMU(6) from Unity log.
        2) Z-score normalize features using provided or computed (mu, sigma).
           Optionally save stats to NPZ.
        3) Apply label shift using pre-computed SNR cross-corr estimate (snr_shift).
        4) Extract windows of length window_size with stride determined by overlap
           (via patch_len granularity).
        5) Assign a window label via majority vote over non-zero labels, mapping
           1…G → 0…G-1; discard pure-rest windows.
        Attributes: X (N, W, C_total), y (N,), labels alias, mu, sigma, augment.

    __len__() -> int
        Number of windows.

    __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]
        Returns (window_tensor, label). If augment=True, applies Gaussian noise
        and random amplitude scaling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import correlate
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
NUM_ELEC = 12 
# --- SNR detection settings ---
SNR_REF = "p95"        # 'p95' or 'max' for per-channel reference
THR_PCT = 0.80     # fraction of reference to define channel thresholds
AGGR_MODE = "kofn"     # 'k-of-n' or 'any'
AGGR_K    = 3          # k for k-of-N (e.g., 3 of 12 channels)
SNR_WIN   = 64         # RMS window (samples). Lower (e.g., 32) => quicker edges.
SR = 500          # assumed sampling rate (Hz); adjust if different

# --- SNR end/bridging settings ---
BELOW_MS  = 80    # how long mask must stay below thr to mark "end"
BRIDGE_MS = 150   # fill gaps below thr shorter than this (bridges small dips)

def _parse_csv_numbers(s: str) -> List[float]:
    """Parse the input file with the EMG and IMU coma- or semicolon-separated (Robust)."""
    toks = s.replace(';', ',').split(',')
    out: List[float] = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            continue
    return out


def _pad_or_trim(row: List[float], C: int) -> List[float]:
    if len(row) >= C:
        return row[:C]
    return row + [0.0] * (C - len(row))


def _load_log_strict(path: str,
                     emg_channels: int = 12,
                     imu_channels: int = 6
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (ts_us [N], labels [N], emg [N,Ce], imu [N,Ci]).
    Accepts lines: ts|label|e1,...,e12|i1,...,i6  OR ts|label|<all sigs>
    """
    ts_list: List[int] = []
    y_list : List[int] = []
    emg_rows: List[List[float]] = []
    imu_rows : List[List[float]] = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue

            # timestamp
            try:
                ts = int(parts[0])
            except ValueError:
                try:
                    ts = int(float(parts[0]))
                except Exception:
                    continue

            # label
            try:
                lab = int(parts[1])
            except ValueError:
                try:
                    lab = int(float(parts[1]))
                except Exception:
                    lab = 0
            if len(parts) == 4:
                emg = _parse_csv_numbers(parts[2])
                #print(emg)
                imu = _parse_csv_numbers(parts[3])
                emg = _pad_or_trim(emg, emg_channels)
            elif len(parts) == 3:
                sigs = _parse_csv_numbers(parts[2])
                if len(sigs) >= emg_channels + imu_channels:
                    emg = sigs[:emg_channels]
                    imu = sigs[emg_channels:emg_channels+imu_channels]
                else:
                    emg = _pad_or_trim(sigs[:emg_channels], emg_channels)
                    imu = _pad_or_trim(sigs[emg_channels:], imu_channels)
            else:
                emg = _parse_csv_numbers(parts[2])
                imu = _parse_csv_numbers('|'.join(parts[3:]))
                emg = _pad_or_trim(emg, emg_channels)
                imu = _pad_or_trim(imu, imu_channels)

            ts_list.append(ts)
            y_list.append(lab)
            emg_rows.append(emg)
            imu_rows.append(imu)

    ts = np.asarray(ts_list, dtype=np.int64)
    y  = np.asarray(y_list, dtype=np.int32)
    emg = np.asarray(emg_rows, dtype=np.float32)
    imu = np.asarray(imu_rows, dtype=np.float32)
    return ts, y, emg, imu

# --------------------------------------------------------------------------- #
#  Lag‑estimation helpers
# --------------------------------------------------------------------------- #


def _running_rms(sig: np.ndarray, win: int) -> np.ndarray:
    """Per-channel running RMS (same length, 'same' conv)."""
    win = max(1, int(win))
    sq = sig.astype(np.float64)**2
    ker = np.ones(win, dtype=np.float64) / float(win)
    out = np.empty_like(sq)
    for c in range(sq.shape[1]):
        out[:, c] = np.sqrt(np.convolve(sq[:, c], ker, mode="same"))
    return out.astype(np.float32)

def _estimate_lag_full_dataset_snr(snr_db: np.ndarray,
                                   thr_db_per_ch: np.ndarray,
                                   y: np.ndarray,
                                   sr: int,
                                   max_lag_ms: int = 1500,
                                   seg_s: float = 5.0,
                                   hop_s: float = 2.5):
    """
    Cross-correlate the full SNR activity mask vs. (y==3) to estimate lag.
    Returns:
      shift_samp (int), shift_ms (float), info (dict)
    Sign: +shift => labels are EARLY and must be shifted FORWARD (to the right).
    """
    # 1) SNR activity mask (k-of-N + gap bridging)
    m_snr = _snr_activity_mask(snr_db, thr_db_per_ch, sr).astype(np.float32)
    m_lbl = (y == 3).astype(np.float32)
    N = min(len(m_snr), len(m_lbl))
    a = m_snr[:N]
    b = m_lbl[:N]

    # 2) Global cross-correlation (like in data_pipeline)
    maxlag = int(round(max_lag_ms * sr / 1000.0))
    lag = _crosscorr_lag(a, b, max_lag=maxlag)  # +lag => b shift LEFT by lag
    shift_samp = -int(lag)                                # convert to our convention
    shift_ms   = shift_samp * 1000.0 / float(sr)

    info = {"xcorr_lag_samples": int(lag), "xcorr_shift_samples": shift_samp, "xcorr_shift_ms": shift_ms}


    return shift_samp, shift_ms, info

def _bridge_mask_gaps(mask: np.ndarray, bridge_ms: int, sr: int) -> np.ndarray:
    """Fill False gaps shorter than bridge_ms that sit between True runs."""
    m = mask.copy()
    N = m.size
    min_gap = max(1, int(round(bridge_ms * sr / 1000.0)))

    i = 0
    while i < N:
        # skip leading False
        while i < N and not m[i]:
            i += 1
        # True run start
        if i >= N: break
        j = i
        while j < N and m[j]:
            j += 1
        # now [i, j) is True, next is a False gap [j, k)
        k = j
        while k < N and not m[k]:
            k += 1
        gap_len = k - j
        # if the gap is short and there is another True run after it, bridge it
        if 0 < gap_len <= min_gap and k < N:
            m[j:k] = True
            i = k
        else:
            i = k
    return m

def _snr_activity_mask(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, sr: int) -> np.ndarray:
    """k-of-N over-threshold + gap bridging -> boolean activity mask (SNR-based)."""
    raw_mask = _over_mask(snr_db, thr_db_per_ch, mode=AGGR_MODE, k=AGGR_K)
    return _bridge_mask_gaps(raw_mask, BRIDGE_MS, sr)

def _over_mask(snr_db: np.ndarray, thr_db_per_ch: np.ndarray,
               mode: str = "kofn", k: int = 3) -> np.ndarray:
    """Boolean series: True when detection condition holds across channels."""
    over_ch = (snr_db >= thr_db_per_ch[None, :])
    if mode == "any":
        return over_ch.any(axis=1)
    elif mode == "kofn":
        return (over_ch.sum(axis=1) >= int(k))
    else:
        raise ValueError("AGGR_MODE must be 'any' or 'kofn'")

def _shift_labels(y: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift labels in time."""
    y = np.asarray(y).copy()
    s = int(shift_samples)
    out = np.zeros_like(y, dtype=y.dtype)
    if s > 0:
        out[s:] = y[:-s]
    elif s < 0:
        out[:s] = y[-s:]
    else:
        out[:] = y
    return out

def _norm_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """
    Normalized cross-correlation lag between a and b.
    Returns lag in samples. Positive lag => b should be shifted LEFT by lag.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    sa = a.std() + 1e-8; sb = b.std() + 1e-8
    a /= sa; b /= sb
    n = len(a)
    max_lag = min(max_lag, n - 2)
    best_lag, best_val = 0, -1e9
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            x = a[lag:]; y = b[:n-lag]
        else:
            x = a[:n+lag]; y = b[-lag:]
        if len(x) < 8:
            continue
        val = float(np.dot(x, y) / len(x))
        if val > best_val:
            best_val, best_lag = val, lag
    return best_lag

def _crosscorr_lag(a_bin: np.ndarray, b_bin: np.ndarray, max_lag: int) -> int:
    return _norm_xcorr_lag(a_bin, b_bin, max_lag=max_lag)


def rest_noise_rms_from_labels(
    y: np.ndarray,
    emg: np.ndarray,
    sr: int = 1000,
    rest_win: int = 256,
    rest_hop: int = 128,
    guard_ms: int = 400,
    method: str = "percentile",
    percentile: float = 10.0,
) -> np.ndarray:
    """
    Estimate rest noise per channel using only clean rest windows (y==0),
    excluding +/- guard_ms around label transitions. Returns shape (C,).
    """
    N, C = emg.shape
    y = np.asarray(y).astype(int)

    # 1) build a clean rest mask with guard around transitions
    rest_mask = (y == 0)
    guard = int(round(sr * guard_ms / 1000.0))
    # transitions where y changes between rest and non-rest
    trans = np.where(np.diff((y == 3).astype(int), prepend=(y[0] != 0)) != 0)[0]
    for t in trans:
        lo = max(0, t - guard)
        hi = min(N, t + guard + 1)
        rest_mask[lo:hi] = False

    if rest_mask.sum() < rest_win:
        # fallback: low-percentile of running RMS over *all* samples
        # (still robust for noise if percentile is small)
        sq = emg.astype(np.float64) ** 2
        ker = np.ones(rest_win, dtype=np.float64) / float(rest_win)
        rr = np.empty_like(sq, dtype=np.float64)
        for c in range(C):
            rr[:, c] = np.sqrt(np.convolve(sq[:, c], ker, mode="same"))
        q = 10.0 if method == "percentile" else 50.0
        noise = np.percentile(rr, q=q, axis=0) if method == "percentile" else np.median(rr, axis=0)
        return np.maximum(noise.astype(np.float32), 1e-6)

    # 2) compute windowed RMS with 'valid' alignment
    w = rest_win
    sq = emg.astype(np.float64) ** 2
    ker = np.ones(w, dtype=np.float64) / float(w)

    # RMS per channel, 'valid' -> shape (N-w+1, C)
    rr_valid = np.empty((N - w + 1, C), dtype=np.float64)
    for c in range(C):
        rr_valid[:, c] = np.sqrt(np.convolve(sq[:, c], ker, mode="valid"))

    # 3) keep only windows fully inside clean rest
    rest_occ_valid = np.convolve(rest_mask.astype(np.int32), np.ones(w, dtype=np.int32), mode="valid")
    valid_idx = np.where(rest_occ_valid == w)[0]
    if valid_idx.size == 0:
        # fall back to 'same' path above if no clean windows survive
        q = 10.0 if method == "percentile" else 50.0
        noise = np.percentile(rr_valid, q=q, axis=0) if method == "percentile" else np.median(rr_valid, axis=0)
        return np.maximum(noise.astype(np.float32), 1e-6)

    # apply hop to reduce correlation
    valid_idx = valid_idx[::max(1, rest_hop)]
    rms_stack = rr_valid[valid_idx, :]   # shape (K, C)

    # 4) aggregate across windows: low percentile (default 10th) or median
    if method == "percentile":
        noise = np.percentile(rms_stack, q=float(percentile), axis=0)
    else:
        noise = np.median(rms_stack, axis=0)

    return np.maximum(noise.astype(np.float32), 1e-6)

def _compute_snr_db(emg: np.ndarray, noise_rms: np.ndarray, rms_win: int = SNR_WIN) -> np.ndarray:
    rr = _running_rms(emg, rms_win).astype(np.float64)           # (N,C)
    den = np.maximum(noise_rms[None, :].astype(np.float64), 1e-9) # (1,C)
    snr_lin = np.maximum(rr, 1e-9) / den
    return (20.0 * np.log10(snr_lin)).astype(np.float32)

def _per_channel_thresholds(snr_db: np.ndarray, pct: float, ref: str = "p95") -> np.ndarray:
    """Return per-channel SNR(dB) thresholds using a robust reference (p95 or max)."""
    if ref == "p95":
        refvals = np.percentile(snr_db, 95.0, axis=0)
    elif ref == "max":
        refvals = np.max(snr_db, axis=0)
    else:
        raise ValueError("SNR_REF must be 'p95' or 'max'")
    return pct * refvals

# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class EMGIMUTextDataset(Dataset):
    """Sliding‑window dataset with automatic lag compensation & scaling.

    Parameters
    ----------
    path : str
        Path to the ``*.txt`` log exported by Unity.
    window_size : int, default 512
        Number of samples per example fed to the transformer.
    patch_len : int, default 8
        Patch length (must divide ``window_size``).
    overlap : float, default 0.98
        Fractional overlap between successive windows.
    sr : float, default 500.0
        Sampling rate of the recording (Hz).
    gesture_for_lag : int, default 3
        Gesture id used when estimating lag via PCA cross-correlation.
    mu, sigma : np.ndarray | None
        Optional **pre-computed** mu / sigma used for z-score.  If both are
        provided they are used *as-is*; otherwise they are computed from the
        current file.
    save_stats_to : str | None
        When computing stats, save them to ``*.npz`` for later reuse.
    """

    def __init__(
        self,
        path: str,
        window_size: int = 512,
        patch_len: int = 8,
        overlap: float = 0.5,
        snr_shift: int = 0,
        *,
        mu: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        save_stats_to: str | None = None,
        augment: bool = False
        
    ):
        super().__init__()

        assert window_size % patch_len == 0, (
            f"window_size ({window_size}) must be divisible by patch_len ({patch_len})"
        )

        # ------------------------------------------------------------------- #
        # 1) Load raw log ----------------------------------------------------
        ts, y, emg, imu = _load_log_strict(path, NUM_ELEC, 6)
        

        data = np.concatenate([np.stack(emg), np.stack(imu)], axis=1).astype(
            np.float32
        )
        labels = np.asarray(y, dtype=int)

        # ------------------------------------------------------------------- #
        # 2) Normalisation ---------------------------------------------------
        if (mu is not None) and (sigma is not None):
            # user provided stats → validate and reuse
            mu = np.asarray(mu, dtype=np.float32)
            sigma = np.asarray(sigma, dtype=np.float32)
            if mu.shape != data.shape[1:]:
                raise ValueError(
                    f"mu shape {mu.shape} ≠ feature dimension {data.shape[1:]}"
                )
            if sigma.shape != mu.shape:
                raise ValueError(
                    f"sigma shape {sigma.shape} ≠ mu shape {mu.shape}"
                )
        else:
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            if save_stats_to is not None:
                np.savez(save_stats_to, mu=mu, sigma=sigma)
        sigma[sigma == 0] = 1e-8  # robustify

        data = (data - mu) / sigma

        # ------------------------------------------------------------------- #
        # 3) Lag compensation -----------------------------------------------
        # estimate lag using SNR xcorr method, to save time this value was calculated once and then is given as input for the function.

        shift_samp = snr_shift
        
        dir_txt = ("forward (delay labels)" if shift_samp > 0
                   else "backward (advance labels)" if shift_samp < 0
                   else "no shift")
        print(f"[FULL SNR xcorr] shift = {shift_samp} samples, {dir_txt}.")
        
        labels = _shift_labels(labels, -shift_samp)

        # ------------------------------------------------------------------- #
        # 4) Window extraction ----------------------------------------------
        num_patches = window_size // patch_len
        stride_patches = max(1, int(round(num_patches * (1 - overlap))))
        stride = stride_patches * patch_len

        pad_amt = (stride - (len(data) % stride)) % stride
        if pad_amt:
            data = np.pad(data, ((0, pad_amt), (0, 0)), mode="edge")
            labels = np.pad(labels, (0, pad_amt), mode="edge")

        win_all = sliding_window_view(data, (window_size, data.shape[1]))
        lab_all = sliding_window_view(labels, window_size)

        windows = win_all[::stride, 0, :, :]  # (N, window_size, C)
        labwins = lab_all[::stride, :]        # (N, window_size)

        # ------------------------------------------------------------------- #
        # 5) Majority vote label --------------------------------------------
        X, y = [], []
        for w, l in zip(windows, labwins):
            nz = l[l != 0]

            if len(nz) == 0:
                 continue  # discard 'rest' windows
            
            lbl = int(np.bincount(nz).argmax()) - 1  # map 1…G → 0…G‑1

            X.append(w)
            y.append(lbl)

        if len(X) == 0:
            raise RuntimeError("No gesture windows were extracted from the file.")

        self.X = torch.tensor(np.stack(X, axis=0))
        self.y = torch.tensor(y, dtype=torch.long)
        self.labels  = self.y
        # store stats for downstream use
        self.mu = mu
        self.sigma = sigma
        self.augment = augment

    # ----------------------------------------------------------------------- #
    #  Dataset interface
    # ----------------------------------------------------------------------- #
    def __len__(self) -> int: 
        return len(self.X)

    def __getitem__(self, idx: int):
        X_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            # Apply augmentations to the PyTorch tensor
            # 1. Add Gaussian Noise
            noise = torch.randn_like(X_sample) * 0.05 # Adjust noise level
            X_sample = X_sample + noise

            # 2. Randomly scale amplitude
            scale = 1.0 + (torch.rand(1) - 0.5) * 0.2 # Scale by +/- 10%
            X_sample = X_sample * scale

        return X_sample, y_sample