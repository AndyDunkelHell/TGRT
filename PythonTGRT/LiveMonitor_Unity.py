
"""
LiveMonitor (Unity): Tkinter GUI to stream/plot HD-EMG (12 ch) + IMU (6 ch) from serial
or tail a Unity-style log file, compute quick SNR/MAV/CV metrics, place SNR-based
thresholds, and run small lag/SNR analyses with exportable plots/CSVs.

I/O & Formats
-------------
• Serial line (firmware): "ts|seq|e1,...,e12|i1,...,i6"
• File line   (Unity):   "time|label|e1,...,e12|i1,...,i6[|hostUs]"
• In-GUI plot: last 1000 samples; optional EMG band-pass.
• Saved outputs (via Tools → Gesture 3…): PNG example plots and CSV metrics.

Key UI Elements
---------------
• Start/Stop feed; Use Serial / Use File toggles; “Bandpass Filter” toggle
• SNR Measurement button (STILL → ACTIVE → DONE cycle)
• SNR View toggle + “thr % of max” entry + “Lock thresholds”
• Per-channel visibility checkboxes + per-channel metric labels
• Tools → Servo Controller (16-servo test window)

Constants / Globals (subset)
----------------------------
NUM_ELEC=12, NUM_CH=18
SNR_REF ∈ {"p95","max"}, THR_PCT (default 0.80), AGGR_MODE ∈ {"kofn","any"}, AGGR_K (default 3)
SNR_WIN=64, SR=500, BELOW_MS=80, BRIDGE_MS=150, REST_CONTROL_LEN=1000

Functions
---------
# --- Stats/normality helpers ---
_shapiro_per_channel(arr: np.ndarray, max_n: int = 5000)
    → (W: np.ndarray, p: np.ndarray, N_used: np.ndarray)
    Shapiro-Wilk per channel with subsampling for stability.

_build_rest_mask_with_guard_from_labels(y: np.ndarray, sr: int, guard_ms: int = 400)
    → np.ndarray[bool]
    REST := (y==0) with ±guard_ms removed around any transition.

_shapiro_rest_lag_corrected(guard_ms: int = 400, max_n: int = 5000, path: str | None = None)
    Compute lag via SNR xcorr, shift labels, run REST-only Shapiro per EMG
    channel, and write CSV (…_normality_rest_shapiro_LAG.csv).

# --- SNR / lag core ---
_running_rms(sig: np.ndarray, win: int) → np.ndarray
_compute_snr_db(emg: np.ndarray, noise_rms: np.ndarray, rms_win: int = SNR_WIN) → np.ndarray
_per_channel_thresholds(snr_db: np.ndarray, pct: float, ref: str = "p95") → np.ndarray
_over_mask(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, mode: str = "kofn", k: int = 3) → np.ndarray
_bridge_mask_gaps(mask: np.ndarray, bridge_ms: int, sr: int) → np.ndarray
_find_snr_end_from_mask(mask: np.ndarray, anchor: int, sr: int, below_ms: int = 60) → int
_estimate_lag_full_dataset_snr(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, y: np.ndarray,
                               sr: int, dp_module, max_lag_ms: int = 800, piecewise: bool = False,
                               seg_s: float = 5.0, hop_s: float = 2.5)
    → (shift_samples: int, shift_ms: float, info: dict)
_estimate_lag_from_first_example_snr(examples_with_anchor, snr_db, thr_db_per_ch,
                                     y, sr, max_lag_ms: int = 600)
    → (shift_samples: int, shift_ms: float)
_estimate_lag(examples_with_anchor, y: np.ndarray, sr: int, strategy: str = "median")
    → (lag_samples: int, lag_ms: float, idx_used: int | None)
_nearest_label_start(anchor_idx: int, y: np.ndarray) → int | None
_nearest_label_span_distance(anchor_idx: int, y: np.ndarray) → int | None
_shift_labels(y: np.ndarray, shift_samples: int) → np.ndarray

# --- Example picking / plotting ---
_pick_examples_nolabels(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, num: int = 3, sr: int = 1000,
                        min_gap_s: float = 2.0, win_s: float = 1.5,
                        y: np.ndarray | None = None, require_label_near: bool = True)
    → list[(s0, e0, anchor_idx, end_idx)]
_expand_window_to_include_label_span(seg: tuple, y: np.ndarray, anchor_idx: int, sr: int,
                                     pad_before_ms: int = 200, pad_after_ms: int = 300)
    → ((s0_new, e0_new), (span_start, span_end) | None)
_extract_still_active_from_seg(seg, emg: np.ndarray)
    → (still_emg: np.ndarray, active_emg: np.ndarray)
_plot_snr_lines_robust(ax, t: np.ndarray, snr_win: np.ndarray,
                       clip_db: tuple[float,float] = (-20.0, 40.0)) → None
_plot_example_pair(save_path: str, ts: np.ndarray, emg: np.ndarray,
                   snr_db: np.ndarray, amp_thr: np.ndarray, thr_db: np.ndarray,
                   seg: tuple, title_prefix: str, sr: int = 1000) → None
_plot_example_pair_with_labels(save_path: str, ts: np.ndarray, y: np.ndarray,
                               emg: np.ndarray, snr_db: np.ndarray,
                               amp_thr: np.ndarray, thr_db: np.ndarray, seg: tuple,
                               title_prefix: str, sr: int = 1000,
                               rest_bounds: tuple | None = None,
                               y_shifted: np.ndarray | None = None,
                               metrics_summary: str | None = None) → None
_write_gui_style_metrics_csv(fh, tag: str, still: np.ndarray, active: np.ndarray)
    → tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_summarize_metrics_for_text(snr_db: np.ndarray, mav_perc: np.ndarray,
                            cv_perc: np.ndarray, rms_perc: np.ndarray,
                            shapiro_p_still: np.ndarray | None = None) → str

# --- “Analyzer vs Control” action (Tools menu) ---
_compare_rest_analyzer_vs_control() → None
    Builds analyzer baseline (REST from labels), control baseline (first 1000),
    selects/plots SNR examples for both, writes per-example metrics CSV, and
    computes full-dataset lag; also emits REST normality CSV.

# --- Filters & transforms (EMG preprocessing) ---
butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4)
    → (b, a)
bandpass_filter(data: np.ndarray, lowcut: float = 20, highcut: float = 450,
                fs: int = 1000, order: int = 4) → np.ndarray
notch_filter_50Hz(signal: np.ndarray, fs: float = 1000.0, quality: float = 60.0) → np.ndarray
teager_kaiser_energy(signal: np.ndarray) → np.ndarray
moving_average(signal: np.ndarray, window_size: int = 5) → np.ndarray

# --- SNR & metrics used by the SNR button ---
compute_snr(still_arr: np.ndarray, active_arr: np.ndarray) → np.ndarray
    SNR_dB per channel from RMS(active) / RMS(still).
compute_metrics(data: np.ndarray) → tuple[np.ndarray, np.ndarray, np.ndarray]
    (RMS, CV, MAV) per channel.

# --- Data ingress (serial/file) & GUI plumbing ---
read_serial_data_hs() → None
    High-speed fixed-frame reader (48-byte frames) into data_buffer.
read_serial_data() → None
    Text-line serial reader "ts|seq|emg|imu" → data_buffer (+ UDP mirror, optional log).
select_data_file() / read_file_data() / stop_file_feed()
    Tail a Unity log file into data_buffer (+ gesture_buffer).
start_feed() / stop_feed() → None
    Open/close serial, spawn/join reader thread, manage logs/metrics files.
toggle_metrics() → None
snr_button() → None
    4-press SNR workflow (STILL → ACTIVE → compute/report → reset).
make_menubar(root: tk.Tk) → None
open_servo_window(root: tk.Tk) / close_servo_window(root, servo_win) → None
update(frame) → list[Artist]
    Matplotlib animation callback: filter/plot EMG/IMU, draw persistent thresholds.

Notes
-----
• Threshold lines reflect SNR(dB) stats (control baseline by default); locking
  freezes them for visual comparison across time.
• Many helpers mirror logic in data_pipeline.py for consistent lag/SNR behavior.
"""
import tkinter as tk
from tkinter import ttk
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import threading
import datetime
import time
from scipy.signal import iirnotch, filtfilt, butter
import os 
from tkinter import filedialog
from matplotlib import lines as mlines
import data_pipeline as dp
from matplotlib.collections import LineCollection
from scipy.stats import shapiro as _shapiro, skew as _skew, kurtosis as _kurtosis




# Global flags and data buffer
NUM_ELEC = 12
NUM_CH = NUM_ELEC + 6  # Number of channels to read from the serial port
running = False         # Controls whether the serial feed is running
toggle_metric = False   # Flag for toggling testing metric calculations
data_buffer = np.empty((0, NUM_CH))  # Buffer for incoming data (assumes NUM_CH channels)
record_file = None
metrics_file = None
_COM_PORT = "COM3"  # Serial port for the Arduino (change as needed)

# Variables for channel toggles and RMS labels
channel_vars = []  
metrics_labels = []    

snr_state = "idle"
still_data = np.empty((0, NUM_CH))
active_data = np.empty((0, NUM_CH))

snr_label = None


# Global for controlling metrics storage rate (once per second)
last_metrics_store_time = 0

ser = None
read_thread = None


data_lock     = threading.Lock()
# File‐tail globals:
file_path     = None
file_handle   = None
file_running  = False
file_thread   = None

dtype_frm  = np.dtype([('emg', '<i2', NUM_ELEC), ('imu', '<i4', 6)])

gesture_buffer = []        # parallel to data_buffer
vlines = []                # to keep track of our vertical‐line artists
last_gesture = 0          # last seen gesture ID (for file reading)


# Set up the Tkinter window
root = tk.Tk()
root.title("HD-EMG Live Feed")

# --- SNR view / thresholds / rest comparison globals ---
snr_view_var = tk.BooleanVar(value=False)   # UI toggle: plot SNR(dB) instead of raw
thr_lines = []                               # drawn horizontal threshold lines
thr_db_per_ch = None                         # per-channel SNR(dB) threshold (0.40 * max)
max_snr_db_per_ch = None                     # running per-channel max of SNR(dB)


REST_CONTROL_LEN = 1000                      # "first N samples are rest" control
thr_percent_var = tk.DoubleVar(value=0.40)   # % of per-channel max SNR for threshold

noise_rms_control = None    # per-channel noise RMS from first 1000 samples
noise_rms_analyzer = None   # per-channel noise RMS from analyzer rest windows

# --- Threshold line persistence ---
thr_lines = []                  # one horizontal line per EMG channel, kept across frames
thr_db_per_ch = None
max_snr_db_per_ch = None
last_thr_percent_val = 0.40

thr_lock_var = tk.BooleanVar(value=False)  # when True, don't update max/thresholds

# --- Marker styling for plots ---
COLOR_SNR_START = "tab:orange"
COLOR_SNR_END   = "tab:red"
COLOR_LABEL_ON  = "tab:green"
COLOR_LABEL_OFF = "tab:purple"
LW_MAJOR = 2.6   # thicker lines for SNR markers
LW_MINOR = 2.2   # thicker lines for label markers

# --- SNR detection settings ---
SNR_REF = "p95"        # 'p95' or 'max' for per-channel reference
THR_PCT = 0.80     # fraction of reference to define channel thresholds
AGGR_MODE = "kofn"     # 'kofn' or 'any'
AGGR_K    = 3          # k for k-of-N (e.g., 3 of 12 channels)
SNR_WIN   = 64         # RMS window (samples). Lower (e.g., 32) => quicker edges.
SR = 500          # sampling rate (Hz)

# --- SNR end/bridging settings ---
BELOW_MS  = 80    # how long mask must stay below thr to mark "end"
BRIDGE_MS = 150   # fill gaps below thr shorter than this (bridges small dips)

NEAR_LABEL_MS = 2000   # only keep SNR events within ±2 s of a label-3 span

# Shifted (lag-corrected) label markers
COLOR_LABEL_ON_SHIFT  = "green"   # solid green (corrected start)
COLOR_LABEL_OFF_SHIFT = "purple"  # solid purple (corrected end)
LW_SHIFT = 2.8


def _shapiro_per_channel(arr: np.ndarray, max_n: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shapiro-Wilk per column for a 2D array (N x C).
    Evenly sub-samples to max_n to keep SciPy stable and fast.
    Returns (W[C], p[C], N_used[C]).
    """
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (samples x channels)")
    N, C = arr.shape
    W = np.full(C, np.nan, dtype=np.float64)
    P = np.full(C, np.nan, dtype=np.float64)
    Nu = np.full(C, 0, dtype=np.int64)
    for c in range(C):
        x = np.asarray(arr[:, c])
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        if x.size > max_n:
            idx = np.linspace(0, x.size - 1, max_n, dtype=int)
            x = x[idx]
        try:
            w, p = _shapiro(x)
        except Exception:
            w, p = np.nan, np.nan
        W[c], P[c], Nu[c] = w, p, x.size
    return W, P, Nu


def _build_rest_mask_with_guard_from_labels(y: np.ndarray, sr: int, guard_ms: int = 400) -> np.ndarray:
    """
    REST := (y==0), with ±guard_ms removed around any transition between REST and NON-REST.
    Works on already lag-corrected labels.
    """
    rest = (y == 0).astype(np.int8)
    mask = rest.astype(bool).copy()
    if rest.size == 0:
        return mask
    guard = int(round(sr * guard_ms / 1000.0))
    # transitions where REST toggles
    edges = np.where(np.diff(rest, prepend=rest[0]) != 0)[0]
    for t in edges:
        lo = max(0, t - guard)
        hi = min(rest.size, t + guard + 1)
        mask[lo:hi] = False
    return mask

def _shapiro_rest_lag_corrected(guard_ms: int = 400, max_n: int = 5000, path: str | None = None):
    """
    REST-only Shapiro using LAG-CORRECTED labels:
      1) estimate lag via SNR xcorr
      2) shift labels
      3) REST mask with guard on shifted labels
      4) Shapiro per channel on REST samples
    Writes: {base}_normality_rest_shapiro_LAG.csv
    """
    if not file_var.get() or not file_path:
        status.config(text="Select a file first (Use File).", fg="red")
        return

    # 1) Load log
    ts, y, emg, imu = dp._load_log_strict(file_path, NUM_ELEC, 6)  
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = path if path else os.path.join(os.path.dirname(file_path), f"{base}_normality_rest_shapiro_LAG.csv")

    # 2) Build SNR(dB) with analyzer baseline (robust rest RMS) → thresholds
    noise_rms_analyzer = rest_noise_rms_from_labels(
        y=y, emg=emg, sr=SR, rest_win=256, rest_hop=128, guard_ms=400,
        method="percentile", percentile=10.0
    )
    snr_db  = _compute_snr_db(emg, noise_rms_analyzer, rms_win=SNR_WIN)
    thr_db  = _per_channel_thresholds(snr_db, pct=THR_PCT, ref=SNR_REF)

    # 3) Estimate full-dataset lag from SNR vs. label==3, then shift labels
    shift_samp, shift_ms, _ = _estimate_lag_full_dataset_snr(
        snr_db, thr_db, y, SR, dp_module=dp, max_lag_ms=1500, piecewise=False
    )
    shift_samp = 64  # TEMP OVERRIDE for testing
    y_shifted = _shift_labels(y, -shift_samp)

    # 4) REST mask (shifted labels) with guard, then Shapiro on those samples
    rest_mask = _build_rest_mask_with_guard_from_labels(y_shifted, SR, guard_ms=guard_ms)
    X = emg[rest_mask, :NUM_ELEC]
    if X.shape[0] < 8:
        status.config(text="Not enough REST samples after lag correction.", fg="red")
        return

    # Shapiro per channel 
    W, P, Nu = _shapiro_per_channel(X, max_n=max_n)
    Sk = _skew(X, axis=0, bias=False)
    Ku = _kurtosis(X, axis=0, fisher=True, bias=False)

    # 5) Save CSV
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("channel;W;p;N_rest;skew;kurt_excess;shift_ms\n")
        for c in range(NUM_ELEC):
            w = W[c]; p = P[c]; n = int(Nu[c])
            sk = Sk[c]; ku = Ku[c]
            fh.write(
                f"{c+1};"
                f"{('' if not np.isfinite(w) else f'{w:.6f}')} ;"
                f"{('' if not np.isfinite(p) else f'{p:.3e}')} ;"
                f"{n};"
                f"{('' if not np.isfinite(sk) else f'{sk:.6f}')} ;"
                f"{('' if not np.isfinite(ku) else f'{ku:.6f}')} ;"
                f"{shift_ms:.1f}\n"
            )

    print("saved:", out_csv)
    status.config(text=f"Saved REST (lag-corrected) Shapiro to: {out_csv}", fg="green")


def _extract_still_active_from_seg(seg, emg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For a seg = (s0,e0,anchor,end_idx) return (still_emg, active_emg) using EMG only (no IMU).
    still := [s0, anchor), active := [anchor, end_idx].
    """
    if len(seg) == 3:
        s0, e0, anchor = seg
        end_idx = e0 - 1
    else:
        s0, e0, anchor, end_idx = seg
    s0 = int(max(0, s0)); e0 = int(min(emg.shape[0], e0))
    anchor = int(np.clip(anchor, s0, e0 - 1))
    end_idx = int(np.clip(end_idx, s0, e0 - 1))
    if end_idx < anchor:
        end_idx = anchor
    still = emg[s0:anchor, :NUM_ELEC]
    active = emg[anchor:end_idx, :NUM_ELEC]
    return still, active


def _summarize_metrics_for_text(snr_db: np.ndarray, mav_perc: np.ndarray,
                                cv_perc: np.ndarray, rms_perc: np.ndarray,
                                shapiro_p_still: np.ndarray | None = None) -> str:
    """
    Build a short, single-line summary for overlay on plots.
    """
    snr_mean = float(np.nanmean(snr_db)) if snr_db.size else float("nan")
    mv_med   = float(np.nanmedian(mav_perc)) if mav_perc.size else float("nan")
    cv_med   = float(np.nanmedian(cv_perc))  if cv_perc.size else float("nan")
    rms_med  = float(np.nanmedian(rms_perc)) if rms_perc.size else float("nan")
    if shapiro_p_still is not None and shapiro_p_still.size:
        nonnorm = int(np.sum(shapiro_p_still < 0.05))
        return (f"SNR≈{snr_mean:.1f} dB | MAV%~{mv_med:.0f} | CV%~{cv_med:.0f} | "
                f"RMS%~{rms_med:.0f} | non-normal(ch p<0.05)={nonnorm}/{shapiro_p_still.size}")
    return f"SNR≈{snr_mean:.1f} dB | MAV%~{mv_med:.0f} | CV%~{cv_med:.0f} | RMS%~{rms_med:.0f}"


def _write_gui_style_metrics_csv(fh, tag: str,
                                 still: np.ndarray, active: np.ndarray):
    """
    Write a section into `fh` that mirrors the GUI SNR button CSV structure:
    - still_data (first/last 10 samples per channel)
    - active_data (first/last 10 samples per channel)
    - snr_values
    - per-channel MAV/CV/RMS percentages (active vs still), incl. general MAV% normalization
    Uses EMG channels only.
    """
    # dump first/last 10 samples like the GUI button does
    fh.write(f"{tag}still_data\nstart\n")
    first_still = still[:10] if still.size else still
    for ch in range(still.shape[1] if still.ndim == 2 else NUM_ELEC):
        for val in (first_still[:, ch] if first_still.size else []):
            fh.write(f"{val};")
        fh.write("\n")
    fh.write("end\n")
    last_still = still[-10:] if still.size else still
    for ch in range(still.shape[1] if still.ndim == 2 else NUM_ELEC):
        for val in (last_still[:, ch] if last_still.size else []):
            fh.write(f"{val};")
        fh.write("\n")
    fh.write("\n")

    fh.write(f"{tag}active_data\nstart\n")
    first_active = active[:10] if active.size else active
    for ch in range(active.shape[1] if active.ndim == 2 else NUM_ELEC):
        for val in (first_active[:, ch] if first_active.size else []):
            fh.write(f"{val};")
        fh.write("\n")
    fh.write("end\n")
    last_active = active[-10:] if active.size else active
    for ch in range(active.shape[1] if active.ndim == 2 else NUM_ELEC):
        for val in (last_active[:, ch] if last_active.size else []):
            fh.write(f"{val};")
        fh.write("\n")
    fh.write("\n")

    # calculations
    snr_vals = compute_snr(still, active) if still.size and active.size else np.array([])
    fh.write("snr_values\n")
    if snr_vals.size:
        fh.write(";".join(f"{v:.2f}" for v in snr_vals) + "\n")
    else:
        fh.write("\n")

    s_rms, s_cv, s_mav = compute_metrics(still)
    a_rms, a_cv, a_mav = compute_metrics(active)

    # Percent metrics
    # Safeguards against divide-by-zero
    s_mav_safe = np.where(s_mav == 0, 1e-12, s_mav)
    s_cv_safe  = np.where(s_cv  == 0, 1e-12, s_cv)
    s_rms_safe = np.where(s_rms == 0, 1e-12, s_rms)
    general_a_max_mav = float(np.max(a_mav)) if a_mav.size and np.max(a_mav) != 0 else 1e-12

    for ch in range(NUM_ELEC if s_mav.size >= NUM_ELEC else s_mav.size):
        mav_perc = (a_mav[ch] / s_mav_safe[ch]) * 100.0
        general_mav_perc = (a_mav[ch] / general_a_max_mav) * 100.0
        cv_perc = (a_cv[ch] / s_cv_safe[ch]) * 100.0
        rms_perc = (a_rms[ch] / s_rms_safe[ch]) * 100.0
        line = (f"Ch{ch+1}_MAV_perc;{mav_perc:.2f};"
                f"Ch{ch+1}_MAV_general_perc;{general_mav_perc:.2f};"
                f"Ch{ch+1}_CV_perc;{cv_perc:.2f};"
                f"Ch{ch+1}_RMS_perc;{rms_perc:.2f}\n")
        fh.write(line)

    # Return arrays for optional overlay text
    mav_perc_arr = (a_mav / s_mav_safe) * 100.0 if a_mav.size else np.array([])
    cv_perc_arr  = (a_cv  / s_cv_safe ) * 100.0 if a_cv.size  else np.array([])
    rms_perc_arr = (a_rms / s_rms_safe) * 100.0 if a_rms.size else np.array([])
    return snr_vals, mav_perc_arr, cv_perc_arr, rms_perc_arr


def _estimate_lag_full_dataset_snr(snr_db: np.ndarray,
                                   thr_db_per_ch: np.ndarray,
                                   y: np.ndarray,
                                   sr: int,
                                   dp_module,
                                   max_lag_ms: int = 800,
                                   piecewise: bool = False,
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
    lag = dp_module._crosscorr_lag(a, b, max_lag=maxlag)  # +lag => b shift LEFT by lag
    shift_samp = -int(lag)                                # convert to our convention
    shift_ms   = shift_samp * 1000.0 / float(sr)

    info = {"xcorr_lag_samples": int(lag), "xcorr_shift_samples": shift_samp, "xcorr_shift_ms": shift_ms}

    # 3) Optional piecewise summary (to see drift)
    if piecewise:
        seg_len = max(1, int(round(seg_s * sr)))
        hop     = max(1, int(round(hop_s * sr)))
        lags = dp_module.segment_lags(a, b, seg_len=seg_len, hop=hop, max_lag=maxlag)
        info["piecewise_samples"] = lags
        if lags.size:
            info["piecewise_median_ms"] = float(np.median(lags) * -1000.0 / sr)
            info["piecewise_mean_ms"]   = float(np.mean(lags)   * -1000.0 / sr)
            info["piecewise_std_ms"]    = float(np.std(lags)    *  1000.0 / sr)

    return shift_samp, shift_ms, info


def _snr_activity_mask(snr_db: np.ndarray, thr_db_per_ch: np.ndarray, sr: int) -> np.ndarray:
    """k-of-N over-threshold + gap bridging -> boolean activity mask (SNR-based)."""
    raw_mask = _over_mask(snr_db, thr_db_per_ch, mode=AGGR_MODE, k=AGGR_K)
    return _bridge_mask_gaps(raw_mask, BRIDGE_MS, sr)

def _estimate_lag_from_first_example_snr(
        examples_with_anchor, snr_db: np.ndarray, thr_db_per_ch: np.ndarray,
        y: np.ndarray, sr: int, max_lag_ms: int = 600
    ) -> tuple[int, float]:
    """
    Use ONLY the first analyzer example window. Cross-correlate SNR activity mask
    vs. label==3 mask to estimate lag (à la data_pipeline).
    Returns (shift_samples, shift_ms) to apply to labels with _shift_labels().
    Sign convention: +shift => delay labels (move RIGHT / later).
    """
    if not examples_with_anchor:
        return 0, 0.0

    # First example's window (use the same region we plotted)
    s0, e0, *_ = examples_with_anchor[0]

    # Build SNR activity mask (k-of-N + bridging) and label==3 mask
    m_snr  = _snr_activity_mask(snr_db, thr_db_per_ch, sr)
    a_bin  = m_snr[s0:e0].astype(np.float32)        # "a" stream (reference)
    b_bin  = (y[s0:e0] == 3).astype(np.float32)     # "b" stream (labels)

    # Cross-corr lag like in data_pipeline: positive lag => b should shift LEFT by lag
    maxlag = int(round(max_lag_ms * sr / 1000.0))
    lag = dp._crosscorr_lag(a_bin, b_bin, max_lag=maxlag)   # from data_pipeline utilities
    # Convert "lag" (b shift LEFT) into our label SHIFT convention (+ = shift RIGHT)
    shift_samples = -int(lag)
    shift_ms      = shift_samples * 1000.0 / float(sr)
    return shift_samples, shift_ms


def _estimate_lag(examples_with_anchor, y: np.ndarray, sr: int, strategy: str = "median"):
    """
    Compute lag between SNR start (anchor) and nearest label-3 start.
    strategy:
      - "first": use the FIRST example that has a label start; returns (lag_samp, lag_ms, idx_used)
      - "median": median over all examples; returns (lag_samp, lag_ms, idx_used=None)
    Positive lag => labels are EARLY (must shift labels FORWARD).
    """
    lags = []
    idxs = []
    for i, seg in enumerate(examples_with_anchor, 1):
        anchor = seg[2]  # (s0, e0, anchor, end)
        ls = _nearest_label_start(anchor, y)
        if ls is None:
            continue
        lag_samp = int(anchor - ls)
        lags.append(lag_samp)
        idxs.append(i)

        if strategy == "first":
            lag_ms = lag_samp * 1000.0 / float(sr)
            return lag_samp, lag_ms, i  # <- example index used

    if not lags:
        return 0, 0.0, None

    # median (default)
    med = int(np.median(lags))
    return med, med * 1000.0 / float(sr), None


def _run_modal(title: str, message: str, fn, *args, **kwargs):
    top = tk.Toplevel(root)
    top.title(title)
    top.transient(root)
    top.grab_set()
    ttk.Label(top, text=message, padding=10).pack()
    pb = ttk.Progressbar(top, mode="indeterminate", length=240)
    pb.pack(padx=10, pady=10)
    pb.start(10)
    root.update_idletasks()
    try:
        fn(*args, **kwargs)
    finally:
        pb.stop()
        top.grab_release()
        top.destroy()


def _shift_labels(y: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift labels in time. Positive shift moves labels *forward* in time (delays them)."""
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

def _nearest_label_start(anchor_idx: int, y: np.ndarray) -> int | None:
    """Nearest rising edge of (y==3). Returns sample index or None."""
    y3  = (y == 3).astype(np.int8)
    starts = np.where(np.diff(y3, prepend=0) == 1)[0]
    if starts.size == 0:
        return None
    return int(starts[np.argmin(np.abs(starts - anchor_idx))])

def _estimate_global_lag_from_examples(examples_with_anchor, y: np.ndarray, sr: int) -> tuple[int, float]:
    """
    From selected examples [(s0,e0,anchor,end), ...], compute per-example lag
    (anchor - nearest label_start). Return (lag_samples_median, lag_ms_median).
    """
    lags = []
    for (_, _, anchor, _) in examples_with_anchor:
        ls = _nearest_label_start(anchor, y)
        if ls is not None:
            lags.append(anchor - ls)   # >0 means labels are early; need to delay labels
    if not lags:
        return 0, 0.0
    lag_samp = int(np.median(lags))
    lag_ms   = lag_samp * 1000.0 / float(sr)
    return lag_samp, lag_ms

def _nearest_label_span_distance(anchor_idx: int, y: np.ndarray) -> int | None:
    if y is None or y.size == 0:
        return None
    y3  = (y == 3).astype(np.int8)
    dy3 = np.diff(y3, prepend=0)
    starts = list(np.where(dy3 == 1)[0])
    ends   = list(np.where(dy3 == -1)[0])
    N = y.size
    if y3[0] == 1: starts = [0] + starts
    if y3[-1] == 1: ends   = ends + [N - 1]
    if not starts or not ends:
        return None
    spans = []
    j = 0
    for st in starts:
        while j < len(ends) and ends[j] < st:
            j += 1
        if j >= len(ends): break
        spans.append((st, ends[j]))
    if not spans:
        return None
    # distance to nearest span (0 if anchor lies inside a span)
    return min(0 if (st <= anchor_idx <= en) else min(abs(anchor_idx - st), abs(anchor_idx - en))
               for st, en in spans)


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


def _plot_snr_lines_robust(ax, t, snr_win, clip_db=(-20.0, 40.0)):
    snr_win = snr_win.astype(np.float64)
    snr_win[~np.isfinite(snr_win)] = np.nan
    if clip_db is not None:
        lo, hi = clip_db
        snr_win = np.clip(snr_win, lo, hi)

    plotted = False
    for c in range(snr_win.shape[1]):
        y = snr_win[:, c]
        m = np.isfinite(y)
        if m.any():
            ax.plot(t[m], y[m])
            plotted = True

    if plotted:
        vals = snr_win[np.isfinite(snr_win)]
        if vals.size:
            q5, q95 = np.percentile(vals, [5.0, 95.0])
            if np.isfinite(q5) and np.isfinite(q95) and q95 > q5:
                pad = 0.08 * (q95 - q5)
                ax.set_ylim(q5 - pad, q95 + pad)
    else:
        ax.set_ylim(-5, 5)


def _per_channel_thresholds(snr_db: np.ndarray, pct: float, ref: str = "p95") -> np.ndarray:
    """Return per-channel SNR(dB) thresholds using a robust reference (p95 or max)."""
    if ref == "p95":
        refvals = np.percentile(snr_db, 95.0, axis=0)
    elif ref == "max":
        refvals = np.max(snr_db, axis=0)
    else:
        raise ValueError("SNR_REF must be 'p95' or 'max'")
    return pct * refvals

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

def _find_snr_end_from_mask(mask: np.ndarray, anchor: int, sr: int, below_ms: int = 60) -> int:
    """First index at/after anchor where mask goes False and stays False for >= below_ms."""
    N = mask.shape[0]; anchor = int(anchor)
    if anchor >= N - 1:
        return N - 1
    min_below = max(1, int(round(below_ms * sr / 1000.0)))
    tail = (~mask[anchor:]).astype(np.int32)
    run = np.convolve(tail, np.ones(min_below, dtype=np.int32), mode="valid")
    idx = np.where(run == min_below)[0]
    return min(N - 1, anchor + int(idx[0])) if idx.size else N - 1


def _plot_example_pair_with_labels(save_path: str,
                                   ts: np.ndarray, y: np.ndarray,
                                   emg: np.ndarray,
                                   snr_db: np.ndarray,
                                   amp_thr: np.ndarray,
                                   thr_db: np.ndarray,
                                   seg, title_prefix: str,
                                   sr: int = 1000,
                                   rest_bounds: tuple | None = None,
                                   y_shifted: np.ndarray | None = None,
                                   metrics_summary: str | None = None):

    # Accept (s0,e0,anchor) or (s0,e0,anchor,end)
    if len(seg) == 3:
        s0, e0, anchor = seg; end_idx = e0 - 1
    else:
        s0, e0, anchor, end_idx = seg

    # Time axis
    if ts is not None and ts.size >= e0 and np.max(ts) > 1e6:
        to_s  = lambda i: (ts[i] - ts[s0]) * 1e-3 / 1000.0
        t     = np.array([to_s(i) for i in range(s0, e0)], dtype=np.float64)
        t_anchor = to_s(anchor); t_end = to_s(min(end_idx, e0-1))
    else:
        t = np.arange(e0 - s0, dtype=np.float64) / float(sr)
        t_anchor = (anchor - s0)/float(sr); t_end = (min(end_idx, e0-1)-s0)/float(sr)

    # Label edges (original)
    y3 = (y == 3).astype(np.int8)
    dy = np.diff(y3, prepend=0)
    ups = [i for i in np.where(dy == 1)[0] if s0 <= i < e0]
    dns = [i for i in np.where(dy == -1)[0] if s0 <= i < e0]

    # Label edges (shifted), if provided
    ups_shift = dns_shift = []
    if y_shifted is not None:
        y3s = (y_shifted == 3).astype(np.int8)
        dys = np.diff(y3s, prepend=0)
        ups_shift = [i for i in np.where(dys == 1)[0] if s0 <= i < e0]
        dns_shift = [i for i in np.where(dys == -1)[0] if s0 <= i < e0]

    # Figure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0])

    # RAW traces + raw thresholds
    ax1.plot(t, emg[s0:e0, :])
    for c in range(emg.shape[1]):
        ax1.axhline(amp_thr[c], linestyle="--", linewidth=0.8)

    # Markers on RAW
    ax1.axvline(t_anchor, color=COLOR_SNR_START, linewidth=LW_MAJOR, zorder=10, label="SNR start")
    ax1.axvline(t_end,    color=COLOR_SNR_END,   linewidth=LW_MAJOR, linestyle="-.", zorder=10, label="SNR end")
    # for L in ups: ax1.axvline((L - s0)/float(sr), color=COLOR_LABEL_ON,  linewidth=LW_MINOR, linestyle=":", zorder=9)
    # for L in dns: ax1.axvline((L - s0)/float(sr), color=COLOR_LABEL_OFF, linewidth=LW_MINOR, linestyle=":", zorder=9)
    # # shifted labels (solid)
    # for L in ups_shift: ax1.axvline((L - s0)/float(sr), color=COLOR_LABEL_ON_SHIFT,  linewidth=LW_SHIFT, linestyle="-", zorder=11)
    # for L in dns_shift: ax1.axvline((L - s0)/float(sr), color=COLOR_LABEL_OFF_SHIFT, linewidth=LW_SHIFT, linestyle="-", zorder=11)

    # --- Label as block/step waveform overlaid at the top of RAW ---
    # Map label values (0..max_label) into a narrow band at the top of the axis
    y_win = y[s0:e0].astype(float)
    max_label = int(np.nanmax(y_win)) 

    ymin, ymax = ax1.get_ylim()
    rng = (ymax - ymin) if ymax > ymin else 1.0
    top_band   = ymax - 0.02 * rng          # top edge of the label band
    band_height = 0.14 * rng                # ~14% of the axis for labels
    step_h = band_height / max(1, max_label if max_label > 0 else 1)

    # Convert label values to y-positions inside the top band (higher label -> slightly lower y)
    y_step = top_band - y_win * step_h

    # Draw the step/block label trace
    # Draw the step/block label trace with color depending on y value

    # Compute the segments for steps
    segments = []
    colors_list = []
    for i in range(len(t) - 1):
        # Each step is from t[i] to t[i+1] at y_step[i]
        segments.append([[t[i], y_step[i]], [t[i+1], y_step[i]]])
        # Choose color based on y value at this step
        if y_win[i] == 3:
            colors_list.append("red")  # or any color for label==3
        elif y_win[i] == 2:
            colors_list.append("orange")
        elif y_win[i] == 1:
            colors_list.append("blue")
        else:
            colors_list.append(COLOR_LABEL_ON)  # fallback/default

    lc = LineCollection(segments, colors=colors_list, linewidths=2.4, alpha=0.95, zorder=12)
    ax1.add_collection(lc)
    # (Optional) fill to the very top for a stronger "block" look
    ax1.fill_between(t, y_step, top_band,
                     step="post", alpha=0.12, zorder=11, color=COLOR_LABEL_ON)
                    
    if y_shifted is not None:
                        y_shift_win = y_shifted[s0:e0].astype(float)
                        y_step_s = (top_band - 0.06 * rng) - y_shift_win * step_h  # second lane, a bit below

                        # Compute the segments for steps (shifted)
                        segments_shifted = []
                        colors_shifted = []
                        for i in range(len(t) - 1):
                            segments_shifted.append([[t[i], y_step_s[i]], [t[i+1], y_step_s[i]]])
                            # Choose color based on y_shifted value at this step
                            if y_shift_win[i] == 3:
                                colors_shifted.append("purple")
                            elif y_shift_win[i] == 2:
                                colors_shifted.append("black")
                            elif y_shift_win[i] == 1:
                                colors_shifted.append("yellow")
                            else:
                                colors_shifted.append("green")  # fallback/default

                        lc_shifted = LineCollection(segments_shifted, colors=colors_shifted, linewidths=2.4, alpha=0.95, zorder=13)
                        ax1.add_collection(lc_shifted)
                        ax1.fill_between(t, y_step_s, (top_band - 0.06 * rng),
                                         step="post", alpha=0.10, zorder=12, color="red")


    # legend
    h_snr_start = mlines.Line2D([], [], color=COLOR_SNR_START, linewidth=LW_MAJOR, label="SNR start")
    h_snr_end   = mlines.Line2D([], [], color=COLOR_SNR_END,   linewidth=LW_MAJOR, linestyle="-.", label="SNR end")
    h_lbl_on    = mlines.Line2D([], [], color=COLOR_LABEL_ON,  linewidth=LW_MINOR, linestyle=":",   label="Label 3 start")
    h_lbl_off   = mlines.Line2D([], [], color=COLOR_LABEL_OFF, linewidth=LW_MINOR, linestyle=":",   label="Label 3 end")
    h_lbl_on_s  = mlines.Line2D([], [], color=COLOR_LABEL_ON_SHIFT,  linewidth=LW_SHIFT, linestyle="-", label="Label 3 start (shifted)")
    h_lbl_off_s = mlines.Line2D([], [], color=COLOR_LABEL_OFF_SHIFT, linewidth=LW_SHIFT, linestyle="-", label="Label 3 end (shifted)")
    ax1.legend(handles=[h_snr_start, h_snr_end, h_lbl_on, h_lbl_off, h_lbl_on_s, h_lbl_off_s],
               loc="upper right", framealpha=0.65)

    # SNR(dB) panel + SNR thresholds
    _plot_snr_lines_robust(ax2, t, snr_db[s0:e0, :])
    for c in range(emg.shape[1]):
        ax2.axhline(thr_db[c], linestyle="--", linewidth=0.8)
    ax2.axvline(t_anchor, color=COLOR_SNR_START, linewidth=LW_MAJOR, zorder=10)
    ax2.axvline(t_end,    color=COLOR_SNR_END,   linewidth=LW_MAJOR, linestyle="-.", zorder=10)
    for L in ups:       ax2.axvline((L - s0)/float(sr), color=COLOR_LABEL_ON,        linewidth=LW_MINOR, linestyle=":", zorder=9)
    for L in dns:       ax2.axvline((L - s0)/float(sr), color=COLOR_LABEL_OFF,       linewidth=LW_MINOR, linestyle=":", zorder=9)
    for L in ups_shift: ax2.axvline((L - s0)/float(sr), color=COLOR_LABEL_ON_SHIFT,  linewidth=LW_SHIFT, linestyle="-", zorder=11)
    for L in dns_shift: ax2.axvline((L - s0)/float(sr), color=COLOR_LABEL_OFF_SHIFT, linewidth=LW_SHIFT, linestyle="-", zorder=11)

    if metrics_summary:
        ax2.text(0.01, 0.02, metrics_summary,
            transform=ax2.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _find_snr_end(any_over: np.ndarray, anchor: int, sr: int, below_ms: int = 60) -> int:
    """
    Return the first index at/after `anchor` where SNR falls below threshold
    and stays below for at least `below_ms`. Provides hysteresis against flicker.
    """
    N = any_over.shape[0]
    anchor = int(anchor)
    if anchor >= N - 1:
        return N - 1
    min_below = max(1, int(round(below_ms * sr / 1000.0)))
    tail = (~any_over[anchor:]).astype(np.int32)
    # first position where a run of length >= min_below starts
    run = np.convolve(tail, np.ones(min_below, dtype=np.int32), mode="valid")
    idx = np.where(run == min_below)[0]
    if idx.size:
        return min(N - 1, anchor + int(idx[0]))
    # fallback: no clear falling edge, use window end
    return N - 1


def _compute_snr_db(emg: np.ndarray, noise_rms: np.ndarray, rms_win: int = SNR_WIN) -> np.ndarray:
    rr = _running_rms(emg, rms_win).astype(np.float64)           # (N,C)
    den = np.maximum(noise_rms[None, :].astype(np.float64), 1e-9) # (1,C)
    snr_lin = np.maximum(rr, 1e-9) / den
    return (20.0 * np.log10(snr_lin)).astype(np.float32)

def _pick_examples_nolabels(snr_db: np.ndarray,
                            thr_db_per_ch: np.ndarray,
                            num: int = 3, sr: int = 1000,
                            min_gap_s: float = 2.0,
                            win_s: float = 1.5,
                            y: np.ndarray | None = None,
                            require_label_near: bool = True):
    """
    Anchor at the first rising edge of the *bridged* k-of-N mask (no refinement).
    Only keep examples whose anchor is within ±NEAR_LABEL_MS of a label-3 span
    if require_label_near is True. Returns (s0, e0, anchor_idx, end_idx).
    """
    raw_mask = _over_mask(snr_db, thr_db_per_ch, mode=AGGR_MODE, k=AGGR_K)
    mask     = _bridge_mask_gaps(raw_mask, BRIDGE_MS, sr)
    edges    = np.where(np.diff(mask.astype(np.int8), prepend=0) == 1)[0]

    win = int(round(win_s * sr))
    gap = int(round(min_gap_s * sr))
    near_samp = int(round(NEAR_LABEL_MS * sr / 1000.0))

    picks, last_end = [], -gap
    for anchor in edges:
        if require_label_near and y is not None:
            dist = _nearest_label_span_distance(anchor, y)
            if dist is None or dist > near_samp:
                continue

        e_fall = _find_snr_end_from_mask(mask, anchor, sr=sr, below_ms=BELOW_MS)
        s0 = max(0, anchor - win // 4)
        e0 = min(max(s0 + win, e_fall + 1), snr_db.shape[0])

        if s0 - last_end < gap:
            continue
        picks.append((s0, e0, anchor, e_fall))
        last_end = e0
        if len(picks) >= num:
            break
    return picks





def _expand_window_to_include_label_span(seg, y: np.ndarray,
                                         anchor_idx: int, sr: int,
                                         pad_before_ms: int = 200,
                                         pad_after_ms: int = 300):
    """
    Expand (s0,e0) to include the entire nearest label==3 SPAN (start & end),
    plus padding. Returns ((s0,e0), (span_start, span_end)) or ((s0,e0), None)
    if no label spans exist.
    """
    s0, e0 = int(seg[0]), int(seg[1])
    if y is None or y.size == 0:
        return (s0, e0), None

    y3  = (y == 3).astype(np.int8)
    dy3 = np.diff(y3, prepend=0)
    starts = list(np.where(dy3 == 1)[0])
    ends   = list(np.where(dy3 == -1)[0])

    N = y.shape[0]
    # Handle open spans at the boundaries
    if y3[0] == 1:
        starts = [0] + starts
    if y3[-1] == 1:
        ends = ends + [N - 1]

    if len(starts) == 0 or len(ends) == 0:
        return (s0, e0), None

    # Build spans (start <= end)
    spans = []
    j = 0
    for i, st in enumerate(starts):
        while j < len(ends) and ends[j] < st:
            j += 1
        if j >= len(ends):
            break
        en = ends[j]
        spans.append((st, en))

    if not spans:
        return (s0, e0), None

    # Pick the span whose boundary (start or end) is closest to the anchor
    nearest_span = min(spans, key=lambda se: min(abs(se[0] - anchor_idx), abs(se[1] - anchor_idx)))
    span_start, span_end = nearest_span

    pre  = int(round(pad_before_ms * sr / 1000.0))
    post = int(round(pad_after_ms  * sr / 1000.0))

    s0_new = min(s0, max(0, span_start - pre))
    e0_new = max(e0, min(N, span_end + post))

    if e0_new <= s0_new:
        e0_new = min(N, s0_new + 10)  # safety

    return (s0_new, e0_new), (span_start, span_end)

def _plot_example_pair(save_path: str, ts: np.ndarray, emg: np.ndarray,
                       snr_db: np.ndarray, amp_thr: np.ndarray,
                       thr_db: np.ndarray, seg, title_prefix: str, sr: int = 1000):
    # seg can be (s0,e0), (s0,e0,anchor), or (s0,e0,anchor,end_idx)
    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
        s0, e0 = int(seg[0]), int(seg[1])
    else:
        raise ValueError("seg must be tuple/list with at least (s0,e0)")

    # Time axis
    if ts is not None and ts.size >= e0 and np.max(ts) > 1e6:
        t = (ts[s0:e0] - ts[s0]) * 1e-3 / 1000.0  # µs -> s
    else:
        t = np.arange(e0 - s0, dtype=np.float64) / float(sr)

    # Figure: RAW (top) + SNR(dB) (bottom)
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0])

    # RAW + per-channel raw thresholds
    ax1.plot(t, emg[s0:e0, :])
    for c in range(emg.shape[1]):
        ax1.axhline(amp_thr[c], linestyle="--", linewidth=0.8)
    ax1.set_title(f"{title_prefix} RAW (with per-channel thresholds)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("EMG (a.u.)")

    # SNR(dB) + per-channel SNR thresholds
    _plot_snr_lines_robust(ax2, t, snr_db[s0:e0, :])
    for c in range(emg.shape[1]):
        ax2.axhline(thr_db[c], linestyle="--", linewidth=0.8)
    ax2.set_title("SNR per channel (dB)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("SNR (dB)")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _ensure_threshold_lines(ax):
    """Create (or recreate) persistent threshold lines once; keep them and just update y."""
    global thr_lines
    if len(thr_lines) == NUM_ELEC:
        return
    # purge any leftovers
    for ln in thr_lines:
        try: ln.remove()
        except Exception: pass
    thr_lines = []
    # create one dashed line per channel, hidden initially
    for _ in range(NUM_ELEC):
        ln = ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.9, zorder=6)
        ln.set_visible(False)
        thr_lines.append(ln)


def _running_rms(sig: np.ndarray, win: int) -> np.ndarray:
    """Per-channel running RMS (same length, 'same' conv)."""
    win = max(1, int(win))
    sq = sig.astype(np.float64)**2
    ker = np.ones(win, dtype=np.float64) / float(win)
    out = np.empty_like(sq)
    for c in range(sq.shape[1]):
        out[:, c] = np.sqrt(np.convolve(sq[:, c], ker, mode="same"))
    return out.astype(np.float32)


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

def _compare_rest_analyzer_vs_control():
    """
    Compare analyzer vs control rest noise; then, WITHOUT labels, find 3 SNR-based
    examples per style and save plots. Additionally:
      * Run Shapiro–Wilk per EMG channel on REST (analyzer vs control) -> CSV
      * For every saved example window, write a GUI-style metrics CSV section
        (still->active->calculations) and overlay a one-line summary on plots.
    """
    global noise_rms_control, noise_rms_analyzer, thr_db_per_ch, max_snr_db_per_ch

    if not file_var.get() or not file_path:
        status.config(text="Select a file first (Use File).", fg="red")
        return

    ts, y, emg, imu = dp._load_log_strict(file_path, NUM_ELEC, 6)

    # --- Analyzer rest baseline (labels only for BASELINE, not for detection)
    noise_rms_analyzer = rest_noise_rms_from_labels(
        y=y, emg=emg, sr=SR, rest_win=256, rest_hop=128,
        guard_ms=400, method="percentile", percentile=10.0
    )

    # --- Control baseline (first REST_CONTROL_LEN samples assumed rest)
    n0 = min(REST_CONTROL_LEN, emg.shape[0])
    noise_rms_control = np.sqrt(np.mean(emg[:-n0, :]**2, axis=0))

    # --- Print comparison (unchanged)
    diff_db = 20.0 * np.log10(np.maximum(noise_rms_analyzer, 1e-12) /
                              np.maximum(noise_rms_control, 1e-12))
    print("\n=== Rest baseline comparison ===")
    print(f"File: {os.path.basename(file_path)}   N={emg.shape[0]}  control N={n0}")
    print("Analyzer rest RMS per ch:", np.round(noise_rms_analyzer, 6))
    print("Control  rest RMS per ch:", np.round(noise_rms_control,  6))
    print("Analyzer - Control (dB): ", np.round(diff_db, 2))
    print("Mean |Δ| (dB):           ", np.round(np.mean(np.abs(diff_db)), 2))
    status.config(text=f"Rest Δ(dB) mean={np.mean(np.abs(diff_db)):.2f}, max={np.max(np.abs(diff_db)):.2f}", fg="blue")

    # Reset dynamic thresholds for live plot
    max_snr_db_per_ch = None
    thr_db_per_ch     = None

    # --- Build output dir
    base    = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(os.path.dirname(file_path), f"{base}_snr_examples")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # NEW: Shapiro–Wilk per EMG channel for REST (analyzer vs control)
    # ------------------------------------------------------------------
    # Recreate the analyzer "clean rest" mask (same logic as baseline) and collect samples
    N = emg.shape[0]
    y_i = (y == 3).astype(int)
    rest_mask = (y == 0)
    guard = int(round(SR * 400 / 1000.0))
    trans = np.where(np.diff(y_i, prepend=(y[0] != 0)) != 0)[0]
    for t in trans:
        lo = max(0, t - guard)
        hi = min(N, t + guard + 1)
        rest_mask[lo:hi] = False

    rest_analyzer_samples = emg[rest_mask, :NUM_ELEC]
    rest_control_samples  = emg[:max(0, N - n0), :NUM_ELEC]  # mirrors control baseline

    #stat_an, p_an = _shapiro_per_channel(rest_analyzer_samples)
    #stat_co, p_co = _shapiro_per_channel(rest_control_samples)

    shapiro_csv = os.path.join(out_dir, f"{base}_rest_normality.csv")
    _shapiro_rest_lag_corrected(path=shapiro_csv)

    print("saved:", shapiro_csv)

    # ---------------------------------------------------
    # SNR detection & example selection
    # ---------------------------------------------------
    WIN_S = 1.5
    GAP_S = 2.0

    # CONTROL SNR
    snr_ctrl_db  = _compute_snr_db(emg, noise_rms_control, rms_win=SNR_WIN)
    thr_ctrl_db  = _per_channel_thresholds(snr_ctrl_db, pct=THR_PCT, ref=SNR_REF)
    amp_thr_ctrl = noise_rms_control * (10.0 ** (thr_ctrl_db / 20.0))
    ex_ctrl = _pick_examples_nolabels(snr_ctrl_db, thr_ctrl_db, num=3, sr=SR,
                                      min_gap_s=GAP_S, win_s=WIN_S, y=y, require_label_near=True)

    # ANALYZER SNR
    snr_an_db  = _compute_snr_db(emg, noise_rms_analyzer, rms_win=SNR_WIN)
    thr_an_db  = _per_channel_thresholds(snr_an_db, pct=THR_PCT, ref=SNR_REF)
    amp_thr_an = noise_rms_analyzer * (10.0 ** (thr_an_db / 20.0))
    ex_an = _pick_examples_nolabels(snr_an_db, thr_an_db, num=3, sr=SR,
                                    min_gap_s=GAP_S, win_s=WIN_S, y=y, require_label_near=True)

    # Paths and a consolidated metrics CSV
    examples_csv = os.path.join(out_dir, f"{base}_examples_metrics.csv")
    with open(examples_csv, "w", encoding="utf-8") as fh:
        fh.write(f"file;{os.path.basename(file_path)};sr;{SR};SNR_REF;{SNR_REF};THR_PCT;{THR_PCT};AGGR_MODE;{AGGR_MODE};AGGR_K;{AGGR_K}\n")

        # -------- Save CONTROL example plots + metrics
        for i, seg in enumerate(ex_ctrl, 1):
            save_path = os.path.join(out_dir, f"{base}_control_example_{i}.png")
            _plot_example_pair(save_path, ts, emg, snr_ctrl_db, amp_thr_ctrl, thr_ctrl_db, seg,
                               title_prefix=f"CONTROL ({int(THR_PCT*100)}% of {SNR_REF.upper()})", sr=SR)
            print("saved:", save_path)

            still, active = _extract_still_active_from_seg(seg, emg)
            fh.write(f"EXAMPLE;control;{i}\n")
            snr_vals, mav_perc, cv_perc, rms_perc = _write_gui_style_metrics_csv(fh, "", still, active)
            fh.write("\n")

        # -------- Save ANALYZER example plots + metrics
        for i, seg in enumerate(ex_an, 1):
            save_path = os.path.join(out_dir, f"{base}_analyzer_example_{i}.png")
            _plot_example_pair(save_path, ts, emg, snr_an_db, amp_thr_an, thr_an_db, seg,
                               title_prefix=f"ANALYZER p10 + 400ms guard ({int(THR_PCT*100)}% of {SNR_REF.upper()})", sr=SR)
            print("saved:", save_path)

            still, active = _extract_still_active_from_seg(seg, emg)
            fh.write(f"EXAMPLE;analyzer;{i}\n")
            snr_vals, mav_perc, cv_perc, rms_perc = _write_gui_style_metrics_csv(fh, "", still, active)
            fh.write("\n")

        # -------- Re-pick analyzer with anchor/labelsand plot WITH summary overlays
        ex_an_with_anchor = _pick_examples_nolabels(snr_an_db, thr_an_db, num=3, sr=SR,
                                                    min_gap_s=GAP_S, win_s=WIN_S, y=y, require_label_near=True)

        for i, seg in enumerate(ex_an_with_anchor, 1):
            (s0, e0), span = _expand_window_to_include_label_span((seg[0], seg[1]), y, seg[2], SR,
                                                                  pad_before_ms=200, pad_after_ms=300)
            seg_expanded = (s0, e0, seg[2], seg[3])
            save_path = os.path.join(out_dir, f"{base}_analyzer_with_labels_example_{i}.png")

            # Compute metrics and overlay summary
            still, active = _extract_still_active_from_seg(seg_expanded, emg)
            _, mav_perc, cv_perc, rms_perc = _write_gui_style_metrics_csv(fh, f"EXAMPLE;analyzer_with_labels;{i};", still, active)
            snr_vals = compute_snr(still, active) if still.size and active.size else np.array([])
            # Shapiro for STILL in this window (EMG only)
            #_, p_still = _shapiro_per_channel(still) if still.size else (np.array([]), np.array([]))
            summary = _summarize_metrics_for_text(snr_vals, mav_perc, cv_perc, rms_perc)

            _plot_example_pair_with_labels(save_path, ts, y, emg, snr_an_db, amp_thr_an, thr_an_db,
                                           seg_expanded,
                                           title_prefix=f"ANALYZER p10 + {400}ms guard ({int(THR_PCT*100)}% of {SNR_REF.upper()})",
                                           sr=SR, metrics_summary=summary)
            print("saved:", save_path)
            fh.write("\n")

        # ---- FULL-DATA lag by SNR xcorr
        shift_samp, shift_ms, full_info = _estimate_lag_full_dataset_snr(
            snr_an_db, thr_an_db, y, SR, dp_module=dp, max_lag_ms=1500, piecewise=True, seg_s=5.0, hop_s=2.5
        )
        dir_txt = ("forward (delay labels)" if shift_samp > 0
                   else "backward (advance labels)" if shift_samp < 0
                   else "no shift")
        print(f"[FULL SNR xcorr] shift = {shift_samp} samples ({shift_ms:+.1f} ms), {dir_txt}.")
        if "piecewise_median_ms" in full_info:
            med = full_info["piecewise_median_ms"]; mean = full_info["piecewise_mean_ms"]; std = full_info["piecewise_std_ms"]
            print(f"[Piecewise 5s/2.5s] median = {med:+.1f} ms, mean = {mean:+.1f} ms, std = {std:.1f} ms")

        # Apply shift to labels for visualization
        y_shifted = _shift_labels(y, -shift_samp)

        for i, seg in enumerate(ex_an_with_anchor, 1):
            save_path = os.path.join(out_dir, f"{base}_analyzer_with_labels_SHIFTED_example_{i}.png")

            # Metrics on the same raw EMG window
            still, active = _extract_still_active_from_seg(seg, emg)
            _, mav_perc, cv_perc, rms_perc = _write_gui_style_metrics_csv(fh, f"EXAMPLE;analyzer_with_labels_SHIFTED;{i};", still, active)
            snr_vals = compute_snr(still, active) if still.size and active.size else np.array([])
            #_, p_still = _shapiro_per_channel(still) if still.size else (np.array([]), np.array([]))
            summary = _summarize_metrics_for_text(snr_vals, mav_perc, cv_perc, rms_perc)

            _plot_example_pair_with_labels(
                save_path, ts, y, emg, snr_an_db, amp_thr_an, thr_an_db, seg,
                title_prefix=(f"ANALYZER p10 + 400ms guard ({int(THR_PCT*100)}% of {SNR_REF.upper()})  "
                              f"[Lag = {shift_ms:+.1f} ms from first example]"),
                sr=SR, y_shifted=y_shifted, metrics_summary=summary
            )
            print("saved:", save_path)

    status.config(text=f"Saved SNR examples + metrics to: {out_dir}", fg="green")





########################
# Filtering Functions  #
########################

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.
    :param lowcut: Lower cutoff frequency in Hz.
    :param highcut: Upper cutoff frequency in Hz.
    :param fs: Sampling frequency in Hz.
    :param order: Filter order.
    :return: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
    """
    Apply a bandpass filter to each channel of the data.
    :param data: 2D numpy array of shape (samples, channels).
    :param lowcut: Lower cutoff frequency in Hz (e.g., 20).
    :param highcut: Upper cutoff frequency in Hz (e.g., 450).
    :param fs: Sampling frequency in Hz.
    :param order: Filter order (e.g., 4).
    :return: Filtered 2D numpy array.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered_data

def notch_filter_50Hz(signal, fs=1000.0, quality=60.0):
    """Apply a 50 Hz notch filter to each column in 'signal'.
    :param signal: 2D numpy array of shape (samples, channels)
    :param fs: Sampling frequency (Hz)
    :param quality: Q-factor for the notch filter
    :return: Filtered 2D numpy array
    """
    # Design a notch filter at 50 Hz
    w0 = 50.0 / (fs / 2.0)  # Normalized frequency
    b, a = iirnotch(w0, quality)

    # Filter each channel
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signal[:, ch])

    return filtered


def teager_kaiser_energy(signal):
    """Apply the Teager-Kaiser Energy operator to each column.
    TKE formula for x[n]: y[n] = x[n]^2 - x[n+1]*x[n-1].
    We'll set boundary points (0, -1) to zero for simplicity.

    :param signal: 2D numpy array of shape (samples, channels)
    :return: 2D numpy array of TKE outputs
    """
    tke_out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        for n in range(1, len(x) - 1):
            tke_out[n, ch] = x[n] * x[n] - x[n + 1] * x[n - 1]
        # The first and last remain 0.
    return tke_out


def moving_average(signal, window_size=5):
    """Apply a simple moving average to each column.

    :param signal: 2D numpy array of shape (samples, channels)
    :param window_size: Number of samples for the moving average
    :return: 2D numpy array of smoothed outputs
    """
    smoothed = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        # Use 'same' mode so the output has the same length.
        smoothed[:, ch] = np.convolve(x, np.ones(window_size) / window_size, mode='same')
    return smoothed


# Serial reading function (runs in a separate thread)
def read_serial_data():
    global running, data_buffer, record_file, ser
        # This will hold the last-seen IMU reading
    latest_imu = None

    while running:
        if not serial_var.get():
            time.sleep(0.1)
            continue 
        try:
            # Read a line and parse it (expecting comma-separated floats)
            line = ser.readline().decode('utf-8', errors='replace').strip()
            # status.config(text=line)
            # print(line)
            if not line:
                continue
            parts = line.split('|')
            #print(parts)
            if len(parts) == 2:
                emg_csv, imu_csv = parts
                ts_us, seq = None, None
            elif len(parts) == 4:
                ts_us, seq, emg_csv, imu_csv = parts
            else:
                continue
            #print(emg_csv, imu_csv)
            parts_elec = emg_csv.split(',')
            parts_imu = imu_csv.split(',')
            # print(len(parts_elec), len(parts_imu))
            if len(parts_imu) == 6:
                # convert once, store for next ELEC
                latest_imu = [float(x) for x in parts_imu]
                # print(latest_imu)
                
            if len(parts_elec) == NUM_ELEC and latest_imu is not None:
                # Ensure we have data for all NUM_CH channels
                unshift = lambda a, k: a[k:] + a[:k]
                elec_floats = [float(x) for x in parts_elec]
                # print(elec_floats)
                # print(unshift(elec_floats,2))
                # print(elec_floats)

                new_row = unshift(elec_floats,2) + latest_imu
                
                # Append new data and keep a fixed-size buffer (last 1000 samples)
                data_buffer = np.vstack([data_buffer, new_row])
                if data_buffer.shape[0] > 1000:
                    # print(data_buffer.shape)
                    data_buffer = data_buffer[-1000:]
                if record_file is not None:
                    record_file.write(line + '\n')

        except (IOError, ValueError) as e:
            # status.config(text="Error writing to file: " + str(e), fg="red")
            break
        except Exception as e:
            status.config(text="Error reading/parsing data:" + str(e),fg= "red")
            # print("Error reading/parsing data:", e)
    # ser.write('!DC\r'.encode())
    # ser.close()

    if record_file is not None:
        record_file.close()

def select_data_file():
    global file_path, file_handle, file_running, file_thread
    path = filedialog.askopenfilename(
        title="Select Unity data file",
        filetypes=[("Text files","*.txt *.csv"),("All files","*.*")]
    )
    if not path:
        return
    file_path = path
    # open & seek to end
    file_handle = open(file_path, "r")
    # file_handle.seek(0, os.SEEK_END)
    file_running = True
    file_thread = threading.Thread(target=read_file_data, daemon=True)
    file_thread.start()
    status.config(text=f"Monitoring file: {os.path.basename(file_path)}", fg="blue")

def stop_file_feed():
    global file_running, file_thread
    if not file_running:
        return
    file_running = False
    if file_thread:
        file_thread.join(timeout=1.0)
    status.config(text="Stopped file monitoring.", fg="orange")

def read_file_data():
    global file_handle, data_buffer, file_running, gesture_buffer
    latest_imu_file = None
    while file_running:
        if not file_var.get():
            time.sleep(0.1)
            continue 
        line = file_handle.readline()
        if not line:
            time.sleep(0.1)
            continue
        parts = line.strip().split('|')
        if len(parts) == 5:
            parts = parts[:-1]  # discard timestamp
        if len(parts) != 4:
            continue
        x_time, gesture_str ,elec_str, imu_str =parts
        # print(x_time, gesture_str, elec_str, imu_str)
        gesture = int(gesture_str)
        elec_parts = elec_str.split(',')
        imu_parts  = imu_str.split(',')
        try:
            if len(imu_parts) == 6:
                latest_imu_file = [float(x) for x in imu_parts]
            if len(elec_parts) == NUM_ELEC and latest_imu_file is not None:
                new_row = [float(x) for x in elec_parts] + latest_imu_file
                with data_lock:
                    data_buffer = np.vstack([data_buffer, new_row])
                    gesture_buffer.append(gesture)
                    if data_buffer.shape[0] > 1000:
                        data_buffer = data_buffer[-1000:]
                        gesture_buffer = gesture_buffer[-1000:]
        except ValueError:
            continue
    file_handle.close()

# Button callback: start the feed
def start_feed():
    global running, record_file, last_metrics_store_time, metrics_file, ser, read_thread, _COM_PORT
    if running:
        return
    # 1) open the port
    if serial_var.get():
        try:
            ser = serial.Serial(_COM_PORT,250000 , timeout=1,
                                dsrdtr=False, rtscts=False)  # disable hardware handshakes
            time.sleep(5)                                     # wait for USB-CDC stabilization
            ser.reset_input_buffer()                          # clear stale data
            ser.reset_output_buffer()                              # allow command to flush 
            ser.write(b'!connect\r')    
        except Exception as e:
            tk.messagebox.showerror("Serial Error", f"Could not open {_COM_PORT}:\n{e}")
            # status.config(text="Error opening serial port:" + str(e),fg= "red") 
            # print("Error opening serial port:", e)
            return
    if file_var.get():
        select_data_file()
        if file_path is None:
            status.config(text="No file selected.",fg= "red")
            return
        # Open the file and seek to
    
    running = True
    # Start the serial reading in a new daemon thread
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    r_filename = "Assets\CUSTOM\BBH_ELECTRODES\TGRT_dataLog.txt"
    record_file = open(r_filename, 'w')
    print(f"Recording to {r_filename}")

    metrics_filename = f"Assets\CUSTOM\BBH_ELECTRODES\metrics\Metrics_{now_str}.csv"
    metrics_file = open(metrics_filename, "w")
    # Write header for metrics file
    metrics_file.write("Timestamp;")
    metrics_file.write("\n")
    last_metrics_store_time = time.time()

    read_thread = threading.Thread(target=read_serial_data, daemon=True)
    read_thread.start()

    status.config(text="Started data feed from serial port." + str(r_filename),fg= "green")
    # print(f"Started data feed from serial port. Recording to {r_filename}")


# Button callback: stop the feed
def stop_feed():
    global running, ser, read_thread, record_file, metrics_file
    
    if not running:
        return
    running = False
    # 1) tell the board to stop streaming
    try:
        ser.write(b'!DC\r')
    except Exception as e:
        status.config(text="Error sending DC command:" + str(e),fg= "red")
        # print("Error sending DC command:", e)

    # 2) stop the reader loop
    running = False
    # 3) wait for thread to finish
    if read_thread is not None:
        read_thread.join(timeout=1.0)

    # 4) close files
    if record_file:
        record_file.close()
    if metrics_file:
        metrics_file.close()

    # 5) close serial port
    try:
        time.sleep(0.1)                       # allow command to flush
        ser.reset_input_buffer()              # drop any incoming data
        ser.reset_output_buffer()             # discard any outgoing data
        ser.close()                           # close port handle
        time.sleep(0.5)   
    except:
        pass

    ser = None
    read_thread = None
    serial_var.set(False)
    file_var.set(False)
    stop_file_feed()
    # file_running = False  
    status.config(text="Stopped data feed.",fg= "orange")
    # print("Stopped data feed.")

# Button callback: toggle metric computation/display
def toggle_metrics():
    global toggle_metric
    toggle_metric = not toggle_metric
    status.config(text="Toggle Metrics is now " + ("ON" if toggle_metric else "OFF"),fg= "green")
    # print("Toggle Metrics is now", "ON" if toggle_metric else "OFF")

############################################################
#             SNR Measurement Button/Logic                 #
############################################################

def snr_button():
    """
    Cycles through STILL -> ACTIVE -> DONE states to compute SNR.
    1) First press: start STILL phase
    2) Second press: start ACTIVE phase
    3) Third press: compute SNR and display
    4) Fourth press: resets to idle
    """
    global snr_state, still_data, active_data
    if snr_state == "idle":
        snr_state = "still"
        still_data = np.empty((0, NUM_CH))
        snr_label.config(text="SNR State: Collecting STILL data... (Press again for ACTIVE)")
    elif snr_state == "still":
        snr_state = "active"
        active_data = np.empty((0, NUM_CH))
        snr_label.config(text="SNR State: Collecting ACTIVE data... (Press again to compute SNR)")
    elif snr_state == "active":
        # Compute SNR
        snr_state = "done"
        snr_values = compute_snr(still_data, active_data)
        s_rms_vals, s_cv_vals, s_mav_vals = compute_metrics(still_data)
        a_rms_vals, a_cv_vals, a_mav_vals = compute_metrics(active_data)

        # Display SNR results
        text_lines = ["SNR Results:"]
        for ch in range(NUM_CH):
            text_lines.append(f"Ch {ch+1}: {snr_values[ch]:.2f} dB")
        snr_label.config(text="\n".join(text_lines) + "\n(Press again to reset)")
    
        # Store SNR results to metrics file
        if metrics_file is not None:
            
            # snr_line = f"SNR;{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')};"
            metrics_file.write("still_data\n")

            start_still = still_data[:10]

            metrics_file.write("start\n")
            for ch in range(NUM_CH):
                for value in start_still[:, ch]:
                    metrics_file.write(f"{value};")
                metrics_file.write("\n")

            metrics_file.write("end\n")
            end_still = still_data[-10:]

            for ch in range(NUM_CH):
                for value in end_still[:, ch]:
                    metrics_file.write(f"{value};")
                metrics_file.write("\n")
            metrics_file.write("\n")
            ts_us = time.time_ns() // 1_000
            metrics_file.write(f"{ts_us};active_data\n")
            start_active = active_data[:10]
            metrics_file.write("start\n")
            for ch in range(NUM_CH):
                for value in start_active[:, ch]:
                    metrics_file.write(f"{value};")
                metrics_file.write("\n")
            
            metrics_file.write("end\n")
            snr_line = f"SNR;{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')};"
            end_active = active_data[-10:]
            for ch in range(NUM_CH):
                for value in end_active[:, ch]:
                    metrics_file.write(f"{value};")  

            metrics_file.write("\n")

            metrics_file.write("snr_values\n")
            for val in snr_values:
                snr_line += f"{val:.2f};"
            snr_line = snr_line.rstrip(';') + "\n"
            metrics_file.write(snr_line)
            general_a_max_mav = np.max(a_mav_vals) if np.max(a_mav_vals) != 0 else 1e-12

            for ch in range(NUM_CH):
                    curr_mav_sdata = s_mav_vals[ch] 
                    curr_mav_adata = a_mav_vals[ch]
                    curr_cv_sdata = s_cv_vals[ch]
                    curr_cv_adata = a_cv_vals[ch]
                    curr_rms_sdata = s_rms_vals[ch]
                    curr_rms_adata = a_rms_vals[ch]

                    s_mean_mav = np.mean(curr_mav_sdata) if np.mean(curr_mav_sdata) != 0 else 1e-12
                    a_max_mav = np.max(curr_mav_adata) if np.max(curr_mav_adata) != 0 else 1e-12
                    s_mean_cv = np.mean(curr_cv_sdata) if np.mean(curr_cv_sdata) != 0 else 1e-12
                    a_max_cv = np.max(curr_cv_adata) if np.max(curr_cv_adata) != 0 else 1e-12
                    s_mean_rms = np.mean(curr_rms_sdata) if np.mean(curr_rms_sdata) != 0 else 1e-12
                    a_max_rms = np.max(curr_rms_adata) if np.max(curr_rms_adata) != 0 else 1e-12
                    
                    mav_perc = (a_max_mav / s_mean_mav) * 100
                    general_mav_perc = (a_max_mav / general_a_max_mav) * 100
                    cv_perc = (a_max_cv / s_mean_cv) * 100
                    rms_perc = (a_max_rms / s_mean_rms) * 100

                    perc_line = f"Ch{ch+1}_MAV_perc;{mav_perc:.2f};Ch{ch+1}_MAV_general_perc;{general_mav_perc:.2f};Ch{ch+1}_CV_perc;{cv_perc:.2f};Ch{ch+1}_RMS_perc;{rms_perc:.2f}\n"
                    metrics_file.write(perc_line)

            metrics_file.flush()
    else:
        # snr_state == "done"
        snr_state = "idle"
        snr_label.config(text="SNR State: Idle. (Press to start STILL)")

def compute_snr(still_arr, active_arr):
    """
    Compute SNR = 20*log10(RMS_active / RMS_still) for each channel.
    If still or active arrays are empty, returns an array of NaNs.
    """
    if still_arr.shape[0] == 0 or active_arr.shape[0] == 0:
        return np.full((NUM_CH,), np.nan)

    # RMS of STILL
    rms_still = np.sqrt(np.mean(still_arr**2, axis=0))
    # RMS of ACTIVE
    rms_active = np.sqrt(np.mean(active_arr**2, axis=0))

    # Avoid division by zero
    rms_still[rms_still == 0] = 1e-12

    # SNR in dB
    snr_db = 20 * np.log10(rms_active / rms_still)
    return snr_db

############################################################
#          Metrics Computation (RMS, CV, MAV)              #
############################################################

def compute_metrics(data):
    """
    Computes RMS, CV, MAV for each channel over 'data'.
    Returns arrays of shape (NUM_CH,).
    """
    # If data is empty, return zeros
    if data.shape[0] == 0:
        ch_count = data.shape[1] if data.shape[1] else NUM_CH
        return (np.zeros(ch_count), np.zeros(ch_count), np.zeros(ch_count))

    # RMS
    rms = np.sqrt(np.mean(data**2, axis=0))

    # CV = std(x) / mean(x)  (watch for zero mean)
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    # To avoid division by zero
    mean_vals[mean_vals == 0] = 1e-12
    cv = std_vals / np.abs(mean_vals)

    # MAV = mean(|x|)
    mav = np.mean(np.abs(data), axis=0)

    return (rms, cv, mav)



# ------------------------------------------------------------------------------
# 1) Add a menubar with “Windows → Servo Controller”
# ------------------------------------------------------------------------------
def make_menubar(root):
    # Reuse existing menu if present, otherwise create one

    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Keep (or create) Tools menu (Servo Controller etc.)
    tools_menu = tk.Menu(menubar, tearoff=False)
    tools_menu.add_command(label="Servo Controller", command=lambda: open_servo_window(root))
    menubar.add_cascade(label="Tools", menu=tools_menu)

    # NEW: Analysis menu
    # analysis_menu = tk.Menu(menubar, tearoff=False)
    tools_menu.add_command(
        label="Gesture 3: SNR vs Labels (plots + lag-corrected)",
        command=lambda: _run_modal(
            "Executing analysis",
            "Generating SNR/label plots… please wait",
            _compare_rest_analyzer_vs_control
        )
    )
    # menubar.add_cascade(label="Analysis", menu=analysis_menu)
    



def open_servo_window(root):
    global _COM_PORT
    # Disable main window while this is open
    root.attributes('-disabled', True)
    stop_feed()  # Stop the feed if it's running

    servo_win = tk.Toplevel(root)
    servo_win.title("Servo Controller v1.0 ARDUINO")
    servo_win.protocol("WM_DELETE_WINDOW", lambda: close_servo_window(root, servo_win))

    frame1     = tk.LabelFrame(servo_win,
                               text='Servo Angle (in Degrees)',
                               padx=5, pady=52,
                               bg='black', fg='white')
    frame2     = tk.LabelFrame(servo_win,
                               text='Controls',
                               padx=1, pady=1)
    sliders    = tk.LabelFrame(frame2, text='Sliders')
    test_servo = tk.LabelFrame(servo_win,
                               text='Test Servo',
                               padx=1, pady=10)

    frame1.grid(    row=0, column=0,           sticky='nsew')
    frame2.grid(    row=0, column=1, columnspan=2, sticky='ew')
    sliders.grid(   row=1, column=1, columnspan=3, pady=10, sticky='ew')
    test_servo.grid(row=2, column=0, columnspan=3, sticky='ew')

    # Try opening the servo serial port
    try:
        servo_serial = serial.Serial(_COM_PORT, 2000000, timeout=None)
    except Exception as e:
        tk.messagebox.showerror("Serial Error", f"Could not open {_COM_PORT}:\n{e}")
        close_servo_window(root, servo_win)
        return

    # Prepare 16 IntVars + the left-pane live-value labels
    vals = [tk.IntVar(value=0) for _ in range(16)]
    val_labels = []
    for i in range(16):
        tk.Label(frame1,
                 text=f"Servo {i+1}:",
                 bg='black', fg='white'
        ).grid(row=i, column=0, sticky='w', padx=5, pady=2)

        vl = tk.Label(frame1,
                      text='0',
                      bg='black', fg='yellow'
        )
        vl.grid(row=i, column=1, sticky='w', padx=5, pady=2)
        val_labels.append(vl)

    # Function to send all 16 values whenever any slider moves
    def send_all(idx, value):
        vals[idx].set(int(value))
        val_labels[idx].configure(text=value)
        packet = "!UD" + "".join(f" {v.get()}" for v in vals) + "\r"
        try:
            servo_serial.write(packet.encode())
        except Exception as e:
            status.config(text="Servo write failed:" + str(e),fg= "red")
            # print("Servo write failed:", e)

    # Create 16 sliders in 4 columns (4 sliders per column) in the Sliders pane
    for i in range(16):
        row = i % 4  # Determine the row (0-3)
        col = i // 4  # Determine the column (0-3)
        s = tk.Scale(sliders,
                     from_=0, to=190,
                     orient='vertical',
                     variable=vals[i],
                     command=lambda v, idx=i: send_all(idx, v),
                     repeatdelay=500
        )
        s.grid(row=row, column=col, sticky='we', padx=2)
        sliders.columnconfigure(col, weight=1)
    # Keep the serial port handle around so we can close it
    servo_win.servo_serial = servo_serial

def close_servo_window(root, servo_win):
    # close serial port
    try:
        servo_win.servo_serial.close()
    except:
        pass
    servo_win.destroy()
    # re-enable main window
    root.attributes('-disabled', False)

# Variable for lowpass filter toggle
lowpass_var = tk.BooleanVar(value=False)
# Create a matplotlib figure for plotting
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Initialize plot lines for NUM_CH channels
lines = []
colors = plt.cm.viridis(np.linspace(0, 1, NUM_CH))
for ch in range(NUM_CH):
    (line,) = ax.plot([], [], color=colors[ch], label=f"Channel {ch+1}")
    lines.append(line)

ax.set_xlim(0, 1000)  # Display last 1000 samples on X-axis

ax.set_ylim(-500, 500)
#ax.set_ylim(-4000, 4000000)

ax.set_xlabel("Sample")
ax.set_ylabel("Filtered Value")
ax.legend(loc="upper right")
serial_var = tk.BooleanVar(value=True)   # “Serial” checkbox
file_var   = tk.BooleanVar(value=False)  # “File”   checkbox

# Create a control frame for the Start, Stop, and Toggle Metrics buttons
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

# Create a new frame for the status bar below everything
status_frame = ttk.Frame(root)
status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, before=control_frame)

# Status bar at bottom, just like Servo_Controller did
status = tk.Label(status_frame,
                  bd=1, relief='sunken',
                  anchor='w', text='Status')
status.pack(side=tk.LEFT, fill=tk.X, expand=True)

start_btn = ttk.Button(control_frame, text="Start Feed", command=start_feed)
start_btn.pack(side=tk.LEFT, padx=5)

stop_btn = ttk.Button(control_frame, text="Stop Feed", command=stop_feed)
stop_btn.pack(side=tk.LEFT, padx=5)

toggle_metrics_btn = ttk.Button(control_frame, text="Toggle Metrics", command=toggle_metrics)
toggle_metrics_btn.pack(side=tk.LEFT, padx=5)

lowpass_chk = ttk.Checkbutton(control_frame, text="Bandpass Filter", variable=lowpass_var)
lowpass_chk.pack(side=tk.LEFT, padx=5)

serial_chk = ttk.Checkbutton(
    control_frame,
    text="Use Serial",
    variable=serial_var
)
serial_chk.pack(side=tk.LEFT, padx=5)

file_chk = ttk.Checkbutton(
    control_frame,
    text="Use File",
    variable=file_var
)
file_chk.pack(side=tk.LEFT, padx=5)

# Button for SNR measurement
snr_btn = ttk.Button(control_frame, text="SNR Measurement", command=lambda: snr_button())
snr_btn.pack(side=tk.LEFT, padx=5)

# Create a frame for channel toggles (checkboxes)
toggle_frame = ttk.Frame(root)
toggle_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# Metrics display frame
metrics_frame = ttk.Frame(root)
metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

# SNR view toggle + threshold % entry + comparison button
snr_chk = ttk.Checkbutton(control_frame, text="SNR View", variable=snr_view_var)
snr_chk.pack(side=tk.LEFT, padx=5)

ttk.Label(control_frame, text="thr % of max").pack(side=tk.LEFT, padx=(10,2))
thr_entry = ttk.Entry(control_frame, width=5, textvariable=thr_percent_var)
thr_entry.pack(side=tk.LEFT, padx=2)

# compare_btn = ttk.Button(control_frame,
#                          text="Compare Rest (Analyzer vs First 1000)",
#                          command=_compare_rest_analyzer_vs_control)
# compare_btn.pack(side=tk.LEFT, padx=8)

lock_chk = ttk.Checkbutton(control_frame, text="Lock thresholds", variable=thr_lock_var)
lock_chk.pack(side=tk.LEFT, padx=4)


make_menubar(root)

# Initialize toggles and RMS labels for each channel
for ch in range(NUM_CH):
    var = tk.BooleanVar(value=True)
    channel_vars.append(var)
    # Checkbutton to toggle channel visibility
    chk = ttk.Checkbutton(toggle_frame, text=f"Channel {ch+1}", variable=var)
    chk.pack(anchor='w')
    
    # Label to display the Metrics values for the channel
    lbl = ttk.Label(metrics_frame, text=f"Channel {ch+1} RMS: N/A")
    lbl.pack(anchor='w')
    metrics_labels.append(lbl)

# Label to show SNR state and results
snr_label = ttk.Label(root, text="SNR State: Idle. (Press to start STILL)")
snr_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

snippet_count = 0
max_snippets   = 10
output_dir     = "gesture_snippets"

os.makedirs(output_dir, exist_ok=True)

def update(frame):
    global data_buffer, still_data, active_data
    global snr_state, last_metrics_store_time
    global vlines, last_gesture, snippet_count, max_snippets

    # 1) clear old live‐plot v‐lines
    for ln in vlines:
        ln.remove()
    vlines.clear()

    # 2) nothing yet?
    if data_buffer.shape[0] == 0:
        return lines + vlines

    # 3) optional band‐pass filtering
    if lowpass_var.get():
        elec = data_buffer[:, :NUM_ELEC]
        imu  = data_buffer[:, NUM_ELEC:]
        bp   = bandpass_filter(elec, 20, 450, fs=1000, order=4)
        n    = min(bp.shape[0], imu.shape[0])
        data_to_plot = np.concatenate([bp[-n:], imu[-n:]], axis=1)
    else:
        data_to_plot = data_buffer

    # 4) update the live EMG/IMU traces
    # --- RAW PLOT with persistent SNR-derived thresholds overlaid ---
    x = np.arange(data_to_plot.shape[0])
    emg = data_to_plot[:, :NUM_ELEC]
    if file_var.get() and gesture_buffer:
        # # Ensure persistent threshold lines exist (created once, reused)
        _ensure_threshold_lines(ax)

        # == Build/refresh SNR stats used ONLY to position thresholds ==
        global noise_rms_control, max_snr_db_per_ch, thr_db_per_ch, last_thr_percent_val

        # Establish control noise baseline from the first REST_CONTROL_LEN samples (if not yet set)
        if noise_rms_control is None and emg.shape[0] >= max(32, min(REST_CONTROL_LEN, emg.shape[0])):
            n0 = min(REST_CONTROL_LEN, emg.shape[0])
            noise_rms_control = np.sqrt(np.mean(emg[:-n0, :]**2, axis=0))

        # Running RMS and SNR(dB) (not plotted; only for thresholds)
        rms = _running_rms(emg, SNR_WIN)
        denom = noise_rms_control if noise_rms_control is not None else (
            np.std(emg[:max(64, emg.shape[0]//4), :], axis=0) + 1e-6
        )
        snr_lin = np.maximum(rms, 1e-9) / np.maximum(denom[None, :], 1e-9)
        snr_db  = 20.0 * np.log10(snr_lin)

        # Track per-channel peak SNR(dB) unless locked
        if max_snr_db_per_ch is None:
            max_snr_db_per_ch = np.full(NUM_ELEC, -np.inf, dtype=np.float32)
        if not thr_lock_var.get():
            max_snr_db_per_ch = np.maximum(max_snr_db_per_ch, np.max(snr_db, axis=0))

        # Recompute SNR(dB) thresholds when % changes (thr = % * max SNR in dB)
        pct = float(thr_percent_var.get())  # e.g. 0.40
        if (thr_db_per_ch is None) or (not np.isclose(pct, last_thr_percent_val)):
            last_thr_percent_val = pct
            thr_db_per_ch = pct * max_snr_db_per_ch

        # Convert SNR thresholds (dB) -> raw-amplitude RMS thresholds
        amp_thr = None
        if noise_rms_control is not None and thr_db_per_ch is not None:
            amp_thr = noise_rms_control * (10.0 ** (thr_db_per_ch / 20.0))

    # ----- Plot RAW signals -----
    for ch, line in enumerate(lines):
        if channel_vars[ch].get():
            line.set_visible(True)
            line.set_data(x, data_to_plot[:, ch])
        else:
            line.set_visible(False)

    ax.set_ylabel("Filtered Value")




    # 5) handle gesture edges & snippet saving
    if file_var.get() and gesture_buffer:
        # # ----- Overlay per-channel horizontal threshold lines (in RAW amplitude) -----
        # # Show a single positive horizontal line per visible channel.

        for c in range(NUM_ELEC):
            show = channel_vars[c].get()
            thr_lines[c].set_visible(show and amp_thr is not None and np.isfinite(amp_thr[c]))
            if thr_lines[c].get_visible():
                ythr = float(amp_thr[c])
                thr_lines[c].set_ydata([ythr, ythr])
        g   = gesture_buffer[-1]
        idx = data_to_plot.shape[0] - 1  # <— define the current sample index

        # only act on 0→1 rising edge, and up to max_snippets
        if g == 3 and last_gesture != 3 and snippet_count < max_snippets:
            # live‐plot entry line
            ln_ent = ax.axvline(idx, color='red', linestyle='--')
            vlines.append(ln_ent)

            # extract ±512 samples around idx
            start_idx = max(idx - 512, 0)
            end_idx   = min(idx + 512 + 1, data_to_plot.shape[0])
            seg       = data_to_plot[start_idx:end_idx]
            ges_seg   = gesture_buffer[start_idx:end_idx]

            # compute relative time in µs (1 sample = 1ms = 1000µs)
            rel_t = np.arange(len(seg)) * 1e3

            # find entry/exit within snippet
            nz = np.where(ges_seg != 0)[0]
            entry_i = nz[0] if nz.size else None
            # first zero after that entry
            exit_i = None
            if entry_i is not None:
                zeros_after = np.where(ges_seg[entry_i:] == 0)[0]
                if zeros_after.size:
                    exit_i = entry_i + zeros_after[0]

            # plot raw channels
            fig, ax_ex = plt.subplots()
            for chn in range(NUM_ELEC):
                ax_ex.plot(rel_t, seg[:, chn], label=f'EMG ch{chn+1}')
            for j in range(seg.shape[1] - NUM_ELEC):
                ax_ex.plot(rel_t, seg[:, NUM_ELEC + j],
                           linestyle=':', label=f'IMU ch{j+1}')

            # draw start/end lines in snippet
            if entry_i is not None:
                ax_ex.axvline(rel_t[entry_i],
                              color='red', linestyle='--',
                              label='Gesture start')
            if exit_i is not None:
                ax_ex.axvline(rel_t[exit_i],
                              color='blue', linestyle='--',
                              label='Gesture end')
            ax_ex.set_ylim(-500, 500)
            ax_ex.set_xlabel('Time (µs)')
            ax_ex.set_ylabel('Signal amplitude')
            ax_ex.set_title(f'Gesture 1 example #{snippet_count+1}')
            ax_ex.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()

            # save & close
            fname = os.path.join(output_dir,
                                 f"snippet_{snippet_count+1}.png")
            fig.savefig(fname)
            plt.close(fig)

            snippet_count += 1

        # mark live‐plot exit line if falling edge
        elif g == 0 and last_gesture == 1:
            ln_exit = ax.axvline(idx, color='blue', linestyle='--')
            vlines.append(ln_exit)

        last_gesture = g

    # 6) accumulate for SNR
    if snr_state == "still":
        still_data  = np.vstack([still_data, data_to_plot])
    elif snr_state == "active":
        active_data = np.vstack([active_data, data_to_plot])

    # 7) update metrics
    if toggle_metric:
        rms_vals, cv_vals, mav_vals = compute_metrics(data_to_plot)
        for ch, lbl in enumerate(metrics_labels):
            lbl.config(text=(
                f"Ch {ch+1} -> RMS: {rms_vals[ch]:.2f}; "
                f"CV: {cv_vals[ch]:.2f}; MAV: {mav_vals[ch]:.2f}"
            ))
    if file_var.get() and gesture_buffer:    
        lines_result = lines + vlines + thr_lines

    else:
        lines_result = lines
    return lines

# Create the animation (updates every 30 ms)
ani = animation.FuncAnimation(fig, update, interval=60, blit=True)
root.mainloop()
