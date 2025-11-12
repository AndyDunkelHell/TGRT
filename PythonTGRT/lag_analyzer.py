"""
Lag analyzer: estimate and visualize label-to-signal lag using SNR-based activity,
and (optionally) export example plots around detected onsets.

Constants
---------
SNR_REF : {"p95","max"}
    Per-channel SNR reference for thresholds (default "p95").
THR_PCT : float
    Fraction of the reference used as channel threshold (default 0.80).
AGGR_MODE : {"kofn","any"}
    Channel aggregation rule for over-threshold detection (default "kofn").
AGGR_K : int
    k for the k-of-N rule (default 3).
SNR_WIN : int
    Running RMS window (samples) for SNR (default 64).
SR : int
    Sampling rate (Hz) for timing (default 500).
WIN_S : float
    Plot window length in seconds (default 1.5).
GAP_S : float
    Minimum gap between picked examples in seconds (default 2.0).
BELOW_MS : int
    Duration mask must stay below threshold to mark "end" (ms, default 80).
BRIDGE_MS : int
    Bridge False gaps shorter than this between True runs (ms, default 150).
NEAR_LABEL_MS : int
    Keep SNR events only if near a label-3 span (±ms, default 2000).
COLOR_* / LW_* :
    Styling for plot markers and line widths.

Functions
---------
_expand_window_to_include_label_span(
    seg: tuple[int,int], y: np.ndarray, anchor_idx: int, sr: int,
    pad_before_ms: int = 200, pad_after_ms: int = 300
) -> tuple[tuple[int,int], tuple[int,int] | None]
    Expand (s0,e0) to fully include the nearest label==3 span around anchor_idx
    with padding. Returns ((s0_new,e0_new), (span_start,span_end)) or ((s0,e0), None).

_find_snr_end_from_mask(
    mask: np.ndarray, anchor: int, sr: int, below_ms: int = 60
) -> int
    First index at/after anchor where mask becomes False and stays False for
    ≥ below_ms.

_nearest_label_span_distance(anchor_idx: int, y: np.ndarray) -> int | None
    Distance in samples from anchor to the nearest label==3 span (0 if inside);
    None if no spans.

_plot_snr_lines_robust(
    ax, t: np.ndarray, snr_win: np.ndarray, clip_db: tuple[float,float] = (-20.0, 40.0)
) -> None
    Plot multi-channel SNR(dB) robustly (handles NaNs/inf and autoscaling).

_pick_examples_nolabels(
    snr_db: np.ndarray, thr_db_per_ch: np.ndarray, num: int = 3, sr: int = 1000,
    min_gap_s: float = 2.0, win_s: float = 1.5, y: np.ndarray | None = None,
    require_label_near: bool = True
) -> list[tuple[int,int,int,int]]
    Pick example segments anchored at bridged mask rising edges.
    Returns list of (s0, e0, anchor_idx, end_idx); filters by proximity to
    label==3 spans when require_label_near is True.

_over_mask(
    snr_db: np.ndarray, thr_db_per_ch: np.ndarray, mode: str = "kofn", k: int = 3
) -> np.ndarray
    Boolean series where per-time over-threshold condition holds across channels
    using "any" or "kofn" aggregation.

_bridge_mask_gaps(mask: np.ndarray, bridge_ms: int, sr: int) -> np.ndarray
    Bridge short False gaps (≤ bridge_ms) between True runs.

_plot_example_pair(
    save_path: str, ts: np.ndarray, emg: np.ndarray,
    snr_db: np.ndarray, amp_thr: np.ndarray, thr_db: np.ndarray,
    seg: tuple, title_prefix: str, sr: int = 1000
) -> None
    Save a two-panel figure (RAW with amplitude thresholds; SNR(dB) with SNR
    thresholds) for the given segment. Uses ts (µs) if available; otherwise sr.

_plot_example_pair_with_labels(
    save_path: str, ts: np.ndarray, y: np.ndarray, emg: np.ndarray,
    snr_db: np.ndarray, amp_thr: np.ndarray, thr_db: np.ndarray,
    seg: tuple, title_prefix: str, sr: int = 1000,
    rest_bounds: tuple | None = None, y_shifted: np.ndarray | None = None
) -> None
    Like _plot_example_pair, additionally overlays label markers/step bands for
    original labels and (optionally) a shifted label sequence.

SNR_lag(path: str, plot: bool = False) -> int
    High-level entry point. Loads a Unity log via data_pipeline._load_log_strict,
    estimates per-channel noise from rest (p10 with ±400 ms guards), computes
    SNR(dB), derives thresholds (THR_PCT * SNR_REF), and estimates dataset-level
    label lag using cross-correlation. If plot=True, saves example figures
    (…/<basename>_snr_examples/). Returns the lag in samples
"""


# Lag analyzer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import datetime
import time
from scipy.signal import iirnotch, filtfilt, butter
import os 
from matplotlib import lines as mlines
import data_pipeline as dp
from matplotlib.collections import LineCollection



# --- SNR detection settings ---
SNR_REF = "p95"        # 'p95' or 'max' for per-channel reference
THR_PCT = 0.80     # fraction of reference to define channel thresholds
AGGR_MODE = "kofn"     # 'kofn' or 'any'
AGGR_K    = 3          # k for k-of-N (e.g., 3 of 12 channels)
SNR_WIN   = 64         # RMS window (samples). Lower (e.g., 32) => quicker edges.
SR = 500          # assumed sampling rate (Hz); adjust if different
WIN_S   = 1.5                 # length of each example plot window
GAP_S   = 2.0                 # minimum gap between examples

# --- SNR end/bridging settings ---
BELOW_MS  = 80    # how long mask must stay below thr to mark "end"
BRIDGE_MS = 150   # fill gaps below thr shorter than this (bridges small dips)

NEAR_LABEL_MS = 2000   # only keep SNR events within ±2 s of a label-3 span

# --- Marker styling for plots ---
COLOR_SNR_START = "tab:orange"
COLOR_SNR_END   = "tab:red"
COLOR_LABEL_ON  = "tab:green"
COLOR_LABEL_OFF = "tab:purple"
LW_MAJOR = 2.6   # thicker lines for SNR markers
LW_MINOR = 2.2   # thicker lines for label markers

# Shifted (lag-corrected) label markers
COLOR_LABEL_ON_SHIFT  = "green"   # solid green (corrected start)
COLOR_LABEL_OFF_SHIFT = "purple"  # solid purple (corrected end)
LW_SHIFT = 2.8


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

    
def _pick_examples_nolabels(snr_db: np.ndarray,
                            thr_db_per_ch: np.ndarray,
                            num: int = 3, sr: int = 1000,
                            min_gap_s: float = 2.0,
                            win_s: float = 1.5,
                            y: np.ndarray | None = None,
                            require_label_near: bool = True):
    """
    Anchor at the first rising edge of the *bridged* k-of-N mask.
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
    
def _plot_example_pair_with_labels(save_path: str,
                                   ts: np.ndarray, y: np.ndarray,
                                   emg: np.ndarray,
                                   snr_db: np.ndarray,
                                   amp_thr: np.ndarray,
                                   thr_db: np.ndarray,
                                   seg, title_prefix: str,
                                   sr: int = 1000,
                                   rest_bounds: tuple | None = None,
                                   y_shifted: np.ndarray | None = None):
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
                    # If you also provide a shifted label array, plot it as a second (slightly lower) band
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

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def SNR_lag(path, plot = False):
    ts, y, emg, imu = dp._load_log_strict(path, 12, 6)
    # 3) Lag compensation -----------------------------------------------      
    # --- Analyzer rest baseline (labels only for BASELINE, not for detection)
    print(f"Analyzing file {path} with the SNR lag algorithm")
    noise_rms_analyzer = dp.rest_noise_rms_from_labels(
                y=y, emg=emg, sr=500, rest_win=256, rest_hop=128,
                guard_ms=400, method="percentile", percentile=10.0
            )
    snr_an_db    = dp._compute_snr_db(emg, noise_rms_analyzer, rms_win=SNR_WIN)
    thr_an_db    = dp._per_channel_thresholds(snr_an_db, pct=THR_PCT, ref=SNR_REF)
    amp_thr_an   = noise_rms_analyzer * (10.0 ** (thr_an_db / 20.0))
        
    shift_samp, shift_ms, full_info = dp._estimate_lag_full_dataset_snr(
                snr_an_db, thr_an_db, y, SR, max_lag_ms=1500, seg_s=5.0, hop_s=2.5
    )

    ex_an   = _pick_examples_nolabels(snr_an_db,   thr_an_db,   num=3, sr=SR,
                                  min_gap_s=GAP_S, win_s=WIN_S, y=y, require_label_near=True)
    if plot == True:
        # Output dir next to the file
        base    = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(os.path.dirname(path), f"{base}_snr_examples")
        os.makedirs(out_dir, exist_ok=True)
    
        for i, seg in enumerate(ex_an, 1):
            save_path = os.path.join(out_dir, f"{base}_analyzer_example_{i}.png")
            # analyzer examples (plain)
            _plot_example_pair(save_path, ts, emg, snr_an_db, amp_thr_an, thr_an_db, seg,
                            title_prefix=f"ANALYZER p10 + 400ms guard ({int(THR_PCT*100)}% of {SNR_REF.upper()})", sr=SR)
            print("saved:", save_path)
        
        ex_an_with_anchor = _pick_examples_nolabels(snr_an_db, thr_an_db, num=3, sr=SR,
                                                    min_gap_s=GAP_S, win_s=WIN_S, y=y, require_label_near=True)
    
        for i, seg in enumerate(ex_an_with_anchor, 1):
            s0, e0, anchor_idx, end_idx = seg
    
            # Find the nearest label span around anchor_idx
            (s0_new, e0_new), span = _expand_window_to_include_label_span(
                (s0, e0), y, anchor_idx, SR, pad_before_ms=200, pad_after_ms=300
            )
            seg_expanded = (s0_new, e0_new, anchor_idx, end_idx)
    
    
            save_path = os.path.join(out_dir, f"{base}_analyzer_with_labels_example_{i}.png")
            # analyzer with labels
            _plot_example_pair_with_labels(save_path, ts, y, emg, snr_an_db, amp_thr_an, thr_an_db, seg_expanded,
                                        title_prefix=f"ANALYZER p10 + {400}ms guard ({int(THR_PCT*100)}% of {SNR_REF.upper()})",
                                        sr=SR)
            print("saved:", save_path)
        y_shifted = dp._shift_labels(y, -shift_samp)
        for i, seg in enumerate(ex_an_with_anchor, 1):
            save_path = os.path.join(out_dir, f"{base}_analyzer_with_labels_SHIFTED_example_{i}.png")
            _plot_example_pair_with_labels(
                save_path, ts, y, emg, snr_an_db, amp_thr_an, thr_an_db, seg,
                title_prefix=(f"ANALYZER p10 + 400ms guard "
                              f"({int(THR_PCT*100)}% of {SNR_REF.upper()})  "
                              f"[Lag = {shift_ms:+.1f} ms from first example]"), 
                sr=SR, y_shifted=y_shifted
            )
            print("saved:", save_path)

    
    return shift_samp
