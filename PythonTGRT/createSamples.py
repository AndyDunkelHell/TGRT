"""
Create a C header with normalized test windows and train-time normalization stats
(mu and sigma) for use in TFLite Micro firmware. Also embeds FIR band-pass taps and a
moving-average window length to mirror training preprocessing.

Config
------
DATA_FILE   : str   # raw/test log used to draw samples
FILT_FILE   : str   # raw log for SNR lag analysis
STATS_FILE  : str   # npz with 'mu' and 'sigma' from training
WINDOW_SIZE : int
N_CHANNELS  : int
PATCH_LEN   : int
NUM_SAMPLES_TO_EXTRACT : int
OUTPUT_FILENAME : str
SR_HZ, BP_LO_HZ, BP_HI_HZ, BP_TAPS : ints
MA_WIN : int
bp : np.ndarray[float32]  # FIR band-pass coefficients (generated via firwin)

Process (top-level script)
--------------------------
1) Load μ/σ from STATS_FILE and sanity-check lengths vs N_CHANNELS.
2) Compute snr_shift = SNR_lag(FILT_FILE) and build EMGIMUTextDataset with
   identical windowing/normalization to training.
3) Randomly select NUM_SAMPLES_TO_EXTRACT windows, already z-scored by the
   dataset (μ/σ set to pass-through if desired).
4) Write OUTPUT_FILENAME as a C header containing:
   - TEST_SAMPLE_WINDOW_SIZE, TEST_SAMPLE_N_CHANNELS, NUM_TEST_SAMPLES
   - MU[channels], SIGMA[channels] and inline zscore()
   - test_sample_i[W][C] arrays and labels
   - all_test_samples[] pointers and test_sample_labels[]
   - FIR band-pass taps BP_TAPS and #define MA_WIN

Helper
------
_fmt_arr(arr: Iterable[float]) -> str
    Format a 1-D float array as C-style initializer with 8 decimals and 'f' suffix.
"""


# createSamples.py — write test samples + mu/sigma into test_samples.h
import os, numpy as np
from data_pipeline import EMGIMUTextDataset
from lag_analyzer import SNR_lag
import random

# --- Config ---
DATA_FILE   = 'Python_TGRT/data/eLog0.txt'
FILT_FILE   = 'Python_TGRT/data/eLog0_filt.txt'   # for lag analysis
STATS_FILE  = 'Python_TGRT/data/stats_train.npz'                   # the one used in training
WINDOW_SIZE = 512
N_CHANNELS  = 18
PATCH_LEN   = 8
NUM_SAMPLES_TO_EXTRACT = 2
OUTPUT_FILENAME = "test_samples.h"

from scipy.signal import firwin
SR_HZ     = 500      # EMG sample rate
BP_LO_HZ  = 20
BP_HI_HZ  = 200
BP_TAPS   = 129       
MA_WIN    = 15         # moving average length used in training
bp = firwin(BP_TAPS, [BP_LO_HZ, BP_HI_HZ], pass_zero=False, fs=SR_HZ).astype(np.float32)


# --- Load train-time stats (mu/sigma) so everything matches the model ---
if not os.path.exists(STATS_FILE):
    raise FileNotFoundError(f"Could not find {STATS_FILE}. Make sure you saved train stats.")

stats = np.load(STATS_FILE)
mu    = stats["mu"].astype(np.float32).reshape(-1)
sigma = stats["sigma"].astype(np.float32).reshape(-1)
if mu.shape[0] != N_CHANNELS or sigma.shape[0] != N_CHANNELS:
    raise ValueError(f"mu/sigma length mismatch: got {mu.shape[0]}/{sigma.shape[0]}, expected {N_CHANNELS}")

mu_for_ds    = np.zeros_like(mu, dtype=np.float32)
sigma_for_ds = np.ones_like(sigma, dtype=np.float32)

# --- Build dataset with identical normalization & lag as training ---
dataLag = int(SNR_lag(FILT_FILE, plot=False))
test_ds = EMGIMUTextDataset(
    DATA_FILE, WINDOW_SIZE, PATCH_LEN,
    snr_shift=dataLag,
    mu=mu_for_ds, sigma=sigma_for_ds,          # <- ensure exact same scaling as training
    augment=False
)

if len(test_ds) < NUM_SAMPLES_TO_EXTRACT:
    raise RuntimeError(f"Dataset has {len(test_ds)} windows, need {NUM_SAMPLES_TO_EXTRACT}.")

def _fmt_arr(arr):
    # C-friendly floats with 8 decimals + 'f'
    return ", ".join([f"{float(v):.8f}f" for v in arr])

with open(OUTPUT_FILENAME, "w") as f:
    f.write("// Auto-generated test samples + normalization for TFLite Micro\n")
    f.write("#ifndef TEST_SAMPLES_H\n#define TEST_SAMPLES_H\n\n")
    f.write("#include <cstdint>\n\n")

    f.write(f"// Window / channels\n")
    f.write(f"static const int TEST_SAMPLE_WINDOW_SIZE = {WINDOW_SIZE};\n")
    f.write(f"static const int TEST_SAMPLE_N_CHANNELS = {N_CHANNELS};\n")
    f.write(f"static const int NUM_TEST_SAMPLES = {NUM_SAMPLES_TO_EXTRACT};\n\n")

    # Normalization arrays (train-time μ/σ)
    f.write("// Train-time normalization (apply to live data before inference)\n")
    f.write(f"static const float MU[TEST_SAMPLE_N_CHANNELS] = {{ { _fmt_arr(mu) } }};\n")
    f.write(f"static const float SIGMA[TEST_SAMPLE_N_CHANNELS] = {{ { _fmt_arr(sigma) } }};\n")
    f.write(
        "static inline float zscore(float x, int ch) {\n"
        "  const float s = SIGMA[ch];\n"
        "  return (s != 0.0f) ? (x - MU[ch]) / s : 0.0f;\n"
        "}\n\n"
    )
    idxs = [0,0]
    # Normalized test samples 
    for i in range(NUM_SAMPLES_TO_EXTRACT):
        idx = random.randint(0, len(test_ds) - 1)
        idxs[i] = idx
        sample_tensor, label = test_ds[idxs[i]]   # shape: (WINDOW_SIZE, N_CHANNELS), dtype float32, normalized
        arr = sample_tensor.numpy()
        f.write(f"// Test Sample {i} (True Label: {int(label)})\n")
        f.write(f"static const float test_sample_{i}[TEST_SAMPLE_WINDOW_SIZE][TEST_SAMPLE_N_CHANNELS] = {{\n")
        for r in range(arr.shape[0]):
            row = ", ".join(f"{val:.8f}f" for val in arr[r])
            f.write(f"  {{ {row} }}{',' if r < arr.shape[0]-1 else ''}\n")
        f.write("};\n\n")

    # Pointers + labels
    f.write("// Array of pointers to samples\n")
    f.write("static const float* const all_test_samples[NUM_TEST_SAMPLES] = {\n")
    for i in range(NUM_SAMPLES_TO_EXTRACT):
        f.write(f"  (const float*)test_sample_{i}{',' if i < NUM_SAMPLES_TO_EXTRACT-1 else ''}\n")
    f.write("};\n\n")

    f.write("static const int test_sample_labels[NUM_TEST_SAMPLES] = {\n")
    for i in range(NUM_SAMPLES_TO_EXTRACT):
        _, label = test_ds[idxs[i]]
        f.write(f"  {int(label)}{',' if i < NUM_SAMPLES_TO_EXTRACT-1 else ''}\n")
    f.write("};\n\n")

    f.write("#endif // TEST_SAMPLES_H\n")
    f.write("// ---- FIR band-pass for EMG ----\n")
    f.write(f"#define BP_NUM_TAPS {len(bp)}\n")
    f.write(f"#define MA_WIN {MA_WIN}\n")
    f.write("static const float BP_TAPS[BP_NUM_TAPS] = { " +
            ", ".join(f"{float(v):.8f}f" for v in bp) + " };\n\n")


print(f"Wrote {NUM_SAMPLES_TO_EXTRACT} samples + MU/SIGMA to {OUTPUT_FILENAME}.")
