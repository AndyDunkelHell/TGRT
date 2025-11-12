
"""
TKE + Moving Average preprocessing for 12-ch EMG (with optional band-pass), preserving
the original IMU stream. Reads Unity logs and writes a transformed log with
per-channel TKE→moving-average features.

I/O format
----------
Input  line:  time|gesture|e1,...,e12|i1,...,i6[|hostUs]
Output line:  time|gesture|MA_e1,...,MA_e12|i1,...,i6
    • time and gesture are copied verbatim.
    • EMG becomes integer-rounded moving-average of TKE per channel.
    • IMU is passed through unchanged.

Functions
---------
butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4)
    -> tuple[np.ndarray, np.ndarray]
    Design a digital Butterworth band-pass. Returns (b, a).

bandpass_filter(
    data: np.ndarray, lowcut: float = 20, highcut: float = 200,
    fs: int = 500, order: int = 4
) -> np.ndarray
    Apply filtfilt() band-pass per EMG channel. Expects shape (samples, 12).
    Returns filtered array with same shape.

parse_line(line: str) -> tuple[str, str, list[float], str] | None
    Parse one log line; validates EMG has 12 values and IMU has 6.
    Returns (time, gesture, emg_list, imu_csv) or None on failure.

tke_window(buf: list[float]) -> float
    Three-point Teager-Kaiser energy: buf[1]**2 − buf[0]*buf[2].

process_file(
    infile: str, outfile: str, ma_window: int = 15,
    lowcut: float = 20, highcut: float = 200, fs: int = 500, order: int = 4
) -> None
    Pipeline:
      1) Parse all lines, collect time, gesture, EMG (12 ch), IMU (6 vals).
      2) Band-pass EMG with Butterworth (filtfilt) using provided params.
      3) For each sample/channel, compute TKE over a 3-sample rolling buffer.
      4) For each channel, compute moving average of last `ma_window` TKE values
         (integer-rounded; emits 0 until the buffer has `ma_window` values).
      5) Write transformed log as described above.
    Shows tqdm progress bars for parsing and processing.

"""

import argparse
from scipy.signal import iirnotch, filtfilt, butter
import numpy as np
import tqdm

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Butterworth bandpass filter.
    - lowcut: Lower cutoff frequency in Hz.
    - highcut: Upper cutoff frequency in Hz.
    - fs: Sampling frequency in Hz.
    - order: Filter order.
    return: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def bandpass_filter(data, lowcut=20, highcut=200, fs=500, order=4):
    """
    Apply a bandpass filter to each channel of the data.
    - data: 2D numpy array of shape (samples, channels).
    - lowcut: Lower cutoff frequency in Hz (e.g., 20).
    - highcut: Upper cutoff frequency in Hz (e.g., 450).
    - fs: Sampling frequency in Hz.
    - order: Filter order (e.g., 4).
    return: Filtered 2D numpy array.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered_data

def parse_line(line):
    """Parse a line into time, gesture, emg list, imu string."""
    try:
        # split on |, then split EMG on comma
        time, gesture, emg_str, imu, hostUs = line.strip().split('|')
        emg = [float(x) for x in emg_str.split(',')]
        imu_vals = [float(x) for x in imu.split(',')]
        if len(emg) != 12 or len(imu_vals) != 6:
            raise ValueError("Invalid EMG or IMU data length")
        return time, gesture, emg, imu
    except:
        return None


def tke_window(buf):
    """Compute TKE on a 3-sample buffer: buf[1]**2 – buf[0]*buf[2]."""
    return buf[1]**2 - buf[0]*buf[2]

def process_file(infile, outfile, ma_window=15,
                 lowcut=20, highcut=200, fs=500, order=4):
    # First, read and parse all data into lists
    times, gestures, imu_list, emg_list = [], [], [], []
    with open(infile, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm.tqdm(lines, desc="Parsing lines"):
            if not line.strip():
                continue
            parsed = parse_line(line)
            if parsed is None:
                continue
            time, gesture, emg, imu = parsed
            times.append(time)
            gestures.append(gesture)
            emg_list.append(emg)
            imu_list.append(imu)

    # Convert list of EMG samples to numpy array and apply bandpass
    emg_array = np.array(emg_list)  # shape: (n_samples, 12)
    filtered_emg = bandpass_filter(emg_array, lowcut, highcut, fs, order)
    # filtered_emg = emg_array

    # Prepare buffers for TKE and moving average
    tke_hist = [[] for _ in range(12)]
    win_buf = [[0.0, 0.0, 0.0] for _ in range(12)]

    with open(outfile, 'w') as fout:
        for idx, (time, gesture, imu_vals) in enumerate(tqdm.tqdm(zip(times, gestures, imu_list), total=len(times), desc="Processing samples")):
            # Compute TKE per channel on filtered data
            tke_vals = []
            for ch in range(12):
                win_buf[ch].pop(0)
                win_buf[ch].append(filtered_emg[idx, ch])
                tke = tke_window(win_buf[ch])
                tke_hist[ch].append(tke)
                tke_vals.append(tke)
                #tke_vals.append(filtered_emg[idx, ch])

            # Compute moving average of TKE per channel
            ma_vals = []
            for ch in range(12):
                hist = tke_hist[ch]
                if len(hist) >= ma_window:
                    ma_vals.append(int(sum(hist[-ma_window:]) / ma_window))
                else:
                    ma_vals.append(0)

            # Write output line: time|gesture|MA values|IMU values
            out_emg = ",".join(str(x) for x in ma_vals)
            #out_emg = ",".join(str(x) for x in tke_vals)
            fout.write(f"{time}|{gesture}|{out_emg}|{imu_vals}\n")


if __name__ == "__main__":
#     p = argparse.ArgumentParser(
#         description="Apply TKE + Moving Average to 12-channel EMG data."
#     )
#     p.add_argument("input", help="Path to input TXT file")
#     p.add_argument("output", help="Path to output TXT file")
#     p.add_argument(
#         "--ma_window", "-w", type=int, default=15,
#         help="Moving-average window size (default: 15)"
#     )
#     args = p.parse_args()
#     args_input = args.input
#     args_output = args.output
#     args_ma_window = args.ma_window
    input_u = "eLog3.txt"
    output_u = "eLog3_filt_tke_ma.txt"
    window_u = 7
    # process_file(args.input, args.output, ma_window=args.ma_window)
    process_file(input_u, output_u, ma_window=window_u)
