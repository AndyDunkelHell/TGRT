# evaluate_models.py
"""
Evaluate and compare PyTorch teacher/student models and the TFLite student on the
same dataset. Reports accuracy/precision/recall/F1, latency, confusion matrices,
and resource metrics (params, file size, peak RAM). Saves CSV and plots.

Constants / Paths
-----------------
WEIGHTS : dict[str, str]
    Paths for 'teacher', 'student', 'tflite'.
WINDOW_SIZE, PATCH_LEN, N_CLASSES, OVERLAP : ints/floats
DATA_FILE, FILT_FILE, STATS_FILE : str
BATCH_SIZE : int
DEVICE : torch.device
D_TEACH, H_TEACH, L_TEACH : int
D_STUD,  H_STUD,  L_STUD, K_STUD : int

Functions
---------
load_ds() -> tuple[EMGIMUTextDataset, DataLoader, DataLoader]
    Loads mu and sigma from STATS_FILE, computes snr_shift via SNR_lag(FILT_FILE),
    builds dataset with identical normalization and overlap, and returns:
    (dataset, PyTorch loader with BATCH_SIZE, TFLite loader with batch=1).

build_models() -> dict[str, torch.nn.Module]
    Instantiates teacher and student like in train.py, loads weights, sets eval().

peak_rss(model: torch.nn.Module, sample: torch.Tensor) -> float
    Run a single forward pass and return host RAM delta in MB.

peak_rss_tflite(interpreter: tf.lite.Interpreter,
                sample_np: np.ndarray,
                input_index: int) -> float
    Resize input if needed, invoke once, return host RAM delta in MB.

evaluate_pytorch(model: torch.nn.Module,
                 loader: DataLoader) -> dict
    Returns dict with acc, prec, rec, f1, latency (sec/batch),
    and full y_true/y_pred lists.

evaluate_tflite(interpreter: tf.lite.Interpreter,
                loader: DataLoader,
                input_index: int,
                output_index: int) -> dict
    Same metrics as above, handling dynamic last-batch resize.

save_classification_report_csv(model: torch.nn.Module,
                               loader: DataLoader,
                               class_names: list[str],
                               csv_path: str,
                               cfg: dict,
                               device: torch.device)
    -> tuple[pd.DataFrame, dict, tuple[np.ndarray, np.ndarray]]
    Generates sklearn classification_report (per-class + macro/weighted),
    saves to CSV, appends provenance/extras (dataset params, snr_shift, #params,
    model_size_MB, peak RAM/VRAM), and returns (df, extras, (y_true, y_pred)).

main() -> None
    Orchestrates dataset/model loading, PyTorch & TFLite evaluation, confusion
    matrices (PNG), bar plots for quality and resources, and a summary CSV.
"""


import time
import pathlib
import psutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from data_pipeline import EMGIMUTextDataset
from models import GestureTransformerNoCLS, GestureLinformerTransformerNoCLS
from lag_analyzer import SNR_lag


WEIGHTS = dict(
    teacher='Models/TeacherGold_Final.pth',
    student='Models/StudentGold_Final.pth',
    tflite='Models/model_StudentGold_Final.tflite'
)
WINDOW_SIZE = 512
PATCH_LEN   = 8
N_CLASSES   = 4
DATA_FILE   = 'Python_TGRT/data/eLog0_filt_tke_ma.txt'
FILT_FILE = 'Python_TGRT/data/eLog0_filt.txt'  # for lag analysis
BATCH_SIZE  = 64                   # batch size for eval & memory tests
DEVICE      = torch.device('cpu')   # CPU for fair RAM measurements
D_TEACH      = 64
H_TEACH      = 4
L_TEACH      = 4
D_STUD      = 32
H_STUD      = 4
L_STUD      = 1
K_STUD      = 16
OVERLAP = 0.5                     # data window overlap for evaluation
STATS_FILE   = "Python_TGRT/data/stats_train.npz"  # µ/σ are stored here for reuse
DATA_LAG = 0                     # will be set after analysis
# ------------------------------------------------------------------------


def load_ds():
    print("Eval file:", DATA_FILE, "| exists:", Path(DATA_FILE).exists())
    print("Stats file:", STATS_FILE, "| exists:", Path(STATS_FILE).exists())
    dataLag = SNR_lag(FILT_FILE, plot = False)  # analyze lag to improve accuracy
    DATA_LAG = int(dataLag)
    stats = np.load(STATS_FILE)
    ds = EMGIMUTextDataset(DATA_FILE, WINDOW_SIZE, PATCH_LEN, overlap=OVERLAP,snr_shift=dataLag, mu=stats['mu'], sigma=stats['sigma'])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    tf_liteLoader = DataLoader(ds, batch_size=1, shuffle=False) 
    return ds, loader, tf_liteLoader


def build_models():
    """Load and prepare PyTorch teacher and student models."""
    teacher = GestureTransformerNoCLS(18, WINDOW_SIZE, D_TEACH, H_TEACH, L_TEACH, PATCH_LEN, N_CLASSES).to(DEVICE)
    student = GestureLinformerTransformerNoCLS(18, WINDOW_SIZE, D_STUD, H_STUD, L_STUD, PATCH_LEN, K_STUD, N_CLASSES).to(DEVICE)
    teacher.load_state_dict(torch.load(WEIGHTS['teacher'], map_location=DEVICE))
    student.load_state_dict(torch.load(WEIGHTS['student'], map_location=DEVICE))
    teacher.eval()
    student.eval()
    return dict(teacher=teacher, student=student)


@torch.no_grad()
def peak_rss(model, sample):
    """Measure RAM increase of a single PyTorch forward pass."""
    process = psutil.Process()
    rss_before = process.memory_info().rss
    _ = model(sample)
    torch.cuda.empty_cache()
    rss_after = process.memory_info().rss
    return (rss_after - rss_before) / 1024**2  # MB


def peak_rss_tflite(interpreter, sample_np, input_index):
    """Measure RAM increase of a TFLite invocation, resizing if needed."""
    details = interpreter.get_input_details()[0]
    expected_shape = details['shape']
    if sample_np.shape[0] != expected_shape[0]:
        interpreter.resize_tensor_input(
            input_index,
            [int(sample_np.shape[0]), int(expected_shape[1]), int(expected_shape[2])]
        )
        interpreter.allocate_tensors()

    process = psutil.Process()
    rss_before = process.memory_info().rss
    interpreter.set_tensor(input_index, sample_np)
    interpreter.invoke()
    rss_after = process.memory_info().rss
    return (rss_after - rss_before) / 1024**2  # MB


def evaluate_pytorch(model, loader):
    """Evaluate a PyTorch model on the dataset loader."""
    y_true, y_pred = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    duration = time.perf_counter() - start
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return dict(acc=acc, prec=p, rec=r, f1=f1, latency=duration / len(loader), y_true=y_true, y_pred=y_pred)


def evaluate_tflite(interpreter, loader, input_index, output_index):
    """Evaluate a TFLite model on the dataset loader, resizing for last batch."""
    details = interpreter.get_input_details()[0]
    orig_shape = details['shape']  # [batch, window, channels]
    y_true, y_pred = [], []
    start = time.perf_counter()
    for x, y in loader:
        x_np = x.numpy().astype(np.float32)
        if x_np.shape[0] != orig_shape[0]:
            interpreter.resize_tensor_input(input_index, [x_np.shape[0], orig_shape[1], orig_shape[2]])
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, x_np)
        interpreter.invoke()
        out = interpreter.get_tensor(output_index)
        y_true.extend(y.tolist())
        y_pred.extend(np.argmax(out, axis=1).tolist())
    duration = time.perf_counter() - start
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return dict(acc=acc, prec=p, rec=r, f1=f1, latency=duration / len(loader), y_true=y_true, y_pred=y_pred)

import os, time, hashlib, numpy as np, pandas as pd
try:
    import psutil
except Exception:
    psutil = None
from sklearn.metrics import classification_report

def save_classification_report_csv(model, loader, class_names, csv_path, cfg, device):
    """Runs eval, builds sklearn classification_report, and appends extras."""
    model.eval()

    # Optional: reset CUDA peak memory stats for clean VRAM reading
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    y_true, y_pred = [], []
    peak_ram = 0
    proc = psutil.Process(os.getpid()) if psutil else None

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            y_pred.append(logits.argmax(1).cpu().numpy())
            y_true.append(yb.cpu().numpy())

            if proc:
                rss = proc.memory_info().rss
                if rss > peak_ram:
                    peak_ram = rss

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # sklearn report as dict (full precision; no rounding), consistent zero_division
    labels = list(range(len(class_names)))
    rep = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(rep).transpose()

    # model params + size
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / 1e6

    # VRAM peak (if CUDA)
    peak_vram_mb = None
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device=device) / 1e6

    # Extras / provenance
    extras = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'dataset_len': len(loader.dataset),
        'batch_size': loader.batch_size,
        'window_size': cfg.get('WINDOW_SIZE'),
        'patch_len': cfg.get('PATCH_LEN'),
        'overlap': cfg.get('OVERLAP'),
        'snr_shift': cfg.get('SNR_SHIFT'),
        'data_file': cfg.get('DATA_FILE'),
        'raw_file': cfg.get('RAW_FILE'),
        'stats_file': cfg.get('STATS_FILE'),
        'num_params': int(num_params),
        'model_size_mb': round(model_size_mb, 3),
        'peak_ram_mb': round(peak_ram / 1e6, 3) if peak_ram else None,
        'peak_vram_mb': round(peak_vram_mb, 3) if peak_vram_mb is not None else None,
    }

    # Write report and append extras
    df.to_csv(csv_path, index=True)
    # append extras as a small section below (CSV-friendly)
    extras_df = pd.DataFrame.from_dict(extras, orient='index', columns=['value'])
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write('\n# extras\n')
    extras_df.to_csv(csv_path, mode='a', header=True)
    print(f"[eval] Saved classification report → {csv_path}")

    return df, extras, (y_true, y_pred)



def main():
    ds, loader, tf_liteLoader = load_ds()
    models = build_models()

    # Single-sample for PyTorch RAM test
    sample, _ = ds[0]
    sample = sample.unsqueeze(0).to(DEVICE)
    sample_np = sample.cpu().numpy().astype(np.float32)

    # Batch-sample for TFLite RAM test (to register measurable usage)
    batch_sample, _ = next(iter(loader))
    batch_sample_np = batch_sample.numpy().astype(np.float32)

    # Load and resize TFLite interpreter
    tflite_interpreter = tf.lite.Interpreter(model_path=WEIGHTS['tflite'])
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.resize_tensor_input(input_details[0]['index'], [BATCH_SIZE, WINDOW_SIZE, 18])
    tflite_interpreter.allocate_tensors()

    rows = []

    # Evaluate PyTorch models
    for name, mdl in models.items():
        stats = evaluate_pytorch(mdl, loader)
        cfg = dict(
            DATA_FILE=DATA_FILE,
            RAW_FILE=FILT_FILE,          
            STATS_FILE=STATS_FILE,
            WINDOW_SIZE=WINDOW_SIZE,
            PATCH_LEN=PATCH_LEN,
            OVERLAP=OVERLAP,
            SNR_SHIFT=int(DATA_LAG),
        )
        
        CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

        report_df, extras, (y_true, y_pred) = save_classification_report_csv(
            model=mdl,                
            loader=loader,
            class_names=CLASS_NAMES,
            csv_path=name + '_report.csv',
            cfg=cfg,
            device=DEVICE
        )

        macro_f1   = report_df.loc['macro avg', 'f1-score']
        weighted_f1= report_df.loc['weighted avg', 'f1-score']
        print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        rows.append({
            'model': name,
            'acc': round(stats['acc'], 4),
            'prec': round(stats['prec'], 4),
            'rec': round(stats['rec'], 4),
            'f1': round(stats['f1'], 4),
            'latency': round(stats['latency'], 6),
            'params': sum(p.numel() for p in mdl.parameters()),
            'size_MB': pathlib.Path(WEIGHTS[name]).stat().st_size / 1e6,
            'peak_RAM_MB': round(peak_rss(mdl, sample), 2)
        })
        # Confusion matrix
        cm = confusion_matrix(stats['y_true'], stats['y_pred'])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.title(f'{name.capitalize()} – Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'cm_{name}.png')
        plt.close()

    # Evaluate TFLite model
    # Estimate params from file size (bytes / 4 bytes per weight)
    params_tflite = int(pathlib.Path(WEIGHTS['tflite']).stat().st_size / 4)
    t_stats = evaluate_tflite(
        tflite_interpreter,
        tf_liteLoader,
        input_details[0]['index'],
        output_details[0]['index']
    )
    rows.append({
        'model': 'tflite',
        'acc': round(t_stats['acc'], 4),
        'prec': round(t_stats['prec'], 4),
        'rec': round(t_stats['rec'], 4),
        'f1': round(t_stats['f1'], 4),
        'latency': round(t_stats['latency'], 6),
        'params': params_tflite,
        'size_MB': pathlib.Path(WEIGHTS['tflite']).stat().st_size / 1e6,
        'peak_RAM_MB': round(peak_rss_tflite(
            tflite_interpreter, batch_sample_np, input_details[0]['index']
        ), 2)
    })

    # Confusion matrix for TFLite
    cm = confusion_matrix(t_stats['y_true'], t_stats['y_pred'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Tflite – Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('cm_tflite.png')
    plt.close()

    # Save tabular results
    df = pd.DataFrame(rows).set_index('model')
    df.to_csv('model_comparison.csv')
    print(df)

    # Bar plots: quality metrics
    ax = df[['acc', 'prec', 'rec', 'f1']].plot.bar(rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Quality metrics')
    plt.tight_layout()
    plt.savefig('metrics_bar.png')
    plt.close()

    # Bar plots: resource metrics (log scale)
    ax = df[['params', 'size_MB', 'peak_RAM_MB']].plot.bar(rot=0, logy=True)
    ax.set_ylabel('Log-scale')
    ax.set_title('Resource metrics')
    plt.tight_layout()
    plt.savefig('resources_bar.png')
    plt.close()

    print("\nArtifacts written: model_comparison.csv, cm_*.png, metrics_bar.png, resources_bar.png")

    


if __name__ == '__main__':
    main()
