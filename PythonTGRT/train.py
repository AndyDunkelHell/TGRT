# train.py

import torch, os
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
# import torch_directml as dml

from data_pipeline import EMGIMUTextDataset   
import data_pipeline as dp
from models import GestureTransformerNoCLS, GestureLinformerTransformerNoCLS  
from utils import EarlyStopping 

from pathlib import Path
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import LineCollection
from matplotlib import lines as mlines
from lag_analyzer import SNR_lag

from copy import deepcopy
import torch.nn.functional as F

# --- Config ---
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
WINDOW_SIZE  = 512     # 512 sample
PATCH_LEN    = 8       # yields 512/8 = 64 tokens
SEQ_LEN      = WINDOW_SIZE // PATCH_LEN  
BATCH_SIZE   = 64
TEACH_EPOCHS = 150
STUD_EPOCHS  = 150
PATIENCE     = 10
N_CLASSES    = 4      # 0 = grab, 1 = pinch, 2 = fist, 3 = palm_up
D_TEACH      = 64
H_TEACH      = 4
L_TEACH      = 4
D_STUD       = 32
H_STUD       = 4
L_STUD       = 1
K_STUD       = 16      # rank for Linformer student
ALPHA        = 0.5     # CE vs. distillation weight
T            = 2.0     # temperature
save = True  # Save models after training
OVERLAP = 0.5
DROPOUT = 0.4

# 2) Teacher: full-attention Transformer
teacher = GestureTransformerNoCLS(
    n_channels=18,
    window_size=WINDOW_SIZE,
    d_model=D_TEACH,
    n_heads=H_TEACH,
    num_layers=L_TEACH,
    patch_len=PATCH_LEN,
    n_classes=N_CLASSES,
).to(DEVICE)


TRAIN_FILE_FILT   = "eLog2_filt"   # original capture
ADD_FILE_FILT = "eLog3_filt"  # extra data to append to training set captured on a different session to improve generalization
VAL_FILE_FILT = "eLog0_filt"

ADD_DATA = ADD_FILE_FILT + "_tke_ma.txt"
TRAIN_DATA  = TRAIN_FILE_FILT + "_tke_ma.txt"
VAL_DATA    = VAL_FILE_FILT + "_tke_ma.txt"

STATS_FILE   = "stats_train.npz"  # µ/σ are stored here for reuse
SPLIT_RATIO = 0.8


# ---------------- Build datasets / loaders ----------------------------------
#make_split_files(TRAIN_FILE, TRAIN_SPLIT, VAL_SPLIT, SPLIT_RATIO)


#val_shift = SNR_lag(VAL_FILE_FILT + ".txt") # Value calculated set bellow to save process time
val_shift = 92 
#train_shift = SNR_lag(TRAIN_FILE_FILT + ".txt") # Value calculated set bellow to save process time
train_shift = 64
print("Validation File: " + VAL_FILE_FILT + " lag Calculation = " + str(val_shift))
print("Train File: " + TRAIN_FILE_FILT + " lag Calculation = " + str(train_shift))


# --- toggles for post-append ---
APPEND_EXTRA_WINDOWS = True          

# ---------------- Helper: split the raw log -----------------
def make_split_files(src: str, train_out: str, val_out: str, ratio: float = 0.8):
    """Write slices of ``src`` into ``train_out``/``val_out``.
    Skips work if both outputs already exist.
    ** Finally not used since the train and validation sets are from different sessions and therefore in different files. Left for utility **
    """
    if Path(train_out).exists() and Path(val_out).exists():
        print("Found existing split files.")
        return

    with open(src, "r") as fin:
        lines = fin.readlines()
    cut = int(len(lines) * ratio)

    with open(train_out, "w") as ftr:
        ftr.writelines(lines[:cut])
    with open(val_out, "w") as fva:
        fva.writelines(lines[cut:])

    print(f"Split {src}: {cut} → {train_out}, {len(lines)-cut} → {val_out}")


# (1) TRAIN Data set  – compute and *save* µ/σ
train_ds_base = EMGIMUTextDataset(
    path=TRAIN_DATA,
    window_size=WINDOW_SIZE,
    patch_len=PATCH_LEN,
    overlap=OVERLAP,
    snr_shift = train_shift,
    save_stats_to=STATS_FILE,
    augment = True
    
)
# load saved stats generated from training set since the Z-score normalization must be consistent across sets
stats = np.load(STATS_FILE)

# (2) VAL     – reuse SAME µ/σ
val_ds = EMGIMUTextDataset(
    path=VAL_DATA,
    window_size=WINDOW_SIZE,
    patch_len=PATCH_LEN,
    overlap=OVERLAP,
    snr_shift = val_shift,
    mu=stats['mu'],
    sigma=stats['sigma']
)


# Optionally add EXTRA windows from a separate dataset *after* windowing
datasets_to_concat = [train_ds_base]
if APPEND_EXTRA_WINDOWS:
    #add_shift = SNR_lag(ADD_FILE_FILT + ".txt")    # lag for the extra capture (result calculated bellow to save process time)
    add_shift = 63
    extra_ds  = EMGIMUTextDataset(
        path=ADD_DATA,               # e.g., "eLog3_filt_tke_ma.txt"
        window_size=WINDOW_SIZE,
        patch_len=PATCH_LEN,
        overlap=OVERLAP,
        snr_shift=add_shift,
        mu=stats['mu'],              # IMPORTANT: use base μ/σ, don't recompute
        sigma=stats['sigma'],
        augment=True
    )
    # Wrap as a Subset so loaders only see the selected windows
    extra_idx = np.arange(len(extra_ds))
    extra_subset = Subset(extra_ds, extra_idx.tolist())
    datasets_to_concat.append(extra_subset)

# Final training dataset is either just base or base+extra
if len(datasets_to_concat) == 1:
    train_ds = train_ds_base
else:
    train_ds = ConcatDataset(datasets_to_concat)

# ---- loaders ----
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

def _collect_labels(ds) -> np.ndarray:
    """Return label array for a (possibly composite) dataset."""
    if isinstance(ds, ConcatDataset):
        parts = [_collect_labels(x) for x in ds.datasets]
        return np.concatenate(parts)
    if isinstance(ds, Subset):
        base_labels = _collect_labels(ds.dataset)
        return base_labels[np.asarray(ds.indices, dtype=int)]
    return np.asarray(ds.labels)

labels_train = _collect_labels(train_ds)

n_train, n_val, n_train_base = len(train_ds), len(val_ds), len(train_ds_base)
print(f"Base Train windows: {n_train_base}  ·  Total Train windows: {n_train}  ·  Val windows: {n_val}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

"""Empirically, the student benefits from class-balanced weights calculated with the inverse 
class frequency. This helps especially with under-represented gestures. This is applied only
to the student loss; the teacher uses closer to sqrt-like weights."""

freq = np.bincount(labels_train, minlength=N_CLASSES).astype(np.float32)
freq[freq == 0] = 1.0                      # guard against empty class
weights_stud = 1.0 / freq                      # inverse frequency
#weights[0] = min(weights[0], 0.5)          # optional: cap “rest” weight
weights_stud = torch.tensor(weights_stud, dtype=torch.float32).to(DEVICE)


beta = 0.999  # 0.99–0.9999 ; higher -> closer to sqrt-like
eff_num = (1.0 - np.power(beta, freq)) / (1.0 - beta)
weights = 1.0 / eff_num
weights = weights / weights.sum() * N_CLASSES
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# 3) Train teacher
opt_t    = optim.Adam(teacher.parameters(), lr=2e-4, weight_decay=1e-4)
teach_crit_ce  = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
stud_crit_ce   = nn.CrossEntropyLoss(weight=weights_stud, label_smoothing=0.02)   # ← balanced loss

early_t  = EarlyStopping(patience=PATIENCE)

for ep in range(1, TEACH_EPOCHS+1):
    teacher.train()
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt_t.zero_grad()

        # current logits
        logits = teacher(X)

        # hard-label CE (with teacher weights + label_smoothing)
        loss = teach_crit_ce(logits, y)

        loss.backward()
        opt_t.step()

    # validation (unchanged)
    teacher.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(DEVICE), yv.to(DEVICE)
            val_loss += teach_crit_ce(teacher(Xv), yv).item() * yv.size(0)
    val_loss /= n_val
    print(f"[Teacher] Epoch {ep} | Val Loss: {val_loss:.4f}")

    early_t(val_loss, teacher)
    if early_t.early_stop:
        print(f"[Teacher] Early stop at epoch {ep}")
        break

# restore best teacher weights
teacher.load_state_dict(early_t.best_state)
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for Xv,yv in val_loader:
        Xv = Xv.to(DEVICE)
        preds = teacher(Xv).argmax(1).cpu().tolist()
        y_true += yv.tolist()
        y_pred += preds

print(classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gesture Classification Confusion Matrix")
plt.show()

if save:
    torch.save(teacher.state_dict(), "teacher2D_nocls.pth")
    print("Saved teacher.pth")


# 4) Student: Linformer-based Transformer
student = GestureLinformerTransformerNoCLS(
    n_channels=18,
    window_size=WINDOW_SIZE,
    d_model=D_STUD,
    n_heads=H_STUD,
    num_layers=L_STUD,
    patch_len=PATCH_LEN,
    k=K_STUD,
    n_classes=N_CLASSES,
).to(DEVICE)

opt_s   = optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=STUD_EPOCHS, eta_min=1e-5)

early_s = EarlyStopping(patience=PATIENCE)
kl_div  = nn.KLDivLoss(reduction='batchmean')

# --- EMA copy of the student (consistency target) ---
student_ema = deepcopy(student).to(DEVICE)
for p in student_ema.parameters():
    p.requires_grad = False
EMA_M = 0.997  # decay

# --- tiny input jitter for a "strong" view (no dataset change needed) ---
def jitter_amp(x, noise_sigma=0.02, gain_sigma=0.10):
    # x: (B, W, C)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(x) * (noise_sigma * std)
    gain  = 1.0 + gain_sigma * torch.randn(x.size(0), 1, 1, device=x.device)
    return x * gain + noise

# --- MixUp (no label mixing for CE; only used for KD) ---
def mixup(x, alpha=0.4):
    b = x.size(0)
    if b < 2:
        return x, torch.arange(b, device=x.device), torch.ones(b, device=x.device)
    perm = torch.randperm(b, device=x.device)
    lam  = torch.distributions.Beta(alpha, alpha).sample((b,)).to(x.device)
    lam  = torch.maximum(lam, 1.0 - lam)  # >0.5
    lam2 = lam.view(-1, 1, 1)
    xm   = lam2 * x + (1.0 - lam2) * x[perm]
    return xm, perm, lam

# 5) Distillation training loop (upgraded)
ALPHA_START, ALPHA_END = 0.30, 0.70   # start KD-heavy, end CE-heavier
T_START, T_END         = 2.5, 1.5
CONS_MAX               = 0.20         # max weight for Exponential Moving Averag (EMA) consistency

for ep in range(1, STUD_EPOCHS+1):
    student.train()
    ep_frac = (ep - 1) / max(1, STUD_EPOCHS - 1)
    ALPHA   = ALPHA_START + (ALPHA_END - ALPHA_START) * ep_frac
    Tcur    = T_START + (T_END - T_START) * ep_frac
    LAM_CONS = min(1.0, ep / 5.0) * CONS_MAX  # warm-up consistency

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # two views: weak (clean) + strong (jittered)
        Xw = X
        Xs = jitter_amp(X)

        # --- CE on clean inputs (uses label_smoothing + class weights) ---
        logits_ce = student(Xw)
        loss_ce   = stud_crit_ce(logits_ce, y)

        # --- KD on strong view from frozen teacher ---
        with torch.no_grad():
            pt_s = F.softmax(teacher(Xs) / Tcur, dim=1)
        logp_s = F.log_softmax(student(Xs) / Tcur, dim=1)
        loss_kd = kl_div(logp_s, pt_s) * (Tcur * Tcur)

        # --- distill from a *mixed* teacher distribution ---
        Xm, perm, lam = mixup(Xs)
        with torch.no_grad():
            pt1 = pt_s
            pt2 = pt_s[perm]
            p_mix = lam.view(-1, 1) * pt1 + (1.0 - lam).view(-1, 1) * pt2
        logp_mix = F.log_softmax(student(Xm) / Tcur, dim=1)
        loss_kd_mix = kl_div(logp_mix, p_mix) * (Tcur * Tcur)

        # --- Consistency to EMA student on the weak view ---
        with torch.no_grad():
            pe = F.softmax(student_ema(Xw) / 1.5, dim=1)
        logp_cons = F.log_softmax(student(Xw) / 1.5, dim=1)
        loss_cons = kl_div(logp_cons, pe) * (1.5 * 1.5)

        # total
        loss = (
            ALPHA * loss_ce
            + (1.0 - ALPHA) * 0.5 * (loss_kd + loss_kd_mix)
            + LAM_CONS * loss_cons
        )

        opt_s.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_s.step()

        # EMA update AFTER the optimizer step
        with torch.no_grad():
            for pe, p in zip(student_ema.parameters(), student.parameters()):
                pe.data.mul_(EMA_M).add_(p.data, alpha=1.0 - EMA_M)

    # validation (hard-label CE on student)
    student.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(DEVICE), yv.to(DEVICE)
            val_loss += stud_crit_ce(student(Xv), yv).item() * yv.size(0)
    val_loss /= n_val
    print(f"[Student] Epoch {ep} | Val Loss: {val_loss:.4f}")

    early_s(val_loss, student)
    if early_s.early_stop:
        print(f"[Student] Early stop at epoch {ep}")
        break

    sched_s.step()  # cosine anneal

# restore best student
student.load_state_dict(early_s.best_state)

# 6) Final: evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

student.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for Xv,yv in val_loader:
        Xv = Xv.to(DEVICE)
        preds = student(Xv).argmax(1).cpu().tolist()
        y_true += yv.tolist()
        y_pred += preds

print(classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gesture Classification Confusion Matrix")
plt.show()


if save:
    torch.save(student.state_dict(), "student2D_nocls.pth")
    # torch.save(teacher.state_dict(), "teacher2D_nocls.pth")
    print("Saved student.pth")
