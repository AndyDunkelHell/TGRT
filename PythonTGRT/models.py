"""
Model definitions: full-attention teacher (no CLS) and a Linformer-based student,
plus a lightweight attention-style sequence pooling block (instead of CLS because CLS not embedded inference friendly)
and Linformer attention.

Classes
-------
SequencePooling(d_model: int) : nn.Module
    Learnable attention pooling over sequence axis.
    forward(x: Tensor[B,S,d]) -> Tensor[B,d]

GestureTransformerNoCLS(
    n_channels=18, window_size=200,
    d_model=64, n_heads=4, num_layers=2,
    patch_len=50, n_classes=6
) : nn.Module
    Patch-embeds EMG+IMU (EMGIMUPatchEmbed), adds learned positions, encodes with
    nn.TransformerEncoder, pools with SequencePooling, and classifies.
    forward(x: Tensor[B,W,C]) -> Tensor[B,n_classes]

GestureLinformerLayer(d_model: int, n_heads: int, seq_len: int, k: int, dropout=0.4)
    Encoder block using LinformerAttention + FFN with GELU and residual LayerNorms.
    forward(x: Tensor[B,S,d]) -> Tensor[B,S,d]

LinformerAttention(
    d_model: int, n_heads: int, seq_len: int, k: int = 64, dropout=0.3
)
    Linformer-style MSA with low-rank projections E,F on sequence axis
    for K,V (S→k). Q,K,V are linear; output projection merges heads.
    forward(x: Tensor[B,S,d]) -> Tensor[B,S,d]

GestureLinformerTransformerNoCLS(
    n_channels=18, window_size=200,
    d_model=64, n_heads=4, num_layers=2,
    patch_len=50, k=64, n_classes=6
) : nn.Module
    MCU-friendly patch embed (MCUFlatPatchEmbed), learned positions, N*
    GestureLinformerLayer, SequencePooling, classifier head.
    forward(x: Tensor[B,W,C]) -> Tensor[B,n_classes]
"""


import torch, torch.nn as nn, torch.nn.functional as F
import math
from patches import EMGIMUPatchEmbed, MCUFlatPatchEmbed   

class SequencePooling(nn.Module):
    """
    Learnable attention-style pooling.
    x : (B, S, d)  ->  pooled : (B, d)
    """
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)          # (B, d)

# ---------- full-attention teacher ----------
class GestureTransformerNoCLS(nn.Module):
    """
    Identical to GestureTransformer but WITHOUT the extra CLS token.
    Uses SequencePooling for classification.
    """
    def __init__(self,
                 n_channels=18, window_size=200,
                 d_model=64, n_heads=4, num_layers=2,
                 patch_len=50, n_classes=6):
        super().__init__()
        assert window_size % patch_len == 0
        self.patch_emb = EMGIMUPatchEmbed(
            window_size, patch_len, 4, 3, d_model
        )
        n_patches = self.patch_emb.num_patches           
        self.pos_time = nn.Parameter(torch.zeros(1, n_patches, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=0.4, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm   = nn.LayerNorm(d_model)
        self.pool   = SequencePooling(d_model)           
        self.head   = nn.Linear(d_model, n_classes)

    def forward(self, x):                     # x : (B, W, C)
        x = self.patch_emb(x)                 # (B, S, d)
        x = x + self.pos_time                 # learnt positions
        x = self.encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.norm(self.pool(x))           # (B, d)
        return self.head(x)                   # (B, n_classes)

class GestureLinformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, k, dropout=0.4):
        super().__init__()
        # replace the built-in MHA with LinformerAttention
        self.self_attn = LinformerAttention(d_model, n_heads, seq_len, k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(approximate='tanh'),
            #nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, S, d_model)
        y = self.self_attn(x)
        x = self.norm1(x + y)
        y = self.ff(x)
        return self.norm2(x + y)


class LinformerAttention(nn.Module):
    """
    Implements Linformer-style multi-head self-attention with low-rank projections on K and V.
    Projects sequence dimension from S -> k for keys and values.
    """
    def __init__(self, d_model, n_heads, seq_len, k=64, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len
        self.k = k

        # Q, K, V projections
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Low-rank projection matrices (sequence -> k)
        self.E = nn.Parameter(torch.randn(seq_len, k))  # for keys
        self.F = nn.Parameter(torch.randn(seq_len, k))  # for values

        # Output projection
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):  # x: (B, S, d_model)
        B, S, _ = x.shape

        # Project to Q, K, V
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        # Split heads: (B, h, S, d_h)
        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Low-rank projection on sequence axis via matmul (avoid einsum)
        # K: (B, h, S, d_h) -> permute to (B, h, d_h, S)
        Kp = K.permute(0, 1, 3, 2)            # (B, h, d_h, S)
        Kp = torch.matmul(Kp, self.E)         # (B, h, d_h, k)
        Kp = Kp.permute(0, 1, 3, 2)            # (B, h, k, d_h)

        Vp = V.permute(0, 1, 3, 2)             # (B, h, d_h, S)
        Vp = torch.matmul(Vp, self.F)         # (B, h, d_h, k)
        Vp = Vp.permute(0, 1, 3, 2)            # (B, h, k, d_h)

        # Scaled dot-product attention
        scores = torch.matmul(Q, Kp.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, S, k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output: (B, h, S, d_h)
        out = torch.matmul(attn, Vp)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out(out)

# ---------- Linformer-based student ----------
class GestureLinformerTransformerNoCLS(nn.Module):
    """
    Linformer student.
    """
    def __init__(self,
                 n_channels=18, window_size=200,
                 d_model=64, n_heads=4, num_layers=2,
                 patch_len=50, k=64, n_classes=6):
        super().__init__()
        # patch embed MCU-friendly flat version
        self.patch_emb = MCUFlatPatchEmbed(window_size, patch_len, d_model,20)
        seq_len = self.patch_emb.num_patches        
        self.pos_time = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.layers = nn.ModuleList([
            GestureLinformerLayer(d_model, n_heads, seq_len, k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)           
        self.pool = SequencePooling(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):                           # (B, W, C)
        x = self.patch_emb(x)                       # (B, S, d)
        x = x + self.pos_time
        for blk in self.layers:
            x = blk(x)
        x = self.norm(self.pool(x))
        return self.head(x)