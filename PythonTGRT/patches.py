"""
Patch-embedding layers for EMG+IMU sequences and a helper concat that is compatible
with TFLite Micro.

Functions
---------
_cascaded_cat(tensor_list: list[Tensor], dim: int, max_per_cat: int = 8) -> Tensor
    Concatenate many tensors by recursively grouping into chunks of size ≤ max_per_cat
    to avoid a single wide cat node (compatible with TFLM converters/runtimes).

Classes
-------
MCUFlatPatchEmbed(
    window_size: int,
    patch_t: int = 8,
    d_model: int = 64,
    spat_dim: int = 20
) : nn.Module
    MCU-safe pure-float embed:
    • Split input x[B,W,C] into EMG (12 ch) and IMU (6 ch).
    • For EMG: per-time Linear(12→spat_dim), then flatten (patch_t*spat_dim)
      and Linear(→d_model) per temporal patch.
    • For IMU: flatten (patch_t*6) then Linear(→d_model) per patch.
    • Concatenate EMG and IMU patch sequences along the patch axis.
    num_patches = (window_size // patch_t) * 2
    forward(x: Tensor[B,W,18]) -> Tensor[B,num_patches,d_model]

EMGIMUPatchEmbed(
    window_size: int,
    patch_t: int = 8, patch_h: int = 2, patch_w: int = 1,
    d_model: int = 64
) : nn.Module
    Mixed 3-D/1-D conv embed:
    • EMG (first 12 ch) reshaped to (T,4,3), Conv3d over (t,h,w) → EMG tokens.
    • IMU (6 ch) via Conv1d over time → IMU tokens.
    • Concatenate token sequences → (B, N_emg+N_imu, d_model).
    num_patches computed from window_size and patch sizes.
    forward(x: Tensor[B,W,18]) -> Tensor[B,num_patches,d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _cascaded_cat(tensor_list, dim, max_per_cat=8):
    """
    Helper function to concatenate a list of tensors in a cascaded manner
    to avoid a single concatenation node with too many inputs, which can
    be problematic for TFLM.
    """
    if not tensor_list:
        raise ValueError("Input tensor_list for _cascaded_cat is empty")
    if len(tensor_list) == 1:
        return tensor_list[0]

    intermediate_concats = []
    for i in range(0, len(tensor_list), max_per_cat):
        chunk = tensor_list[i : i + max_per_cat]
        if not chunk: 
            continue
        if len(chunk) == 1:
            intermediate_concats.append(chunk[0])
        else:
            intermediate_concats.append(torch.cat(chunk, dim=dim))

    if len(intermediate_concats) > max_per_cat:
        return _cascaded_cat(intermediate_concats, dim, max_per_cat)
    elif len(intermediate_concats) == 1:
        return intermediate_concats[0]
    else:
        return torch.cat(intermediate_concats, dim=dim)

class MCUFlatPatchEmbed(nn.Module):
    """
    Pure-float, MCU-safe patch embed.
    Processes EMG and IMU data into patch embeddings using reshapes and linear layers.
    """
    def __init__(self, window_size, patch_t=8, d_model=64, spat_dim=20):
        super().__init__()
        self.window_size = window_size
        self.patch_t     = patch_t
        self.d_model     = d_model
        self.spat_dim    = spat_dim

        # Linear layers for processing

        self.spatial_fc  = nn.Linear(12, self.spat_dim, bias=False)
        self.flat_fc_emg = nn.Linear(self.patch_t * self.spat_dim, self.d_model, bias=False)
        self.flat_fc_imu = nn.Linear(self.patch_t * 6, self.d_model, bias=False)

        # self.num_patches will be the total number of patches output by this layer
        # (number of temporal segments * 2 for separate EMG and IMU patch sequences)
        self.num_patches = (self.window_size // self.patch_t) * 2

    def forward(self, x): # Input x: (Batch_size, Window_size, Num_features_total) -> (B, 512, 18)
        B, W, C = x.shape
        # W should be self.window_size (e.g., 512)
        # C should be 18 (12 EMG + 6 IMU)

        num_temp_patches = W // self.patch_t # e.g., 512 / 8 = 64 temporal segments

        # Separate EMG and IMU data
        emg_data = x[..., :12]  # Shape: (B, W, 12)
        imu_data = x[..., 12:]  # Shape: (B, W, 6)

        # 1. Reshape data into batches of patches
        emg_data_batched = emg_data.reshape(B, num_temp_patches, self.patch_t, 12)
        imu_data_batched = imu_data.reshape(B, num_temp_patches, self.patch_t, 6)

        # 2. Process EMG patches
        # Apply spatial FC across the 12 EMG channels for each timestep within each patch
        emg_spatial_batched = self.spatial_fc(emg_data_batched) # Shape: (B, num_temp_patches, patch_t, spat_dim)

        # Flatten the patch_t and spat_dim dimensions for each EMG patch
        emg_flat_for_fc = emg_spatial_batched.reshape(B, num_temp_patches, self.patch_t * self.spat_dim)
        # Project each flattened EMG patch to d_model
        final_emg_patches = self.flat_fc_emg(emg_flat_for_fc) # Shape: (B, num_temp_patches, d_model) -> (B, 64, d_model)

        # 3. Process IMU patches
        # Flatten the patch_t and 6 IMU features dimensions for each IMU patch
        imu_flat_for_fc = imu_data_batched.reshape(B, num_temp_patches, self.patch_t * 6)
        # Project each flattened IMU patch to d_model
        final_imu_patches = self.flat_fc_imu(imu_flat_for_fc) # Shape: (B, num_temp_patches, d_model) -> (B, 64, d_model)

        # 4. Concatenate the processed EMG patch sequence and IMU patch sequence
        # Simple concatenation of TWO tensors, which is TFLM-friendly.
        all_patches = torch.cat([final_emg_patches, final_imu_patches], dim=1)
        # Final shape: (B, num_temp_patches * 2, d_model) -> (B, 128, d_model)

        return all_patches


class EMGIMUPatchEmbed(nn.Module):
    """
    • Splits the first 12 channels into a (T,4,3) grid and applies
      a 3-D conv to yield spatial-temporal EMG tokens.
    • Applies a 1-D conv to the 6 IMU channels to yield motion tokens.
    • Concatenates both token sequences → (B, N_emg+N_imu, d_model).
    """
    def __init__(self,
                 window_size,
                 patch_t=8, patch_h=2, patch_w=1,
                 d_model=64):
        super().__init__()
        self.emg_conv = nn.Conv3d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w)
        )
        self.imu_conv = nn.Conv1d(
            in_channels=6,
            out_channels=d_model,
            kernel_size=patch_t,
            stride=patch_t
        )

        self.n_emg_t = window_size // patch_t        # temporal patches
        self.n_emg_h = 4 // patch_h                  # 3-row grid
        self.n_emg_w = 3 // patch_w                  # 4-col grid
        self.n_imu   = window_size // patch_t
        self.num_patches = self.n_emg_t * self.n_emg_h * self.n_emg_w + self.n_imu

    def forward(self, x):            # x: (B, W, 18)
            # ----- EMG branch -----
            emg = x[:, :, :12].contiguous()                 
            emg = emg.reshape(x.size(0), x.size(1), 4, 3)     # (B,T,4,3)
            emg = emg.unsqueeze(1)
            emg_tok = self.emg_conv(emg)                      # Conv3d now sees 5-D
            emg_tok = emg_tok.flatten(2).transpose(1, 2)

            # ----- IMU branch -----
            imu_tok = self.imu_conv(x[:, :, 12:].permute(0, 2, 1)).transpose(1, 2)

            return torch.cat([emg_tok, imu_tok], dim=1)
