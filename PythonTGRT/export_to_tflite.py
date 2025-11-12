"""
Export a trained PyTorch student model to ONNX → TensorFlow SavedModel → TFLite,
and (optionally) analyze the resulting TFLite graph/tensors. Includes a small
representative dataset generator for post-training quantization.

Constants / Config
------------------
WINDOW_SIZE : int
N_CHANNELS  : int
D_STUD      : int
H_STUD      : int
L_STUD      : int
PATCH_LEN   : int
K_STUD      : int
N_CLASSES   : int
DATA_FILE   : str
FILT_FILE   : str
    Model/data hyperparameters and paths used across export stages.

Functions
---------
representative_dataset() -> Iterable[list[np.ndarray]]
    Yields ~100 input samples shaped [1, WINDOW_SIZE, N_CHANNELS] as float32
    for TFLite calibration. Uses SNR_lag(FILT_FILE) and EMGIMUTextDataset(DATA_FILE).

load_student_model(pth_path: str, device: str = "cpu")
    -> GestureLinformerTransformerNoCLS
    Construct the student model with constants above, load state_dict from .pth,
    set to eval(), and return it.

export_to_onnx(model: torch.nn.Module, onnx_path: str) -> None
    Export model with a dummy input [1, WINDOW_SIZE, N_CHANNELS], opset 15.

onnx_to_saved_model(onnx_path: str, export_dir: str) -> None
    Convert ONNX → TensorFlow SavedModel via onnx-tf backend.

saved_model_to_tflite(saved_model_dir: str, tflite_path: str) -> None
    Convert SavedModel → .tflite (float). Hooks for full-int8 flow are present
    but commented (optimizations, representative_dataset, I/O dtypes).

analyze_tflite_model(tflite_path: str) -> None
    Load a .tflite, allocate tensors, and print a table of tensor index/name,
    shape, dtype, and per-tensor byte size, plus an overall byte total.

main() -> None
    CLI: --pth .pth, --onnx .onnx, --tfdir SavedModel dir, --tflite .tflite.
    Runs: load → ONNX export → SavedModel → TFLite → analyze.
"""


import argparse
import torch
import onnx
import numpy as np
# re-create the old alias so downstream libs don’t break
if not hasattr(np, 'bool'):
    np.bool = bool
    
from onnx_tf.backend import prepare
import tensorflow as tf
from torch.utils.data import DataLoader
from models import GestureLinformerTransformerNoCLS
from lag_analyzer import SNR_lag

# match trained model hyperparameters
WINDOW_SIZE = 512
N_CHANNELS = 18
D_STUD      = 32
H_STUD      = 4
L_STUD      = 1
PATCH_LEN   = 8
K_STUD      = 16
N_CLASSES   = 4
DATA_FILE   = 'Python_TGRT/data/eLog0_filt_tke_ma.txt'
FILT_FILE = 'Python_TGRT/data/eLog0_filt.txt'  # for lag analysis


def representative_dataset():
    from data_pipeline import EMGIMUTextDataset
    from torch.utils.data import DataLoader
    dataLag = SNR_lag(FILT_FILE, plot = False)  # analyze lag to improve accuracy
    ds = EMGIMUTextDataset(DATA_FILE, WINDOW_SIZE, PATCH_LEN, snr_shift=dataLag)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    for i, (x, _) in enumerate(loader):
        if i >= 100:  # calibrate on 100 examples
            break
        # x: torch.Tensor of shape [1,512,18]
        yield [x.numpy().astype(np.float32)]

def load_student_model(pth_path, device='cpu'):
    model = GestureLinformerTransformerNoCLS(
        n_channels=N_CHANNELS,
        window_size=WINDOW_SIZE,
        d_model=D_STUD,
        n_heads=H_STUD,
        num_layers=L_STUD,
        patch_len=PATCH_LEN,
        k=K_STUD,
        n_classes=N_CLASSES
    )
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def export_to_onnx(model, onnx_path):
    dummy = torch.randn(1, WINDOW_SIZE, N_CHANNELS)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=15
    )
    print(f"ONNX model saved to {onnx_path}")


def onnx_to_saved_model(onnx_path, export_dir):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(export_dir)
    print(f"TensorFlow SavedModel saved to {export_dir}")


def saved_model_to_tflite(saved_model_dir, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

def analyze_tflite_model(tflite_path):
    """
    Loads a TFLite model and prints a summary of its tensors and memory usage.
    """
    print("\n--- TFLite Model Analysis ---")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    total_bytes = 0
    print(f"{'INDEX':<6} {'NAME':<40} {'SHAPE':<20} {'TYPE':<10} {'SIZE (BYTES)':<15}")
    print("="*100)
    
    for i, tensor in enumerate(tensor_details):
        tensor_name = tensor['name']
        tensor_shape = tensor['shape']
        tensor_type = tensor['dtype']
        
        # Calculate the size of the tensor in bytes
        # np.prod calculates the product of all elements in the shape array
        tensor_size = np.prod(tensor_shape) * np.dtype(tensor_type).itemsize
        total_bytes += tensor_size
        
        print(f"{i:<6} {tensor_name:<40} {str(tensor_shape):<20} {str(np.dtype(tensor_type)):<10} {tensor_size:<15}")

    print("="*100)
    print(f"\nTotal size of all tensors (weights, activations, etc.): {total_bytes / 1024:.2f} KB ({total_bytes} bytes)")
    print("NOTE: The actual 'arena_used_bytes' on the device will be SMALLER than this total.")
    print("This is because TFLite Micro reuses memory for tensors that are not active at the same time.")
    print("However, this total is a very useful upper-bound for comparing model complexity.\n")


def main():
    parser = argparse.ArgumentParser(description="Export student PyTorch model to TFLite")
    parser.add_argument('--pth',    default='Models/StudentGold_Final.pth', help='Path to trained .pth file')
    parser.add_argument('--onnx',   default='Models/StudentGold_Final.onnx', help='Output ONNX filename')
    parser.add_argument('--tfdir',  default='Models/tf_model_StudentGold_Final',     help='Temp SavedModel directory')
    parser.add_argument('--tflite', default='Models/model_StudentGold_Final.tflite', help='Final TFLite filename')
    args = parser.parse_args()

    student = load_student_model(args.pth)
    export_to_onnx(student, args.onnx)
    onnx_to_saved_model(args.onnx, args.tfdir)
    saved_model_to_tflite(args.tfdir, args.tflite)

    analyze_tflite_model(args.tflite)  

    print("All done!\nNext: use 'xxd -i model.tflite > model_data.h' to generate C array for TFLite Micro.")


if __name__ == '__main__':
    main()
