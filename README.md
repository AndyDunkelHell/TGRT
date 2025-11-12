# Tiny Gesture Recognition Transformer (TGRT)

A complete, reproducible pipeline for **hands-free gesture control** using **high-density EMG (12 ch)** + **IMU (6 ch)** with a **Tiny Gesture Recognition Transformer** that runs **fully on the Arduino Portenta H7** (TensorFlow Lite Micro), plus a **Unity AR** workflow for **automatic labeling** and dataset generation. 

## Overview

* **Python tools** for data ingestion, preprocessing, lag compensation, training, evaluation, and deployment:

  * `data_pipeline.py` — build sliding-window EMG-IMU datasets from Unity logs, compute per-channel noise/SNR, thresholds, and lag masks; includes a `torch.utils.data.Dataset`. 
  * `lag_analyzer.py` — estimate session-specific label↔signal lag (SNR-based masks + xcorr) and export example plots. 
  * `LiveMonitor_Unity.py` — Tkinter GUI to stream/plot EMG+IMU from serial or file, compute SNR/MAV/CV, and run small lag analyses. 
  * `tke_ma.py` — EMG preprocessing recipe (band-pass → TKE → moving average) with Unity-style I/O. 
  * `models.py` & `patches.py` — teacher (full attention) and student (Linformer) Transformers with MCU-friendly patch embeddings/pooling.  
  * `evaluate_models.py` — compare teacher/student PyTorch and TFLite models on the same dataset; metrics, plots, confusion matrices. 
  * `export_to_tflite.py` — export student: PyTorch → ONNX → TF SavedModel → TFLite (with optional representative dataset). 
  * `createSamples.py` — write a C header with normalized test windows + μ/σ + FIR taps for firmware tests. 

* **PortentaTGRT** (PlatformIO project)

  * Dual-core firmware (M7: deterministic EMG sampling; M4: IMU + TFLM inference), custom memory layout, slim TFLM resolver; integrates the exported `.tflite` and C headers from `createSamples.py`. (Described in thesis Practical Framework) 

* **Unity TGRT** (scene + scripts)

  * **BBH** (serial/threading) and **PoseDetector** (XR-Hands based) for **AR-assisted automatic labeling**; records `time|label|e1..e12|i1..i6` logs for training.


## Practical Framework

1. **AR-assisted data collection**
   Use the Unity scene (XR Hands / MRTK) to detect poses (e.g., *pinch*, *fist*, *palm up*, *point*) while sampling EMG+IMU from the Portenta. Unity emits lines:
   `time|label|e1,...,e12|i1,...,i6[|hostUs]`. 

2. **Session-aware lag compensation (offline)**
   SNR(dB) per channel is computed via running RMS over a rest-noise baseline; per-channel thresholds (p95/max × %), mask bridging, and **cross-correlation** align labels with signal (no manual tuning). Apply the resulting `snr_shift` to labels before windowing.  

3. **Preprocessing**
   EMG optionally filtered (notch/band-pass), **Teager-Kaiser Energy** + short moving average improves onsets and SNR for training/inspection.  

4. **Dataset building**
   `EMGIMUTextDataset`: normalize (μ/σ), window (e.g., 512×18), overlap, majority-vote labels (discard all-rest). Save stats for reproducibility. 

5. **Models**

   * **Teacher**: ViT-style encoder with sequence pooling.
   * **Student (TGRT)**: **Linformer** attention (linear-time in sequence), MCU-friendly patch embed, sequence pooling; distilled from teacher.   

6. **Training & evaluation**
   Train teacher → distill student; evaluate PyTorch and TFLite on identical windows; export metrics, confusion matrices, and resource profiles (params, model size, peak RAM). 

7. **Export & deployment**
   Export student → ONNX → SavedModel → **TFLite**; optionally create C header test fixtures; integrate with **PortentaTGRT** firmware for **on-device inference** (low-ms latency).   


## Data format & conventions

* **Log line**: `time|label|e1..e12|i1..i6[|hostUs]`
* **Windowing**: typical 512 samples @ 500–1000 Hz; overlap configurable.
* **Normalization**: per-channel μ/σ stored and reused across runs.
* **Lag sign**: positive lag ⇒ labels are **early** (shift *right* to align).  

## License

MIT (code) + assets as noted. See individual folders for any exceptions.

## Implementation and Usage

I will gladly provide further details on implementing the software, setting up the control system, and integrating the hardware and software do not hesitate to contact me! 

## Contributing

I encourage contributions to enhance the software, refine control strategies, and expand the hand's capabilities. Please contact me for more information.

TGRT Project 2025
