# PulseVision

[English](README.md) | [Русский](README.ru.md)

PulseVision is an open-source PyTorch implementation of the **EfficientPhys-C** architecture proposed by Liu et al. in the research paper ["EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement"](https://arxiv.org/abs/2110.04447). The system performs contactless, real-time heart rate estimation (rPPG — remote photoplethysmography) directly from a standard webcam video stream.

## Method Overview

Remote photoplethysmography measures subtle variations in skin reflectance caused by cardiovascular pulse waves. The processing pipeline consists of four main stages:

1. **Face Detection and ROI Stabilization:**  
   During an initial 300-frame calibration period (~10 seconds), facial landmarks are tracked with MediaPipe FaceLandmarker to compute an aggregated, motion-tolerant bounding box. Statically cropping the face during measurement eliminates artificial pixel fluctuations and noise caused by frame-to-frame detector jitter.

2. **Spatiotemporal Processing (EfficientPhys-C):**  
   This repository implements the EfficientPhys-C model described in the original paper. Face crops are resized to $72 \times 72$ pixels and passed to the model in sliding windows of 100 frames. The network uses standard 2D convolutions augmented with Temporal Shift Modules (TSM) and spatial attention layers, capturing temporal blood flow dynamics without the high computational cost of 3D convolutions.

3. **Derivative Signal Target and Integration:**  
   The network is trained to predict the temporal first derivative of the blood volume pulse ($\Delta y_t$). Reconstructing the signal via cumulative summation (`cumsum`) functions as an intrinsic low-pass filter, significantly attenuating high-frequency sensor noise.

4. **Digital Signal Processing (DSP):**  
   The reconstructed pulse wave is centered, filtered with a zero-phase 2nd-order Butterworth bandpass filter ($0.75 - 2.5$ Hz, corresponding to $45 - 150$ BPM), and analyzed using Fast Fourier Transform (FFT) to extract the dominant cardiac frequency.

## Quick Start

### Requirements
* Python 3.10+
* Standard USB or built-in webcam

### Installation
Clone the repository and install required dependencies:

```bash
git clone https://github.com/mmmishaa/pulse-vision.git
cd pulse-vision
pip install -r requirements.txt
```

### Environment Configuration (Optional)
If you only plan to run real-time webcam inference, no environment setup or API keys are required.

If you want to reproduce dataset preparation or publish datasets to Kaggle, copy the template and provide your credentials:

```bash
cp .env.example .env
```

* `HF_TOKEN`: Hugging Face access token for downloading the raw MCD dataset (`kyegorov/mcd_rppg`) via `src/data/prepare_mcd.py`.
* `KAGGLE_USERNAME` / `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`): Credentials for uploading processed datasets to Kaggle via the Kaggle API.

### Running Inference
Run the real-time application:

```bash
python main.py
```

Pretrained weights for MediaPipe and EfficientPhys-C are downloaded automatically on the first launch.

### Controls
* Keep your head relatively steady during the initial 10-second calibration.
* Press `q` or `Esc` to terminate the application.

## Project Structure

```text
pulse-vision/
├── docs/                     # Research papers and reference documentation
├── notebooks/                # Training and validation notebooks
│   ├── efficientphys-c-train.ipynb
│   └── efficientphys-c-evaluation.ipynb
├── scripts/                  # Utility scripts
│   └── download_weights.py   # Automated weights downloader
├── src/
│   ├── capture/              # Video capture wrappers
│   ├── data/                 # Dataset preprocessing and loaders
│   ├── dsp/                  # Digital signal processing and FFT estimation
│   ├── models/               # EfficientPhys-C implementation, layers, and loss
│   ├── train/                # Training harness and learning rate scheduler
│   ├── utils/                # Project paths configuration
│   └── vision/               # MediaPipe FaceLandmarker detection
├── main.py                   # Real-time inference entrypoint
└── requirements.txt          # Python dependencies
```

## Datasets and Notebooks

* **Original Paper:** Liu et al. (2021). *EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement*. [arXiv:2110.04447](https://arxiv.org/abs/2110.04447).
* **Original Dataset:** [MCD-rPPG on Hugging Face](https://huggingface.co/datasets/kyegorov/mcd_rppg)
* **Preprocessed Dataset:** [MCD-rPPG Dataset on Kaggle](https://www.kaggle.com/datasets/mishasavinov/mcd-rppg-dataset)
* **Training Notebook:** [efficientphys-c-train.ipynb](notebooks/efficientphys-c-train.ipynb) (Kaggle: [mishasavinov/efficientphys-c-train](https://www.kaggle.com/code/mishasavinov/efficientphys-c-train))
* **Evaluation Notebook:** [efficientphys-c-evaluation.ipynb](notebooks/efficientphys-c-evaluation.ipynb) (Kaggle: [mishasavinov/efficientphys-c-evaluation](https://www.kaggle.com/code/mishasavinov/efficientphys-c-evaluation))
* **Pretrained Model Checkpoint:** [GitHub Release v0.1.0](https://github.com/mmmishaa/pulse-vision/releases/tag/v0.1.0)
