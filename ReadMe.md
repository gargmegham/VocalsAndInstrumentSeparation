# Vocals and Instrument Separation

A research and implementation notebook for separating singing vocals from instrumental accompaniment in music recordings using deep learning. The project explores multiple approaches, from classical signal-processing baselines to a full PyTorch implementation of the Open-Unmix architecture.

---

## Table of Contents

- [Overview](#overview)
- [Approaches](#approaches)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Architecture & Configuration](#model-architecture--configuration)
- [Research References](#research-references)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Music source separation is the task of decomposing a mixed audio signal into its constituent sources — vocals, drums, bass, and other instruments. This project implements and demonstrates:

- A **ratio mask baseline** using oracle spectrograms
- A **deep neural network** approach based on [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch), trained on the MUSDB18 dataset
- Wiener filter post-processing for improved separation quality
- Flexible input handling: local audio files, YouTube downloads, or direct uploads (Colab)

The entire implementation lives in `main.ipynb`, designed to run in Google Colab with GPU acceleration.

---

## Approaches

### 1. Oracle / Ratio Mask Filtering
A signal-processing baseline that constructs soft masks from ground-truth spectrograms. Useful for understanding the upper-bound quality of mask-based separation, but requires access to the original isolated sources.

### 2. Open-Unmix (Deep Learning)
A bidirectional LSTM-based neural network that estimates a magnitude spectrogram mask for each target source. Key steps:

1. Compute STFT of the mixed audio
2. Normalize the spectrogram with training-set statistics (mean/scale)
3. Pass through the neural network to predict a mask
4. Apply the mask and reconstruct the waveform with inverse STFT
5. Optionally refine with iterative Wiener filtering (via [Norbert](https://github.com/sigsep/norbert))

Supports four separation targets: **vocals**, **drums**, **bass**, and **other**. An accompaniment signal can be derived residually from these estimates.

---

## Dataset

The project uses **[MUSDB18](https://sigsep.github.io/datasets/musdb.html)**, a dataset of 150 full-length music tracks with isolated stems.

- A 7-second preview subset is used for quick experimentation
- The full training set is used for model training
- The `musdb` Python library handles loading and iteration

---

## Dependencies

All dependencies are installed within the notebook. No separate `requirements.txt` is needed.

**System packages:**
- `libsndfile1` — audio file I/O
- `ffmpeg` — multimedia encoding/decoding

**Python packages:**

| Package | Purpose |
|---|---|
| `torch` (GPU) | Deep learning framework |
| `librosa` 0.6.3 | Audio analysis, STFT utilities |
| `musdb` 0.3.1 | MUSDB18 dataset loader |
| `museval` 0.3.0 | BSS separation metrics (SDR, SIR, SAR) |
| `norbert` | Wiener filter post-processing |
| `stempeg` 0.1.8 | STEM file format I/O |
| `soundfile` | WAV read/write |
| `numpy`, `scipy` | Numerical and signal processing |
| `pandas` 0.25.1 | Data handling |
| `matplotlib`, `seaborn` | Visualization |
| `youtube-dl` | YouTube audio download |
| `scikit-learn`, `scikit_posthocs` | Utilities and statistical tests |

---

## Getting Started

1. Open `main.ipynb` in [Google Colab](https://colab.research.google.com/) (recommended — GPU required for training).
2. Run the setup cells to install system and Python dependencies.
3. Choose your workflow: use pretrained Open-Unmix weights or train from scratch on MUSDB18.

To run locally, ensure you have a CUDA-capable GPU and install the packages listed above manually.

---

## Usage

### Separate audio using pretrained Open-Unmix

```python
import torch

# Load pretrained model from torch.hub
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', target='vocals')

# Separate — input is a path to a WAV file or a tensor
estimates = test.separate(
    audio,
    targets=['vocals', 'drums', 'bass', 'other'],
    model_name='umxhq',
    niter=1,               # Wiener filter iterations
    residual_model=False,
    device='cuda'
)
```

### Process audio from YouTube

```python
# Download and separate a YouTube track
# (youtube-dl must be installed)
!youtube-dl -x --audio-format wav -o "input.wav" "<youtube-url>"
```

### Export separated stems

```python
import stempeg
# Write multi-stem file
stempeg.write_stems("output.stem.mp4", stems, sample_rate=44100)

# Or write individual WAV files
soundfile.write("vocals.wav", vocals_audio, samplerate=44100)
```

### Train from scratch

The notebook contains a full training loop:

```python
# Wraps MUSDB18 into a PyTorch Dataset
dataset = SimpleMUSDBDataset(mus, target='vocals', seq_dur=5.0)

# Train OpenUnmix for 50 epochs
model = OpenUnmix(n_fft=2048, n_hop=1024, hidden_size=256)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
# ... training loop in notebook
```

---

## Model Architecture & Configuration

| Parameter | Value |
|---|---|
| FFT window (`n_fft`) | 2048 |
| Hop size (`n_hop`) | 1024 |
| Hidden size | 256 |
| Max frequency bin (`max_bin`) | 64 |
| Sample rate | 44,100 Hz |
| Training epochs | 50 |
| Batch size | 8 |
| Sequence duration | 5.0 s |
| Optimizer | RMSprop (lr=0.001) |
| Loss | MSE |
| Wiener filter iterations | 1–2 |

Input spectrograms are standardized using per-frequency mean and scale computed over the training set.

---

## Research References

The following papers and resources informed this project (PDFs included in the repository):

- **Audio Source Separation** — survey of classical and DNN-based methods
- **Multichannel Audio Source Separation with Deep Neural Networks** — DNN mask estimation for multichannel mixtures
- **Audio Classification**, **Audio Fingerprinting**, **Audio Segmentation** — supporting audio ML literature
- [Open-Unmix (UMX)](https://hal.inria.fr/hal-02293689) — Stöter et al., JOSS 2019
- [MUSDB18 Dataset](https://sigsep.github.io/datasets/musdb.html) — Rafii et al.
- [Norbert](https://github.com/sigsep/norbert) — Wiener filtering for audio source separation

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the [MIT License](LICENSE). Copyright 2024 Megham Garg.

---

## Contact

[meghamgarg@gmail.com](mailto:meghamgarg@gmail.com)
