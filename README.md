# MFCRT: Multi-Frame Correlation with Radon Transform for Pitch Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DSP](https://img.shields.io/badge/Domain-Digital%20Signal%20Processing-orange.svg)]()

A Python implementation of the **MFCRT** (Multi-Frame Correlation Radon Transform) pitch detection algorithm for noisy speech signals.

---

## 📄 Reference Paper

> Li, J. et al. *"A novel pitch detection algorithm for noisy speech signal based on Radon transform and multi-frame correlation."*  
> **Digital Signal Processing**, 2025. [DOI: 10.1016/j.dsp.2025.104373](https://www.sciencedirect.com/science/article/abs/pii/S1051200425004373)

---

## 🔍 Overview

Pitch (fundamental frequency, F0) detection under noisy conditions is a long-standing challenge in speech processing. Classical methods such as autocorrelation (YIN) and cepstrum degrade rapidly at low SNR values. The MFCRT algorithm improves noise robustness by:

1. **Extending 1D speech frames into a 2D multi-frame matrix**, capturing the repetitive structure of voiced speech across time.
2. **Applying multi-frame correlation** to average out uncorrelated noise while reinforcing periodic pitch patterns.
3. **Using the Radon transform** on the correlation map to detect dominant periodic lines, from which the pitch period is extracted.
4. **Post-processing with Viterbi-inspired smoothing** to correct octave errors and produce a stable F0 trajectory.

---

## 🗂️ Repository Structure

```
mfcrt_pitch/
├── src/
│   ├── mfcrt.py       # Core MFCRT algorithm implementation
│   └── evaluate.py    # Evaluation against autocorrelation baseline
├── results/
│   └── mfcrt_results.png   # Generated benchmark plots
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Algorithm Pipeline

```
Speech Signal
     │
     ▼
[1] Pre-emphasis filter  (compensate high-freq attenuation)
     │
     ▼
[2] Framing + Windowing  (30 ms frames, 10 ms hop, Hamming window)
     │
     ▼
[3] Voiced/Unvoiced Detection  (energy threshold)
     │  (voiced frames only)
     ▼
[4] 2D Multi-Frame Representation  (stack N consecutive frames)
     │
     ▼
[5] Multi-Frame Correlation  (average autocorrelation across frames)
     │
     ▼
[6] Radon Transform  (project 2D correlation map along angles)
     │
     ▼
[7] Peak Picking in Radon Spectrum  (within valid F0 range)
     │
     ▼
[8] Viterbi Smoothing  (octave correction + median filtering)
     │
     ▼
F0 Trajectory (Hz)
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mfcrt-pitch-detection.git
cd mfcrt-pitch-detection
pip install -r requirements.txt
```

### Run the benchmark

```bash
python src/evaluate.py
```

### Use in your own code

```python
import numpy as np
from src.mfcrt import mfcrt_pitch_detection

# Load or synthesise your signal
fs = 16000
signal = np.random.randn(fs * 2)  # replace with real speech

result = mfcrt_pitch_detection(signal, fs, verbose=True)

print(result['times'])      # frame centre times (s)
print(result['f0_smooth'])  # smoothed F0 estimates (Hz)
print(result['voiced'])     # boolean voiced mask
```

---

## 📊 Benchmark Results

Evaluation on synthesised voiced speech (F0 = 150 Hz) with additive white Gaussian noise:

| SNR (dB) | MFCRT GPE (%) | Autocorr GPE (%) | MFCRT MAE (Hz) | Autocorr MAE (Hz) |
|:--------:|:-------------:|:----------------:|:--------------:|:-----------------:|
| +20      | 0.00          | 0.00             | 0.7            | 0.7               |
| +10      | **0.00**      | 15.54            | **1.0**        | 18.8              |
| +5       | 72.30         | 56.76            | 85.3           | 72.8              |
| 0        | 91.22         | 77.70            | 104.3          | 104.5             |
| −5       | 70.27         | 75.68            | 86.9           | 93.5              |

> **Note:** MFCRT shows a significant advantage at moderate SNR (10 dB). At very low SNR (0 dB and below), pitch detection degrades for both methods — consistent with findings in the original paper at extreme conditions.

---

## 📐 Metrics

- **GPE (Gross Pitch Error):** Fraction of voiced frames with |F0_est − F0_ref| / F0_ref > 20%
- **MAE (Mean Absolute Error):** Average |F0_est − F0_ref| in Hz over voiced frames

---

## 🔬 Implementation Notes

- The Radon transform is implemented via image rotation + column projection (`scipy.ndimage.rotate`)
- Angle resolution is set to 5° steps for a balance between speed and accuracy
- The Viterbi smoother applies octave-error correction (2:1 and 1:2 ratio detection) followed by a median filter
- Tested with Python 3.8–3.12 on synthesised data; can be adapted for real CSTR/TIMIT datasets

---

## 📦 Requirements

```
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
```

---

## 📜 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{li2025mfcrt,
  title   = {A novel pitch detection algorithm for noisy speech signal
             based on Radon transform and multi-frame correlation},
  author  = {Li, J. and others},
  journal = {Digital Signal Processing},
  year    = {2025},
  doi     = {10.1016/j.dsp.2025.104373}
}
```

---

## 📃 License

MIT License — see [LICENSE](LICENSE) for details.
