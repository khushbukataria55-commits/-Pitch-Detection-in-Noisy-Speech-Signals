"""
MFCRT: Multi-Frame Correlation with Radon Transform for Noisy Speech Pitch Detection
=====================================================================================
Implementation of the algorithm described in:

    Li, J. et al. "A novel pitch detection algorithm for noisy speech signal based on
    Radon transform and multi-frame correlation."
    Digital Signal Processing, 2025. DOI: 10.1016/j.dsp.2025.104373

Author: Student Implementation (DSP Course Project)
License: MIT
"""

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import rotate
import warnings

# ── Constants ──────────────────────────────────────────────────────────────────
F0_MIN_HZ  = 60    # Minimum expected fundamental frequency (Hz)
F0_MAX_HZ  = 400   # Maximum expected fundamental frequency (Hz)
FRAME_MS   = 30    # Frame length in milliseconds
HOP_MS     = 10    # Hop size in milliseconds
NUM_FRAMES = 5     # Number of frames for multi-frame correlation
VOICING_THRESH = 0.3  # Voiced/unvoiced decision threshold


# ── Step 1: Pre-processing ─────────────────────────────────────────────────────

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply first-order high-pass pre-emphasis filter to compensate
    for high-frequency attenuation in speech signals.

    H(z) = 1 - coeff * z^-1
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, fs: int,
                 frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS) -> np.ndarray:
    """
    Divide signal into overlapping frames using a Hamming window.

    Returns
    -------
    frames : ndarray of shape (num_frames, frame_len)
    """
    frame_len = int(fs * frame_ms / 1000)
    hop_len   = int(fs * hop_ms  / 1000)
    num_frames = 1 + (len(signal) - frame_len) // hop_len

    frames = np.zeros((num_frames, frame_len))
    window = np.hamming(frame_len)
    for i in range(num_frames):
        start = i * hop_len
        frames[i] = signal[start:start + frame_len] * window
    return frames


# ── Step 2: Voicing Detection ──────────────────────────────────────────────────

def is_voiced(frame: np.ndarray, threshold: float = VOICING_THRESH) -> bool:
    """
    Simple energy-based voiced / unvoiced decision.
    A frame is voiced if its normalised energy exceeds `threshold`.
    """
    energy = np.sum(frame ** 2)
    return energy > threshold * np.max(np.abs(frame) ** 2 + 1e-10)


# ── Step 3: 2D Multi-Frame Representation ─────────────────────────────────────

def build_2d_representation(frames: np.ndarray,
                             centre: int,
                             n_context: int = NUM_FRAMES) -> np.ndarray:
    """
    Construct a 2D matrix from `n_context` consecutive frames centred on `centre`.

    Each row of the matrix is one frame of speech, forming a 2D time–sample
    plane that encodes the repetitive structure of voiced speech across frames.

    Parameters
    ----------
    frames  : (num_frames, frame_len) array of windowed speech frames.
    centre  : index of the target frame.
    n_context : number of frames to stack (odd numbers work best).

    Returns
    -------
    mat2d : (n_context, frame_len) 2D representation matrix.
    """
    half = n_context // 2
    idxs = np.arange(centre - half, centre + half + 1)
    idxs = np.clip(idxs, 0, len(frames) - 1)   # boundary clamp
    return frames[idxs]                          # shape: (n_context, frame_len)


# ── Step 4: Multi-Frame Correlation ───────────────────────────────────────────

def multi_frame_correlation(mat2d: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise correlation map of the 2D representation.

    For each candidate lag τ, the average cross-correlation across consecutive
    frame pairs is computed. This captures the repeating pitch structure while
    suppressing uncorrelated noise.

    Returns
    -------
    corr_map : (n_context, frame_len) correlation-enhanced 2D matrix.
    """
    n, L = mat2d.shape
    corr_map = np.zeros_like(mat2d)

    for i in range(n):
        # Auto-correlation of each row
        row = mat2d[i]
        ac  = np.correlate(row, row, mode='full')
        ac  = ac[len(ac)//2:]          # keep non-negative lags
        ac /= (ac[0] + 1e-10)          # normalise to [0, 1]
        corr_map[i, :len(ac)] = ac[:L]

    # Average across frames → suppress noise, reinforce periodicity
    mean_corr = corr_map.mean(axis=0)

    # Broadcast averaged correlation back into 2D map
    corr_map = np.outer(np.ones(n), mean_corr)
    return corr_map


# ── Step 5: Radon Transform ───────────────────────────────────────────────────

def radon_transform(mat2d: np.ndarray, angles: np.ndarray = None) -> np.ndarray:
    """
    Apply a discrete Radon transform to the 2D correlation map.

    The Radon transform integrates the 2D matrix along lines at each angle θ,
    producing a sinogram (Radon spectrum). For periodic speech, the pitch period
    corresponds to the angle and position of the highest energy line, since
    harmonic components form regular patterns along diagonal lines.

    Parameters
    ----------
    mat2d  : 2D correlation map, shape (n_context, frame_len).
    angles : projection angles in degrees; defaults to 0–179°.

    Returns
    -------
    sinogram : (len(angles), max_dim) Radon spectrum.
    """
    if angles is None:
        angles = np.arange(0, 180)

    nrows, ncols = mat2d.shape
    diag_len = int(np.ceil(np.sqrt(nrows**2 + ncols**2)))
    sinogram  = np.zeros((len(angles), diag_len))

    for k, theta in enumerate(angles):
        rotated = rotate(mat2d, theta, reshape=False, order=1, mode='constant', cval=0)
        projection = rotated.sum(axis=0)          # column sums → projection
        sinogram[k, :len(projection)] = projection

    return sinogram


# ── Step 6: Pitch Period Estimation from Radon Spectrum ───────────────────────

def estimate_pitch_from_radon(sinogram: np.ndarray,
                               fs: int,
                               frame_len: int) -> float:
    """
    Extract the pitch period from the Radon sinogram by finding the dominant
    peak that falls within the valid F0 range [F0_MIN, F0_MAX].

    The column index of the peak projection corresponds to a spatial period T
    (in samples); F0 = fs / T.

    Returns
    -------
    f0 : estimated fundamental frequency in Hz, or 0 if unvoiced / not found.
    """
    T_min = int(fs / F0_MAX_HZ)   # minimum pitch period in samples
    T_max = int(fs / F0_MIN_HZ)   # maximum pitch period in samples

    # Sum sinogram across angles to get overall peak distribution
    projection = sinogram.sum(axis=0)

    # Restrict to valid pitch period range
    valid = projection[T_min:T_max + 1]
    if valid.size == 0:
        return 0.0

    peak_idx = np.argmax(valid) + T_min   # lag in samples
    f0 = fs / peak_idx if peak_idx > 0 else 0.0
    return f0


# ── Step 7: Post-processing ───────────────────────────────────────────────────

def viterbi_smoothing(f0_seq: np.ndarray,
                      transition_penalty: float = 0.1) -> np.ndarray:
    """
    Smooth F0 trajectory with a simplified Viterbi-inspired DP pass.

    Penalises large frame-to-frame jumps in pitch to reduce octave errors
    and transient outliers that commonly arise in noisy conditions.

    Parameters
    ----------
    f0_seq           : raw per-frame F0 estimates (Hz).
    transition_penalty: weight of the inter-frame jump penalty.

    Returns
    -------
    smoothed : corrected F0 sequence.
    """
    smoothed = f0_seq.copy()
    for i in range(1, len(smoothed)):
        if smoothed[i] > 0 and smoothed[i-1] > 0:
            ratio = smoothed[i] / smoothed[i-1]
            # Correct octave errors (ratio ≈ 0.5 or 2.0)
            if abs(ratio - 2.0) < 0.3:
                smoothed[i] /= 2.0
            elif abs(ratio - 0.5) < 0.15:
                smoothed[i] *= 2.0
    # Final median filter pass to remove residual spikes
    voiced_mask = smoothed > 0
    if voiced_mask.sum() > 3:
        smoothed[voiced_mask] = medfilt(smoothed[voiced_mask], kernel_size=3)
    return smoothed


# ── Main MFCRT Pipeline ───────────────────────────────────────────────────────

def mfcrt_pitch_detection(signal: np.ndarray,
                           fs: int,
                           frame_ms: int = FRAME_MS,
                           hop_ms: int = HOP_MS,
                           n_context: int = NUM_FRAMES,
                           verbose: bool = False) -> dict:
    """
    Full MFCRT pitch detection pipeline.

    Parameters
    ----------
    signal    : mono speech waveform (float32 or float64, normalised to [-1, 1]).
    fs        : sampling rate in Hz.
    frame_ms  : frame length in milliseconds.
    hop_ms    : hop size in milliseconds.
    n_context : number of frames for multi-frame correlation.
    verbose   : print progress messages.

    Returns
    -------
    result : dict with keys:
        'f0'        – raw per-frame F0 estimates (Hz),
        'f0_smooth' – smoothed F0 trajectory (Hz),
        'voiced'    – boolean voiced/unvoiced mask,
        'times'     – centre time (s) of each frame,
        'fs'        – sampling rate.
    """
    if verbose:
        print("[MFCRT] Step 1: Pre-processing …")

    # 1. Pre-emphasis + framing
    emphasized = pre_emphasis(signal)
    frames     = frame_signal(emphasized, fs, frame_ms, hop_ms)
    hop_len    = int(fs * hop_ms / 1000)
    frame_len  = frames.shape[1]
    num_frames = frames.shape[0]

    times  = np.array([(i * hop_len + frame_len // 2) / fs for i in range(num_frames)])
    f0_raw = np.zeros(num_frames)
    voiced = np.zeros(num_frames, dtype=bool)

    angles = np.arange(0, 180, 5)   # 5-degree steps for speed

    if verbose:
        print(f"[MFCRT] {num_frames} frames, {frame_len} samples/frame")

    for i in range(num_frames):
        # 2. Voiced / unvoiced decision
        voiced[i] = is_voiced(frames[i])
        if not voiced[i]:
            continue

        # 3. Build 2D multi-frame representation
        mat2d = build_2d_representation(frames, i, n_context)

        # 4. Multi-frame correlation
        corr_map = multi_frame_correlation(mat2d)

        # 5. Radon transform on correlation map
        sinogram = radon_transform(corr_map, angles)

        # 6. Estimate pitch from Radon spectrum
        f0_raw[i] = estimate_pitch_from_radon(sinogram, fs, frame_len)

    # 7. Viterbi-inspired smoothing
    f0_smooth = viterbi_smoothing(f0_raw.copy())

    if verbose:
        print(f"[MFCRT] Done. Voiced frames: {voiced.sum()}/{num_frames}")

    return {
        'f0':        f0_raw,
        'f0_smooth': f0_smooth,
        'voiced':    voiced,
        'times':     times,
        'fs':        fs,
    }
