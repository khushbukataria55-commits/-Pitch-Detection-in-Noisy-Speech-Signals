"""
evaluate.py – Evaluation and result visualisation for MFCRT pitch detection.

Generates synthetic voiced + noisy speech, runs MFCRT, compares against
baseline (autocorrelation) and plots results.

Usage:
    python evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import chirp
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from mfcrt import mfcrt_pitch_detection, pre_emphasis, frame_signal, is_voiced


# ── Synthesis helpers ──────────────────────────────────────────────────────────

def synthesise_voiced_speech(duration: float, fs: int,
                              f0_hz: float = 150.0,
                              n_harmonics: int = 10) -> np.ndarray:
    """
    Synthesise a voiced speech-like signal as a sum of harmonics with
    random amplitude weighting to mimic natural vowel formants.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / (k ** 0.8)   # spectral tilt
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * k * f0_hz * t + phase)
    signal /= np.max(np.abs(signal) + 1e-10)
    return signal


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (dB)."""
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


# ── Baseline: Autocorrelation pitch detector ──────────────────────────────────

def autocorr_pitch(signal: np.ndarray, fs: int,
                   frame_ms: int = 30, hop_ms: int = 10) -> dict:
    """Standard autocorrelation-based pitch detector (YIN-style)."""
    from mfcrt import F0_MIN_HZ, F0_MAX_HZ, pre_emphasis, frame_signal, is_voiced

    T_min = int(fs / F0_MAX_HZ)
    T_max = int(fs / F0_MIN_HZ)

    emphasized = pre_emphasis(signal)
    frames     = frame_signal(emphasized, fs, frame_ms, hop_ms)
    hop_len    = int(fs * hop_ms / 1000)
    frame_len  = frames.shape[1]
    num_frames = frames.shape[0]
    times      = np.array([(i * hop_len + frame_len // 2) / fs
                            for i in range(num_frames)])
    f0_raw     = np.zeros(num_frames)
    voiced     = np.zeros(num_frames, dtype=bool)

    for i, frame in enumerate(frames):
        voiced[i] = is_voiced(frame)
        if not voiced[i]:
            continue
        ac     = np.correlate(frame, frame, mode='full')
        ac     = ac[len(ac) // 2:]
        ac    /= ac[0] + 1e-10
        valid  = ac[T_min:T_max + 1]
        if valid.size == 0:
            continue
        peak   = np.argmax(valid) + T_min
        f0_raw[i] = fs / peak if peak > 0 else 0.0

    return {'f0': f0_raw, 'times': times, 'voiced': voiced}


# ── Metrics ────────────────────────────────────────────────────────────────────

def gross_pitch_error(f0_est: np.ndarray, f0_ref: float,
                      voiced_mask: np.ndarray, p: float = 0.20) -> float:
    """
    Gross Pitch Error (GPE): fraction of voiced frames whose estimate
    deviates from the reference by more than p × f0_ref.
    """
    est = f0_est[voiced_mask & (f0_est > 0)]
    if len(est) == 0:
        return 1.0
    errors = np.abs(est - f0_ref) / f0_ref
    return float(np.mean(errors > p))


def mean_absolute_error(f0_est: np.ndarray, f0_ref: float,
                         voiced_mask: np.ndarray) -> float:
    """Mean Absolute Error in Hz for voiced frames."""
    est = f0_est[voiced_mask & (f0_est > 0)]
    return float(np.mean(np.abs(est - f0_ref))) if len(est) > 0 else np.nan


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(42)
    FS      = 16000
    DUR     = 1.5    # seconds
    F0_TRUE = 150.0  # Hz
    SNR_LEVELS = [20, 10, 5, 0, -5]

    print("=" * 60)
    print("MFCRT vs Autocorrelation – Pitch Detection Benchmark")
    print("=" * 60)
    print(f"  True F0    : {F0_TRUE} Hz")
    print(f"  Sample rate: {FS} Hz")
    print(f"  Duration   : {DUR} s")
    print()

    clean_signal = synthesise_voiced_speech(DUR, FS, F0_TRUE)

    gpe_mfcrt, gpe_ac = [], []
    mae_mfcrt, mae_ac = [], []

    for snr in SNR_LEVELS:
        noisy = add_noise(clean_signal, snr)

        res_mfcrt = mfcrt_pitch_detection(noisy, FS)
        res_ac    = autocorr_pitch(noisy, FS)

        v = res_mfcrt['voiced']

        gpe_m = gross_pitch_error(res_mfcrt['f0_smooth'], F0_TRUE, v)
        gpe_a = gross_pitch_error(res_ac['f0'],           F0_TRUE, v)
        mae_m = mean_absolute_error(res_mfcrt['f0_smooth'], F0_TRUE, v)
        mae_a = mean_absolute_error(res_ac['f0'],           F0_TRUE, v)

        gpe_mfcrt.append(gpe_m); gpe_ac.append(gpe_a)
        mae_mfcrt.append(mae_m); mae_ac.append(mae_a)

        print(f"  SNR {snr:+3d} dB │ MFCRT GPE={gpe_m:.2%}  MAE={mae_m:.1f} Hz"
              f"   │  AC GPE={gpe_a:.2%}  MAE={mae_a:.1f} Hz")

    return SNR_LEVELS, gpe_mfcrt, gpe_ac, mae_mfcrt, mae_ac, clean_signal, FS, F0_TRUE


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(snr_levels, gpe_mfcrt, gpe_ac, mae_mfcrt, mae_ac,
                 clean_signal, fs, f0_true):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f1117')

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.92, bottom=0.07)

    ACCENT  = '#4fc3f7'
    ACCENT2 = '#ef5350'
    GRID    = '#2a2d3a'
    TEXT    = '#e0e0e0'
    BG      = '#1a1d27'

    plt.rcParams.update({
        'text.color': TEXT, 'axes.labelcolor': TEXT,
        'xtick.color': TEXT, 'ytick.color': TEXT,
        'axes.facecolor': BG, 'axes.edgecolor': GRID,
        'grid.color': GRID, 'legend.facecolor': BG,
        'legend.edgecolor': GRID,
    })

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=11, pad=8)
        ax.grid(True, alpha=0.3, color=GRID)
        ax.spines[:].set_color(GRID)

    # ── Subplot 1: Waveform ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    noisy_0 = add_noise(clean_signal, 0)
    t = np.linspace(0, len(clean_signal)/fs, len(clean_signal))
    ax1.plot(t, clean_signal, color=ACCENT,  alpha=0.8, lw=0.8, label='Clean')
    ax1.plot(t, noisy_0,      color=ACCENT2, alpha=0.5, lw=0.5, label='Noisy (0 dB SNR)')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=9)
    style_ax(ax1, 'Speech Waveform: Clean vs 0 dB SNR')

    # ── Subplot 2: F0 Trajectory ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    np.random.seed(42)
    noisy_10 = add_noise(clean_signal, 10)
    res_m = mfcrt_pitch_detection(noisy_10, fs)
    res_a = autocorr_pitch(noisy_10, fs)
    ax2.axhline(f0_true, color='white', ls='--', lw=1.2, alpha=0.6, label=f'True F0 = {f0_true} Hz')
    ax2.plot(res_m['times'], res_m['f0_smooth'], color=ACCENT,  lw=1.5, label='MFCRT (smoothed)')
    ax2.plot(res_a['times'], res_a['f0'],        color=ACCENT2, lw=1.0, alpha=0.8, label='Autocorrelation')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('F0 (Hz)')
    ax2.set_ylim(0, 350); ax2.legend(fontsize=9)
    style_ax(ax2, 'F0 Trajectory Comparison at 10 dB SNR')

    # ── Subplot 3: GPE vs SNR ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(snr_levels, [g*100 for g in gpe_mfcrt], 'o-', color=ACCENT,  lw=2, label='MFCRT')
    ax3.plot(snr_levels, [g*100 for g in gpe_ac],    's-', color=ACCENT2, lw=2, label='Autocorr')
    ax3.set_xlabel('SNR (dB)'); ax3.set_ylabel('GPE (%)')
    ax3.legend(fontsize=9); style_ax(ax3, 'Gross Pitch Error vs. SNR')

    # ── Subplot 4: MAE vs SNR ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(snr_levels, mae_mfcrt, 'o-', color=ACCENT,  lw=2, label='MFCRT')
    ax4.plot(snr_levels, mae_ac,    's-', color=ACCENT2, lw=2, label='Autocorr')
    ax4.set_xlabel('SNR (dB)'); ax4.set_ylabel('MAE (Hz)')
    ax4.legend(fontsize=9); style_ax(ax4, 'Mean Absolute Error vs. SNR')

    fig.suptitle('MFCRT Pitch Detection – Results', color=TEXT, fontsize=14, fontweight='bold')
    plt.savefig('/home/claude/mfcrt_pitch/results/mfcrt_results.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n  Plot saved → results/mfcrt_results.png")
    plt.close()


if __name__ == '__main__':
    results = run_experiment()
    plot_results(*results)
    print("\nDone.")
