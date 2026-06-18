# -*- coding: utf-8 -*-
"""
Baja SAE Suspension Free Vibration Analysis
============================================
Processes impact-excitation data from a drop-rig or hammer test to extract
modal parameters of the suspension system during free vibration decay.

Engineering workflow
--------------------
  1. Load raw DAQ data (tab-separated .txt, 10-row header).
  2. Apply a zero-phase Butterworth low-pass filter across the full signal.
  3. Automatically detect the free-vibration window starting at the first
     oscillation immediately after the impact — no manually chosen times.
  4. Shift the time axis so that free vibration starts at t = 0 s.
  5. Detect all significant positive peaks (typically 5-8 for Baja suspension).
  6. Compute logarithmic decrement averaged over all consecutive peak pairs.
  7. Derive damping ratio, damped and undamped natural frequencies.
  8. Produce a four-panel diagnostic figure and a clean engineering report.

Signal channel priority
-----------------------
  Displacement (AI 4) is preferred over acceleration (AI 1) because:
  - The textbook log-decrement formula (delta = ln x1/x2) is derived for
    displacement. Applying it to acceleration introduces a systematic error
    that grows with the damping ratio.
  - Directly measured displacement avoids double integration, which
    amplifies low-frequency noise.
  The script falls back to Acc1 automatically when no displacement channel
  is found or when the displacement signal has negligible amplitude.

Why frequency is estimated from peak spacing, NOT from FFT of the full signal
------------------------------------------------------------------------------
  The FFT of the full 11-second acquisition is dominated by the broadband
  impact transient, not by the structural natural frequency. A peak in the
  full-signal FFT at an incorrect frequency (e.g. 1-2 Hz instead of 7 Hz)
  propagates into the minimum inter-peak distance threshold used by
  find_peaks and causes most true peaks to be rejected as "too close".
  Estimating the period from the median inter-peak spacing of rough peaks
  found *immediately after* the impact measures the actual oscillation
  frequency directly and correctly from the region of interest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, hilbert, savgol_filter


# ================================================================
# CONFIGURATION — only edit this block for a different test file
# ================================================================

# Data file name (must be in the same directory as this script).
FILENAME = "baja_front_rig1_carga_centro_turma2.txt"

# Low-pass filter cutoff [Hz].
# Set to ~5-10x the expected natural frequency. Baja suspension fn ~ 2-15 Hz;
# 50 Hz provides a decade of headroom while removing electrical noise.
FILTER_CUTOFF_HZ = 50.0

# Butterworth filter order. Order 4 gives -80 dB/decade roll-off:
# aggressive enough for noise suppression, stable at high sampling rates.
FILTER_ORDER = 4

# A post-impact positive peak is accepted as the start of free vibration
# only if its amplitude is below this fraction of the impact amplitude.
# 0.90 means the forced transient must have decayed to <= 90% of the impact.
IMPACT_DECAY_THRESHOLD = 0.90

# Free-vibration window ends when the smoothed Hilbert envelope drops below
# this fraction of the first free-vibration peak amplitude.
# 0.01 = 1% keeps the window open until the signal has decayed ~40 dB.
MIN_ENVELOPE_FRACTION = 0.01

# Peak detection thresholds (fractions of the largest peak in the ROI).
#   PROMINENCE: suppresses noise ripples and filter-induced shoulders.
#   HEIGHT:     absolute noise floor below which peaks are not physical.
PEAK_PROMINENCE_FRACTION = 0.05
PEAK_HEIGHT_FRACTION     = 0.01

# Minimum displacement signal peak-to-peak range to be considered connected.
DISP_MEANINGFUL_PP_V = 0.01  # V


# ================================================================
# 1. DATA LOADING
# ================================================================

def load_data(filename: str) -> pd.DataFrame:
    """
    Read the DAQ .txt file and return a DataFrame indexed by time [s].

    Column matching is partial and case-insensitive so that minor header
    variations between test sessions do not require code changes.
    Columns are renamed to short identifiers: Acc1, Acc2, Force, Disp.
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", skiprows=10, encoding="latin1")
    df.set_index("Time (s)", inplace=True)

    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if "ai 1" in cl:
            rename_map[col] = "Acc1"
        elif "ai 2" in cl:
            rename_map[col] = "Acc2"
        elif "ai 3" in cl:
            rename_map[col] = "Force"
        elif "ai 4" in cl:
            rename_map[col] = "Disp"

    df.rename(columns=rename_map, inplace=True)
    return df


def select_signal(df: pd.DataFrame):
    """
    Choose the best channel for damping estimation.
    Returns (signal_array, time_array, channel_label, unit_string).

    Preference order: Disp (if present and meaningful) -> Acc1.
    See module docstring for the engineering rationale.
    """
    time = df.index.to_numpy(dtype=float)

    if "Disp" in df.columns:
        disp_pp = float(df["Disp"].max() - df["Disp"].min())
        if disp_pp >= DISP_MEANINGFUL_PP_V:
            print(f"[CHANNEL] Displacement (AI 4): pp = {disp_pp:.4f} V - preferred.")
            return df["Disp"].to_numpy(dtype=float), time, "Disp", "V"
        print(f"[CHANNEL] Disp (AI 4) negligible (pp = {disp_pp:.4f} V) - fallback to Acc1.")

    if "Acc1" not in df.columns:
        raise KeyError("No recognised signal channel (AI 1 / AI 4) in the file.")

    print("[CHANNEL] Using Acc1 (m/s^2).")
    return df["Acc1"].to_numpy(dtype=float), time, "Acc1", "m/s^2"


# ================================================================
# 2. LOW-PASS FILTERING
# ================================================================

def lowpass_filter(signal: np.ndarray, cutoff_hz: float,
                   fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth low-pass filter (scipy filtfilt).

    filtfilt applies the filter forward then backward:
    - Zero phase shift: peak positions are preserved exactly in time.
      A causal single-pass filter shifts all peaks forward by the group
      delay, corrupting the period measurement used for fd computation.
    - Effective order doubles to 8th order: steeper roll-off than order 4.

    Butterworth is chosen for its maximally flat passband: no ripple means
    no artificial amplitude modulation of the oscillation peaks used for
    logarithmic decrement estimation.
    """
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        raise ValueError(f"Cutoff {cutoff_hz} Hz must be < Nyquist {nyq:.0f} Hz.")
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    return filtfilt(b, a, signal)


# ================================================================
# 3. FREE-VIBRATION REGION DETECTION
# ================================================================

def find_free_vibration_region(signal_filtered: np.ndarray,
                                time: np.ndarray, fs: float):
    """
    Locate the free-vibration window automatically from the signal itself.

    Why 50 ms minimum distance for the rough peak search
    -----------------------------------------------------
    The signal contains content at multiple frequencies after filtering to
    50 Hz (e.g. a structural mode at ~7 Hz and a higher mode at ~24 Hz).
    Using 50 ms prevents sub-cycle peaks from the higher mode being counted
    as structural peaks, while still finding one peak per 7 Hz cycle (T=143 ms).

    Why the consistent-spacing test
    --------------------------------
    The large positive peak immediately after the impact (the impact-bounce
    transient) has a much larger spacing to the next structural peak than the
    structural peaks have among themselves. By finding the first rough peak
    whose *following* spacing is within 50 % of the median structural period,
    we skip the anomalous impact-bounce and start exactly at the first true
    free-vibration oscillation.

    Why idx_start is shifted back 5 samples
    ----------------------------------------
    scipy.signal.find_peaks cannot return a peak at index 0 of an array
    (it requires a strictly lower neighbor on both sides). Shifting the
    ROI start 5 samples before the first free-vibration peak ensures that
    peak appears as an interior point and is correctly detected.

    Returns
    -------
    idx_start : int    first sample of the ROI (5 samples before first FV peak)
    idx_end   : int    last sample of the free-vibration window
    T_est     : float  estimated vibration period [s]
    """
    # --- Step 1: Locate impact as the global maximum of |signal| ---
    idx_impact = int(np.argmax(np.abs(signal_filtered)))
    impact_amp = float(np.abs(signal_filtered[idx_impact]))
    t_impact   = float(time[idx_impact])

    # --- Step 2: Rough peak search — limited to 3 s after the impact ---
    # Searching the full signal would include the noisy tail (small oscillations
    # around zero, potentially from a different mode or noise), whose irregular
    # spacings would corrupt the median period estimate.
    # 50 ms minimum distance (1000 samples at 20 kHz) rejects sub-cycle peaks
    # from higher-frequency modes in the passband while capturing structural peaks.
    rough_dist  = max(1, int(0.05 * fs))
    search_end  = min(idx_impact + int(3.0 * fs), len(signal_filtered))
    post_seg    = signal_filtered[idx_impact:search_end]
    rough_peaks, _ = find_peaks(post_seg, distance=rough_dist)

    if len(rough_peaks) < 2:
        T_est     = 1.0 / 7.0           # 7 Hz physical fallback
        idx_start = max(0, idx_impact + int(0.10 * fs) - 5)
        print("[REGION]  Warning: too few rough peaks; using 7 Hz fallback.")
    else:
        # --- Step 3: Estimate structural period from inter-peak spacings ---
        # Physical range: 50-400 ms (2.5-20 Hz). Spacings outside this window
        # are either sub-cycle noise or low-frequency baseline drift.
        # The first 10 spacings are used (highest SNR, before amplitude decays).
        spacings = np.diff(rough_peaks.astype(float)) / fs  # [s]
        valid    = spacings[(spacings >= 0.05) & (spacings <= 0.40)]
        T_est    = float(np.median(valid[:10])) if len(valid) >= 1 else 1.0 / 7.0

        # --- Step 4: Find the first peak with a consistent following spacing ---
        # The impact-bounce peak has an anomalously large following gap to the
        # first structural peak. We skip it by looking for the first spacing
        # that is within 50 % of T_est — that marks the onset of the consistent
        # free-vibration decay.
        idx_start = max(0, idx_impact + int(rough_peaks[0]) - 5)  # default
        for i, sp in enumerate(spacings):
            if 0.5 * T_est <= sp <= 1.5 * T_est:
                # Shift 5 samples back so the peak is interior (not a boundary)
                idx_start = max(0, idx_impact + int(rough_peaks[i]) - 5)
                break

    # --- Step 5: Free-vibration window end via smoothed Hilbert envelope ---
    # Reference amplitude at idx_start (≈ first free-vibration peak amplitude).
    first_free_amp = float(np.abs(signal_filtered[idx_start + 5]))  # at the true peak
    seg = signal_filtered[idx_start:]
    env = np.abs(hilbert(seg))

    # Smooth over half a vibration period to remove carrier-frequency ripple
    # that would cause premature threshold crossings.
    win = max(3, int(0.5 * T_est * fs))
    win = win if win % 2 == 1 else win + 1
    if len(env) > win:
        env = savgol_filter(env, win, polyorder=3)

    threshold = MIN_ENVELOPE_FRACTION * first_free_amp
    below = np.where(env < threshold)[0]
    idx_end = (idx_start + int(below[0])) if len(below) > 0 else len(signal_filtered) - 1

    t_start = float(time[idx_start])
    t_end   = float(time[idx_end])
    print(f"[REGION]  Impact at t = {t_impact:.3f} s")
    print(f"[REGION]  Free-vibration window: {t_start:.4f} s -> {t_end:.4f} s "
          f"({t_end - t_start:.3f} s duration)")
    print(f"[REGION]  Period estimate (median of {len(valid) if len(rough_peaks) >= 2 else 0} "  # noqa
          f"spacings): T ~ {T_est*1000:.1f} ms  ->  f ~ {1/T_est:.1f} Hz")

    return idx_start, idx_end, T_est


# ================================================================
# 4. PEAK DETECTION
# ================================================================

def detect_peaks(signal_roi: np.ndarray, time_roi: np.ndarray,
                 fs: float, T_est: float):
    """
    Detect significant positive peaks in the free-vibration region.

    Two-pass strategy
    -----------------
    Pass 1 (distance only): establishes the amplitude scale from the
    largest rough peak, solving the chicken-and-egg problem (we need the
    amplitude to set the threshold but need peaks to find the amplitude).

    Pass 2 (distance + prominence + height): physics-informed thresholds
    scaled relative to the signal amplitude in the ROI.

    Parameter rationale
    -------------------
    distance = 0.8 x T_est x fs
        Ensures at most one peak per vibration cycle. The 0.8 factor gives
        tolerance for mild period shortening at high amplitudes (amplitude-
        dependent stiffness) without merging adjacent cycles into one.

    prominence = PEAK_PROMINENCE_FRACTION x amp_scale
        Each peak must rise above its surrounding signal by this amount.
        Suppresses filter-induced shoulders and noise ripples without
        requiring knowledge of the absolute signal level.

    height = PEAK_HEIGHT_FRACTION x amp_scale
        Absolute noise floor. Prevents very late near-zero peaks from
        entering the log-decrement calculation, where a small denominator
        in ln(Ai/Ai+1) would inflate the individual delta value.
    """
    min_dist = max(1, int(0.8 * T_est * fs))

    # Pass 1: distance-only to set amplitude scale
    rough, _ = find_peaks(signal_roi, distance=min_dist)
    if len(rough) == 0:
        raise RuntimeError(
            "No peaks found in the free-vibration region. "
            "Try reducing MIN_ENVELOPE_FRACTION or adjusting FILTER_CUTOFF_HZ."
        )
    amp_scale = float(signal_roi[rough].max())

    # Pass 2: full criteria
    peaks, _ = find_peaks(
        signal_roi,
        distance=min_dist,
        prominence=PEAK_PROMINENCE_FRACTION * amp_scale,
        height=PEAK_HEIGHT_FRACTION * amp_scale,
    )

    print(f"[PEAKS]   {len(peaks)} peaks detected.")
    return peaks, signal_roi[peaks], time_roi[peaks]


# ================================================================
# 5. LOGARITHMIC DECREMENT & VIBRATION PARAMETERS
# ================================================================

def log_decrement_average(amplitudes: np.ndarray):
    """
    Average logarithmic decrement delta over all consecutive peak pairs.

    For each pair:  delta_i = ln(A_i / A_{i+1})
    Average:        delta   = (1/(N-1)) * sum(delta_i)

    Why average consecutive pairs rather than the two-endpoint formula
    delta = (1/N) ln(A_0/A_N)?
    -------------------------------------------------------------------
    - Averaging uses all N-1 pairs, so a single outlier peak affects at
      most two terms of the sum rather than the entire estimate.
    - Both estimators are unbiased for a true exponential decay; the
      average is more informative because the cycle-to-cycle standard
      deviation (also reported) reveals whether damping is constant across
      amplitude levels -- a sign that the viscous model is appropriate.

    Returns (mean_delta, list_of_individual_deltas).
    """
    if len(amplitudes) < 2:
        raise ValueError("Need at least 2 peaks to compute logarithmic decrement.")

    deltas = [
        np.log(amplitudes[i] / amplitudes[i + 1])
        for i in range(len(amplitudes) - 1)
        if amplitudes[i] > 0 and amplitudes[i + 1] > 0
    ]
    if not deltas:
        raise ValueError("All amplitude ratios are non-positive. "
                         "Check signal orientation or filter settings.")

    return float(np.mean(deltas)), deltas


def compute_vibration_params(log_dec: float, peak_times: np.ndarray) -> dict:
    """
    Derive all vibration parameters from delta and the inter-peak intervals.

    Damping ratio (exact for any zeta < 1, linear viscous SDOF):
        zeta = delta / sqrt(4*pi^2 + delta^2)

    Damped period from mean inter-peak interval:
        Td = mean(dt_i)   [s]

    Damped natural frequency:   fd = 1 / Td               [Hz]
    Undamped natural frequency: fn = fd / sqrt(1 - zeta^2) [Hz]
    Natural circular frequency: wn = 2*pi*fn              [rad/s]

    The relation fn = fd/sqrt(1-zeta^2) is exact for linear viscous
    (velocity-proportional) damping -- the standard model for structural
    vibration analysis. Baja suspension systems with hydraulic dampers
    are approximately linear-viscous over the amplitude range of this test.
    """
    zeta = log_dec / np.sqrt(4.0 * np.pi**2 + log_dec**2)
    Td   = float(np.mean(np.diff(peak_times)))
    fd   = 1.0 / Td
    fn   = fd / np.sqrt(1.0 - zeta**2)
    wn   = 2.0 * np.pi * fn
    return {"delta": log_dec, "zeta": zeta,
            "Td": Td, "fd": fd, "fn": fn, "wn": wn}


# ================================================================
# 6. EXPONENTIAL ENVELOPE
# ================================================================

def exponential_envelope(time_shifted: np.ndarray, t_first_peak: float,
                          A0: float, zeta: float, wn: float) -> np.ndarray:
    """
    Theoretical amplitude envelope: A(t) = A0 * exp(-zeta*wn*(t - t0)).

    t0 = time of the first detected peak on the shifted axis.
    A0 = amplitude of the first detected peak.

    Plotting this against the detected peaks validates the viscous model:
    - Peaks on the curve: exponential decay confirmed, zeta is reliable.
    - Concave deviation: Coulomb (dry) friction component present.
    - Convex deviation: quadratic (air-drag) component present.
    """
    return A0 * np.exp(-zeta * wn * (time_shifted - t_first_peak))


# ================================================================
# 7. PLOTS
# ================================================================

def plot_analysis(time_full, signal_raw_full, signal_filtered_full,
                  time_roi_shifted, signal_raw_roi, signal_filtered_roi,
                  peak_times_shifted, peak_amps,
                  envelope,
                  t_start_orig, t_end_orig,
                  channel_label, unit_label,
                  params: dict):
    """
    Four-panel engineering figure.

    Panels 1-2 use the original (absolute) time axis for the full acquisition.
    Panels 3-4 use the shifted time axis (t = 0 at free-vibration onset),
    so the duration of the decay is directly readable and the axis starts
    at zero -- appropriate for a Mechanical Engineering laboratory report.

    Panel 1 - Raw signal, full acquisition.
    Panel 2 - Filtered signal with free-vibration window highlighted.
              (Shares x-axis with Panel 1 for aligned comparison.)
    Panel 3 - Free-vibration decay: filtered + peaks + exponential envelope.
    Panel 4 - FFT spectrum of the free-vibration region with fd and fn marked.
    """
    plt.rcParams.update({
        "font.family"    : "serif",
        "font.size"      : 10,
        "axes.titlesize" : 11,
        "axes.labelsize" : 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth" : 0.8,
        "grid.linewidth" : 0.4,
        "grid.alpha"     : 0.6,
        "lines.linewidth": 1.2,
    })

    C_RAW  = "#9db8d2"  # muted blue -- raw signal
    C_FILT = "#1a5276"  # dark navy  -- filtered signal
    C_PEAK = "#c0392b"  # red        -- detected peaks
    C_ENV  = "#1c1c1c"  # near-black -- exponential envelope
    C_FD   = "#c0392b"  # red        -- damped frequency marker
    C_FN   = "#d35400"  # orange     -- undamped frequency marker
    C_WIN  = "#f39c12"  # amber      -- free-vibration window

    ylabel = f"{channel_label}  [{unit_label}]"

    fig = plt.figure(figsize=(12, 16))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.06,
                        hspace=0.58)

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    fig.suptitle(
        "Baja SAE Suspension -- Free Vibration Analysis\n"
        f"Channel: {channel_label}    "
        f"fd = {params['fd']:.3f} Hz    "
        f"fn = {params['fn']:.3f} Hz    "
        f"zeta = {params['zeta']:.4f}    "
        f"delta = {params['delta']:.4f}",
        fontsize=12, fontweight="bold", y=0.975,
    )

    # ------------------------------------------------------------------
    # Panel 1 -- Raw signal, full acquisition (absolute time)
    # ------------------------------------------------------------------
    ax1.plot(time_full, signal_raw_full,
             color=C_RAW, lw=0.8, label="Raw signal")
    ax1.set_ylabel(ylabel)
    ax1.set_title("(1)  Raw Signal -- Full Acquisition")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_xlim(time_full[0], time_full[-1])
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ------------------------------------------------------------------
    # Panel 2 -- Filtered signal with free-vibration window (absolute time)
    # ------------------------------------------------------------------
    ax2.plot(time_full, signal_filtered_full,
             color=C_FILT, lw=1.0, label="Filtered signal")
    ax2.axvspan(t_start_orig, t_end_orig, alpha=0.18, color=C_WIN,
                label=f"Free-vibration window  ({t_start_orig:.3f}-{t_end_orig:.3f} s)")
    ax2.axvline(t_start_orig, color=C_WIN, lw=1.0, ls="--", alpha=0.85)
    ax2.axvline(t_end_orig,   color=C_WIN, lw=1.0, ls="--", alpha=0.85)
    ax2.set_xlabel("Time  (s)")
    ax2.set_ylabel(ylabel)
    ax2.set_title("(2)  Filtered Signal -- Automatically Detected Free-Vibration Region")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # ------------------------------------------------------------------
    # Panel 3 -- Free-vibration decay on shifted time axis (t = 0 at onset)
    # ------------------------------------------------------------------
    ax3.plot(time_roi_shifted, signal_filtered_roi,
             color=C_FILT, lw=1.1, label="Filtered signal (free-vibration region)")
    ax3.plot(time_roi_shifted, envelope,
             color=C_ENV, lw=1.3, ls="--", zorder=3,
             label="Exponential envelope  A0*exp(-zeta*wn*t)")
    ax3.plot(peak_times_shifted, peak_amps,
             marker="o", ms=6, ls="none", color=C_PEAK, zorder=4,
             label=f"Detected peaks  (n = {len(peak_amps)})")
    for i, (tp, ap) in enumerate(zip(peak_times_shifted, peak_amps)):
        ax3.annotate(str(i + 1), xy=(tp, ap),
                     xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=7, color=C_PEAK)
    ax3.set_xlabel("Time from free-vibration onset  (s)")
    ax3.set_ylabel(ylabel)
    ax3.set_title(
        f"(3)  Free-Vibration Decay -- Peaks & Exponential Envelope    "
        f"(delta = {params['delta']:.4f},  zeta = {params['zeta']:.4f})"
    )
    ax3.legend(loc="upper right")
    ax3.grid(True)

    # ------------------------------------------------------------------
    # Panel 4 -- FFT spectrum of the free-vibration region
    # Single-sided amplitude spectrum: factor 2/N normalises to physical amplitude.
    # ------------------------------------------------------------------
    n_fft    = len(signal_filtered_roi)
    # dt is inferred from the original-time ROI (spacing is the same as shifted)
    dt       = float(params["Td"] / round(params["Td"] * (1.0 / (time_roi_shifted[1] - time_roi_shifted[0]))))  # noqa
    # Simpler: use the time step directly from the shifted array
    dt       = float(time_roi_shifted[1] - time_roi_shifted[0])
    freqs    = np.fft.rfftfreq(n_fft, d=dt)
    spectrum = 2.0 / n_fft * np.abs(np.fft.rfft(signal_filtered_roi))

    ax4.plot(freqs, spectrum, color=C_FILT, lw=0.9, label="Amplitude spectrum")
    ax4.axvline(params["fd"], color=C_FD, ls="--", lw=1.4,
                label=f"fd = {params['fd']:.3f} Hz  (damped natural frequency)")
    ax4.axvline(params["fn"], color=C_FN, ls="-.", lw=1.4,
                label=f"fn = {params['fn']:.3f} Hz  (undamped natural frequency)")
    ax4.set_xlim(0, min(8.0 * params["fn"], freqs[-1]))
    ax4.set_xlabel("Frequency  (Hz)")
    ax4.set_ylabel("Amplitude  (single-sided)")
    ax4.set_title("(4)  FFT Spectrum -- Free-Vibration Region  |  Modal Frequencies")
    ax4.legend(loc="upper right")
    ax4.grid(True)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vibration_analysis.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[PLOT]    Saved -> {out_path}")


# ================================================================
# 8. ENGINEERING REPORT
# ================================================================

def print_report(params: dict, n_peaks: int, channel: str,
                 individual_deltas: list):
    """Print a clean, self-contained engineering results summary."""
    sep = "-" * 48
    print()
    print(sep)
    print("  Baja Suspension Free Vibration Analysis")
    print(sep)
    print(f"  Signal channel              : {channel}")
    print(f"  Number of peaks detected    : {n_peaks}")
    print(f"  Log-decrement samples (N-1) : {n_peaks - 1}  (consecutive pairs)")
    print(f"  Avg logarithmic decrement d : {params['delta']:.5f}")
    if len(individual_deltas) > 1:
        print(f"  Std dev of d (cycle-to-cycle): "
              f"{float(np.std(individual_deltas, ddof=1)):.5f}")
    print(f"  Damping ratio  zeta         : {params['zeta']:.5f}")
    print(f"  Damped period  Td           : {params['Td'] * 1e3:.2f} ms")
    print(f"  Damped frequency    fd      : {params['fd']:.4f} Hz")
    print(f"  Natural frequency   fn      : {params['fn']:.4f} Hz")
    print(f"  Natural circ. freq  wn      : {params['wn']:.4f} rad/s")
    print(sep)
    print()


# ================================================================
# MAIN
# ================================================================

def main():
    # ---- 1. Load data ----
    print(f"\n[LOAD]    Reading: {FILENAME}")
    df = load_data(FILENAME)
    signal_full, time_full, channel_label, unit_label = select_signal(df)

    fs = 1.0 / float(time_full[1] - time_full[0])
    print(f"[LOAD]    Fs = {fs:.0f} Hz | Duration = {time_full[-1]:.3f} s "
          f"| Samples = {len(time_full):,}")

    # ---- 2. Filter the entire signal ----
    # Filtering before region detection prevents high-frequency noise from
    # biasing the Hilbert envelope and the rough peak search.
    signal_filtered = lowpass_filter(signal_full, FILTER_CUTOFF_HZ, fs)

    # ---- 3. Detect the free-vibration region ----
    # Returns idx_start (first sample of free vibration), idx_end, and T_est.
    # T_est is estimated from median inter-peak spacing of rough post-impact
    # peaks -- NOT from the FFT of the full signal (see module docstring).
    idx_start, idx_end, T_est = find_free_vibration_region(
        signal_filtered, time_full, fs
    )

    time_roi_orig   = time_full[idx_start : idx_end + 1]
    signal_raw_roi  = signal_full[idx_start : idx_end + 1]
    signal_filt_roi = signal_filtered[idx_start : idx_end + 1]

    if len(time_roi_orig) < 10:
        raise RuntimeError("Free-vibration window too short. "
                           "Reduce MIN_ENVELOPE_FRACTION and re-run.")

    # ---- 4. Shift time axis so free vibration starts at t = 0 s ----
    # The shifted axis shows the duration of the decay directly and is
    # more appropriate than absolute test time for a lab report figure.
    t_offset         = float(time_roi_orig[0])
    time_roi_shifted = time_roi_orig - t_offset

    # ---- 5. Detect peaks using the period estimate from step 3 ----
    _, peak_amps, peak_times_orig = detect_peaks(
        signal_filt_roi, time_roi_orig, fs, T_est
    )
    peak_times_shifted = peak_times_orig - t_offset

    if len(peak_amps) < 2:
        print("[ERROR]   Fewer than 2 peaks found. "
              "Increase IMPACT_DECAY_THRESHOLD or decrease MIN_ENVELOPE_FRACTION.")
        return

    # ---- 6. Logarithmic decrement (all consecutive pairs) ----
    log_dec, individual_deltas = log_decrement_average(peak_amps)

    # ---- 7. Full vibration parameter set ----
    params = compute_vibration_params(log_dec, peak_times_orig)

    # ---- 8. Exponential envelope on the shifted time axis ----
    # t_first_peak_shifted: position of peak 1 on the shifted axis.
    # The envelope is defined from the first peak onward; before that
    # it extrapolates backward (useful context for the plot).
    t_first = float(peak_times_shifted[0])
    envelope = exponential_envelope(
        time_roi_shifted, t_first, float(peak_amps[0]),
        params["zeta"], params["wn"]
    )

    # ---- 9. Print engineering report ----
    print_report(params, len(peak_amps), channel_label, individual_deltas)

    # ---- 10. Four-panel diagnostic figure ----
    plot_analysis(
        time_full, signal_full, signal_filtered,
        time_roi_shifted, signal_raw_roi, signal_filt_roi,
        peak_times_shifted, peak_amps,
        envelope,
        float(time_roi_orig[0]), float(time_roi_orig[-1]),
        channel_label, unit_label,
        params,
    )


if __name__ == "__main__":
    main()
