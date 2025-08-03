from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
from scipy import signal
import mne

# ---------- frequency bands ------------------
DefaultBands: Dict[Literal["delta", "theta", "alpha", "beta", "gamma"], Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}


@dataclass
class WindowBandResult:
    window_index: int
    start_sec: float
    end_sec: float
    band_name: str
    channel_label: str  # Single channel for this result
    power: float  # PSD μV²/Hz for this channel


def compute_window_band_power(
    raw: mne.io.BaseRaw,
    window_size_s: float = 4.0,
    step_size_s: float = 2.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[List[WindowBandResult], Dict[int, Dict[str, float]]]:
    """
    Sliding-window Welch PSD -> band power for all channels.
    Returns:
        List[WindowBandResult]: One entry per channel, per band, per window. Power is μV²/Hz.
        Dict[int, Dict[str, float]]: Convenience dict: window_index → band_name → mean_power (μV²/Hz averaged over channels).
    """
    if bands is None:
        bands = DefaultBands

    sfreq = raw.info["sfreq"]
    # ensure data is scaled to Volts for SciPy, then we'll scale power to μV²/Hz
    data_volts = raw.get_data(units="V")  # shape (n_channels, n_samples)

    window_samples = int(window_size_s * sfreq)
    step_samples = int(step_size_s * sfreq)
    total_samples = data_volts.shape[1]
    
    if window_samples == 0:
        raise ValueError("Window size in samples is 0. Check sfreq and window_size_s.")
    if total_samples < window_samples: # Not enough data for even one window
        return [], {}

    n_windows = (total_samples - window_samples) // step_samples + 1
    if n_windows <= 0: # Handles cases where total_samples might be just equal or slightly more than window_samples but not enough for step
        n_windows = 1 if total_samples >= window_samples else 0
        if n_windows == 0: return [], {}


    chan_labels = raw.ch_names
    per_window_channel_results: List[WindowBandResult] = []
    # For storing power values per channel for averaging
    # window_idx -> band_name -> list_of_channel_powers
    temp_window_band_powers: Dict[int, Dict[str, List[float]]] = {
        w: {band_name: [] for band_name in bands} for w in range(n_windows)
    }

    for w in range(n_windows):
        start_sample = w * step_samples
        end_sample = start_sample + window_samples
        if end_sample > total_samples: # Ensure we don't go past the data
            continue
        
        window_data = data_volts[:, start_sample:end_sample]

        for ch_idx, ch_name in enumerate(chan_labels):
            # nperseg should not exceed window_samples
            nperseg_val = min(256, window_samples) # Default Welch nperseg is 256, good for general purpose
            if sfreq < 50 and window_samples < 128 : # Adjust for low sfreq or very short windows
                 nperseg_val = window_samples // 2 if window_samples // 2 > 0 else window_samples

            freqs, psd_density = signal.welch(
                window_data[ch_idx],
                fs=sfreq,
                nperseg=nperseg_val,
                noverlap=nperseg_val // 2, # 50% overlap is common
                scaling="density", # V^2/Hz
            )

            for band_name, (fmin, fmax) in bands.items():
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                if np.any(idx_band):
                    # Integrate PSD over the band (approximate by mean * band_width, or just mean for density)
                    # Paper by Bendat & Piersol (2010) suggests mean for density.
                    mean_psd_in_band = np.mean(psd_density[idx_band])
                    power_val_uv_sq_hz = mean_psd_in_band * 1e12  # Convert V^2/Hz to μV²/Hz
                else:
                    power_val_uv_sq_hz = 0.0

                per_window_channel_results.append(
                    WindowBandResult(
                        window_index=w,
                        start_sec=float(start_sample / sfreq),
                        end_sec=float(end_sample / sfreq),
                        band_name=band_name,
                        channel_label=ch_name,
                        power=power_val_uv_sq_hz,
                    )
                )
                temp_window_band_powers[w][band_name].append(power_val_uv_sq_hz)

    # Calculate final average across channels for the convenience dict
    window_band_avg_across_channels: Dict[int, Dict[str, float]] = {w: {} for w in range(n_windows)}
    for w_idx in range(n_windows):
        for band_name in bands:
            channel_powers = temp_window_band_powers[w_idx][band_name]
            if channel_powers:
                window_band_avg_across_channels[w_idx][band_name] = float(np.mean(channel_powers))
            else:
                window_band_avg_across_channels[w_idx][band_name] = 0.0
                
    return per_window_channel_results, window_band_avg_across_channels

def calculate_psd_for_window(
    window_data: np.ndarray,  # Shape: (n_samples, n_channels)
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Calculates the average Power Spectral Density (PSD) across all channels 
    for a given numpy window. This is adapted for real-time use by the consumer.
    
    Returns:
        A dictionary mapping band_name to its average power (μV²/Hz).
    """
    if bands is None:
        bands = DefaultBands

    window_data_ch_first = window_data.T

    n_channels, n_samples = window_data_ch_first.shape
    if n_samples == 0:
        return {band_name: 0.0 for band_name in bands}
    
    nperseg_val = min(256, n_samples)
    freqs, psd_density = signal.welch(
        window_data_ch_first,
        fs=sfreq,
        nperseg=nperseg_val,
        scaling="density",  # V^2/Hz
        axis=-1 
    )
    psd_density_uv = psd_density * 1e12

    avg_psd_per_band: Dict[str, float] = {}
    for band_name, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.any(idx_band):
            avg_power = np.mean(psd_density_uv[:, idx_band])
            avg_psd_per_band[band_name] = float(avg_power)
        else:
            avg_psd_per_band[band_name] = 0.0
            
    return avg_psd_per_band
