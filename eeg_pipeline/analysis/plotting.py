from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_avg_band_power(
    band_averages: Dict[str, float], # band_name -> avg_power_value
    outfile: Path,
    title: str = 'Average Power in Frequency Bands',
    ylabel: str = 'Power Spectral Density (μV²/Hz)'
) -> None:
    """Plots average band power as a bar chart."""
    if not band_averages:
        print("No data to plot for average band power.")
        return

    bands = list(band_averages.keys())
    values = [band_averages[b] for b in bands]

    plt.figure(figsize=(10, 6))
    plt.bar(bands, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'mediumpurple'])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(outfile)
        print(f"Saved plot: {outfile}")
    except Exception as e:
        print(f"Error saving plot {outfile}: {e}")
    plt.close()


def plot_band_power_over_time(
    window_band_avg: Dict[int, Dict[str, float]], # window_index -> band_name -> avg_power_value
    step_size_sec: float,
    window_size_sec: float,
    outfile: Path,
    title: str = 'Frequency Band Power Over Time',
    xlabel: str = 'Time (s)',
    ylabel: str = 'Power Spectral Density (μV²/Hz)'
) -> None:
    """Plots band power over time for multiple bands."""
    if not window_band_avg:
        print("No data to plot for band power over time.")
        return

    # determine the set of all bands present
    all_bands = set()
    for win_data in window_band_avg.values():
        all_bands.update(win_data.keys())
    if not all_bands:
        print("No bands found in window_band_avg data.")
        return
    
    # prepare times axis: center of each window
    window_indices = sorted(window_band_avg.keys())
    if not window_indices:
        print("No window indices found.")
        return

    times = np.array([idx * step_size_sec + (window_size_sec / 2) for idx in window_indices])

    plt.figure(figsize=(15, 7))
    
    colors = plt.cm.get_cmap('viridis', len(all_bands)) # Use a colormap

    for i, band_name in enumerate(sorted(list(all_bands))):
        values = [window_band_avg[w_idx].get(band_name, np.nan) for w_idx in window_indices] 
        # Filter out NaNs
        valid_times = times[~np.isnan(values)]
        valid_values = np.array(values)[~np.isnan(values)]
        if len(valid_values) > 0:
             plt.plot(valid_times, valid_values, label=band_name, color=colors(i))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(outfile)
        print(f"Saved plot: {outfile}")
    except Exception as e:
        print(f"Error saving plot {outfile}: {e}")
    plt.close()