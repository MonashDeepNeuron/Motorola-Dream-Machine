from typing import Dict, Tuple, Optional

import numpy as np
import mne

def event_band_statistics(
    raw: mne.io.BaseRaw,
    freq_bands: Dict[str, Tuple[float, float]],
    event_id: Optional[Dict[str, int]] = None,
    tmin: float = -1.0,
    tmax: float = 2.0,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = (-1.0, 0.0),
) -> Dict[str, Dict[str, float]]:
    """
    Computes event-locked band-power statistics.
    Power is calculated as the mean of the squared signal amplitude within the band (μV²).
    Returns a nested dict {event_code: {band_name: mean_power_uv_sq}}.
    """
    try:
        events, inferred_event_id = mne.events_from_annotations(raw)
        if event_id is None:
            event_id = inferred_event_id
    except ValueError as e: # No annotations found
        print(f"Warning: Could not extract events from annotations: {e}")
        return {code: {b: 0.0 for b in freq_bands} for code in (event_id if event_id else {})}


    # Bail early if no events or event_id is empty
    if events.size == 0 or not event_id:
        print("Warning: No events found or event_id is empty. Skipping event-band statistics.")
        return {code: {b: 0.0 for b in freq_bands} for code in (event_id if event_id else {})}

    # Filter event_id to only include events present in 'events'
    present_event_codes = np.unique(events[:, 2])
    valid_event_id = {name: code for name, code in event_id.items() if code in present_event_codes}

    if not valid_event_id:
        print("Warning: None of the specified event_ids are present in the data. Skipping event-band statistics.")
        return {name: {b: 0.0 for b in freq_bands} for name in event_id}


    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=valid_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False, # Suppress extensive MNE output
            on_missing='warn' # Warn if some event types have no epochs
        )
    except Exception as e:
        print(f"Error creating epochs: {e}. Skipping event-band statistics.")
        return {name: {b: 0.0 for b in freq_bands} for name in event_id}


    result: Dict[str, Dict[str, float]] = {}
    for event_name in event_id.keys(): # Iterate over original event_id to ensure all keys are present
        result[event_name] = {}
        if event_name not in epochs.event_id or len(epochs[event_name]) == 0:
            # print(f"No epochs found for event: {event_name}. Setting powers to 0.")
            for band_name in freq_bands:
                result[event_name][band_name] = 0.0
            continue

        for band_name, (fmin, fmax) in freq_bands.items():
            # Filter data in V, then square, then scale to uV^2
            # MNE's filter works in place, so copy
            band_epochs = epochs[event_name].copy().filter(
                l_freq=fmin, h_freq=fmax, fir_design='firwin', skip_by_annotation='edge', verbose=False
            )
            # Get data in Volts, then scale to microvolts before squaring
            data_uv = band_epochs.get_data(units="uV") # (n_epochs, n_channels, n_times)
            
            # Power = mean of squared signal amplitude
            # Average across time points, then across channels, then across epochs
            if data_uv.size > 0:
                power_uv_sq = np.mean(data_uv ** 2)
            else:
                power_uv_sq = 0.0
            result[event_name][band_name] = float(power_uv_sq)
            
    return result