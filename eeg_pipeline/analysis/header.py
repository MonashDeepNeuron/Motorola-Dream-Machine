from datetime import datetime
from typing import List, Any
import numpy as np

import mne


def _fmt(field: Any, width: int) -> str:
    return f"{str(field):<{width}}"  # pad / truncate to fixed width


def header_dump(raw: mne.io.BaseRaw) -> List[str]:
    """
    Return the EDF header lines largely following the order of the official spec.
    The caller decides whether to `print` or write them to a file.
    MNE provides access to most, but not all, EDF header fields directly.
    This function retrieves what's available via MNE.
    """
    lines: List[str] = []
    
    # mne.io.Raw object stores header info in raw.info and raw._raw_extras
    # _raw_extras[0] usually contains the EDF specific fields for the first data file part
    extra = raw._raw_extras[0] if raw._raw_extras else {}

    # --- global header -----------------------------------------------------
    # EDF+ files might have 'EDF+C' or 'EDF+D'. pyedfread might store this.
    # MNE doesn't seem to store the 8-char version string directly in a parsed way.
    # try to get it from `extra` if it's loaded by MNE's internal EDF reader
    edf_version_bytes = extra.get("version_bytes", b"0       ") 
    try:
        ver = edf_version_bytes.decode('ascii').strip()
    except UnicodeDecodeError:
        ver = "N/A" # Fallback if decoding fails
    if not ver: ver = "0" 
    patient_info_str = extra.get('patient', 'X X X X') 
    if isinstance(patient_info_str, bytes):
        patient_info_str = patient_info_str.decode('ascii', errors='replace')
    
    # Recording ID: similarly, from 'recording' field
    recording_info_str = extra.get('recording', 'Startdate ...')
    if isinstance(recording_info_str, bytes):
        recording_info_str = recording_info_str.decode('ascii', errors='replace')

    meas_date = raw.info.get("meas_date")
    if isinstance(meas_date, datetime):
        date_str = meas_date.strftime("%d.%m.%y")
        time_str = meas_date.strftime("%H.%M.%S")
    else: # Fallback if no measurement date
        date_str = extra.get('startdate', '00.00.00')
        time_str = extra.get('starttime', '00.00.00')
        if isinstance(date_str, bytes): date_str = date_str.decode('ascii', errors='replace')
        if isinstance(time_str, bytes): time_str = time_str.decode('ascii', errors='replace')


    # Number of bytes in header record: (ns + 1) * 256 bytes
    # ns is number of signals.
    num_channels = len(raw.ch_names)
    header_len_bytes = (num_channels + 1) * 256
    
    # reserved field (often 'EDF+C' or 'EDF+D' for EDF+ continuous/discontinuous)
    reserved_global = extra.get('reserved_global_header', '-') 
    if isinstance(reserved_global, bytes): reserved_global = reserved_global.decode('ascii', errors='replace')


    # Number of data records & Duration of a data record
    record_length_val = extra.get('record_length')
    default_n_records_calc = 'N/A'
    if raw.info['sfreq'] and record_length_val is not None:

        actual_record_length = record_length_val[0] if isinstance(record_length_val, (tuple, list, np.ndarray)) and len(record_length_val) > 0 else record_length_val
        if isinstance(actual_record_length, (int, float)) and actual_record_length > 0: # Check it's a positive scalar
            default_n_records_calc = raw.n_times // (raw.info['sfreq'] * actual_record_length)

    n_records = extra.get('n_records', default_n_records_calc)

    record_duration_sec = extra.get('record_length', (1,))[0] # Duration of a data record in seconds


    lines.append(f"--- Global Header Information (as per MNE) ---")
    lines.append(f"{_fmt(ver, 16)}EDF Version (8 ascii)") # Original script format
    lines.append(f"{_fmt(patient_info_str[:80], 80)}Patient ID (80 ascii)")
    lines.append(f"{_fmt(recording_info_str[:80], 80)}Recording ID (80 ascii)")
    lines.append(f"{_fmt(date_str, 16)}Start Date dd.mm.yy (8 ascii)")
    lines.append(f"{_fmt(time_str, 16)}Start Time hh.mm.ss (8 ascii)")
    lines.append(f"{_fmt(str(header_len_bytes), 16)}Number of bytes in header (8 ascii)")
    lines.append(f"{_fmt(reserved_global[:44], 44)}Reserved (44 ascii)")
    lines.append(f"{_fmt(str(n_records), 16)}Number of data records (8 ascii)")
    lines.append(f"{_fmt(str(record_duration_sec), 16)}Duration of a data record in sec (8 ascii)")
    lines.append(f"{_fmt(str(num_channels), 16)}Number of signals (channels) (4 ascii)")
    lines.append("")
    lines.append("--- Channel Specific Information (as per MNE) ---")
    lines.append("=" * 40)

    # --- per-channel header -----------------------------------------------
    # MNE stores channel info in raw.info['chs'] list of dicts
    for ch_idx, ch_dict in enumerate(raw.info['chs']):
        label = ch_dict['ch_name'][:16] # EDF label is 16 ascii
        
        # Transducer type: MNE doesn't typically store this from EDF. Placeholder.
        # If available in _raw_extras (e.g., from pyedfread internals if MNE used it)
        transducer = extra.get('transducer_type', ["unknown"] * num_channels)[ch_idx][:80]
        if isinstance(transducer, bytes): transducer = transducer.decode('ascii', errors='replace')

        # Physical dimension (unit)
        # The `extra.get('units')` is more directly from the EDF header string.
        phys_dim_from_edf_header = "uV" # Default or common EEG unit
        edf_channel_units_list = extra.get('units')

        # Check if edf_channel_units_list is not None and has elements
        if edf_channel_units_list is not None and \
           ( (isinstance(edf_channel_units_list, np.ndarray) and edf_channel_units_list.size > 0) or \
             (not isinstance(edf_channel_units_list, np.ndarray) and len(edf_channel_units_list) > 0) ) and \
           ch_idx < len(edf_channel_units_list):
            
            unit_val = edf_channel_units_list[ch_idx]
            if isinstance(unit_val, bytes):
                phys_dim_from_edf_header = unit_val.decode('ascii', errors='replace').strip()[:8]
            elif isinstance(unit_val, str):
                phys_dim_from_edf_header = unit_val.strip()[:8]
        
        phys_dim_extra = phys_dim_from_edf_header


        # Physical min/max & Digital min/max
        # These are important for scaling. MNE uses them to convert to physical values.
        # raw.info['chs'][ch_idx]['cal'] gives the calibration factor.
        # Digital min/max usually -32768 to 32767 for 16-bit EDF.
        # Physical min/max are in extra['physical_min'], extra['physical_max']
        # Digital min/max are in extra['digital_min'], extra['digital_max']
        phys_min = extra.get('physical_min', [ch_dict.get('range', 0)] * num_channels)[ch_idx] # Placeholder if not found
        phys_max = extra.get('physical_max', [ch_dict.get('range', 0)] * num_channels)[ch_idx]
        dig_min = extra.get('digital_min', [-32768] * num_channels)[ch_idx]
        dig_max = extra.get('digital_max', [32767] * num_channels)[ch_idx]

        # Prefiltering: e.g., "HP:0.1Hz LP:75Hz N:50Hz"
        # Stored in raw.info['highpass'], raw.info['lowpass'] after parsing.
        # The raw string is in extra['prefiltering']
        prefilter_str = extra.get('prefilter', ["unknown"] * num_channels)[ch_idx][:80]
        if isinstance(prefilter_str, bytes): prefilter_str = prefilter_str.decode('ascii', errors='replace')
        # Or assemble from MNE's parsed info if available and more reliable
        # hp = raw.info['highpass']
        # lp = raw.info['lowpass']
        # prefilter_mne = f"HP:{hp}Hz LP:{lp}Hz" if hp and lp else "N/A"

        # Number of samples in each data record (for this channel)
        # This is raw._raw_extras[0]['n_samps'][ch_idx]
        n_samps_per_rec = extra.get('n_samps', [int(raw.info['sfreq'] * record_duration_sec)]*num_channels)[ch_idx]
        
        # Reserved per channel
        reserved_chan = extra.get('reserved_channel', ["-"] * num_channels)[ch_idx][:32]
        if isinstance(reserved_chan, bytes): reserved_chan = reserved_chan.decode('ascii', errors='replace')


        lines.append(f"{_fmt(label, 16)}Label (16 ascii)")
        lines.append(f"{_fmt(transducer, 80)}Transducer Type (80 ascii)")
        lines.append(f"{_fmt(phys_dim_extra, 8)}Physical Dimension (8 ascii)")
        lines.append(f"{_fmt(phys_min, 8)}Physical Minimum (8 ascii)")
        lines.append(f"{_fmt(phys_max, 8)}Physical Maximum (8 ascii)")
        lines.append(f"{_fmt(dig_min, 8)}Digital Minimum (8 ascii)")
        lines.append(f"{_fmt(dig_max, 8)}Digital Maximum (8 ascii)")
        lines.append(f"{_fmt(prefilter_str, 80)}Prefiltering (80 ascii)")
        lines.append(f"{_fmt(n_samps_per_rec, 8)}Samples per data record (8 ascii)")
        lines.append(f"{_fmt(reserved_chan, 32)}Reserved (32 ascii)")
        lines.append("") 

    return lines