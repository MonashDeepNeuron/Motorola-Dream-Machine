from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# Standard band names (lowercase, extensible)
BandNameLiteral = Literal["delta", "theta", "alpha", "beta", "gamma"]

# ---------------------------------------------------------------------------#
# 1.  Raw EEG Batches                                                        #
# ---------------------------------------------------------------------------#
class EEGBatch(BaseModel):
    """
    Represents a batch of raw EEG data.
    Data is expected to be in Volts.
    """
    device_id: str = Field(..., description="Identifier of the data acquisition device or producer.")
    session_id: UUID = Field(..., description="Unique identifier for the recording session.")
    timestamp: datetime = Field(..., description="Timestamp of the first sample in this batch (UTC).")

    seq_number: int = Field(..., ge=0, description="Sequence number for this batch within the session.")
    sample_rate: float = Field(..., gt=0, description="Sampling rate in Hz.")
    channels: List[str] = Field(..., min_length=1, description="List of EEG channel labels.")
    
    # Data is List[List[float]], where outer list is samples, inner list is channels.
    # Shape: [batch_size_actual, num_channels]
    data: List[List[float]] = Field(
        ..., 
        description="EEG data as a list of samples, where each sample is a list of float values (Volts) for each channel."
    )
    classification_labels: Optional[List[str]] = Field(
        None, 
        description="Optional list of classification labels, one for each sample in the batch (e.g., 'T0', 'T1', 'background'). Length must match number of samples in 'data'."
    )

    @field_validator("data")
    def validate_data_dimensions(cls, v_data: List[List[float]], info: ValidationInfo):
        if not v_data:
            return v_data
        num_channels_expected = len(info.data.get("channels", []))
        if num_channels_expected == 0 and v_data:
            pass
        for i, sample_row in enumerate(v_data):
            if len(sample_row) != num_channels_expected:
                raise ValueError(
                    f"Sample at index {i} in 'data' has {len(sample_row)} values, "
                    f"but {num_channels_expected} channels were declared. All data rows must match channel count."
                )
        return v_data

    @field_validator("classification_labels")
    def validate_classification_labels_length(cls, v_labels: Optional[List[str]], info: ValidationInfo):
        if v_labels is not None:
            num_samples_in_data = len(info.data.get("data", []))
            if len(v_labels) != num_samples_in_data:
                raise ValueError(
                    f"'classification_labels' has {len(v_labels)} entries, "
                    f"but 'data' has {num_samples_in_data} samples. Lengths must match."
                )
        return v_labels

# ---------------------------------------------------------------------------#
# 2.  Band-power per Window (per channel)                                    #
# ---------------------------------------------------------------------------#
class WindowBandPower(BaseModel):
    """
    Represents the power in a specific frequency band for one or more channels,
    calculated over a single time window. Power is typically in μV²/Hz (PSD).
    The producer sends one message per (channel, band, window).
    """
    device_id: str = Field(..., description="Identifier of the processing device (producer).")
    session_id: UUID = Field(..., description="Unique identifier for the recording session this data derives from.")

    window_index: int = Field(..., ge=0, description="Index of the time window.")
    start_time_sec: float = Field(..., description="Start time of the window in seconds, relative to session start.")
    end_time_sec: float = Field(..., description="End time of the window in seconds, relative to session start.")

    band: BandNameLiteral = Field(..., description="The frequency band identifier (e.g., 'alpha').")
    
    channel_labels: List[str] = Field(..., min_length=1, description="List of EEG channel labels this power value corresponds to (usually one per message from producer).")
    power: List[float] = Field(..., min_length=1, description="List of power spectral density (PSD) values (μV²/Hz), one for each channel in 'channel_labels'.")

    @field_validator("power")
    def validate_power_matches_channel_labels(cls, v_power: List[float], info: ValidationInfo):
        num_labels = len(info.data.get("channel_labels", []))
        if len(v_power) != num_labels:
            raise ValueError(
                f"Number of power values ({len(v_power)}) must match the number of channel_labels ({num_labels})."
            )
        return v_power

    @field_validator("end_time_sec")
    def validate_end_time_after_start_time(cls, v_end_time: float, info: ValidationInfo):
        start_time = info.data.get("start_time_sec")
        if start_time is not None and v_end_time <= start_time:
            raise ValueError("'end_time_sec' must be greater than 'start_time_sec'.")
        return v_end_time

# ---------------------------------------------------------------------------#
# 3.  (Optional) EEGBandPower - Kept for potential different aggregation    #
#     This schema is not actively used by the current producer/consumer for #
#     the primary band power stream but kept as an example or for future use.#
# ---------------------------------------------------------------------------#
class EEGBandPowerAggregate(BaseModel):
    """
    Represents aggregated band power across multiple channels for a specific band
    at a single point in time or aggregated over a period.
    This is a more general schema, distinct from WindowBandPower.
    """
    device_id: str
    session_id: UUID
    timestamp: datetime

    band: BandNameLiteral
    channel_labels: List[str] = Field(..., min_length=1)
    power: List[float]

    @field_validator("power")
    def validate_power_length(cls, v_power: List[float], info: ValidationInfo):
        labels = info.data.get("channel_labels")
        if labels and len(v_power) != len(labels):
            raise ValueError(
                "Length of 'power' list must match the length of 'channel_labels' list."
            )
        return v_power