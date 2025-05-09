from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ValidationInfo

class EEGBatch(BaseModel):
    device_id: str
    session_id : UUID
    timestamp: datetime

    seq_number: int = Field(..., ge=0, description="Counter for batch order")
    sample_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    channels: List[str] = Field(..., min_items=1, description="List of EEG channel labels")
    data: List[List[float]] = Field(
        ..., description="[batch_size x num_channels] of EEG readings in mV" #TODO: confirm
    )
    classification_labels: Optional[List[str]] = Field(None, description = "Optional list of classfiication labels, one for each sample in the batch (e.g., 'moving_up', 'resting')")

    @field_validator("data")
    def validate_data_dimensions(cls, v, info: ValidationInfo):
        chans = info.data.get("channels") if getattr(info, "data", None) else None
        if chans and any(len(row) != len(chans) for row in v):
            raise ValueError(
                "Each row in data must have len equal to num of channels"
            )
        return v

    @field_validator("classification_labels")
    def validate_classification_labels(cls, v_labels, info: ValidationInfo):
        if v_labels is not None:
            data_samples = info.data.get("data") if getattr(info, "data", None) else None
            if data_samples and len(v_labels) != len(data_samples):
                raise ValueError(
                    "Length of `classification_labels` must match the number of samples in `data` (i.e., len(data))"
                )
        return v_labels

class EEGBandPower(BaseModel):
    device_id: str
    session_id : UUID
    timestamp: datetime

    band: Literal['delta', 'theta', 'alpha', 'beta', 'gamma'] # freq band identifier
    channel_labels: List[str] = Field(..., min_items=1, description="List of EEG channel labls")
    power: List[float] # band power vals/channel

    @field_validator("power")
    def validate_power_length(cls, v, info: ValidationInfo):
        labels = info.data.get("channel_labels") if getattr(info, "data", None) else None
        if labels and len(v) != len(labels):
            raise ValueError(
                "Length of `power` must match length of `channel_labels`"
            )
        return v