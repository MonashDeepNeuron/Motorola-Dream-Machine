"""
Robot Command Schemas
Messages for controlling robot arms from AI predictions.
"""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

# Command types
CommandType = Literal["REST", "LEFT", "RIGHT", "FORWARD", "BACKWARD", "STOP", "EMERGENCY_STOP"]

class RobotCommand(BaseModel):
    """
    Command to control robot arm based on EEG/AI prediction.
    """
    command_type: CommandType = Field(..., description="Type of movement command")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI prediction confidence (0-1)")
    timestamp: datetime = Field(..., description="When the command was generated (UTC)")
    
    # Optional velocity/position parameters
    velocity: Optional[List[float]] = Field(None, description="Velocity vector [vx, vy, vz, wx, wy, wz] (m/s, rad/s)")
    position: Optional[List[float]] = Field(None, description="Target position [x, y, z, rx, ry, rz] (m, rad)")
    
    # Metadata
    session_id: Optional[UUID] = Field(None, description="EEG session this command originated from")
    prediction_probabilities: Optional[List[float]] = Field(None, description="Full probability distribution from AI model")
    
    @field_validator("velocity")
    def validate_velocity_length(cls, v):
        if v is not None and len(v) != 6:
            raise ValueError("Velocity must have 6 components [vx, vy, vz, wx, wy, wz]")
        return v
    
    @field_validator("position")
    def validate_position_length(cls, v):
        if v is not None and len(v) != 6:
            raise ValueError("Position must have 6 components [x, y, z, rx, ry, rz]")
        return v


class RobotState(BaseModel):
    """
    Current state of the robot arm.
    """
    timestamp: datetime = Field(..., description="State timestamp (UTC)")
    position: List[float] = Field(..., description="Current TCP position [x, y, z, rx, ry, rz]")
    velocity: List[float] = Field(..., description="Current TCP velocity [vx, vy, vz, wx, wy, wz]")
    is_moving: bool = Field(..., description="Whether robot is currently moving")
    in_safety_limits: bool = Field(True, description="Whether robot is within safety limits")
    error_message: Optional[str] = Field(None, description="Error message if any")
    
    @field_validator("position", "velocity")
    def validate_6dof(cls, v):
        if len(v) != 6:
            raise ValueError("Must have 6 DOF components")
        return v


class SafetyLimits(BaseModel):
    """
    Safety constraints for robot motion.
    """
    max_velocity: float = Field(0.5, ge=0, description="Maximum Cartesian velocity (m/s)")
    max_acceleration: float = Field(0.5, ge=0, description="Maximum acceleration (m/s²)")
    workspace_min: List[float] = Field(default_factory=lambda: [-0.5, -0.5, 0.0, -3.14, -3.14, -3.14], 
                                      description="Workspace minimum bounds [x, y, z, rx, ry, rz]")
    workspace_max: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5, 3.14, 3.14, 3.14],
                                      description="Workspace maximum bounds [x, y, z, rx, ry, rz]")
    command_timeout_ms: int = Field(2000, ge=0, description="Stop if no command received (ms)")
    min_confidence: float = Field(0.5, ge=0, le=1, description="Minimum confidence to execute command")
    
    @field_validator("workspace_min", "workspace_max")
    def validate_bounds(cls, v):
        if len(v) != 6:
            raise ValueError("Workspace bounds must have 6 components")
        return v
