from typing import Optional, Dict
from pydantic import BaseModel, Field

class TripData(BaseModel):
    speed: float = Field(..., ge=0, description="Current speed in km/h")
    rpm: float = Field(..., ge=0, description="Engine RPM")
    acceleration: float = Field(..., description="Vehicle acceleration in m/s²")
    throttle_position: float = Field(..., ge=0, le=100, description="Throttle position percentage")
    engine_temperature: float = Field(..., description="Engine temperature in °C")
    system_voltage: float = Field(..., ge=0, description="System voltage")
    engine_load_value: float = Field(..., ge=0, le=100, description="Engine load percentage")
    distance_travelled: Optional[float] = Field(None, ge=0, description="Distance travelled in km")

    class Config:
        from_attributes = True