package com.example.DriveSafeAI.dto;

import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class TripRequestDTO {
    public Float speed, rpm, acceleration, throttlePosition, engineTemperature;
    public Float systemVoltage, engineLoadValue, distanceTravelled, brake;
    public Long vehicleId;
}

