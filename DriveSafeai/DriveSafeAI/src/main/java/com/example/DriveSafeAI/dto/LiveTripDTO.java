package com.example.DriveSafeAI.dto;

import lombok.*;

@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class LiveTripDTO {
    public String sessionId;
    public Integer observationHour;
    public Long speed;
    public Long rpm;
    public Double acceleration;
    public Double throttlePosition;
    public Long engineTemperature;
    public Double systemVoltage;
    public Double distanceTravelled;
    public Double engineLoadValue;
    public Double latitude;
    public Double longitude;
    public Long altitude;
    public Long idVehicle;
    public Long bodyTemperature;
    public Long idDriver;
    public Long currentWeather;
    public Integer hasPrecipitation;
    public Integer isDayTime;
    public Double temperature;
    public Double windSpeed;
    public Long windDirection;
    public Long relativeHumidity;
    public Double visibility;
    public Long uvIndex;
    public Long cloudCover;
    public Long ceiling;
    public Long pressure;
    public Double precipitation;
    public Long accidentsOnsite;
    public Long designSpeed;
    public Long accidentsTime;
}
