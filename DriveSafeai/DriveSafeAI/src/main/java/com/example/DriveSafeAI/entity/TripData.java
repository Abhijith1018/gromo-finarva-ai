package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "trip_data")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class TripData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Vehicle telemetry data
    private Float speed, rpm, acceleration, throttlePosition, engineTemperature;
    private Float systemVoltage, engineLoadValue, distanceTravelled;

    // Location data
    private Double latitude, longitude;
    private Long altitude;

    // Driver and vehicle identifiers
    private Long bodyTemperature;
    private Long idDriver;

    // Weather data
    private Long currentWeather;
    private Integer hasPrecipitation;
    private Integer isDayTime;
    private Double temperature;
    private Double windSpeed;
    private Long windDirection;
    private Long relativeHumidity;
    private Double visibility;
    private Long uvIndex;
    private Long cloudCover;
    private Long ceiling;
    private Long pressure;
    private Double precipitation;

    // Road and traffic data
    private Long accidentsOnsite;
    private Long designSpeed;
    private Long accidentsTime;

    // Observation metadata
    private Integer observationHour;

    // Risk Features (calculated)
    private Float speedRisk, harshAcceleration, rpmEfficiency, highRpmRisk, throttleAggression;
    private Float engineTempRisk, voltageRisk, engineLoadRisk, weatherRisk;
    private Float speedVariance, accelerationVariance, excessiveThrottleTime, engineStrain;




    private LocalDateTime recordedAt = LocalDateTime.now();

    @ManyToOne
    @JoinColumn(name = "vehicle_id")
    private Vehicle vehicle;

    @Column(name = "trip_no")
    private Integer tripNo;
}