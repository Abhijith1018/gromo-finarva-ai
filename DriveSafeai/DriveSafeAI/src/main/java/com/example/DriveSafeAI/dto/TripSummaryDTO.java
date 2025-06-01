package com.example.DriveSafeAI.dto;

import lombok.*;

@Data
@AllArgsConstructor
@Setter
@Getter
@NoArgsConstructor
public class TripSummaryDTO {
    private Integer tripNo;
    private Float driveScore;
    private Float maxSpeed;
    private Float avgSpeed;
    private Float maxAcceleration;
    private Float distanceTravelled;
    private Boolean isRainy;
    private Boolean isDay;
}
