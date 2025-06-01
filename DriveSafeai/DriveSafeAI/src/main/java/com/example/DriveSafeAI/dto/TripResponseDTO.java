package com.example.DriveSafeAI.dto;
import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor

public class TripResponseDTO {
    public Long tripId;
    public Float driveScore;
    public String feedback;
    private Integer rewardPoints;
}

