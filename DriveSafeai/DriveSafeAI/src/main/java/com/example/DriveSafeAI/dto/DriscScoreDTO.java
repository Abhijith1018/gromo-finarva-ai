package com.example.DriveSafeAI.dto;
import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class DriscScoreDTO {
    public Float score;
    public int tripsConsidered;
}
