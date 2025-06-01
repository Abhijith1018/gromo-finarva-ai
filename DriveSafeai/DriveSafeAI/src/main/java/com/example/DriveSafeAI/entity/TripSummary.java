package com.example.DriveSafeAI.entity;



import jakarta.persistence.*;
import lombok.*;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Table(name = "trip_summary")
public class TripSummary {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Integer tripNo;

    private Float driveScore;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "vehicle_id")
    private Vehicle vehicle;

    private Float maxSpeed;
    private Float avgSpeed;
    private Float maxAcceleration;
    private Float distanceTravelled;

    private Boolean isRainy;
    private Boolean isDay;
}

