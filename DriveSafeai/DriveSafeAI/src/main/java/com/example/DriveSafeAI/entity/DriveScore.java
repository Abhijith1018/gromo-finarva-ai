package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "drive_score")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class DriveScore {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Float score;

    private LocalDateTime createdAt = LocalDateTime.now();

    @ManyToOne
    @JoinColumn(name = "vehicle_id")
    private Vehicle vehicle;//this also can be changed to user

    @OneToOne
    @JoinColumn(name = "tripdata_id", unique = true)
    private TripData tripData;

    @Column(name = "reward_points")
    private Integer rewardPoints;
    private Float distanceTravelled;
}

