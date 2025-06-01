package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "drisc_score")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class DriscScore {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Float score;
    private Integer tripsConsidered;//the "N" This can be iterated to calculate the score based on multiple trips fixed by insurance companies
    private LocalDateTime calculatedAt = LocalDateTime.now();

    @ManyToOne
    @JoinColumn(name = "User_id")
    private User userid;
}

