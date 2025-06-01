package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "risk_category")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class RiskCategory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String categoryName;
    private Float minScore;
    private Float maxScore;
    private Float premiumMultiplier;
    private String description;

    private LocalDateTime createdAt = LocalDateTime.now();
}

