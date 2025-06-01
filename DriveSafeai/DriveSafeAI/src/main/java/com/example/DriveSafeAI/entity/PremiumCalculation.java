package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "premium_calculation")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class PremiumCalculation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "policy_id")
    private InsurancePolicy policy;

    @ManyToOne
    @JoinColumn(name = "drisc_score_id")
    private DriscScore driscScore;

    @ManyToOne
    @JoinColumn(name = "risk_category_id")
    private RiskCategory riskCategory;

    private BigDecimal basePremium;
    private Float riskMultiplier;
    private BigDecimal calculatedPremium;

    private LocalDateTime calculationDate = LocalDateTime.now();
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private Boolean isActive = true;
}

