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
@Table(name = "insurance_claims")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class InsuranceClaim {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "policy_id")
    private InsurancePolicy policy;

    @Column(unique = true)
    private String claimNumber;

    private LocalDate claimDate;
    private LocalDate incidentDate;
    private BigDecimal claimAmount;
    private BigDecimal approvedAmount;
    private String description;

    @Enumerated(EnumType.STRING)
    private ClaimStatus claimStatus = ClaimStatus.SUBMITTED;

    private LocalDateTime createdAt = LocalDateTime.now();
    private LocalDateTime updatedAt = LocalDateTime.now();
}

