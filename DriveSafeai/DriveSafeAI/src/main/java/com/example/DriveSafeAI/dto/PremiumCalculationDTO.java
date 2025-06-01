package com.example.DriveSafeAI.dto;

import java.math.BigDecimal;
import java.sql.Date;
import java.time.LocalDate;

import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor

public class PremiumCalculationDTO {
    public Long policyId;
    public Long basePremium;
    public Long covarageAmount;
    public String coverageType;
    public LocalDate policyEndDate;
    public LocalDate policyStartDate;
    public String policyNumber;
    public Float driscScore;
    public Float finalPremium;
}
