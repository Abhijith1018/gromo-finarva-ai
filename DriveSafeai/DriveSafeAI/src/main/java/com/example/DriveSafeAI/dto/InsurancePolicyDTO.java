package com.example.DriveSafeAI.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor


public class InsurancePolicyDTO {
    public String policyNumber;
    public Long userId;
    public Long vehicleId;
    public LocalDate policyStartDate;
    public LocalDate policyEndDate;
    public String coverageType;
    public BigDecimal coverageAmount;
    public BigDecimal basePremium;
}
