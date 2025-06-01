package com.example.DriveSafeAI.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor


public class InsuranceClaimDTO {
    public Long policyId;
    public String claimNumber;
    public LocalDate claimDate;
    public LocalDate incidentDate;
    public BigDecimal claimAmount;
    public String description;
}
