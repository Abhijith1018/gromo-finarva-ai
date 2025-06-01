package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.PremiumCalculation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PremiumCalculationRepository extends JpaRepository<PremiumCalculation, Long> {
    List<PremiumCalculation> findByPolicyIdAndIsActiveTrue(Long policyId);
}

