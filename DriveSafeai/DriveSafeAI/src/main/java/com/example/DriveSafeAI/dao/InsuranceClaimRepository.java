package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.InsuranceClaim;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface InsuranceClaimRepository extends JpaRepository<InsuranceClaim, Long> {
    List<InsuranceClaim> findByPolicyId(Long policyId);
}

