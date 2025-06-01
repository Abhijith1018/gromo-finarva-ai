package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.RiskCategory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface RiskCategoryRepository extends JpaRepository<RiskCategory, Long> {
    Optional<RiskCategory> findByMinScoreLessThanEqualAndMaxScoreGreaterThanEqual(Float score1, Float score2);
}

