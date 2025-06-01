package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.DriveScore;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface DriveScoreRepository extends JpaRepository<DriveScore, Long> {
    List<DriveScore> findTop10ByVehicleIdOrderByCreatedAtDesc(Long vehicleId);
}

