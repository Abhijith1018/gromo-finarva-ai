package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.DriscScore;

import com.example.DriveSafeAI.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface DriscScoreRepository extends JpaRepository<DriscScore, Long> {
   // List<DriscScore> findByVehicle_Id(Long vehicleId);
    Optional<DriscScore> findTopByUseridOrderByCalculatedAtDesc(User userid);
}
