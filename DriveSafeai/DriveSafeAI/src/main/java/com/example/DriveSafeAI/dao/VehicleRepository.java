package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.Vehicle;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface VehicleRepository extends JpaRepository<Vehicle, Long> {
    Optional<Vehicle>findByUserId(Long userId);
}

