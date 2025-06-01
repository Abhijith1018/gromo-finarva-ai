package com.example.DriveSafeAI.dao;


import com.example.DriveSafeAI.entity.TripSummary;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TripSummaryRepository extends JpaRepository<TripSummary, Long> {
    List<TripSummary> findByVehicleId(Long vehicleId);
    TripSummary findByVehicleIdAndTripNo(Long vehicleId, Integer tripNo);
    List<TripSummary> findByVehicleIdOrderByIdDesc(Long vehicleId, Pageable pageable);
}