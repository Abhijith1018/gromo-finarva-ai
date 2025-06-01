package com.example.DriveSafeAI.dao;

import com.example.DriveSafeAI.entity.TripData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface TripRepository extends JpaRepository<TripData, Long> {
    List<TripData> findByVehicleId(Long vehicleId);

    @Query("SELECT MAX(t.tripNo) FROM TripData t WHERE t.vehicle.id = :vehicleId")
    Integer findMaxTripNoByVehicleId(@Param("vehicleId") Long vehicleId);

}

