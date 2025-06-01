package com.example.DriveSafeAI.controller;

import com.example.DriveSafeAI.dao.TripSummaryRepository;
import com.example.DriveSafeAI.dao.VehicleRepository;
import com.example.DriveSafeAI.dto.*;
import com.example.DriveSafeAI.entity.Vehicle;
import com.example.DriveSafeAI.service.DriveSafeService;
import com.example.DriveSafeAI.service.impl.MLModelClient;
import com.example.DriveSafeAI.util.TripSessionBuffer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.stream.Collectors;


@RestController
@RequestMapping("/api")
public class DriveSafeController {

    @Autowired
    private DriveSafeService driveSafeService;

    @Autowired
    private TripSummaryRepository TripSummaryRepository;
    @Autowired
    private VehicleRepository vehicleRepository;

    @Autowired
    private MLModelClient mlClient;

    // 1️⃣ Register new user + vehicle
    @PostMapping("/register")
    public ResponseEntity<UserResponseDTO> registerUser(@RequestBody UserRegisterDTO dto) {
        return ResponseEntity.ok(driveSafeService.registerUser(dto));
    }

    // User login
    @PostMapping("/login")
    public ResponseEntity<String> login(@RequestBody LoginRequestDTO dto) {
        return ResponseEntity.ok(driveSafeService.login(dto));
    }




//    // 2️⃣ Submit trip data and get DriveScore
//    @PostMapping("/trip")
//    public ResponseEntity<TripResponseDTO> submitTrip(@RequestBody TripRequestDTO dto) {
//        return ResponseEntity.ok(driveSafeService.submitTrip(dto));
//    }

    // 3️⃣ Calculate DriscScore (risk score) for user
    @GetMapping("/drisc-score/{userId}")
    public ResponseEntity<DriscScoreDTO> calculateDriscScore(@PathVariable Long userId, @RequestParam Long N) {
        return ResponseEntity.ok(driveSafeService.calculateDriscScore(userId, N));
    }

    // 4️⃣ Get all notifications for user
    @GetMapping("/notifications/{userId}")
    public ResponseEntity<List<NotificationDTO>> getNotifications(@PathVariable Long userId) {
        return ResponseEntity.ok(driveSafeService.getUserNotifications(userId));
    }

    // 5️⃣ Create insurance policy
    @PostMapping("/insurance/policy")
    public ResponseEntity<String> createPolicy(@RequestBody InsurancePolicyDTO dto) {
        return ResponseEntity.ok(driveSafeService.createPolicy(dto));
    }

    // 6️⃣ Calculate premium based on DriscScore and risk category
    @GetMapping("/insurance/premium/{userId}")
    public ResponseEntity<PremiumCalculationDTO> calculatePremium(@PathVariable Long userId) {
        return ResponseEntity.ok(driveSafeService.calculatePremium(userId));
    }

    // 7️⃣ File an insurance claim
   @PostMapping("/insurance/claim")
    public ResponseEntity<String> fileClaim(@RequestBody InsuranceClaimDTO dto) {
        return ResponseEntity.ok(driveSafeService.fileClaim(dto));
    }

    // 8️⃣ Get all claims for a policy
   @GetMapping("/insurance/claim/{policyId}")
    public ResponseEntity<List<InsuranceClaimDTO>> getClaimsByPolicy(@PathVariable Long policyId) {
        return ResponseEntity.ok(driveSafeService.getClaimsByPolicy(policyId));
    }
//Adding CSV Upload Endpoint
@PostMapping("/upload-trips/{vehicleId}")
public ResponseEntity<String> uploadTrips(@RequestParam("file") MultipartFile file,
                                          @PathVariable Long vehicleId) {
    return ResponseEntity.ok(driveSafeService.uploadTripCsv(file, vehicleId));
}

    @PostMapping("/live")
    public ResponseEntity<String> receiveLiveTrip(@RequestBody LiveTripDTO dto) {
        TripSessionBuffer.addToSession(dto.getSessionId(), dto);
        return ResponseEntity.ok("Received one data point for session: " + dto.getSessionId());
    }

    @PostMapping("/end-session/{sessionId}")
    public ResponseEntity<TripResponseDTO> endSession(@PathVariable String sessionId) {
        return ResponseEntity.ok(driveSafeService.processLiveTripSession(sessionId));
    }

//get user
@GetMapping("/user/{userId}")
public ResponseEntity<UserResponseDTO> getUserById(@PathVariable Long userId) {
    return ResponseEntity.ok(driveSafeService.getUserById(userId));
}

    @GetMapping("/rewards/{userId}")
    public ResponseEntity<Integer> getTotalRewards(@PathVariable Long userId) {
        return ResponseEntity.ok(driveSafeService.getTotalRewardPoints(userId));
    }

    //trip summary endpoint
    @GetMapping("/trip-summary/{userId}")
    public List<TripSummaryDTO> getTripSummaries(@PathVariable Long userId) {
        Long vehicleId = vehicleRepository.findByUserId(userId)
                .map(Vehicle::getId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found"));
        return TripSummaryRepository.findByVehicleId(vehicleId).stream()
                .map(s -> new TripSummaryDTO(
                        s.getTripNo(), s.getDriveScore(), s.getMaxSpeed(), s.getAvgSpeed(),
                        s.getMaxAcceleration(), s.getDistanceTravelled(), s.getIsRainy(), s.getIsDay()
                ))
                .collect(Collectors.toList());
    }

//    @GetMapping("/rewards/trip/{tripNo}")
//    public ResponseEntity<Integer> getTripRewards(@PathVariable Long tripNo) {
//        return ResponseEntity.ok(driveSafeService.getTotalRewardPoints(userId));
//    }





}
