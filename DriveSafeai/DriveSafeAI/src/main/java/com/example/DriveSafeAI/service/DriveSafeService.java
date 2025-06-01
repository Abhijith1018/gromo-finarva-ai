package com.example.DriveSafeAI.service;

import com.example.DriveSafeAI.dto.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface DriveSafeService {
    UserResponseDTO registerUser(UserRegisterDTO dto);

    String login(LoginRequestDTO dto);

    //getuser
    UserResponseDTO getUserById(Long userId);



   // TripResponseDTO submitTrip(TripRequestDTO dto);

    DriscScoreDTO calculateDriscScore(Long userId, Long N);  //updated to use userid

    List<NotificationDTO> getUserNotifications(Long userId);

    String createPolicy(InsurancePolicyDTO dto);

    PremiumCalculationDTO calculatePremium(Long userId);

  String fileClaim(InsuranceClaimDTO dto);

    List<InsuranceClaimDTO> getClaimsByPolicy(Long policyId);


//upload trip csv file
    String uploadTripCsv(MultipartFile file, Long vehicleId);

    //for continous live data from frontend
    TripResponseDTO processLiveTripSession(String sessionId);

    //get total reward points for a user
    public int getTotalRewardPoints(Long userId);


}
