package com.example.DriveSafeAI.service.impl;

import com.example.DriveSafeAI.dao.*;
import com.example.DriveSafeAI.dto.*;
import com.example.DriveSafeAI.entity.*;
//import com.example.DriveSafeAI.enums.*;
import com.example.DriveSafeAI.service.DriveSafeService;
//import com.example.DriveSafeAI.service.MLModelClient;
import com.example.DriveSafeAI.service.security.JWTService;
//import org.apache.commons.csv.CSVFormat;
//import org.apache.commons.csv.CSVParser;
//import org.apache.commons.csv.CSVRecord;
import com.example.DriveSafeAI.util.TripSessionBuffer;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import java.io.InputStreamReader;
import java.io.Reader;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.web.multipart.MultipartFile;

@Service
public class DriveSafeServiceImpl implements DriveSafeService {

    @Autowired private UserRepository userRepo;
    @Autowired private VehicleRepository vehicleRepo;
    @Autowired private TripRepository tripRepo;
    @Autowired private DriveScoreRepository driveScoreRepo;
    @Autowired private DriscScoreRepository driscScoreRepository;
    @Autowired private NotificationRepository notificationRepo;
    @Autowired private InsurancePolicyRepository policyRepo;
    @Autowired private PremiumCalculationRepository premiumRepo;
    @Autowired private RiskCategoryRepository riskCategoryRepo;
    @Autowired private InsuranceClaimRepository claimRepo;
    @Autowired private AuthenticationManager authenticationManager;
    @Autowired private JWTService jwtService;
   @Autowired private MLModelClient mlClient;
    @Autowired
    private TripSummaryRepository TripSummaryRepository;

    //User Registration
    @Override
    public UserResponseDTO registerUser(UserRegisterDTO dto) {
        User user = new User();
        user.setFullName(dto.fullName);
        user.setEmail(dto.email);
        user.setDrivingLicense(dto.drivingLicense);
        final BCryptPasswordEncoder encoder = new BCryptPasswordEncoder(12);
        user.setPassword(encoder.encode(dto.password));
        userRepo.save(user);

        Vehicle vehicle = new Vehicle();
        vehicle.setChasisNo(dto.chasisNo);
        vehicle.setVehicleNo(dto.vehicleNo);
        vehicle.setModel(dto.model);
        vehicle.setManufacturer(dto.manufacturer);
        vehicle.setUser(user);

        vehicleRepo.save(vehicle);

        return new UserResponseDTO(user.getId(), user.getEmail(), vehicle.getVehicleNo(), user.getFullName());
    }

  //User Login
    //@Override
  //  public UserResponseDTO login(LoginRequestDTO dto) {
//        User user = userRepo.findByEmailAndPassword(dto.email, dto.password)
//                .orElseThrow(() -> new RuntimeException("Invalid credentials"));
//
//        Vehicle vehicle = vehicleRepo.findByUserId(user.getId())
//                .orElseThrow(() -> new RuntimeException("Vehicle not found"));
//
//        return new UserResponseDTO(user.getId(), user.getEmail(), vehicle.getVehicleNo());
//    }

    @Override
    public String login(LoginRequestDTO loginRequestDTO) {
        User user = userRepo.findByEmail(loginRequestDTO.getEmail())
                .orElseThrow(() -> new RuntimeException("User not found"));

        Authentication authentication = authenticationManager.authenticate(new UsernamePasswordAuthenticationToken(loginRequestDTO.getEmail(), loginRequestDTO.getPassword()));
        if (authentication.isAuthenticated()) {
            return jwtService.generateToken(loginRequestDTO.getEmail(), null, user.getId());
        }
        throw new RuntimeException("Invalid credentials");
    }

    //get user
    @Override
    public UserResponseDTO getUserById(Long userId) {
        User user = userRepo.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        Vehicle vehicle = vehicleRepo.findByUserId(userId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found for user"));

        return new UserResponseDTO(user.getId(), user.getEmail(), vehicle.getVehicleNo(),user.getFullName());
    }



  /*  //Trip Submission and DriveScore Generation
    @Override
    public TripResponseDTO submitTrip(TripRequestDTO dto) {
        TripData trip = new TripData();
        trip.setSpeed(dto.speed);
        trip.setRpm(dto.rpm);
        trip.setAcceleration(dto.acceleration);
        trip.setThrottlePosition(dto.throttlePosition);
        trip.setEngineTemperature(dto.engineTemperature);
        trip.setSystemVoltage(dto.systemVoltage);
        trip.setEngineLoadValue(dto.engineLoadValue);
        trip.setDistanceTravelled(dto.distanceTravelled);
        trip.setBrake(dto.brake);

        Vehicle vehicle = vehicleRepo.findById(dto.vehicleId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found"));
        trip.setVehicle(vehicle);

        tripRepo.save(trip);

        Float driveScore = mlClient.getDriveScore(trip);



        int rewardPoints = 0;
        if (trip.getDistanceTravelled() >= 1.0) {
            if (driveScore >= 90) rewardPoints = 50;
            else if (driveScore >= 80) rewardPoints = 30;
            else if (driveScore >= 70) rewardPoints = 20;
            else if (driveScore >= 60) rewardPoints = 10;
            else if (driveScore >= 50) rewardPoints = 5;
            else rewardPoints = 0; // No reward for low scores
        }

        DriveScore score = new DriveScore();
        score.setScore(driveScore);
        score.setTripData(trip);
        score.setVehicle(vehicle);
        score.setRewardPoints(rewardPoints); // ✅ Set reward
        driveScoreRepo.save(score);

// Notification
        Notification n = new Notification();
        n.setUser(vehicle.getUser());
        n.setMessage("You earned " + rewardPoints + " reward points for this trip!");
        n.setMessage("Your Drive Score: " + driveScore);
        notificationRepo.save(n);

        return new TripResponseDTO(trip.getId(), driveScore,
                driveScore > 80 ? "Excellent driving!" : "Improve your braking or acceleration.",rewardPoints);
    }*/

    // DriscScore Calculation
    @Override
    public DriscScoreDTO calculateDriscScore(Long userId, Long N) {
        User user = userRepo.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        Vehicle vehicle = vehicleRepo.findByUserId(userId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found for user"));

         // Variable: insurer defines recent trip count

        // Fetch N latest trip summaries
        List<TripSummary> recentTrips = TripSummaryRepository
                .findByVehicleIdOrderByIdDesc(vehicle.getId(), PageRequest.of(0, N.intValue()));

        if (recentTrips.isEmpty()) {
            throw new RuntimeException("Not enough trip summaries to calculate DriscScore.");
        }

        double weightedSum = 0.0;
        double totalDistance = 0.0;

        for (TripSummary trip : recentTrips) {
            if (trip.getDistanceTravelled() > 0) {
                weightedSum += trip.getDriveScore() * trip.getDistanceTravelled();
                totalDistance += trip.getDistanceTravelled();
            }
        }

        float driscScore = (totalDistance == 0) ? 0f : (float) (weightedSum / totalDistance);

        DriscScore drisc = new DriscScore();
        drisc.setScore(driscScore);
        drisc.setUserid(user);
        drisc.setTripsConsidered(recentTrips.size());

        driscScoreRepository.save(drisc);

        Notification n = new Notification();
        n.setUser(user);
        n.setMessage("✅ DRISC Score updated: " + driscScore);
        notificationRepo.save(n);

        return new DriscScoreDTO(driscScore, recentTrips.size());
    }

    //Get Notifications
    @Override
    public List<NotificationDTO> getUserNotifications(Long userId) {
        return notificationRepo.findByUserId(userId).stream()
                .map(n -> new NotificationDTO(n.getMessage(), n.getIsRead(), n.getCreatedAt()))
                .collect(Collectors.toList());
    }

    //Create Insurance Policy
    @Override
    public String createPolicy(InsurancePolicyDTO dto) {
        InsurancePolicy policy = new InsurancePolicy();
        policy.setPolicyNumber(dto.policyNumber);
        policy.setUser(userRepo.findById(dto.userId).orElseThrow());
        policy.setVehicle(vehicleRepo.findById(dto.vehicleId).orElseThrow());
        policy.setPolicyStartDate(dto.policyStartDate);
        policy.setPolicyEndDate(dto.policyEndDate);
        policy.setCoverageType(dto.coverageType);
        policy.setCoverageAmount(dto.coverageAmount);
        policy.setBasePremium(dto.basePremium);
        policy.setCurrentPremium(dto.basePremium);
        policy.setStatus(PolicyStatus.PENDING);

        policyRepo.save(policy);
        return "Policy created with number: " + policy.getPolicyNumber();
    }

    // 6️⃣ Calculate Premium based on DriscScore + Risk Category
    @Override
    public PremiumCalculationDTO calculatePremium(Long userId) {
        User user = userRepo.findById(userId).orElseThrow();
        Vehicle vehicle = vehicleRepo.findByUserId(userId).orElseThrow();
        InsurancePolicy policy = policyRepo.findByVehicleId(vehicle.getId()).orElseThrow();

        DriscScore latest = driscScoreRepository.findTopByUseridOrderByCalculatedAtDesc(user)
                .orElseThrow(() -> new RuntimeException("No DriscScore found"));

        float finalPremium = 4000 + ((100 - latest.getScore()/8)/100)*4000;
        return new PremiumCalculationDTO(policy.getId(), policy.getBasePremium().longValue(), policy.getCoverageAmount().longValue(),
                policy.getCoverageType(), policy.getPolicyEndDate(), policy.getPolicyStartDate(), policy.getPolicyNumber(), latest.getScore(),finalPremium );
    }

    // 7️⃣ File Insurance Claim
   @Override
    public String fileClaim(InsuranceClaimDTO dto) {
        InsurancePolicy policy = policyRepo.findById(dto.policyId).orElseThrow();

        InsuranceClaim claim = new InsuranceClaim();
        claim.setPolicy(policy);
        claim.setClaimNumber(dto.claimNumber);
        claim.setClaimDate(dto.claimDate);
        claim.setIncidentDate(dto.incidentDate);
        claim.setClaimAmount(dto.claimAmount);
        claim.setDescription(dto.description);
       claim.setClaimStatus(ClaimStatus.SUBMITTED);
        claim.setCreatedAt(LocalDateTime.now());
        claimRepo.save(claim);
       return "Claim filed successfully with number: " + claim.getClaimNumber();
    }

   // 8️⃣ Get All Claims by Policy
    @Override
    public List<InsuranceClaimDTO> getClaimsByPolicy(Long policyId) {
        return claimRepo.findByPolicyId(policyId).stream()
                .map(claim -> new InsuranceClaimDTO(
                        claim.getPolicy().getId(),
                       claim.getClaimNumber(),
                        claim.getClaimDate(),
                        claim.getIncidentDate(),
                        claim.getClaimAmount(),
                        claim.getDescription()))
                .collect(Collectors.toList());
    }


    // Upload Trip CSV File


  @Override
   public String uploadTripCsv(MultipartFile file, Long vehicleId) {
        Vehicle vehicle = vehicleRepo.findById(vehicleId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found"));

        List<TripData> trips = new ArrayList<>();

        try (Reader reader = new InputStreamReader(file.getInputStream());
             CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT
                     .withFirstRecordAsHeader()
                     .withIgnoreHeaderCase()
                     .withTrim())) {

            for (CSVRecord record : parser) {
                TripData trip = new TripData();
                trip.setSpeed(Float.parseFloat(record.get("speed")));
                trip.setRpm(Float.parseFloat(record.get("rpm")));
                trip.setAcceleration(Float.parseFloat(record.get("acceleration")));
                trip.setThrottlePosition(Float.parseFloat(record.get("throttle_position")));
                trip.setEngineTemperature(Float.parseFloat(record.get("engine_temperature")));
                trip.setSystemVoltage(Float.parseFloat(record.get("system_voltage")));
                trip.setEngineLoadValue(Float.parseFloat(record.get("engine_load_value")));
                trip.setDistanceTravelled(Float.parseFloat(record.get("distance_travelled")));
               // trip.setBrake(Float.parseFloat(record.get("brake")));
                trip.setVehicle(vehicle);
                trips.add(trip);
            }

           tripRepo.saveAll(trips); // ✅ Batch save
            return "Uploaded " + trips.size() + " trips successfully.";

        } catch (Exception e) {
            throw new RuntimeException("Failed to parse CSV: " + e.getMessage());
       }

    }

    //for live data fromm frontend

    @Override
    public TripResponseDTO processLiveTripSession(String sessionId) {
        List<LiveTripDTO> dataList = TripSessionBuffer.endSession(sessionId);
        if (dataList == null || dataList.isEmpty()) {
            throw new RuntimeException("No trip data found for session: " + sessionId);
        }

        Long vehicleId = dataList.get(0).getIdVehicle();
        System.out.println(vehicleId);
        Vehicle vehicle = vehicleRepo.findById(vehicleId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found"));

        Integer lastTripNo = tripRepo.findMaxTripNoByVehicleId(vehicleId);
        int currentTripNo = (lastTripNo != null ? lastTripNo + 1 : 1);


        List<TripData> tripList = new ArrayList<>();
        for (LiveTripDTO dto : dataList) {
            TripData t = new TripData();

            t.setSpeed(dto.getSpeed().floatValue());
            t.setRpm(dto.getRpm().floatValue());
            t.setAcceleration(dto.getAcceleration().floatValue());
            t.setThrottlePosition(dto.getThrottlePosition().floatValue());
            t.setEngineTemperature(dto.getEngineTemperature().floatValue());
            t.setSystemVoltage(dto.getSystemVoltage().floatValue());
            t.setDistanceTravelled(dto.getDistanceTravelled().floatValue());
            t.setEngineLoadValue(dto.getEngineLoadValue().floatValue());

            t.setLatitude(dto.getLatitude());
            t.setLongitude(dto.getLongitude());
            t.setAltitude(dto.getAltitude());

            t.setBodyTemperature(dto.getBodyTemperature());
            t.setCurrentWeather(dto.getCurrentWeather());
            t.setHasPrecipitation(dto.getHasPrecipitation());
            t.setIsDayTime(dto.getIsDayTime());
            t.setTemperature(dto.getTemperature());
            t.setWindSpeed(dto.getWindSpeed());
            t.setWindDirection(dto.getWindDirection());
            t.setRelativeHumidity(dto.getRelativeHumidity());
            t.setVisibility(dto.getVisibility());
            t.setUvIndex(dto.getUvIndex());
            t.setCloudCover(dto.getCloudCover());
            t.setCeiling(dto.getCeiling());
            t.setPressure(dto.getPressure());
            t.setPrecipitation(dto.getPrecipitation());
            t.setAccidentsOnsite(dto.getAccidentsOnsite());
            t.setDesignSpeed(dto.getDesignSpeed());
            t.setAccidentsTime(dto.getAccidentsTime());
            t.setTripNo(currentTripNo);  // 👈 NEW
            tripList.add(t);
        }

        tripRepo.saveAll(tripList);

        Float driveScore = mlClient.getDriveScoreFromList();
        int rewardPoints = 0;
         {
            if (driveScore >= 90) rewardPoints = 50;
            else if (driveScore >= 80) rewardPoints = 30;
            else if (driveScore >= 70) rewardPoints = 20;
            else if (driveScore >= 60) rewardPoints = 10;
            else if (driveScore >= 50) rewardPoints = 5;
            else rewardPoints = 0; // No reward for low scores
        }

        DriveScore score = new DriveScore();
        score.setScore(driveScore);
        score.setVehicle(vehicle);
        score.setRewardPoints(rewardPoints);
        score.setTripData(tripList.get(tripList.size() - 1)); // Use last row
        driveScoreRepo.save(score);

        //create and save trip summary
        TripSummary summary = TripSummary.builder()
                .tripNo(currentTripNo)
                .vehicle(vehicle)
                .driveScore(driveScore)
                .maxSpeed((float) tripList.stream().mapToDouble(TripData::getSpeed).max().orElse(0))
                .avgSpeed((float) tripList.stream().mapToDouble(TripData::getSpeed).average().orElse(0))
                .maxAcceleration((float) tripList.stream().mapToDouble(TripData::getAcceleration).max().orElse(0))
                .distanceTravelled((float) tripList.stream().mapToDouble(TripData::getDistanceTravelled).sum())
                .isRainy(false) // TODO: Detect from weather data
                .isDay(true)    // TODO: Detect from timestamp
                .build();

        TripSummaryRepository.save(summary);

        Notification note = new Notification();
        note.setUser(vehicle.getUser());
        note.setMessage("You earned " + rewardPoints + " reward points for this trip!");
        note.setMessage("Live trip session completed. Drive Score: " + driveScore);
        notificationRepo.save(note);

        return new TripResponseDTO(score.getTripData().getId(), driveScore,
                driveScore > 80 ? "Excellent driving!" : "Needs improvement",rewardPoints);


    }

    @Override
    public int getTotalRewardPoints(Long userId) {
        Vehicle vehicle = vehicleRepo.findByUserId(userId)
                .orElseThrow(() -> new RuntimeException("Vehicle not found"));

        return driveScoreRepo.findTop10ByVehicleIdOrderByCreatedAtDesc(Long .valueOf(vehicle.getId()))
                .stream()
                .mapToInt(ds -> ds.getRewardPoints() == null ? 0 : ds.getRewardPoints())
                .sum();
    }


}
