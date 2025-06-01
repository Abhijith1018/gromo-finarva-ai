package com.example.DriveSafeAI.service.impl;

import com.example.DriveSafeAI.dto.MLFeedbackResponse;
import com.example.DriveSafeAI.dto.MLPredictionResponse;
import com.example.DriveSafeAI.entity.TripData;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Client for communicating with the Driver Risk Analysis ML API
 * Handles both JSON data and CSV file uploads for predictions
 */
@Component
public class MLModelClient {

    private static final Logger logger = LoggerFactory.getLogger(MLModelClient.class);

    @Value("${ml.api.base-url:http://localhost:5000}")
    private String mlApiBaseUrl;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public MLModelClient() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Health check for the ML API
     */

    /**
     * Get predictions with AI feedback from CSV file
     */


    public Float getDriveScoreFromList() {
        try {
            String url = mlApiBaseUrl + "/predictfsql";


            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<List<Map<String, Object>>> request = new HttpEntity<>(null, headers);

            ResponseEntity<MLPredictionResponse> response = restTemplate.postForEntity(
                    url, request, MLPredictionResponse.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                logger.info("Batch prediction successful for {} records");
                return (float) response.getBody().getPrediction().getDriveScore();
            } else {
                logger.error("Batch prediction failed with status: {}", response.getStatusCode());
                throw new RuntimeException("Batch prediction failed with status: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Error during batch prediction", e);
            throw new RuntimeException("Batch prediction error: " + e.getMessage());
        }

    }

    }

    // Response DTOs

   /* @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MLPredictionResponse {
        private Double driveScore;
        private Double riskScore;
        private String riskCategory;
        private String inputMethod;
        private Map<String, Object> prediction;
        private String error;

        public static MLPredictionResponse error(String error) {
            return MLPredictionResponse.builder()
                    .error(error)
                    .build();
        }
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static  class MLFeedbackResponse {
        private List<Map<String, Object>> aiFeedback;
        private List<Map<String, Object>> tripSummaries;
        private Map<String, Object> overallSummary;
        private String processedAt;
        private String error;

        public static MLFeedbackResponse error(String error) {
            return MLFeedbackResponse.builder()
                    .error(error)
                    .build();
        }
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MLModelInfoResponse {
        private String modelStatus;
        private int trainingFeaturesCount;
        private Map<String, Object> riskThresholds;
        private List<Map<String, Object>> topFeatures;
        private Map<String, Object> modelParams;
        private boolean openaiConfigured;
        private String error;

        public static MLModelInfoResponse error(String error) {
            return MLModelInfoResponse.builder()
                    .error(error)
                    .build();
        }*/

