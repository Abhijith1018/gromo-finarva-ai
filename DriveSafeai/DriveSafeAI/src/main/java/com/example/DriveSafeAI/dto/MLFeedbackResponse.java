package com.example.DriveSafeAI.dto;

import lombok.*;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
@Setter
public  class MLFeedbackResponse {
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
