package com.example.DriveSafeAI.dto;

import lombok.*;

import java.time.LocalDateTime;

@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class NotificationDTO {
    public String message;
    public Boolean isRead;
    public LocalDateTime createdAt;
}

