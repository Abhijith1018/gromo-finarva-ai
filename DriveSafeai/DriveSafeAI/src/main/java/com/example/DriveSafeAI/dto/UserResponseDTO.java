package com.example.DriveSafeAI.dto;
import lombok.*;

@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class UserResponseDTO {
    public Long userId;
    public String email;
    public String vehicleNo;
    public String fullName;
}

