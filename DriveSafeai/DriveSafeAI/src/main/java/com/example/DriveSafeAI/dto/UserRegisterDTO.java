package com.example.DriveSafeAI.dto;

import lombok.*;

@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class UserRegisterDTO {
    public String fullName;
    public String email;
    public String drivingLicense;
    public String password;
    public String chasisNo;
    public String vehicleNo;
    public String model;
    public String manufacturer;
}
