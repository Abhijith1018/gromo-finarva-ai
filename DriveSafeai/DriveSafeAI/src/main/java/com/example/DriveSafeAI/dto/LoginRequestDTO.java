package com.example.DriveSafeAI.dto;
import lombok.*;
@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor

public class LoginRequestDTO {
    public String email;
    public String password;
}
