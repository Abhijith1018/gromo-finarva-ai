package com.example.DriveSafeAI.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "vehicle")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Vehicle {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String chasisNo;

    @Column(unique = true, nullable = false)
    private String vehicleNo;

    private String model;
    private String manufacturer;

    @OneToOne
    @JoinColumn(name = "user_id", unique = true)
    private User user;
}

