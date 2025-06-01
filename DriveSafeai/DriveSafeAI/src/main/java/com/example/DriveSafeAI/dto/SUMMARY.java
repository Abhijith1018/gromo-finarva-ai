package com.example.DriveSafeAI.dto;
//DTO Class Name	Use-Case in MVP	Required?	Notes
//✅ UserRegisterDTO	User + Vehicle registration	Yes	Required to onboard users into the system
//✅ UserResponseDTO	Send back basic user info (email, vehicle)	Yes	Used after registration/login
//✅ TripRequestDTO	Submit telemetry data for scoring	Yes	Core to calculating DRIVE score
//✅ TripResponseDTO	Return drive score and feedback	Yes	Required for frontend to show results
//✅ RiskScoreDTO	Return DRIX score from backend	Yes	Essential for premium and dashboard
//✅ NotificationDTO	Send alerts like "Drive Score too low"	Yes	Improves user experience and feedback loop
//⚠️ InsurancePolicyDTO	Create new insurance policy	Optional	Include only if you're allowing policy creation in MVP
//⚠️ PremiumCalculationDTO	Calculate premium based on DRIX + category	Optional	Needed only if you're calculating/displaying premium now
//⚠️ InsuranceClaimDTO	File/view claims	Optional	Include if claim management is part of MVP
