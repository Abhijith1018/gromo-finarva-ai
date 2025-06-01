package com.example.DriveSafeAI.dto;

public class MLPredictionResponse {

    private Prediction prediction;

    // Getters and Setters

    public Prediction getPrediction() {
        return prediction;
    }

    public void setPrediction(Prediction prediction) {
        this.prediction = prediction;
    }

    // Inner class for the prediction object
    public static class Prediction {
        private double distance_travelled;
        private double driveScore;
        private int driver_id;
        private double trip_id;

        // Getters and Setters

        public double getDistance_travelled() {
            return distance_travelled;
        }

        public void setDistance_travelled(double distance_travelled) {
            this.distance_travelled = distance_travelled;
        }

        public double getDriveScore() {
            return driveScore;
        }

        public void setDriveScore(double driveScore) {
            this.driveScore = driveScore;
        }

        public int getDriver_id() {
            return driver_id;
        }

        public void setDriver_id(int driver_id) {
            this.driver_id = driver_id;
        }

        public double getTrip_id() {
            return trip_id;
        }

        public void setTrip_id(double trip_id) {
            this.trip_id = trip_id;
        }
    }
}
