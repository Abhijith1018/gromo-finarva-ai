# 🚗 DriveSafe AI – Real-Time, Behavior-Based Motor Insurance Scoring

**DriveSafe AI** is a smart, AI-driven motor insurance risk engine that leverages real-time driving behavior, machine learning, and edge-to-cloud integration to calculate personalized risk scores. It empowers insurers (B2B) to underwrite smarter policies and helps drivers (B2C) receive fairer premiums through transparent, gamified feedback.

---

## 🎯 Project Objective

DriveSafe AI transforms India’s traditional motor insurance model by shifting from flat-rate pricing to **dynamic, behavior-based premiums**.

### ✅ Key Benefits:
- **For Insurers (B2B)**: Real-time risk scoring, reduced fraud, data-backed premium modeling
- **For Drivers (B2C)**: Personalized drive scores, safety coaching, and premium rewards for responsible driving

---

## ⚙️ Tech Stack Overview

| Component        | Technology                               |
|------------------|-------------------------------------------|
| Machine Learning | Python, **LightGBM**, **Isolation Forest** |
| ML Serving       | **Flask API** for real-time predictions   |
| Backend Layer    | **Spring Boot (Java)** for score logic    |
| Frontend Input   | Simulated live driving events             |
| Database         | **MySQL** for persistent trip storage     |
| API Testing      | Postman, cURL, REST                      |
| Deployment       | GitHub + Localhost (for MVP demo)         |

---

## 🔁 System Architecture

```plaintext
Frontend → Simulated drive data (live)
    ↓
Spring Boot Backend (Java)
    ↓ calls
Flask ML API (Python) → Predicts drive_score
    ↓
Backend calculates: risk_score = weighted_avg(drive_score, distance)
    ↓
Results stored in MySQL database```

## EXAMPLE REQUEST 

POST /predict-score
{
  "speed": 60,
  "rpm": 2100,
  "acceleration": 1.2,
  "throttle": 40.5
}


## EXAMPLE RESPONSE

{
  "drive_score": 76.2
}
