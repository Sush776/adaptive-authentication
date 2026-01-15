# Adaptive Multi-Modal Behavioral Authentication

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)

---

## Overview

This repository contains the **Master’s dissertation project** on adaptive authentication using behavioral and contextual machine learning features.  
The system predicts login **risk scores** and recommends **adaptive MFA** strategies in real-time.

- Predictive features: **keystroke, mouse_speed, failed login attempts, new device/location flags, device/browser type**.  
- Adaptive MFA thresholds:
  - **Low Risk (<0.3):** Password only  
  - **Medium Risk (0.3–0.7):** OTP / Email  
  - **High Risk (>0.7):** Step-up / Biometric MFA  

---

##  Project Structure
# Adaptive Multi-Modal Behavioral Authentication

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)

---

##  Overview

This repository contains the **Master’s dissertation project** on adaptive authentication using behavioral and contextual machine learning features.  
The system predicts login **risk scores** and recommends **adaptive MFA** strategies in real-time.

- Predictive features: **keystroke, mouse_speed, failed login attempts, new device/location flags, device/browser type**.  
- Adaptive MFA thresholds:
  - **Low Risk (<0.3):** Password only  
  - **Medium Risk (0.3–0.7):** OTP / Email  
  - **High Risk (>0.7):** Step-up / Biometric MFA  

---
Create a Python virtual environment:


python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Model Training:
python src/train_models.py

Locally via FastAPI:

uvicorn src/app:app --reload

Build Docker image:

docker build -t adaptive-mfa-api .


Run container:

docker run -d -p 8000:8000 --name adaptive-mfa adaptive-mfa-api



