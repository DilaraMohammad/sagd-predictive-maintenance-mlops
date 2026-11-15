# SAGD Predictive Maintenance & Intelligence Platform

## Overview
A production-ready MLOps and automated intelligence pipeline designed for Steam-Assisted Gravity Drainage (SAGD) operations. The platform transforms raw well sensor streams into real-time failure predictions to ensure operational continuity and mitigate critical asset risks.

## Core Capabilities
* **Data Ingestion & Validation:** Streamlines telemetry inputs, enforcing strict checks for missing values, infinite anomalies, and schema alignment before model processing.
* **Real-Time Inference API:** Built with FastAPI to deliver low-latency failure predictions with robust error handling and standardized response tiers.
* **Containerization & Deployment:** Fully containerized via Docker with optimized build contexts and orchestrated locally using Minikube to mirror enterprise cloud standards.
* **Quality Assurance:** Integrated automated test suites utilizing Pytest and FastAPI TestClient to validate system health and endpoint reliability.

## Technology Stack
* **Language:** Python 3.10
* **API Framework:** FastAPI, Uvicorn
* **Machine Learning & Processing:** Scikit-Learn, Pandas, NumPy, Joblib
* **Testing:** Pytest, TestClient
* **MLOps & Infrastructure:** Docker, Kubernetes (Minikube), GitHub Actions CI/CD

## Architecture Highlights
* Explicit multi-tier exception management separating data validation errors from internal server faults.
* Optimized build protocols leveraging .dockerignore to secure runtime environments and minimize image footprints.
* Automated CI/CD integration ensuring continuous code quality and build verification prior to deployment.