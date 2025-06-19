# Diabetes Prediction Web Application

A clean and interactive web application that predicts whether a user is likely to have diabetes based on medical inputs. The application is built using **FastAPI** for the backend, styled with **HTML/CSS**, and containerized with **Docker** for scalable deployment.

---

## Key Features

- Predicts diabetes using a pre-trained machine learning model (`model.pkl`)
- Simple and user-friendly web interface (HTML + JavaScript)
- REST API endpoint for real-time predictions (`/predict`)
- Docker-compatible for easy local or cloud-based deployment

---

## Requirements

- Python 3.11+
- Docker (optional, for containerization)
- `model.pkl` file (placed in the root directory)

---

## How to Run Locally with Docker

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetes-app.git
cd diabetes-app
2. Build the Docker image
bash
Copy
Edit
docker build -t diabetes-app .
3. Start the container
bash
Copy
Edit
docker run -p 5000:5000 diabetes-app
4. Open your browser
Go to: http://localhost:5000

## How It Works
User enters medical data (e.g., glucose level, insulin, BMI)

Data is submitted to the backend via a POST request to /predict

The pre-trained model analyzes the input and returns one of two outcomes:

0: Likely not diabetic

1: Likely diabetic

⚙️ Technology Stack
Backend: FastAPI

Frontend: HTML / CSS / JavaScript

Machine Learning: Pandas, Scikit-learn

Containerization: Docker

Deployment: Google Cloud Run (Cloud-hosted API)

🌐 Deployment
The application is deployed on Google Cloud Run in the us-central1 region:

Live URL:
https://diabetes-671958340399.us-central1.run.app/

👨‍💻 Author
Duc Phu Nguyen
📧 dpnguyen4@myseneca.ca
🔗 LinkedIn