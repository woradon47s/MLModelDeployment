# 🧠 ML Model Deployment with Flask and Docker

## 🔍 Overview
This project demonstrates how to train, serve, and test machine learning models using **Flask** and **Docker**. It includes:

1. **Classification Model (Iris Dataset)**
* Model: `RandomForestClassifier`
* Output: Class prediction and confidence score
* Input shape: 4 numeric features

2. **Regression Model (Housing Dataset)**
* Model: `RandomForestRegressor` inside a scikit-learn Pipeline
* Output: Price prediction
* Input shape: 20 numeric/categorical features (after one-hot encoding)

## 🛠 Setup Steps

1. Clone the Repository
```bash
git clone https://github.com/woradon47s/MLModelDeployment
cd MLDeployWithDocker
```

2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

3. Train the Models Classification Model
```bash
python train.py
```
Generates `model.pkl.` Regression Model
```bash
python train_regression.py
```
Generates `reg_model.pkl.`

4. Run the Flask API
```bash
python python app.py
```
The API will be available at: http://localhost:9000

5. (Optional) Run via Docker Build Docker Image
```bash
docker build -t my-ml-model .
```
Run Docker Container
```bash
docker run -p 9000:9000 my-ml-model
```

## 🚀 API Endpoints

### ✅ Health Check
* **GET** `/health`
* **Request:**
```bash
curl http://localhost:9000/health
```
* **Response:**
```json
{ "status": "ok" }
```
### 🌸 Classification Endpoint
**POST** `/predict`

**Single Input**
* **Request:**
```bash
curl -X POST http://localhost:9000/predict \
-H "Content-Type: application/json" \
-d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```
* **Response:**
```json
{ "predictions": [0], "confidences": [0.97] }
```
**Multiple Inputs**
* **Request:**
```bash
curl -X POST http://localhost:9000/predict \
-H "Content-Type: application/json" \
-d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}'
```
* **Response:**
```json
{ "predictions": [0, 2], "confidences": [0.97, 0.94] }
```

### 🏠 Regression Endpoint (if implemented in Flask)
If you create a separate `/predict_regression` endpoint or use a dashboard: **Note:** The regression endpoint is not yet integrated in your `app.py`, but the model `reg_model.pkl` is trained and ready.

### 📦 File Structure
```bash
├── app.py                  # Flask API
├── train.py                # Train classification model
├── train_regression.py     # Train regression model
├── model.pkl               # Saved classification model
├── reg_model.pkl           # Saved regression pipeline
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker setup
```
### 💡 Notes
* Ensure that `model.pkl` exists before starting the Flask app.
* Port `9000` is used both locally and in Docker.
* `predict_proba()` is used for classification confidence.
* Input validation ensures correct shape and data types.

