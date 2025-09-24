# 🚗 Car Price Prediction – End-to-End Machine Learning Project

##   Overview

This project predicts the selling price of German used cars using machine learning. It is built as a complete end-to-end pipeline, covering data cleaning, preprocessing, model training, evaluation, and deployment.

The pipeline is modular and production-ready, with Docker containerization already implemented and future scope for AWS deployment.

## 🎯 Objectives

- Build an end-to-end ML pipeline for regression.
- Apply EDA, feature engineering, and model selection.
- Train and evaluate multiple ML models.
- Deploy as a web application using Flask.

## 📊 Dataset

The dataset contains detailed information about German used cars, including technical specifications, price, and listing details.

### 🔑 Columns Description

| Column                 | Description                                      |
| :--------------------- | :----------------------------------------------- |
| Brand                  | Manufacturer (e.g., BMW, Audi, VW)               |
| Model                  | Specific car model                               |
| Color                  | Exterior color                                   |
| Registration Date      | First registration date                          |
| Year                   | Manufacturing year                               |
| Price in Euro          | Selling price (target variable)                  |
| Power kW               | Engine power in kilowatts                        |
| Power PS               | Engine power in horsepower                       |
| Transmission Type      | Manual / Automatic                               |
| Fuel Type              | Petrol, Diesel, Electric, etc.                   |
| Fuel Consumption (L/100km) | Fuel use in liters per 100 km                |
| Fuel Consumption (g/km) | CO₂ emissions                                   |
| Mileage in km          | Distance traveled                                |
| Offer Description      | Extra info about the car listing                 |

### File Info:

- **Name:** `gcar_data.csv`
- **Size:** ~13.76 MB
- **License:** Apache 2.0

## 📁 Project Structure


| Folder / File                           | Description                                    |
| --------------------------------------- | ---------------------------------------------- |
| `data/`                                 | Dataset (gitignored)                           |
| `notebooks/`                            | Jupyter notebooks (EDA, experiments)           |
| `src/`                                  | Source code                                    |
| `src/components/`                       | Modular scripts                                |
| `src/components/data_ingestion.py`      | Script for data ingestion                      |
| `src/components/data_transformation.py` | Script for data preprocessing & transformation |
| `src/components/model_trainer.py`       | Script to train ML models                      |
| `src/pipeline/`                         | Pipeline scripts                               |
| `src/pipeline/train_pipeline.py`        | Main training pipeline                         |
| `src/pipeline/pred.py`                  | Script for making predictions                  |
| `artifacts/`                            | Trained models, scalers, encoders (gitignored) |
| `app.py`                                | Flask app for deployment                       |
| `requirements.txt`                      | Python dependencies                            |
| `Dockerfile`                            | Docker setup                                   |
| `.gitignore`                            | Git ignore file                                |
| `README.md`                             | Project documentation                          |
| `pyproject.toml`                        | Optional project config                        |
| `templates/`                            | Flask HTML templates                           |
| `static/`                               | images for Flask                      |


## ⚙️ Setup & Installation
# 1️ Clone the Repository
- git clone https://github.com/your-username/car-price-prediction.git
- cd car-price-prediction

# 2️ Create Conda Environment (Python 3.8)
- conda create -n car_price python=3.8 -y
- conda activate car_price

# 3️ Install Dependencies
pip install -r requirements.txt

# 4️ Run Locally

Train the model:

python src/pipeline/train_pipeline.py


Run the Flask web app:

python app.py


Open in browser: http://127.0.0.1:5000/

# 🐳 Docker Deployment

# Build Docker Image:

docker build -t car-price-prediction .


# Run Docker Container:

- docker run -p 5000:5000 car-price-prediction


- Open browser: http://127.0.0.1:5000/

# Check running containers:

docker ps


Stop & remove container (if needed):

- docker stop <container_id_or_name>
- docker rm <container_id_or_name>

## 📦 requirements.txt
numpy
pandas
scikit-learn
matplotlib
seaborn
flask
joblib
xgboost
lightgbm
catboost



## 📊 Workflow

- Data Cleaning: Handle missing values, outliers, duplicates.

- EDA: Analyze price trends by age, mileage, brand, fuel type.

- Feature Engineering: Encoding, scaling, feature selection.

- Model Training: Linear Regression, Random Forest, Gradient Boosting.

- Evaluation: RMSE, MAE, R² Score.

- Deployment: Web app with Flask, Docker containerization (done).

## 📈 Results

- Best model: Stacking Regressor (R² ≈ 0.90)

- Example prediction: 2015 Diesel BMW with 80,000 km → Predicted Price: €12,500

- Other models tested: XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, Stacking Regressor

- Train vs Test Gap: ~2% → well-regularized, minimal overfitting

## 🔮 Future Improvements

- Full deployment on AWS/GCP/Heroku

- Add more features like engine size, number of previous owners

- Automate model retraining with new datasets

## 🤖 Modeling Strategy

- Experimented with multiple ML models to predict car prices.

- Filtered dataset to recent cars (2015–2023) for realistic price predictions.

- Cleaned full dataset (1995–2023) for optional historical analysis.