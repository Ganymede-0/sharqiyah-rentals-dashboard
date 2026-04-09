An end-to-end data science project and interactive web application that analyzes historical real estate transactions in Saudi Arabia’s Eastern Province (Sharqiyah) and deploys an AI-powered pricing engine for investors.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sharqiyah-rentals-dashboard-sf8t3e6zhiwpeynryplk2w.streamlit.app/)

### Quick Links
* **Live Web App:** [Access the Dashboard Here](https://sharqiyah-rentals-dashboard-sf8t3e6zhiwpeynryplk2w.streamlit.app/)
* **Raw Dataset:** [Saudi Open Data Portal](https://open.data.gov.sa/en/datasets/view/a108f1ed-0091-4264-bb82-71a4ad0989f8/resources)

---

## Project Overview

The Eastern Province is one of the most economically active regions in Saudi Arabia. This project bridges the gap between raw, unstructured rental data and actionable business intelligence. It features a two-phase architecture:

1. **Historical Market Analysis (EDA):** A dynamic dashboard exploring over 1 million rental deals (2019–2024), uncovering pricing hierarchies, volume dominance (Dammam/Al Khobar), and post-pandemic market recovery trends.
2. **Predictive Pricing Engine:** A deployed `CatBoost` machine learning model that acts as a real-time property valuation tool for investors, enhanced by a custom time-series forecasting engine.

---

## Dashboard Previews

### Phase 1: Historical Market Analysis
*Filter by city, property type, and year to extract market insights.*

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/acfabcf2-ab1a-4f50-8ebe-2209b7b49e5f" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2cdde2c1-ecfe-435f-8a1d-0830606ebf38" />


### Phase 2: AI Price Estimator
*Input property parameters to receive an AI-generated fair market valuation.*

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/208c51ea-53a0-4c2d-bae1-3e417327e04e" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8aaedb94-f081-4cdc-8053-c0fc05f8072e" />


---

## Machine Learning Architecture

* **Algorithm:** `CatBoostRegressor` (Chosen over XGBoost for superior handling of text-heavy categorical data).
* **Target Variable Optimization:** Trained on a Log-Transformed target variable (`numpy.log1p`) to handle extreme luxury property outliers, with inverse transformations (`numpy.expm1`) applied during live inference.
* **Accuracy:** Achieved a baseline accuracy of **R² = 0.794**.
* **Data Pipeline:** Utilizes a serialized Scikit-Learn `ColumnTransformer` to enforce strict One-Hot Encoding on user inputs without data leakage.

### The "Hybrid Forecasting Engine"
A core limitation of Gradient Boosting Decision Trees (GBDTs) is their inability to extrapolate temporal data into future, unseen years. 

To solve this for real estate investors, this application utilizes a **Hybrid Forecasting Engine**:
1. The CatBoost model generates a highly accurate "Base Value" using historical market boundaries (capped at 2024).
2. The application applies financial time-series mathematics via a user-controlled **Compound Annual Growth Rate (CAGR)** slider, transforming the ML model into a dynamic scenario-analysis tool for future projections.

---

## Tech Stack
* **Language:** Python 3.11
* **Machine Learning:** Scikit-Learn (1.6.1), CatBoost, XGBoost
* **Data Processing:** Pandas, NumPy, Joblib
* **Data Visualization:** Plotly Express
* **Web Deployment:** Streamlit, Streamlit Community Cloud

---

## How to Run Locally

If you wish to run this dashboard on your own machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/Ganymede-0/sharqiyah-rentals-dashboard.git](https://github.com/Ganymede-0/sharqiyah-rentals-dashboard.git)
