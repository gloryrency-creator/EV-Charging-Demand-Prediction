# EV Charging Demand Prediction and Usage Pattern Analysis

## Project Overview
This project analyzes electric vehicle (EV) charging transactions to understand charging usage patterns and predict charging demand using machine learning.

## Objectives
- Analyze EV charging usage patterns.
- Identify demand trends by location, time, and charger type.
- Build regression models to predict numerical charging demand.
- Build classification models to categorize demand as High or Normal.
- Develop an interactive Streamlit dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost and LightGBM
- SHAP and LIME
- Streamlit
- Jupyter Notebook

## Machine Learning Models

### Regression
Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM.

Best model: LightGBM

- MAE: 10.0030
- RMSE: 13.0932
- R² Score: 0.1147

### Classification
Logistic Regression, Decision Tree, Random Forest, KNN, Linear SVM, XGBoost, and LightGBM.

Best model: LightGBM

- Accuracy: 64.51%
- Precision: 61.70%
- Recall: 72.72%
- F1 Score: 66.76%
- ROC-AUC: 69.31%

## Explainable AI
- Feature Importance
- SHAP
- LIME

## Dashboard Features
- Project overview and KPIs
- Charging usage analysis
- Interactive filters
- Model comparison
- Charging demand prediction

## Project Structure
- `app.py` – Streamlit dashboard
- `dataset/` – Dataset and model results
- `models/` – Saved trained models
- `notebook/` – Data analysis and machine learning notebooks
- `requirements.txt` – Project dependencies

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Conclusion
This project demonstrates an end-to-end machine learning workflow for EV charging demand analysis, including data preprocessing, exploratory analysis, model development, explainability, and interactive dashboard deployment.