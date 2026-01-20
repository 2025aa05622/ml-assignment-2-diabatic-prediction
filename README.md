# MLAssignment2

# Name - Kameswara Rao K
# Bits Id - 2025aa05622

# Diabetis Prediction

This repository contains an end‑to‑end machine learning project implementing six models and a Streamlit web application for Diabetis Prediction.

## Project Links
- GitHub Repository: https://github.com/2025aa05622/ml-assignment-2-diabatic-prediction
- Streamlit Web App: https://ml-assignment-2-diabatic-prediction-mzuhcuwudafmrx7hbfs2xu.streamlit.app/

---
## a) Problem Statement
The objective of this project is to design and evaluate six distinct machine‑learning classification models aimed at predicting diabetes based on the Diabetes Health Indicators (BRFSS) dataset available on Kaggle. The models will be assessed using appropriate performance metrics to compare their predictive capabilities. In addition, an interactive Streamlit application will be developed to showcase model predictions, provide visualizations of evaluation results, and enable users to upload custom CSV files for real‑time testing and exploration.

---
## b) Dataset Description
- Dataset Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Target Variable: `Diabetes_binary` (0 → `No Diabetes`, 1 → `Diabetes`)
- Total Features in data set: 21
- Total features used in building model: 16
- Key Feature Types:
  - Numerical: `BMI`, `PhysHlth`
  - Categorical: `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `DiffWalk`
- Features removed from data set: `Sex`, `AnyHealthcare`, `NoDocbcCost`, `MentHlth`, `Stroke`

---


## c) Models & Evaluation


<!-- METRICS START -->
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.730 | 0.820 | 0.309 | 0.758 | 0.439 | 0.354 |
| Decision Tree | 0.819 | 0.574 | 0.308 | 0.237 | 0.268 | 0.169 |
| Naive Bayes | 0.781 | 0.789 | 0.332 | 0.566 | 0.418 | 0.310 |
| KNN | 0.857 | 0.710 | 0.435 | 0.099 | 0.162 | 0.155 |
| Random Forest | 0.738 | 0.824 | 0.316 | 0.755 | 0.445 | 0.360 |
| XGBoost | 0.865 | 0.828 | 0.651 | 0.061 | 0.111 | 0.170 |
<!--METRICS END -->

<!-- OBS_TABLE_START -->
| Model             | Observation |
|-------------------|-------------|
| Logistic Regression | Logistic Regression provides a strong baseline with high recall and good AUC, making it effective for identifying diabetic cases. |
| Decision Tree       | The Decision Tree achieves decent accuracy but performs poorly on AUC, recall, and MCC. This indicates a bias toward the majority class and limited generalization capability. |
| KNN                 | KNN shows high accuracy but extremely low recall, indicating that it largely predicts the majority (non-diabetic) class. |
| Naive Bayes         | Naive Bayes offers a balanced and stable performance with moderate recall and F1-score. Despite its strong independence assumptions, it performs competitively and serves as a fast and simple probabilistic baseline model. |
| Random Forest       | Random Forest demonstrates the best overall balance across recall, F1-score, AUC, and MCC. |
| XGBoost             | XGBoost achieves the highest accuracy and AUC, indicating strong ranking capability. However, its very low recall shows that it is overly conservative without class weighting or threshold tuning. |
<!-- OBS_TABLE_END -->


---
## Observations
- With larger datasets it took lot of time to model
- When using Random Forest on larger datasets, the resulting model file size can become excessively large if compression is not applied and only moderate parameter settings are chosen.

---
## Running the Project Locally
### 1. Clone the repository
```
git clone https://github.com/kameswararaokakaraparti/MLAssignment2.git
cd MLAssignment2
```
### 2. Create a virtual environment (optional)
```
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. (Optional) Train all models
```
python -m models.train_logreg
python -m models.train_decision_tree
python -m models.train_knn
python -m models.train_naive_bayes
python -m models.train_random_forest
python -m models.train_xgboost
```
### 5. Run the Streamlit app
```
streamlit run streamlit_app.py
```

---
## Folder Structure
```
data/                 # contains training and test csv files
model/                # cotains model files(*.pkl) and source py files 
streamlit_app.py      # Streamlit frontend
requirements.txt
README.md
```

---
## Summary
This project focuses on the end‑to‑end development of machine‑learning solutions for diabetes prediction using the BRFSS dataset. It encompasses data preprocessing, model construction, and systematic evaluation across multiple algorithms to highlight comparative performance

## Screenshots
![Screenshot] (https://github.com/2025aa05622/ml-assignment-2-diabatic-prediction/blob/main/Screenshot%202026-01-20%20222412.png)

![Screenshot] (https://github.com/2025aa05622/ml-assignment-2-diabatic-prediction/blob/main/Screenshot%202026-01-20%20222612.png)
