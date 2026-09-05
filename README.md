# 🤖 ML Analysis Studio

ML Analysis Studio is a web-based machine learning application that allows users to upload a CSV dataset, analyze it, select a suitable target column, identify the machine learning problem type, train a model, and view its performance.

## 🚀 Features

- Upload CSV datasets
- Analyze dataset structure
- Display:
  - Number of rows
  - Number of columns
  - Column data types
  - Missing values
  - Unique values
- Suggest suitable target columns using confidence-based target selection
- Automatically determine:
  - Classification
  - Regression
- Suggest suitable machine learning models
- Train machine learning models
- Evaluate model performance

## 🧠 Supported Machine Learning Models

### Classification

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

### Regression

- Linear Regression

## 📊 Model Evaluation

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🛠️ Technologies Used

- Python
- FastAPI
- Pandas
- Scikit-learn
- HTML
- CSS
- JavaScript

## 📁 Project Structure

```text
ml-analysis-studio/
│
├── dataset/
│   ├── classification_sample.csv
│   ├── regression_sample.csv
│   ├── student_data.csv
│   └── tested.csv
│
├── models/
│   └── model.pkl
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── utils/
│   ├── analysis.py
│   ├── model_selection.py
│   ├── model_training.py
│   └── preprocessing.py
│
├── main.py
├── requirements.txt
└── README.md
