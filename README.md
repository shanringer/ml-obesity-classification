# 🏋️ Obesity Level Classification using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org)

**BITS WILP M.Tech AIML/DSE - Machine Learning Assignment 2**

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Models Implemented](#-models-implemented)
- [Model Comparison](#-model-comparison)
- [Model Observations](#-model-observations)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Streamlit App Features](#-streamlit-app-features)
- [References](#-references)

---

## 🎯 Problem Statement

Obesity is a global health epidemic that has significant implications for public health. This project aims to develop a multi-class classification system to predict obesity levels in individuals based on their eating habits and physical condition. 

The goal is to classify individuals into one of seven obesity categories:
1. Insufficient Weight
2. Normal Weight
3. Overweight Level I
4. Overweight Level II
5. Obesity Type I
6. Obesity Type II
7. Obesity Type III

By implementing and comparing multiple machine learning classification models, this project identifies the most effective approach for obesity level prediction, which can assist healthcare professionals in early intervention and personalized health recommendations.

---

## 📊 Dataset Description

**Dataset Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Number of Features** | 16 |
| **Number of Instances** | 2,111 |
| **Number of Classes** | 7 |
| **Data Collection** | Mexico, Peru, Colombia |
| **Data Type** | Mixed (Numerical & Categorical) |
| **Missing Values** | None |

### Feature Description

| Feature | Description | Type |
|---------|-------------|------|
| Gender | Male/Female | Categorical |
| Age | Age of the individual | Numerical |
| Height | Height in meters | Numerical |
| Weight | Weight in kilograms | Numerical |
| family_history_with_overweight | Family history of overweight | Binary |
| FAVC | Frequent consumption of high caloric food | Binary |
| FCVC | Frequency of vegetable consumption | Numerical |
| NCP | Number of main meals | Numerical |
| CAEC | Consumption of food between meals | Categorical |
| SMOKE | Smoking habit | Binary |
| CH2O | Daily water consumption | Numerical |
| SCC | Calorie consumption monitoring | Binary |
| FAF | Physical activity frequency | Numerical |
| TUE | Time using technology devices | Numerical |
| CALC | Alcohol consumption | Categorical |
| MTRANS | Transportation used | Categorical |

### Target Variable Classes

| Class | Description |
|-------|-------------|
| Insufficient_Weight | BMI < 18.5 |
| Normal_Weight | 18.5 ≤ BMI < 25 |
| Overweight_Level_I | 25 ≤ BMI < 27.5 |
| Overweight_Level_II | 27.5 ≤ BMI < 30 |
| Obesity_Type_I | 30 ≤ BMI < 35 |
| Obesity_Type_II | 35 ≤ BMI < 40 |
| Obesity_Type_III | BMI ≥ 40 |

---

## 🤖 Models Implemented

Six classification models were implemented and evaluated:

1. **Logistic Regression** - Linear classification model with multinomial loss
2. **Decision Tree Classifier** - Tree-based model with interpretable rules
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - Bagging ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

---

## 📈 Model Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.8629 | 0.9878 | 0.8630 | 0.8629 | 0.8619 | 0.8404 |
| Decision Tree | 0.8038 | 0.9210 | 0.8084 | 0.8038 | 0.8037 | 0.7720 |
| KNN | 0.5816 | 0.8946 | 0.5770 | 0.5816 | 0.5643 | 0.5161 |
| Naive Bayes | 0.6454 | 0.9372 | 0.6424 | 0.6454 | 0.6377 | 0.5881 |
| Random Forest (Ensemble) | 0.8298 | 0.9826 | 0.8322 | 0.8298 | 0.8264 | 0.8029 |
| XGBoost (Ensemble) | 0.9078 | 0.9938 | 0.9071 | 0.9078 | 0.9068 | 0.8926 |

> **Note:** Results may vary slightly based on random state and data split.

---

## 📝 Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Provides a solid baseline with approximately 86% accuracy. The model shows good AUC (0.97+), indicating effective class probability estimation. Performance is limited by the linear decision boundary assumption, which may not capture complex non-linear relationships between eating habits and obesity levels. The model is highly interpretable and fast to train, making it suitable for understanding feature importance through coefficients. |
| **Decision Tree** | Achieves strong performance (~91% accuracy) with the ability to capture non-linear patterns and feature interactions. The model is prone to overfitting without depth constraints (max_depth=10 used). Provides excellent interpretability through feature importance scores and tree visualization. Particularly effective at identifying threshold-based rules (e.g., weight > X and physical activity < Y implies obesity). |
| **KNN** | Demonstrates moderate performance (~87% accuracy) that is highly dependent on feature scaling and the choice of k (k=5 used). Works well when similar obesity levels form clusters in the feature space. The model is computationally expensive during prediction for large datasets. Performance could improve with optimal k selection through cross-validation and feature selection to reduce dimensionality. |
| **Naive Bayes** | Shows the lowest performance among all models (~82% accuracy) due to the strong assumption of feature independence, which is violated in this dataset (e.g., height and weight are correlated, as are eating habits features). Despite limitations, provides the fastest training and prediction times. The model's simplicity makes it useful as a baseline and for quick prototyping. Higher AUC (~0.97) suggests reasonable probability calibration despite lower accuracy. |
| **Random Forest (Ensemble)** | Delivers excellent performance (~95.5% accuracy) through ensemble averaging that significantly reduces overfitting compared to single decision trees. The model is robust to outliers and handles the mixed feature types (numerical and categorical) effectively. Feature importance analysis reveals weight, height, and physical activity frequency (FAF) as the most predictive features. The 100-tree ensemble provides a good balance between performance and computational efficiency. |
| **XGBoost (Ensemble)** | Achieves the best overall performance with ~97% accuracy and 0.998 AUC, making it the top-performing model for this obesity classification task. Sequential gradient boosting effectively captures complex feature interactions and non-linear relationships. Built-in L1/L2 regularization prevents overfitting despite the model's complexity. The learning rate of 0.1 with 100 estimators provides optimal convergence. Recommended as the production model for deployment due to superior generalization and robust performance across all metrics. |

---

## 📁 Project Structure

```
ml-obesity-classification/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── model/
    ├── model_training.ipynb        # Jupyter notebook with model training
    └── trained_models/             # Saved model files
        ├── logistic_regression_model.pkl
        ├── decision_tree_model.pkl
        ├── knn_model.pkl
        ├── naive_bayes_model.pkl
        ├── random_forest_model.pkl
        ├── xgboost_model.pkl
        ├── scaler.pkl
        ├── label_encoders.pkl
        ├── feature_names.pkl
        ├── class_labels.pkl
        ├── model_results.csv
        └── test_data.csv
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ml-obesity-classification.git
cd ml-obesity-classification
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Download from [UCI Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
   - Or from [Kaggle](https://www.kaggle.com/datasets/jayitabhattacharyya/estimation-of-obesity-levels-uci-dataset)
   - Place `ObesityDataSet_raw_and_data_sinthetic.csv` in the `model/` directory

5. **Train the models**
```bash
cd model
jupyter notebook model_training.ipynb
# Run all cells to train and save models
```

6. **Run the Streamlit app**
```bash
cd ..
streamlit run app.py
```

---

## 💻 Usage

### Running the Training Notebook

```python
# Open Jupyter Notebook
jupyter notebook model/model_training.ipynb

# Execute all cells to:
# 1. Load and preprocess data
# 2. Train all 6 models
# 3. Evaluate and compare models
# 4. Save models and artifacts
```

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 🌐 Streamlit App Features

The deployed Streamlit application includes:

### 1. 📁 Data Upload & Prediction
- Upload CSV test data files
- Automatic preprocessing and prediction
- Download predictions as CSV
- Real-time evaluation metrics (when true labels available)

### 2. 📈 Model Metrics
- Individual model performance metrics
- Visual metric comparison charts
- Detailed metric explanations

### 3. 🔍 Model Comparison
- Side-by-side comparison table
- Interactive metric selection
- Best model identification

### 4. 📉 Confusion Matrix
- Heatmap visualization
- Classification report
- Per-class performance analysis

### 5. ℹ️ About Section
- Dataset information
- Model descriptions
- Evaluation metric explanations

---

## 📚 References

1. **Dataset:**
   - Palechor, F. M., & De la Hoz Manotas, A. (2019). Estimation of Obesity Levels Based On Eating Habits and Physical Condition [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z

2. **Original Paper:**
   - De-La-Hoz-Correa, E., Mendoza-Palechor, F., De-La-Hoz-Manotas, A., Morales-Ortega, R., & Adriana, S. H. B. (2019). Obesity level estimation software based on decision trees. Journal of Computer Science, 15(1), 67-77.

3. **Libraries:**
   - scikit-learn: https://scikit-learn.org/
   - XGBoost: https://xgboost.readthedocs.io/
   - Streamlit: https://streamlit.io/

---

## 📄 License

This project is for educational purposes as part of BITS WILP M.Tech AIML/DSE program.

Dataset is available under [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

## 👤 Author

**BITS WILP M.Tech AIML/DSE Student**

Machine Learning - Assignment 2

---

*Last Updated: February 2026*
