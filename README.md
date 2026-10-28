# California Housing Price Prediction — Regression Project

## Project Overview

This project uses the California Housing dataset to practise a complete machine learning regression workflow. The objective is to predict median house values using housing-related features.

This project is part of my ongoing journey towards becoming a Machine Learning / AI Engineer, with a focus on developing practical skills using Python and Scikit-learn.

## Dataset

- **Dataset:** California Housing
- **Source:** Scikit-learn (`fetch_california_housing`)
- **Problem Type:** Supervised Learning — Regression
- **Target Variable:** `MedHouseVal`
- **Number of Rows:** 20,640
- **Number of Input Features:** 8

The target variable represents median house values in units of $100,000.

## Project Workflow

1. Load and inspect the dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Analyse feature distributions and potential outliers.
4. Explore correlations between features and the target variable.
5. Separate the input features (`X`) and target (`y`).
6. Split the data into training and testing sets.
7. Build preprocessing pipelines using `SimpleImputer` and `StandardScaler`.
8. Combine preprocessing and regression models using Scikit-learn pipelines.
9. Compare models using cross-validation.
10. Select a baseline model and perform hyperparameter tuning where applicable.
11. Evaluate the selected model on the test set.

## Machine Learning Models

The project compares the following regression models:

- Linear Regression
- Random Forest Regressor
- Ridge Regression

## Evaluation Metrics

The models are evaluated using:

- **R² Score:** Measures how well the model explains variation in the target.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target's original units.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run the Project

### 1. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Python script

```bash
python california_housing_regression.py
```

The dataset is loaded through Scikit-learn. An internet connection may be required for the initial download.

## Key Learning Objectives

- Understand the end-to-end regression workflow.
- Practise exploratory data analysis.
- Apply preprocessing using pipelines.
- Compare models using cross-validation.
- Understand hyperparameter tuning.
- Evaluate model performance on unseen data.

## Project Status

This project is part of my ongoing machine learning portfolio. I will continue building and improving practical projects as I develop my skills.

## Author

Machine Learning / AI Engineering Learner
