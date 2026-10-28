# ===================================================================
# Step # 1: Importing libraries
# ===================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ===================================================================
# Step # 2: Load the Dataset
# ===================================================================
"""
The data will be imported from the Scikit library. The first step will be to load
the dataset and convert it into a dataframe and convert it into a csv file.
"""
housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.to_csv("california_housing_data.csv", index=False)

# Since we have the dataframe in CSV file we can also easily load it.
"""
df = pd.read_csv("california_housing_data.csv")
"""

# ===================================================================
#           EDA Explaintory Data Analysis Step-3 to Step-9
# ===================================================================

# ===================================================================
# Step # 3: Inspect the Dataset
# ===================================================================

print("\n First Five Rows of the DataFrame:")
print(df.head())

print("\n Last Five Rows of the DataFrame:")
print(df.tail())

print("\n Dataset Shape:")
print(df.shape)

print("\n Column Names:")
print(df.columns.tolist())

print("\n Check Datatype and non-missing values:")
df.info()

print("\n Summary Statistics for numerical and categorical")
print(df.describe(include="all").T)

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Percentage of Missing Values")
print((df.isnull().mean() * 100).round(2))

print("\n Check Duplicated Rows")
print(df.duplicated().sum())

print("\n Display only Columns with Missing Values")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ===================================================================
# Step # 4: Understanding Target Variable
# ===================================================================

print(df["MedHouseVal"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MedHouseVal", bins=40, kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value(units of $100,000)")
plt.ylabel("Frequency")
plt.show()

# ===================================================================
# Step # 5: Understanding Feature Distributions
# ===================================================================

# Since I need to plot all the features together the best approach would be to use pandas plotting


df.drop(columns=["MedHouseVal"]).hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.show()

# ===================================================================
# Step # 6: Investigating Potential Outliers
# ===================================================================

plt.figure(figsize=(14, 6))
sns.boxplot(data=df.drop(columns=["MedHouseVal"]))
plt.title("Boxplot of Input Features")
plt.xticks(rotation=45)
plt.show()

# ===================================================================
# Step # 7: Examine Correlations
# ===================================================================
correlation_matrix = df.corr()
print(correlation_matrix.round(2))

plt.figure(figsize=(11, 8))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ===================================================================
# Step # 8: Investigating a Feature("MedInc") vs Target("MedHouseVal")
# ===================================================================
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.3)
plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value(unit of $100,000)")
plt.show()

# ======================================================================
# Step # 9: Investigating Two Features("MedInc") vs Target("MedHouseVal")
# ======================================================================

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    df["MedInc"], df["HouseAge"], c=df["MedHouseVal"], cmap="viridis", alpha=0.5
)
plt.colorbar(scatter, label="MedHouseVal")
plt.xlabel("Median Income")
plt.ylabel("House Age")
plt.title("Income and House Age vs House Value")
plt.show()


# ===================================================================
#                           Preprocessing
# ===================================================================

# ===================================================================
# Step # 10: Seperating X features and y target
# ===================================================================

X = df.drop(columns=["MedHouseVal"])  # Features and dropping the target
y = df["MedHouseVal"]  # Target

# ===================================================================
# Step # 11: Training and Testing The Data
# ===================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Training Features:", X_train.shape)
print("\n Testing Features:", X_test.shape)

# ===================================================================
# Step # 12: Identify Numeric Features
# ===================================================================
# Since there are no non-numeric feature categorical features and pipeline won't be needed.

numeric_features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]  # In a list

print("\n Training Columns")
print(X_train.columns.tolist())

print("\n Numeric Features", numeric_features)

# ===================================================================
# Step # 13: Numeric Pipeline
# ===================================================================

numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# ===================================================================
# Step # 14: Combining Preprocessing using ColumnTransformer
# ===================================================================

preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_features)])

# ===================================================================
# Step # 15: Model Pipeline
# ===================================================================

# Linear Regression
lr_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# RandomForest Regressor
rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Ridge
r_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

# ===================================================================
# Step # 16: Comparing Models using Cross-Validation
# ===================================================================

models = {
    "Linear Regression": lr_pipeline,
    "Random Forest Regressor": rf_pipeline,
    "Ridge": r_pipeline,
}
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    rmse_scores = -scores
    cv_results[name] = rmse_scores.mean()

    print(f"\n{name}")
    print("RMSE per Fold:", rmse_scores.round(4))
    print("Mean CV RMSE:", round(rmse_scores.mean(), 4))
    print("Std CV RMSE", round(rmse_scores.std(), 4))

# ===================================================================
# Step # 17: Selecting The Best Baseline Candidate
# ===================================================================

best_model_name = min(cv_results, key=cv_results.get)
best_pipeline = models[best_model_name]

print("\n Best Baseline Candidate:", best_model_name)


# ===================================================================
# Step # 18: Define Hyperparameter Search Space
# ===================================================================

if best_model_name == "Random Forest Regressor":
    param_grid = {
        "regressor__n_estimators": [100, 300],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_leaf": [1, 3],
    }
elif best_model_name == "Ridge":
    param_grid = {
        "regressor__alpha": [0.01, 0.1, 1, 10, 100],
    }
else:  # Linear Regression
    param_grid = {}


# ===================================================================
# Step # 19: Tune Hyperparameter using GridSearchCV
# ===================================================================

grid_search = GridSearchCV(
    estimator=best_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    refit=True,
)

# Model Training
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n Best Hyperparameters:")
print(grid_search.best_params_)

print("\n Best Cross-Validation RMSE:")
print(round(-grid_search.best_score_, 4))


# ===================================================================
# Step # 20: Evaluate on Test Dataset
# ===================================================================

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Final Test Results are:")

print("R2 Results:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error 'MAE':", round(mean_absolute_error(y_test, y_pred), 4))
print("Mean Squared Error 'MSE':", round(mean_squared_error(y_test, y_pred), 4))
print("Root Means Squared Error 'RMSE':", round(rmse, 4))
